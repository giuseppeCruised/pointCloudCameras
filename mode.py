import torch
import math
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import open3d as o3d
import os
import json
import numpy as np

from util import farthest_point_sampling, knn_group

from timm import create_model

class CameraPointCloudAutoEncoder(nn.Module):
    def __init__(self, N, D=256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(D, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, N * 3)
        )

        self.scene_fusion = SceneTokenFusion()
        self.view_encoder = ViewEncoder()
        self.projector = LearnableCameraProjector()

    def forward(self, points): # points: (B, N, 3)
        projections = self.projector(points) # (B, C, N, 2)
        encoded_views = self.view_encoder(projections) # (B, C, dim)
        fused_scene = self.scene_fusion(encoded_views) # (B, dim)
        rec = self.decoder(fused_scene) # (B, N, 3)
        return rec


class SceneTokenFusion(nn.Module):
    def __init__(
        self,
        D_in: int = 256,
        D_model: int = 256,
        pose_dim: int = 6,
        n_heads: int = 8,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # --- 1.  Pose → embedding  -------------------------------------------------
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, D_in),
        )

        # --- 2.  Optional projection to common model dim --------------------------
        self.cam_proj = (
            nn.Identity() if D_in == D_model else nn.Linear(D_in, D_model, bias=False)
        )

        # --- 3.  Learnable scene token  -------------------------------------------
        self.scene_token = nn.Parameter(torch.randn(1, 1, D_model) / D_model**0.5)

        # --- 4.  Multi‑head cross‑attention  --------------------------------------
        self.attn = nn.MultiheadAttention(
            embed_dim=D_model, num_heads=n_heads, batch_first=True
        )

    def forward(self, cam_feat: torch.Tensor, cam_pose: torch.Tensor) -> torch.Tensor:
        B, K, _ = cam_feat.shape

        # 1) add pose‑conditioned bias  (geometry awareness)
        pose_emb = self.pose_mlp(cam_pose)            # (B, K, D_in)
        feat     = cam_feat + pose_emb                # (B, K, D_in)

        # 2) project to attention dimension
        feat = self.cam_proj(feat)                    # (B, K, D_model)

        # 3) replicate the learnable query for the minibatch
        q = self.scene_token.expand(B, -1, -1)        # (B, 1, D_model)

        # 4) cross‑attention   (query = scene token, key/value = cameras)
        z, _ = self.attn(q, feat, feat)               # (B, 1, D_model)

        return z.squeeze(1)                           # (B, D_model)


class ViewEncoder(nn.Module):
    def __init__(self, d_token=256):
        super().__init__()
        self.backbone = create_model(
            'mobilevit_s', pretrained=True, features_only=True
        )                       # returns 5 feature maps
        self.stage_id = 3       # pick 0‑4; 3 ⇒ (BK, 64, 14, 14)

        C = 64                  # channels of chosen stage
        self.proj = nn.Linear(C, d_token, bias=False)

        # single learnable query that will pull info out of the view
        self.query = nn.Parameter(torch.randn(1, 1, d_token) / d_token**0.5)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_token, num_heads=8, batch_first=True
        )

    def forward(self, x):           # x: (B, K, H, W)  1‑channel raster
        B, K, H, W = x.shape
        x = x.unsqueeze(2).repeat(1, 1, 3, 1, 1)      # → fake RGB
        x = x.view(B * K, 3, H, W)

        feats = self.backbone(x)[self.stage_id]        # (BK, C, H', W')
        BK, C, Hs, Ws = feats.shape
        S = Hs * Ws

        feats = feats.view(BK, C, S).permute(0, 2, 1)  # (BK, S, C)
        feats = self.proj(feats)                       # (BK, S, D)

        # tile query to batch‑size
        q = self.query.expand(BK, -1, -1)              # (BK, 1, D)
        z, _ = self.attn(q, feats, feats)              # (BK, 1, D)
        z = z.squeeze(1).view(B, K, -1)                # (B, K, D)

        return z                                       # latent per view


class LearnableCameraProjector(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K

        # Learnable rotation vectors (Lie algebra, axis-angle)
        self.rotvecs = nn.Parameter(torch.randn(K, 3) * 0.01)

        # Learnable translations (camera positions)
        self.trans = nn.Parameter(torch.randn(K, 3) * 0.01)

        # Learnable focal lengths (shared fx = fy)
        self.focal = nn.Parameter(torch.ones(K) * 1.5)


    def forward(self, points):  # points: (B, N, 3)
        B, N, _ = points.shape
        device = points.device

        # Expand points to (B, K, N, 3)
        points = points.unsqueeze(1).expand(B, self.K, N, 3)  # (B, K, N, 3)

        # Rotation matrices: (K, 3, 3)
        R = self._so3_exp_map(self.rotvecs)  # (K, 3, 3)

        # Transform points: apply (R * (x - t))
        R = R.unsqueeze(0)           # (1, K, 3, 3)
        t = self.trans.unsqueeze(0) # (1, K, 3)
        points_local = torch.matmul(R, (points - t.unsqueeze(2)).unsqueeze(-1))  # (B, K, N, 3, 1)
        points_local = points_local.squeeze(-1)  # (B, K, N, 3)

        # Apply perspective projection
        x = points_local[..., 0]
        y = points_local[..., 1]
        z = points_local[..., 2].clamp(min=1e-4)  # prevent divide-by-zero

        f = self.focal.view(1, self.K, 1)  # (1, K, 1)
        x_proj = f * (x / z)
        y_proj = f * (y / z)

        projected = torch.stack([x_proj, y_proj], dim=-1)  # (B, K, N, 2)
        return projected

    def _so3_exp_map(self, rotvecs):
        """
        Convert axis-angle vectors (K, 3) to rotation matrices (K, 3, 3)
        using Rodrigues' rotation formula.
        """
        theta = rotvecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        k = rotvecs / theta

        K = torch.zeros(rotvecs.shape[0], 3, 3, device=rotvecs.device)
        K[:, 0, 1] = -k[:, 2]
        K[:, 0, 2] =  k[:, 1]
        K[:, 1, 0] =  k[:, 2]
        K[:, 1, 2] = -k[:, 0]
        K[:, 2, 0] = -k[:, 1]
        K[:, 2, 1] =  k[:, 0]

        I = torch.eye(3, device=rotvecs.device).unsqueeze(0)
        R = I + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta).unsqueeze(-1)) * torch.matmul(K, K)
        return R





class FourierPosEnc(nn.Module):
    def __init__(self, dim_out, num_freqs=10):
        super().__init__()
        self.num_freqs = num_freqs
        self.linear = nn.Linear(num_freqs * 6, dim_out)

    def forward(self, xyz):  # (B, P, 3)
        B, P, _ = xyz.shape
        device = xyz.device

        # Frequencies: 2^i * π
        freqs = 2 ** torch.arange(self.num_freqs, device=device).float() * math.pi  # (F,)
        xyz = xyz.unsqueeze(-1) * freqs            # (B, P, 3, F)
        sin = torch.sin(xyz)
        cos = torch.cos(xyz)
        pe  = torch.cat([sin, cos], dim=-1)        # (B, P, 3, 2F)
        pe  = pe.view(B, P, -1)                    # (B, P, 6F)
        return self.linear(pe)                     # (B, P, dim_out)

class PatchTransformer(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):  # x: (B, P, dim)
        # Self-attention block
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual

        # MLP block
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + residual
        return x


class MaskedReconstructionModule(nn.Module):
    def __init__(self, dim, patch_size):
        super().__init__()
        self.dim = dim                         # <‑‑ store it

        # Patch encoder (PointNet‑style)
        self.point_net = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim))

        # Shared positional encoder (center xyz → dim)
        self.pos_enc_layer = FourierPosEnc(dim, num_freqs=10)

        # Very small “encoder” & “decoder”
        self.feature_encoder = nn.Sequential(
            PatchTransformer(dim), PatchTransformer(dim),
            PatchTransformer(dim), PatchTransformer(dim),
            PatchTransformer(dim), PatchTransformer(dim),
            PatchTransformer(dim), PatchTransformer(dim)
        )
        self.decoder = nn.Sequential(
            PatchTransformer(dim), PatchTransformer(dim),
            PatchTransformer(dim), PatchTransformer(dim)
        )

        # MLP to turn decoder token → k×3 points (for reconstruction)
        self.patch_decoder = nn.Linear(dim, patch_size * 3)   # assuming k = 16

    def forward(self, masked_patches, visible_patches):
        # ---- encode visible patches ---------------------------------
        vis_centers = visible_patches.mean(dim=2, keepdim=True)      # (B,P,1,3)
        vis_centered = visible_patches - vis_centers                 # center
        B, P_vis, k, _ = vis_centered.shape
        feat = self.point_net(vis_centered).max(dim=2)[0]            # (B,P_vis,D)

        pos_vis = self.pos_enc_layer(vis_centers.squeeze(2))         # (B,P_vis,D)
        vis_tokens = feat + pos_vis

        enc_out = self.feature_encoder(vis_tokens)                   # (B,P_vis,D)

        # ---- build masked tokens ------------------------------------
        m_centers = masked_patches.mean(dim=2).detach()              # (B,P_mask,3)
        pos_mask = self.pos_enc_layer(m_centers)                     # (B,P_mask,D)
        mask_tok = self.mask_token.expand(B, m_centers.size(1), self.dim)
        mask_tokens = mask_tok + pos_mask                            # (B,P_mask,D)

        # ---- decoder ------------------------------------------------
        dec_in   = torch.cat([enc_out, mask_tokens], dim=1)          # (B,P_vis+P_mask,D)
        dec_out  = self.decoder(dec_in)                              # same shape
        masked_decoded = dec_out[:, -m_centers.size(1):, :]          # (B,P_mask,D)

        # ---- reconstruct geometry -----------------------------------
        pred_patch = self.patch_decoder(masked_decoded)              # (B,P_mask,16*3)
        pred_patch = pred_patch.view(B, m_centers.size(1), k, 3)     # (B,P_mask,k,3)
        pred_patch = pred_patch + m_centers.unsqueeze(2)             # add center back

        return pred_patch
