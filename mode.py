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

class Patchify3D(torch.nn.Module):
    def __init__(self, num_patches, patch_size):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size

    def forward(self, x):  # x: (B, N, 3)
        centers = farthest_point_sampling(x, self.num_patches)      # (B, P)
        patches = knn_group(x, centers, self.patch_size)            # (B, P, k, 3)
        return patches


# class PatchTransformer(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#
#         self.dim = dim
#
#         self.to_q = nn.Linear(dim, dim)
#         self.to_k = nn.Linear(dim, dim)
#         self.to_v = nn.Linear(dim, dim)
#
#     def forward(self, tokens):  # tokens: (B, P, dim)
#         Q = self.to_q(tokens)
#         K = self.to_k(tokens)
#         V = self.to_v(tokens)
#
#         attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / self.dim ** 0.5
#         attn_weights = F.softmax(attn_scores, dim=-1)
#
#         out = torch.matmul(attn_weights, V)
#         return out

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
