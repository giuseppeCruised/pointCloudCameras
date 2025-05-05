import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.simple_view_encoder import SimpleViewEncoder
from model.token_transformer import TokenTransformer


class ViewAwareEncoder(nn.Module):
    """
    Encodes multi-view images and poses into fused memory tokens.

    Args:
        img_emb_dim (int): Dimension of image features from CNN backbone.
        pose_emb_dim (int): Hidden dimension for pose embedding MLP.
        tokens_per_view (int): Number of tokens generated per view by the backbone.
        transformer_nhead (int): Number of heads for the TokenTransformer.
        transformer_layers (int): Number of layers for the TokenTransformer.
        transformer_ffn_dim (int): Feedforward dim for the TokenTransformer.
    """
    def __init__(self, img_emb_dim: int = 256, pose_emb_dim: int = 128, tokens_per_view: int = 4,
                 transformer_nhead: int = 8, transformer_layers: int = 4, transformer_ffn_dim: int = 256,
                 tokens_grid_size: int = 2, # Pass down to SimpleViewEncoder
                 intermediate_cnn_channels: tuple = (32, 64, 128) # Pass down to SimpleViewEncoder
                ):
        super().__init__()
        self.tokens_per_view = tokens_per_view

        # Simple CNN backbone
        self.backbone = SimpleViewEncoder(
            output_dim=img_emb_dim,
            tokens_grid_size=tokens_grid_size,
            intermediate_channels=intermediate_cnn_channels
            )
        assert self.tokens_per_view == self.backbone.num_tokens, \
               f"tokens_per_view ({tokens_per_view}) must match backbone num_tokens ({self.backbone.num_tokens})"

        # MLP to embed 6D poses (rot_vec + trans + foc)
        self.pose_proj = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, img_emb_dim)
        )

        # MLP to embed 6D poses (rot_vec + trans)
        self.pose_scale = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, img_emb_dim)
        )

        self.norm = nn.LayerNorm(img_emb_dim)

        # Transformer to fuse tokens across views
        self.transformer_encoder = TokenTransformer(
            d_model=img_emb_dim,
            nhead=4,
            num_layers=1,
            dim_feedforward=transformer_ffn_dim
        )

    def forward(self, x, poses): # x: (B, K, 3, H, W), poses: (B, K, 6)
        B, K, C_in, H, W = x.shape
        _, _, P = poses.shape
        assert P == 6, f"Expected poses dim 6, got {P}"

        # 1. Encode images -> (B, K, T, D)
        tokens = self.backbone(x)
        B, K, T, D = tokens.shape # T = tokens_per_view, D = img_emb_dim

        pose_token_input = poses.unsqueeze(2).expand(-1, -1, self.tokens_per_view, -1)

        # 2. Encode poses -> (B, K, T*D) -> (B, K, T, D)
        pose_enc = self.pose_proj(pose_token_input)      # (B, K, T*D)
        pose_enc = pose_enc.view(B, K, T, D)  # (B, K, T, D)

        pose_scales = self.pose_scale(pose_token_input)
        pose_scales = pose_scales.view(B, K, T, D)  # (B, K, T, D)

        # 3. Add pose encoding to image tokens
        tokens = pose_scales * tokens + pose_enc

        tokens = self.norm(tokens)

        # 4. Flatten tokens and fuse with Transformer
        # (B, K, T, D) -> (B, K*T, D)
        tokens = tokens.reshape(B, K * T, D)
        # (B, K*T, D)
        encoded_memory = self.transformer_encoder(tokens)

        return encoded_memory # (B, K*T, D)
