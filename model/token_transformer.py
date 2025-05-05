import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class TransformerEncoderLayerWithAttn(nn.Module):
    def __init__(self, dim, heads, ff_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim)
        )

    def forward(self, x):
        attn_out, attn_weights = self.self_attn(x, x, x, need_weights=True, average_attn_weights=False)
        self.last_attn_weights = attn_weights  # (B, H, T, T)
        x = self.norm1(x+attn_out)
        x = self.norm2(x+self.ff(x))
        return x, attn_weights

class TokenTransformer(nn.Module):
    """ Standard Transformer Encoder """
    def __init__(self, d_model=256, nhead=8, num_layers=2, dim_feedforward=256, dropout=0):
        super().__init__()
        self.encoder = nn.ModuleList([
            TransformerEncoderLayerWithAttn(
                dim=d_model,
                ff_dim=dim_feedforward,
                heads=nhead
            ) for _ in range(num_layers)
        ])

        self.last_attn_weights = None

    def forward(self, x): # x: (B, Seq, Dim) e.g. (B, K*T, D)
        for layer in self.encoder:
            x, attn_weights = layer(x)

        self.last_attn_weights = attn_weights

        return x
