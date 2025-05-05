import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleViewEncoder(nn.Module):
    """
    A drop-in Vision-Transformer (ViT) encoder that **keeps exactly the same
    public interface** as your previous CNN version, so you can paste this file
    in place of the old one and nothing else in your code has to change.

    ---------------------------------------------------------------------------
    Architectural sketch
    ---------------------------------------------------------------------------
    • Patch embedding  : Conv2d(3 → C, kernel = patch_size, stride = patch_size)
    • Positional enc.  : 2-D fixed sin-cos, generated on-the-fly (no size limit)
    • Transformer body : PyTorch nn.TransformerEncoder (depth × MH-attention)
    • Token reduction  : AdaptiveAvgPool1d → exactly G×G tokens (G=tokens_grid_size)

    ---------------------------------------------------------------------------
    Notes on sparsity
    ---------------------------------------------------------------------------
    * Even an all-zero patch becomes a *token*; the Transformer can learn to
      ignore it instead of being structurally skipped, alleviating the gradient
      starvation that hurts CNNs on ultra-sparse images.
    * Because we pool **after** the encoder, the model first reasons over the
      full set of patches, then compresses to your required token grid.

    ---------------------------------------------------------------------------
    Args (unchanged w.r.t. your CNN)
    ---------------------------------------------------------------------------
    output_dim          (int)  – embedding dimension C of every token       (256)
    tokens_grid_size    (int)  – grid side G  ⇒  #tokens = G×G               (2)
    intermediate_channels(tuple) – kept for API compatibility (unused)  ((32,64,128))
    num_groups          (int)  – kept for API compatibility (unused)        (8)
    negative_slope      (float)– kept for API compatibility (unused)       (0.10)

    ---------------------------------------------------------------------------
    New optional args   (all have safe defaults, so old calls still work)
    ---------------------------------------------------------------------------
    patch_size          (int)  – square patch edge length                    (16)
    depth               (int)  – number of Transformer layers                 (4)
    num_heads           (int)  – attention heads per layer                    (8)
    mlp_ratio           (float)– width multiplier for feed-forward net        (4.0)
    drop_rate           (float)– dropout inside Transformer                   (0.0)
    """

    def __init__(
        self,
        output_dim: int = 256,
        tokens_grid_size: int = 2,
        intermediate_channels: tuple = (32, 64, 128),   # unused, kept for API
        num_groups: int = 8,                            # unused, kept for API
        negative_slope: float = 0.10,                   # unused, kept for API
        *,
        patch_size: int = 16,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
    ):
        super().__init__()

        # ------------------------------------------------------------------ #
        # Public attributes that *downstream code* might depend on
        # ------------------------------------------------------------------ #
        self.num_tokens: int = tokens_grid_size * tokens_grid_size   # G×G
        self.embed_dim: int  = output_dim                            # ≡ C

        # ----------------------------- layers ----------------------------- #
        # Patch embedding (Conv2d for speed; equivalent to linear on patches)
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(
            3,
            self.embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=int(self.embed_dim * mlp_ratio),
            dropout=drop_rate,
            activation="gelu",
            batch_first=True,        # (B, S, D) layout
            norm_first=True          # Pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Dropout just before encoder (token dropout, optional)
        self.token_dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        # --- an adaptive pool that squeezes sequence length to G×G tokens
        self._G = tokens_grid_size                    # store for forward
        # (No nn.Module needed; we use F.adaptive_avg_pool1d)

        # Positional-embedding cache (grid_size → tensor).  Avoids regen.
        self._pos_cache = {}

        # Weight initialization (following ViT style)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    # ---------------------------------------------------------------------- #
    #                     forward:  (B, K, 3, H, W) ─► (B, K, T, C)          #
    # ---------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : tensor of shape (B, K, 3, H, W)

        returns
        -------
        tokens : (B, K, G×G, C)   where C = output_dim
        """
        B, K, C_in, H, W = x.shape
        x = x.view(B * K, C_in, H, W)                      # (B·K, 3, H, W)

        # -------------------- patch embedding -------------------- #
        x = self.patch_embed(x)                            # (B·K, C, H', W')
        H_p, W_p = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)                   # (B·K, N, C)  where N = H'×W'

        # --------------- add 2-D sin-cos positional enc. --------- #
        pos = self._get_2d_sincos_pos_embed(H_p, W_p, device=x.device)  # (1,N,C)
        x = x + pos

        x = self.token_dropout(x)                          # (B·K, N, C)

        # -------------------- Transformer encoder ---------------- #
        x = self.encoder(x)                                # (B·K, N, C)

        # ------------- compress to exactly G×G tokens ------------ #
        # AdaptiveAvgPool1d needs (B, C, N)
        x = x.permute(0, 2, 1)                             # (B·K, C, N)
        x = F.adaptive_avg_pool1d(x, self.num_tokens)      # (B·K, C, T)
        x = x.permute(0, 2, 1).contiguous()                # (B·K, T, C)

        # ---------------- reshape back to (B, K, T, C) ----------- #
        tokens = x.view(B, K, self.num_tokens, self.embed_dim)
        return tokens                                       # (B, K, T, C)

    # ------------------------------------------------------------------ #
    #      Positional-encoding helper  (2-D fixed sin-cos, no limits)    #
    # ------------------------------------------------------------------ #
    def _get_2d_sincos_pos_embed(self, H: int, W: int, device) -> torch.Tensor:
        """
        Return tensor of shape **(1, H*W, C)** where *C == self.embed_dim*,
        using a standard 2-D sin-cos formulation (like ViT/MAE).

        Cached per (H, W) so subsequent calls are free.
        """
        key = (H, W)
        if key in self._pos_cache:
            return self._pos_cache[key]

        C = self.embed_dim
        assert C % 4 == 0, "embed_dim must be divisible by 4"

        # ------------------------------------------------------------------
        # Build 1-D embeddings for y and x with dimension C/2 each
        # (each of those splits into sin+cos internally).
        # ------------------------------------------------------------------
        def _sincos_1d(n_positions: int, dim: int):
            """Return (n_positions, dim)  where dim is even."""
            omega = torch.arange(dim // 2, device=device) / (dim // 2)
            omega = 1.0 / (10000 ** omega)                         # (dim/2,)
            positions = torch.arange(n_positions, device=device).float()  # (n,)
            out = torch.einsum('n,d->nd', positions, omega)               # (n, dim/2)
            return torch.cat([out.sin(), out.cos()], dim=1)               # (n, dim)

        dim_each = C // 2                         # y gets C/2, x gets C/2
        emb_y = _sincos_1d(H, dim_each)           # (H, C/2)
        emb_x = _sincos_1d(W, dim_each)           # (W, C/2)

        # ------------------------------------------------------------------
        # Combine to 2-D grid: for every (y,x) concatenate their embeddings
        # ------------------------------------------------------------------
        emb_y = emb_y.unsqueeze(1).expand(-1, W, -1)   # (H, W, C/2)
        emb_x = emb_x.unsqueeze(0).expand(H, -1, -1)   # (H, W, C/2)
        pos    = torch.cat([emb_y, emb_x], dim=2)      # (H, W, C)
        pos    = pos.reshape(1, H * W, C)              # (1, N, C)

        self._pos_cache[key] = pos
        return pos

    # ------------------------------------------------------------------ #
    #  The following helper is **unused** in ViT but kept so that code   #
    #  referring to SimpleViewEncoder._group_norm still finds it.        #
    # ------------------------------------------------------------------ #
    def _group_norm(self, channels: int) -> nn.GroupNorm:
        """Stub kept for backward compatibility with the old CNN version."""
        return nn.GroupNorm(1, channels)
