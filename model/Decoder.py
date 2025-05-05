import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SceneDecoder(nn.Module):
    """
    Coordinate-conditioned cross-attention decoder. Predicts point cloud from memory tokens.

    Args:
        num_points (int): Number of points to reconstruct (N_pts).
        memory_dim (int): Dimension of input memory tokens (D from encoder).
        n_heads (int): Number of attention heads for cross-attention.
        num_fourier_bands (int): Number of frequency bands for positional encoding.
        ffn_scale (int): Scale factor for the FFN hidden dimension.
        coord_range (tuple): Range for initializing query coordinates. Default: (-1.0, 1.0).
    """
    def __init__(
        self,
        num_points: int = 256,
        memory_dim: int = 256,
        n_heads: int = 8,
        num_fourier_bands: int = 6,
        ffn_scale: int = 4,
        coord_range: tuple = (-1.0, 1.0)
    ):
        super().__init__()
        self.num_points = num_points
        self.d_model = memory_dim # Internal dim should match memory dim
        self.num_fourier_bands = num_fourier_bands

        # --- learnable 3-D query coordinates ---------------------------------
        coord_min, coord_max = coord_range
        self.coords = nn.Parameter(
            torch.rand(num_points, 3) * (coord_max - coord_min) + coord_min
        ) # Initialize in specified range

        # --- positional / Fourier encoding -----------------------------------
        # Input dim for projection: 3 (xyz) + 3 (dims) * 2 (sin/cos) * bands
        pe_dim = 3 + 3 * 2 * num_fourier_bands
        self.input_proj = nn.Linear(pe_dim, self.d_model)

        # --- single cross-attention + FFN block ------------------------------
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=n_heads,
            batch_first=True, # Expects (B, Seq, Dim)
        )
        self.norm1 = nn.LayerNorm(self.d_model)

        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, ffn_scale * self.d_model),
            nn.GELU(),
            nn.Linear(ffn_scale * self.d_model, self.d_model),
        )
        self.norm2 = nn.LayerNorm(self.d_model)

        # --- final projection to xyz -----------------------------------------
        self.out_proj = nn.Linear(self.d_model, 3)



    # -------------------------------------------------------------------------
    def _fourier_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies positional encoding using Fourier features.
        x : (B, N_pts, 3) -> (B, N_pts, 3 + 3*2*num_bands)
        """
        B, N, _ = x.shape
        device = x.device
        # (num_bands,)
        bands = torch.linspace(
            1.0,
            2.0 ** (self.num_fourier_bands - 1),
            self.num_fourier_bands,
            device=device,
        )
        # (B, N_pts, 3, 1) * (num_bands,) -> (B, N_pts, 3, num_bands)
        xb = x.unsqueeze(-1) * bands
        # sin/cos -> (B, N_pts, 3, 2*num_bands)
        enc = torch.cat(
            (torch.sin(math.pi * xb), torch.cos(math.pi * xb)), dim=-1
        )
        # (B, N_pts, 3*2*num_bands)
        enc = enc.flatten(start_dim=2)
        # (B, N_pts, 3 + 3*2*num_bands)
        return torch.cat((x, enc), dim=-1)

    # -------------------------------------------------------------------------
    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """
        Decodes memory tokens into a point cloud.

        Args:
            memory (torch.Tensor): Encoder output tokens (B, K*T, D).

        Returns:
            torch.Tensor: Reconstructed point cloud (B, N_pts, 3).
        """
        tokens = memory.detach()  # (B, K, D)
        pairwise_dots = torch.matmul(tokens, tokens.transpose(1, 2))  # (B, K, K)
        std_across_rows = pairwise_dots.std(dim=-1).mean()
        print("Encoder token similarity std:", std_across_rows.item())

        B = memory.size(0)
        D = memory.size(2)
        assert D == self.d_model, f"Memory dim mismatch: expected {self.d_model}, got {D}"

        # 1. broadcast learnable queries to batch size
        # (1, N_pts, 3) -> (B, N_pts, 3)
        q_coords = self.coords.unsqueeze(0).expand(B, -1, -1)

        # 2. encode coordinates + linear projection to d_model
        # (B, N_pts, 3) -> (B, N_pts, pe_dim) -> (B, N_pts, D)
        q = self.input_proj(self._fourier_encode(q_coords))

        # 3. cross-attention (queries = coord tokens, keys/values = memory)
        # attn_out: (B, N_pts, D)
        attn_out, attn_weights = self.cross_attn(query=q, key=memory, value=memory, need_weights=True, average_attn_weights=False)
        x = self.norm1(q + attn_out) # Add & Norm

        # 4. feed-forward refinement
        x = self.norm2(x + self.ffn(x)) # Add & Norm

        # 5. predict xyz offset (or absolute position, depending on interpretation)
        # It's common to predict offsets from the initial query coords
        # delta_coords = self.out_proj(x) # (B, N_pts, 3)
        # return q_coords + delta_coords # Add predicted offset
        # Or predict absolute coordinates directly:
        return self.out_proj(x), attn_weights # (B, N_pts, 3)
