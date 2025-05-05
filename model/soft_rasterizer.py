import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SoftRasterizer(nn.Module):
    """
    Differentiable rasterization using Gaussian splatting.
    Converts projected 2D points into heatmaps.

    Args:
        height (int): Height of the output raster image.
        width (int): Width of the output raster image.
        sigma (float): Standard deviation for the Gaussian splat. Default: 1.0.
        inv_depth_weighting (bool): Whether to weight contribution by inverse depth. Default: False.
    """
    def __init__(self, height: int, width: int, sigma: float = 1.0, inv_depth_weighting: bool = False):
        super().__init__()
        self.height = height
        self.width = width
        self.sigma = sigma
        self.inv_depth_weighting = inv_depth_weighting

        # Create pixel grid buffers (persistent but not parameters)
        y_grid = torch.arange(height).float()
        x_grid = torch.arange(width).float()
        # (1, 1, 1, H, 1)
        self.register_buffer("y_grid", y_grid.view(1, 1, 1, height, 1), persistent=False)
        # (1, 1, 1, 1, W)
        self.register_buffer("x_grid", x_grid.view(1, 1, 1, 1, width), persistent=False)

    def forward(self, points_2d: torch.Tensor, points_z: torch.Tensor = None) -> torch.Tensor:
        """
        Rasterizes projected points.

        Args:
            points_2d (torch.Tensor): Projected 2D points (B, K', N_pts, 2) in pixel coordinates.
            points_z (torch.Tensor, optional): Z-coordinates in camera space (B, K', N_pts).
                                                Used for inverse depth weighting if enabled.

        Returns:
            torch.Tensor: Rasterized heatmaps (B, K', H, W).
        """
        B, K_prime, N_pts, _ = points_2d.shape
        device = points_2d.device

        if self.inv_depth_weighting and points_z is None:
             raise ValueError("points_z must be provided if inv_depth_weighting is True")
        if self.inv_depth_weighting and points_z.shape != (B, K_prime, N_pts):
             raise ValueError(f"points_z shape mismatch: expected {(B, K_prime, N_pts)}, got {points_z.shape}")


        # Ensure grids are on the correct device
        y_grid = self.y_grid.to(device) # (1, 1, 1, H, 1)
        x_grid = self.x_grid.to(device) # (1, 1, 1, 1, W)

        # Extract x, y and reshape for broadcasting
        x = points_2d[..., 0].view(B, K_prime, N_pts, 1, 1) # (B, K', N_pts, 1, 1)
        y = points_2d[..., 1].view(B, K_prime, N_pts, 1, 1) # (B, K', N_pts, 1, 1)

        # Compute squared distances to each pixel center
        # broadcast (B, K', N_pts, 1, 1) with (1, 1, 1, 1, W) -> (B, K', N_pts, 1, W)
        # broadcast (B, K', N_pts, 1, 1) with (1, 1, 1, H, 1) -> (B, K', N_pts, H, 1)
        # sum -> (B, K', N_pts, H, W)
        dist_sq = (x_grid - x)**2 + (y_grid - y)**2

        # Apply Gaussian kernel weight = exp(-dist_sq / (2 * sigma^2))
        # (B, K', N_pts, H, W)
        weights = torch.exp(-dist_sq / (2 * (self.sigma**2 + 1e-8)))

        # Optional inverse depth weighting (closer points contribute more)
        if self.inv_depth_weighting:
            inv_z = (1.0 / points_z.clamp(min=1e-5)).view(B, K_prime, N_pts, 1, 1) # (B, K', N_pts, 1, 1)
            weights = weights * inv_z # Weight by 1/z

        # Sum weights over the N_pts points for each pixel
        raster_heatmap = weights.sum(dim=2) # (B, K', H, W)

        # Normalize heatmap (optional, depends on loss function)
        # Example normalization: scale max value in each view to 1
        # max_vals = raster_heatmap.flatten(2).max(dim=2)[0].view(B, K_prime, 1, 1).clamp(min=1e-8)
        # raster_heatmap = raster_heatmap / max_vals

        return raster_heatmap
