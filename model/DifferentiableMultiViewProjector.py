import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DifferentiableMultiViewProjector(nn.Module):
    """
    Projects a 3D point cloud onto multiple virtual camera planes using
    learnable camera intrinsics (focal length) and extrinsics (rotation, translation).
    Outputs coordinates suitable for rasterization (e.g., scaled NDC or pixel coordinates).

    Args:
        num_views (int): The number of virtual camera views (K').
        image_height (int): Target image height for coordinate scaling.
        image_width (int): Target image width for coordinate scaling.
        obj_radius (float): Estimated object radius for camera init.
        cam_distance (float): Camera distance from origin for init.
        cluster_std_deg (float): Std deviation for camera clustering init.
        device (torch.device or str): Device for camera parameters.
    """
    def __init__(self, num_views: int, image_height: int, image_width: int,
                                  obj_radius: float = 0.9, cam_distance: float = 1.6,
                                  cluster_std_deg: float = 50.0, device: torch.device = torch.device('cpu')):
        super().__init__()
        if not isinstance(num_views, int) or num_views <= 0:
                raise ValueError("num_views must be a positive integer")
            self.num_views = num_views
        self.image_height = image_height
        self.image_width = image_width

        # Initialize camera parameters on the specified device
        rot_vecs, centres, f_init = init_clustered_cameras(
            num_views,
            obj_radius=obj_radius,
            cam_distance=cam_distance,
            cluster_std_deg=cluster_std_deg,
            device=device
        )

        self.rotation_vectors = nn.Parameter(rot_vecs)  # (K', 3)
        self.translations       = nn.Parameter(centres)   # (K', 3)
        self.focal_lengths      = nn.Parameter(f_init)    # (K',)

    def _so3_exponential_map(self, log_rotations: torch.Tensor) -> torch.Tensor:
        """Converts axis-angle vectors to rotation matrices using Rodrigues' formula."""
        K = log_rotations.shape[0]
        device = log_rotations.device
        theta = torch.norm(log_rotations, dim=1, keepdim=True).clamp(min=1e-8)
        k = log_rotations / theta
        K_ss = torch.zeros(K, 3, 3, device=device)
        K_ss[:, 0, 1], K_ss[:, 0, 2] = -k[:, 2], k[:, 1]
        K_ss[:, 1, 0], K_ss[:, 1, 2] = k[:, 2], -k[:, 0]
        K_ss[:, 2, 0], K_ss[:, 2, 1] = -k[:, 1], k[:, 0]
        I = torch.eye(3, device=device).unsqueeze(0).expand(K, -1, -1)
        sin_theta, cos_theta = torch.sin(theta).unsqueeze(-1), torch.cos(theta).unsqueeze(-1)
        R = I + sin_theta * K_ss + (1 - cos_theta) * torch.matmul(K_ss, K_ss)
        return R

    def get_camera_poses_6d(self) -> torch.Tensor:
        """Returns learned camera poses as (K', 6) tensor (rot_vec, trans)."""
        return torch.cat([self.rotation_vectors, self.translations], dim=1)

    def forward(self, points_world: torch.Tensor) -> torch.Tensor:
        """
        Projects points and scales them to pixel coordinates for rasterization.

        Args:
            points_world (torch.Tensor): Input point cloud (B, N_pts, 3).

        Returns:
            torch.Tensor: Projected 2D coordinates in pixel space (B, K', N_pts, 2),
                          where origin (0,0) is top-left.
            torch.Tensor: Z-coordinates in camera space (B, K', N_pts). Useful for depth checks.
        """
        B, N_pts, _ = points_world.shape
        # (B, 1, N_pts, 3) -> (B, K', N_pts, 3)
        points_expanded = points_world.unsqueeze(1).expand(B, self.num_views, N_pts, 3)

        # (K', 3, 3) -> (1, K', 1, 3, 3)
        R = self._so3_exponential_map(self.rotation_vectors)
        R_expanded = R.view(1, self.num_views, 1, 3, 3)
        # (K', 3) -> (1, K', 1, 3)
        t_expanded = self.translations.view(1, self.num_views, 1, 3)

        # Transform points to camera coordinates
        # (B, K', N_pts, 3, 1) = matmul( (1, K', 1, 3, 3), ((B, K', N_pts, 3) - (1, K', 1, 3)).unsqueeze(-1) )
        points_camera_coords = torch.matmul(R_expanded, (points_expanded - t_expanded).unsqueeze(-1)).squeeze(-1) # (B, K', N_pts, 3)

        x_cam, y_cam = points_camera_coords[..., 0], points_camera_coords[..., 1]
        z_cam = points_camera_coords[..., 2].clamp(min=1e-5) # Avoid division by zero

        f = self.focal_lengths.view(1, self.num_views, 1) # (1, K', 1)

        # Project to normalized plane (still centered at 0)
        x_proj_norm = f * (x_cam / z_cam)
        y_proj_norm = f * (y_cam / z_cam)

        # Convert to pixel coordinates (origin top-left)
        # Assumes principal point is at image center (W/2, H/2)
        x_pixel = x_proj_norm * (self.image_width / 2.0) + (self.image_width / 2.0)
        y_pixel = y_proj_norm * (self.image_height / 2.0) + (self.image_height / 2.0)

        projected_pixels = torch.stack([x_pixel, y_pixel], dim=-1) # (B, K', N_pts, 2)

        return projected_pixels, z_cam # Return Z for potential depth buffering/masking
