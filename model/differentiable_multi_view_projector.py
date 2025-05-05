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


def init_clustered_cameras(num_views: int,
                           obj_radius: float = 1,
                           cam_distance: float = 0.001,
                           cluster_std_deg: float = 10.0,
                           device="cpu"):
    """
    Returns:
      rot_vecs : (N,3)  axis‑angle vectors  (requires_grad=True later)
      trans    : (N,3)  camera centres     (requires_grad=True later)
      f_init   : (N,)   focal lengths
    """
    #
    # # ---- pick a “mean” viewing direction on the sphere -------------
    # mean_dir = torch.randn(3, device=device)
    # mean_dir = mean_dir / mean_dir.norm()
    #
    # # ---- sample N directions near that mean ------------------------
    # std_rad = math.radians(cluster_std_deg)
    # dirs = []
    # for _ in range(num_views):
    #     # tangential Gaussian perturbation
    #     tangent = torch.randn(3, device=device)
    #     tangent -= (tangent @ mean_dir) * mean_dir      # remove radial part
    #     tangent = tangent / tangent.norm() * std_rad
    #     v = mean_dir + tangent
    #     v = v / v.norm()
    #     dirs.append(v)
    # dirs = torch.stack(dirs, dim=0)                     # (N,3)
    #
    # # ---- camera centres -------------------------------------------
    # centres = dirs * cam_distance * obj_radius          # (N,3)
    #
    # # ---- rotations (look‑at) --------------------------------------
    # R_all = torch.stack([look_at_R(c) for c in centres], dim=0)  # (N,3,3)
    # rot_vecs = torch.stack([rotmat_to_axis_angle(R) for R in R_all], dim=0)
    #
    # # ---- focal length initial guess -------------------------------
    # f_init = torch.full((num_views,), 1.0, device=device)
    # ---- fixed predefined camera directions ------------------------
    # return rot_vecs, centres, f_init
    # ---- 8 fixed cameras in a ring around the origin --------------------
    # Slight phase shift (e.g. π/16) to avoid alignment with axes
    phase_shift = math.pi / 16
    angle_step = 2 * math.pi / 8
    tilt = 0.1  # small positive Z value so cams are slightly above the XY plane

    dirs = []
    for i in range(8):
        angle = i * angle_step + phase_shift
        x = math.cos(angle)
        y = math.sin(angle)
        z = tilt
        vec = torch.tensor([x, y, z], dtype=torch.float32, device=device)
        dirs.append(vec / vec.norm())

    dirs = torch.stack(dirs, dim=0)  # (8,3)

    # ---- camera centres -------------------------------------------
    centres = dirs * cam_distance * obj_radius * 0.8  # (8,3)

    # ---- rotations (look‑at) --------------------------------------
    R_all = torch.stack([look_at_R(c) for c in centres], dim=0)  # (8,3,3)
    rot_vecs = torch.stack([rotmat_to_axis_angle(R) for R in R_all], dim=0)  # (8,3)

    # ---- hardcoded focal length -----------------------------------
    f_init = torch.full((8,), 0.8, device=device)

    return rot_vecs, centres, f_init



def look_at_R(cam_pos: torch.Tensor) -> torch.Tensor:
    """
    cam_pos : (3,) xyz position of the camera in world coordinates.
    Returns a 3×3 rotation matrix whose -Z axis looks at the origin and whose
    +Y is as “up” as possible.
    """
    z_axis = cam_pos / cam_pos.norm()               # forward (camera -Z)
    up_tmp = torch.tensor([0., 1., 0.], device=cam_pos.device)
    # In the rare case forward collinear with up_tmp, pick another up
    if torch.abs(torch.dot(z_axis, up_tmp)) > 0.999:
        up_tmp = torch.tensor([0., 0., 1.], device=cam_pos.device)

    x_axis = torch.cross(up_tmp, z_axis)
    x_axis = x_axis / x_axis.norm()
    y_axis = torch.cross(z_axis, x_axis)
    R = torch.stack([x_axis, y_axis, z_axis], dim=1)  # columns = basis vecs
    return R

def rotmat_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """
    R : (3,3) rotation matrix  →  (3,) axis‑angle (Lie algebra so(3))
    """
    cos_theta = (torch.trace(R) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos_theta)
    axis = torch.tensor([
        R[2,1]-R[1,2],
        R[0,2]-R[2,0],
        R[1,0]-R[0,1]
    ], device=R.device) / (2*torch.sin(theta))
    return axis * theta



