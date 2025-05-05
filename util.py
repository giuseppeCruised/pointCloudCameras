import torch
import os
import json
import numpy as np
import wandb
import math

from torch_cluster import knn_graph





# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Main initialiser
# ------------------------------------------------------------------


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


def log_camera_images(
        heatmaps: torch.Tensor,        # (B, K, H, W)  –  single‑channel floats
        rgb_feats: torch.Tensor,       # (B, K, 3, H, W)
        step: int,
        prefix: str = "train"          # "val", "test", … if you like
):
    """
    Logs to W&B:
      • `{prefix}/heatmaps` – list of K grayscale wandb.Image objects
      • `{prefix}/rgb`      – list of K RGB wandb.Image objects

    Only the first element in the batch (index 0) is visualised.
    """

    # ---- sanity ----------------------------------------------------
    assert heatmaps.ndim == 4, "heatmaps should be (B,K,H,W)"
    assert rgb_feats.ndim == 5, "rgb_feats should be (B,K,3,H,W)"
    B, K, H, W = heatmaps.shape
    assert rgb_feats.shape[:3] == (B, K, 3), "shapes disagree!"

    # ---- helpers ---------------------------------------------------
    def to_uint8(arr: np.ndarray) -> np.ndarray:
        """Normalise arr to 0‑255 uint8 (handles 1‑ or 3‑ch)."""
        arr = arr - arr.min()
        arr = arr / (arr.max() + 1e-8)
        return (arr * 255).round().astype(np.uint8)

    # ---- collect images -------------------------------------------
    heat_imgs, rgb_imgs = [], []

    for cam in range(K):
        # ---- heat‑map (grayscale) ---------------------------------
        hm = heatmaps[0, cam].detach().cpu().numpy()          # (H,W)
        hm_uint8 = to_uint8(hm)
        heat_imgs.append(
            wandb.Image(hm_uint8, mode="L", caption=f"{prefix}‑cam{cam:02d}")
        )

        # ---- RGB ---------------------------------------------------
        rgb = rgb_feats[0, cam].detach().cpu().permute(1, 2, 0).numpy()  # (H,W,3)
        if rgb.max() <= 1.0:                        # assume 0‑1, scale to 0‑255
            rgb = (rgb * 255).clip(0, 255)
        rgb_imgs.append(
            wandb.Image(rgb.astype(np.uint8),
                        caption=f"{prefix}‑cam{cam:02d}")
        )

    # ---- log -------------------------------------------------------
    wandb.log(
        {
            f"{prefix}/heatmaps": heat_imgs,
            f"{prefix}/rgb":      rgb_imgs,
        },
        step=step
    )

def soft_rasterize_batched(points_2d, H, W, sigma=1.0):
    """
    Differentiable rasterization for batched, multi-camera 2D points.

    Args:
        points_2d: (B, K, N, 2)  –  2D points projected by K cameras on B samples.
        H: height of the output image.
        W: width of the output image.
        sigma: Gaussian splat stddev (in pixel units).

    Returns:
        raster: (B, K, H, W)  –  Rasterized 2D heatmaps per camera and per batch.
    """
    B, K, N, _ = points_2d.shape
    device = points_2d.device

    # Create a pixel grid (H, W)
    y_grid = torch.arange(H, device=device).view(1, 1, 1, H, 1).float()
    x_grid = torch.arange(W, device=device).view(1, 1, 1, 1, W).float()

    # Extract x, y and reshape for broadcasting
    x = points_2d[..., 0].unsqueeze(-1).unsqueeze(-1)  # (B, K, N, 1, 1)
    y = points_2d[..., 1].unsqueeze(-1).unsqueeze(-1)  # (B, K, N, 1, 1)

    # Compute squared distances to each pixel
    dist2 = (x_grid - x)**2 + (y_grid - y)**2  # (B, K, N, H, W)

    # Apply Gaussian kernel
    weights = torch.exp(-dist2 / (2 * sigma**2))  # (B, K, N, H, W)

    # Sum over N points
    raster = weights.sum(dim=2)  # (B, K, H, W)

    return raster


def farthest_point_sampling(xyz, npoint):
    # x: (B, N, 3) → output: (B, npoint)
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.ones(B, N, device=xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)

    batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest].unsqueeze(1)  # (B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)  # (B, N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids  # (B, npoint)

def knn_group(xyz, centers, k):
    # xyz: (B, N, 3), centers: (B, P) → output: (B, P, k, 3)
    B, N, _ = xyz.shape
    P = centers.shape[1]
    grouped_xyz = []

    for b in range(B):
        center_xyz = xyz[b][centers[b]]  # (P, 3)
        dists = torch.cdist(center_xyz, xyz[b])  # (P, N)
        idx = torch.topk(dists, k, dim=-1, largest=False)[1]  # (P, k)
        grouped = xyz[b][idx]  # (P, k, 3)
        grouped_xyz.append(grouped)

    return torch.stack(grouped_xyz, dim=0)  # (B, P, k, 3)


def chamfer_distance(p1, p2):
    """
    p1, p2: (B, N, 3)
    Returns: average chamfer distance (scalar)
    """
    dists = torch.cdist(p1, p2)  # (B, N, N)
    cd1 = dists.min(dim=2).values.mean(dim=1)  # pred -> gt
    cd2 = dists.min(dim=1).values.mean(dim=1)  # gt -> pred
    return (cd1 + cd2).mean()


def chamfer_l1_safe(x, y):  # x, y: (B, N, 3)
    diff = x[:, :, None, :] - y[:, None, :, :]     # (B, N, M, 3)
    dist = diff.norm(dim=-1)                       # (B, N, M)
    x_to_y = dist.min(dim=2)[0]
    y_to_x = dist.min(dim=1)[0]
    return (x_to_y.mean() + y_to_x.mean())


def differentiable_emd(pred, target):
    """
    Differentiable EMD approximation using min-pair matching.
    Works with (B, N, 3) point clouds.
    """
    assert pred.shape == target.shape, "Shape mismatch"
    B, N, _ = pred.shape

    # Pairwise distance matrix
    dists = torch.cdist(pred, target, p=2)  # (B, N, N)

    # Approximate one-to-one matching by minimizing row and column distances
    forward = dists.min(dim=2)[0]  # (B, N)
    backward = dists.min(dim=1)[0]  # (B, N)

    loss = (forward.mean(dim=1) + backward.mean(dim=1)) / 2  # (B,)
    return loss.mean()  # scalar


def add_noise_to_pointcloud(points, sigma=0.1, clip=0.30):
    noise = torch.randn_like(points) * sigma
    noise = torch.clamp(noise, -clip, clip)
    return points + noise

def log_separate_pointclouds(patches, mask_idx, vis_idx, recon, step):
    """
    Log 3 coloured point‑clouds to W&B:
        • reconstructed patches   – green
        • ground‑truth masked pts – red
        • visible patches         – blue
    """

    # ----- tiny helpers ------------------------------------------------------
    def to_flat_np(t):
        """torch‑tensor → (N,3) float64 ndarray"""
        return t.detach().cpu().numpy().reshape(-1, 3).astype(float)

    def paint(xyz, rgb):
        """(N,3) xyz  →  (N,6) xyzrgb as float ndarray"""
        rgb = np.asarray(rgb, dtype=float)[None]                 # (1,3)
        rgb = np.repeat(rgb, xyz.shape[0], axis=0)               # (N,3)
        return np.hstack([xyz, rgb])                             # (N,6)

    def make_obj3d(name, pc6):
        """
        pc6  : (N,6) ndarray  (x,y,z,r,g,b)
        returns wandb.Object3D
        """
        obj = {
            "type":   "lidar/beta",
            "points": pc6,                # <-- NumPy array (good!)
            "boxes":  np.zeros((0,8))     # empty, but still NumPy
        }

        # Optional debug dump
        os.makedirs("tmp_logs", exist_ok=True)
        with open(f"tmp_logs/{name}_{step}.json", "w") as f:
            json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v
                       for k, v in obj.items()}, f)

        return wandb.Object3D(obj)
    # ------------------------------------------------------------------------

    # grab & colour the clouds
    gt_masked_pts = paint(to_flat_np(patches[mask_idx]), [255,   0,   0])  # red
    visible_pts   = paint(to_flat_np(patches[vis_idx]),  [  0,   0, 255])  # blue
    recon_pts     = paint(to_flat_np(recon),             [  0, 255,   0])  # green

    wandb.log({
        "3d/reconstructed": make_obj3d("reconstructed", recon_pts),
        "3d/masked_gt":     make_obj3d("masked_gt",     gt_masked_pts),
        "3d/visible":       make_obj3d("visible",       visible_pts),
    }, step=step)


def log_original_and_recon(patches, recon, step):
    """
    patches : Tensor  (P, k, 3)      – original geometry
    mask_idx: Tensor / list (P_mask) – indices of masked patches
    vis_idx : Tensor / list (P_vis)  – indices of visible patches
    recon   : Tensor  (P_mask, k, 3) – predicted patches
    """

    # ------------------------------------------------------------------ helpers
    def t2np(t):                           # (…,3) → ndarray float
        return t.detach().cpu().numpy().reshape(-1, 3).astype(float)

    def paint(xyz, rgb):                   # xyz (N,3) + rgb(3) → (N,6)
        rgb = np.repeat(np.asarray(rgb, float)[None], xyz.shape[0], axis=0)
        return np.hstack([xyz, rgb])       # (N,6)

    def obj3d(name, pts_rgb):              # build / dump / wrap
        obj = {"type": "lidar/beta",
               "points": pts_rgb,          # ndarray(!)
               "boxes":  np.zeros((0,8))}

        os.makedirs("tmp_logs", exist_ok=True)
        with open(f"tmp_logs/{name}_{step}.json", "w") as f:
            json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v
                       for k, v in obj.items()}, f)

        return wandb.Object3D(obj)
    # --------------------------------------------------------------------------

    # ---------- cloud #1 : ORIGINAL  (white) ----------------------------------
    orig_xyz   = t2np(patches)                       # (P*k , 3)
    orig_rgb   = paint(orig_xyz, [255, 255, 255])    # white

    recon_xyz  = t2np(recon)                         # recon   → blue
    recon_rgb  = paint(recon_xyz, [  255,   100, 100])   # blue

    comb_rgb   = np.vstack([orig_rgb, recon_rgb])     # concat (N_total,6)

    # ---------- log to W&B -----------------------------------------------------
    wandb.log({
        "3d/comp" : obj3d("orig",  comb_rgb),
    }, step=step)

def log_pointcloud_comparison(original, noisy, reconstructed, step):
    """
    Logs three point clouds (original, noisy, reconstructed) to W&B in .pts.json format.

    original, noisy, reconstructed: NumPy arrays of shape (N, 3)
    step: integer epoch or global step
    """
    os.makedirs("tmp_logs", exist_ok=True)

    def write_pts_json(points, filename, color=(1,1,1)):
        """
        Write out a W&B-compatible .pts.json file with 'type': 'lidar/beta',
        'points': and 'colors': arrays.
        """
        # 'points' can be Nx3, and 'colors' can be Nx3. We'll just replicate one color for all points here.
        points = np.asarray(points)
        num_pts = points.shape[0]

        data = {
            "type": "lidar/beta",
            "points": points.tolist(),
            "colors": np.tile(color, (num_pts, 1)).tolist(),  # replicate one color for all points
            # If you want per-point colors, you'd supply them individually here.
        }

        with open(filename, 'w') as f:
            json.dump(data, f)

    def save_points(points, label, color):
        filename = f"tmp_logs/{label}_{step}.pts.json"
        write_pts_json(points, filename, color=color)
        return filename

    # Write each cloud to a separate .pts.json file
    orig_file = save_points(original,  "original",      (1, 1, 1))    # white
    noisy_file = save_points(noisy,    "noisy",         (1, 0, 0))    # red
    recon_file = save_points(reconstructed, "recon",    (0, 1, 0))    # green

    # Log them as 3D objects in W&B
    wandb.log({
        "pointcloud/original":      wandb.Object3D(orig_file),
        "pointcloud/noisy":         wandb.Object3D(noisy_file),
        "pointcloud/reconstructed": wandb.Object3D(recon_file)
    }, step=step)


def mask_patches(patches, mask_ratio=0.25):
    """
    Randomly mask a subset of patches from a (B, P, k, 3) tensor.
    Returns:
        visible_patches: (B, P_vis, k, 3)
        masked_patches:  (B, P_mask, k, 3)
        visible_idx: (B, P_vis)
        masked_idx:  (B, P_mask)
    """
    torch.manual_seed(42)  # or any number

    B, P, k, _ = patches.shape
    P_mask = int(P * mask_ratio)
    P_vis = P - P_mask

    all_indices = torch.arange(P, device=patches.device).unsqueeze(0).repeat(B, 1)  # (B, P)

    # For each batch, randomly permute patch indices
    rand_idx = torch.rand(B, P, device=patches.device).argsort(dim=1)  # (B, P)
    vis_idx = rand_idx[:, :P_vis]   # (B, P_vis)
    mask_idx = rand_idx[:, P_vis:]  # (B, P_mask)

    # Gather visible and masked patches
    # We use batch-wise indexing (slow-ish, but correct)
    visible_patches = batched_index_select(patches, vis_idx)  # (B, P_vis, k, 3)
    masked_patches  = batched_index_select(patches, mask_idx) # (B, P_mask, k, 3)

    return visible_patches, masked_patches, vis_idx, mask_idx

def batched_index_select(x, idx):
    """
    Selects indices from x along dim=1 in a batch-wise way.

    Args:
        x: (B, N, ...)
        idx: (B, K)

    Returns:
        out: (B, K, ...)
    """
    B = x.shape[0]
    K = idx.shape[1]
    # Create offset indices for batch dim
    batch_offset = torch.arange(B, device=x.device).unsqueeze(1) * x.shape[1]
    flat_idx = (idx + batch_offset).view(-1)  # (B*K,)
    x_flat = x.view(B * x.shape[1], *x.shape[2:])
    out = x_flat[flat_idx].view(B, K, *x.shape[2:])
    return out

