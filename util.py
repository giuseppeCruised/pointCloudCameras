import torch
import os
import json
import numpy as np
import wandb

from torch_cluster import knn_graph


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


def log_original_and_recon(patches, mask_idx, vis_idx, recon, step):
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

    # ---------- cloud #2 : visible + recon ------------------------------------
    vis_xyz    = t2np(patches[vis_idx])              # visible → white
    recon_xyz  = t2np(recon)                         # recon   → blue
    masked_xyz  = t2np(patches[mask_idx])                         # recon   → blue

    vis_rgb    = paint(vis_xyz,   [255, 255, 255])   # white
    recon_rgb  = paint(recon_xyz, [  255,   100, 100])   # blue

    masked_rgb  = paint(masked_xyz, [  150,   255, 150])   # blue

    comb_rgb   = np.vstack([vis_rgb, recon_rgb, masked_rgb])     # concat (N_total,6)

    # ---------- log to W&B -----------------------------------------------------
    wandb.log({
        "3d/original_all_white"  : obj3d("orig",  orig_rgb),
        "3d/vis_white_recon_blue": obj3d("comb",  comb_rgb),
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

