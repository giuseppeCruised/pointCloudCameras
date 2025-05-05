import os, json, math, torch
import numpy as np
import wandb

# -------------------------
#  basic helpers
# -------------------------
def axis_angle_to_R(vec):
    """ vec: (3,) torch axis-angle → (3,3) rotation matrix """
    theta = torch.linalg.norm(vec) + 1e-8
    if theta < 1e-5:
        return torch.eye(3, device=vec.device)
    axis = vec / theta
    K = torch.tensor([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]],
                     device=vec.device)
    return (torch.eye(3, device=vec.device)
            + torch.sin(theta) * K
            + (1 - torch.cos(theta)) * (K @ K))

def to_np(t):
    return t.detach().cpu().numpy().astype(float).reshape(-1, 3)

def paint(xyz, rgb):
    """ xyz: (M,3) np, rgb: 3‑tuple 0–255 → (M,6) """
    M = xyz.shape[0]
    cols = np.repeat(np.array(rgb)[None], M, 0)
    return np.hstack([xyz, cols])

def obj3d(name, pts_rgb, step):
    obj = {
      "type": "lidar/beta",             # string
      "points": pts_rgb,                # ndarray (N×6)
      "boxes":  np.zeros((0,8), float)  # ndarray (0×8)
    }
    os.makedirs("tmp_logs", exist_ok=True)
    # only convert the arrays to lists
    serial = {
      "type":      obj["type"],
      "points":    obj["points"].tolist(),
      "boxes":     obj["boxes"].tolist()
    }
    with open(f"tmp_logs/{name}_{step}.json","w") as f:
        json.dump(serial, f)
    return wandb.Object3D(obj)
# -------------------------
#  main logging fn
# -------------------------
def log_with_cameras(orig_pts, recon_pts,
                     cam_rot_vecs, cam_centres,
                     step,
                     tetra_size=0.05,
                     line_len=0.10,
                     prefix="train"):
    """
    orig_pts:     Tensor (N,3)      the original point cloud
    recon_pts:    Tensor (M,3)      the reconstructed point cloud
    cam_rot_vecs: Tensor (C,3)      axis-angle
    cam_centres:  Tensor (C,3)
    """

    # --- turn clouds into colored arrays ---
    orig_np  = to_np(orig_pts)
    recon_np = to_np(recon_pts)

    orig_col  = paint(orig_np,  [255,255,255])   # white
    recon_col = paint(recon_np,[255,100,100])   # pinkish

    layers = [orig_col, recon_col]

    C = cam_centres.shape[0]
    for i in range(C):
        c = cam_centres[i]
        R = axis_angle_to_R(cam_rot_vecs[i])

        forward = R @ torch.tensor([0,0,-1.],device=c.device)
        right   = R @ torch.tensor([1,0, 0.],device=c.device)
        up      = R @ torch.tensor([0,1, 0.],device=c.device)

        # tetra: tip + three base verts
        verts = torch.stack([
            c,
            c + right   * tetra_size,
            c + up      * tetra_size,
            c + forward * tetra_size * 0.5
        ],0)

        # little line of 5 points
        line = torch.stack([c + forward * (j/4)*line_len for j in range(5)],0)

        # color gradient green→blue
        t   = i/max(C-1,1)
        rgb = [0, int(255*(1-t)), int(255*t)]

        xyz = to_np(torch.cat([verts, line],0))
        layers.append(paint(xyz, rgb))

    all_pts = np.vstack(layers)
    wandb.log({
      f"3d{prefix}/scene": obj3d("scene", all_pts, step)
    }, step=step)
