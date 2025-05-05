import torch
import torch.nn.functional as F

def repulsion_loss(pred, k=5, radius=0.1):
    B, N, _ = pred.shape
    dist = pairwise_dist(pred, pred)           # (B, N, N)
    diag = torch.eye(N, device=pred.device).unsqueeze(0) * 1e6
    dist = dist + diag                            # ignore self
    knn_dists, _ = dist.topk(k, dim=-1, largest=False)  # (B, N, k)
    # Compare *L1* distances against the *linear* radius:
    violation = F.relu(radius - knn_dists)        # positive if too close
    return violation.mean()

def pairwise_dist(x, y):  # x: (B, N, 3), y: (B, M, 3)
    B, N, _ = x.shape
    _, M, _ = y.shape

    # Broadcast differences: (B, N, M, 3)
    diff = x[:, :, None, :] - y[:, None, :, :]  # (B, N, M, 3)

    # Take L1 norm over last dimension
    dist = diff.abs().sum(dim=-1)  # (B, N, M)
    return dist

def chamfer_safe(x, y):  # x, y: (B, N, 3)
    dist = pairwise_dist(x, y)  # (B, N, M)
    x_to_y = dist.min(dim=2)[0]  # (B, N)
    y_to_x = dist.min(dim=1)[0]  # (B, M)
    return (x_to_y.mean(dim=1) + y_to_x.mean(dim=1)).mean()  # scalar loss
