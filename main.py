import torch
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.reconstruction_module import MultiViewReconstructionModel
from util import chamfer_distance, mask_patches, chamfer_l1_safe, log_separate_pointclouds, log_original_and_recon, log_camera_images
from loss import chamfer_safe, repulsion_loss
from log_cameras import log_with_cameras
from dataset import SceneWiseDataLoader

import math
import wandb

from collections import defaultdict


def centroid_regression_loss(gt_points, gt_labels, pred_centroid0, pred_centroid1, eps=1e-6):
    """
    Computes L1 loss between predicted and ground truth centroids for two object classes.

    Args:
        gt_points:     (B, N, 3)  — ground truth point cloud
        gt_labels:     (B, N)     — class labels per point (0 or 1)
        pred_centroid0: (B, 3)    — predicted centroid for class 0
        pred_centroid1: (B, 3)    — predicted centroid for class 1
        eps: float — small constant to avoid division by zero

    Returns:
        loss: scalar tensor — sum of L1 losses for class 0 and class 1
    """

    # Boolean masks per class
    mask0 = (gt_labels == 0)  # (B, N)
    mask1 = (gt_labels == 1)  # (B, N)

    # Avoid division by zero in empty classes
    sum0 = mask0.sum(dim=1, keepdim=True).clamp(min=eps)  # (B, 1)
    sum1 = mask1.sum(dim=1, keepdim=True).clamp(min=eps)  # (B, 1)

    # Compute centroids
    centroid0 = (gt_points * mask0.unsqueeze(-1)).sum(dim=1) / sum0  # (B, 3)
    centroid1 = (gt_points * mask1.unsqueeze(-1)).sum(dim=1) / sum1  # (B, 3)

    # L1 loss to predicted centroids
    loss0 = F.l1_loss(pred_centroid0, centroid0)
    loss1 = F.l1_loss(pred_centroid1, centroid1)

    return loss0 + loss1

def plot_attention_map(attn_weights, batch_index=0, head_indices=None, max_heads_per_row=4):
    """
    Plots attention maps (Query vs Key) for selected heads for a specific batch item.
    Each selected head gets its own subplot showing the full Q x K attention matrix.

    Args:
        attn_weights (torch.Tensor): Tensor of shape (B, H, Q, K) containing attention weights.
        batch_index (int): The index of the batch item to visualize. Default is 0.
        head_indices (list[int], optional): A list of head indices (H dimension) to plot.
                                            If None, all heads are plotted. Defaults to None.
        max_heads_per_row (int): Maximum number of subplots to display per row.

    Returns:
        matplotlib.figure.Figure: The Matplotlib figure object containing the plots.

    Raises:
        AssertionError: If batch_index is out of bounds.
        ValueError: If provided head_indices are out of bounds.
    """
    # --- Input Validation and Data Preparation ---
    B, H, Q, K = attn_weights.shape
    assert 0 <= batch_index < B, f"batch_index {batch_index} out of bounds for batch size {B}"

    # Select the specific batch item and move to CPU
    # Detach from computation graph and convert to numpy
    attn = attn_weights[batch_index].detach().cpu().numpy()  # Shape: (H, Q, K)

    # Determine which heads to plot
    if head_indices is None:
        h_indices_to_plot = list(range(H))
    else:
        if not all(0 <= i < H for i in head_indices):
            raise ValueError(f"Head indices must be between 0 and {H-1}")
        h_indices_to_plot = head_indices

    num_plot_heads = len(h_indices_to_plot)

    if num_plot_heads == 0:
        print("Warning: No heads selected for plotting.")
        return plt.figure() # Return an empty figure

    # --- Plotting ---
    # Determine grid layout for subplots
    num_rows = math.ceil(num_plot_heads / max_heads_per_row)
    num_cols = min(num_plot_heads, max_heads_per_row)

    # Adjust figure size based on grid - ensure plots aren't too squished
    # Heuristic: ~4-5 inches width per plot, ~4-5 inches height per plot row
    fig_width = num_cols * 5
    fig_height = num_rows * 4.5
    fig, axes = plt.subplots(num_rows, num_cols,
                             figsize=(fig_width, fig_height),
                             squeeze=False) # Keep axes as 2D array

    print(f"Plotting attention for Batch Item {batch_index}")
    print(f"Heads to plot (indices): {h_indices_to_plot}")
    print(f"Attention dimensions (Q x K): {Q} x {K}")
    print(f"Creating a grid of {num_rows} rows x {num_cols} columns...")

    # Flatten axes array for easier iteration
    axes_flat = axes.flatten()

    # Find global min/max across selected heads for consistent color scaling
    min_val = np.min(attn[h_indices_to_plot, :, :])
    max_val = np.max(attn[h_indices_to_plot, :, :])

    # Keep track of the last image plotted for the colorbar
    im = None

    # Iterate through the *selected* head indices
    for i, h_idx in enumerate(h_indices_to_plot):
        ax = axes_flat[i]
        # Select the full attention map for this head: (Q, K)
        heatmap = attn[h_idx, :, :]

        # Plot the QxK heatmap
        # Use vmin/vmax for consistent color scaling
        im = ax.imshow(heatmap, aspect='auto', cmap='viridis', vmin=min_val, vmax=max_val,
                       interpolation='nearest') # 'nearest' avoids blurring pixels

        ax.set_title(f"Head {h_idx}")
        ax.set_ylabel("Query Index (Q)")
        ax.set_xlabel("Key Index (K)")

        # Optionally add grid lines for clarity, especially for smaller Q/K
        # ax.set_xticks(np.arange(-.5, K, 1), minor=True)
        # ax.set_yticks(np.arange(-.5, Q, 1), minor=True)
        # ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
        # ax.tick_params(which="minor", size=0)

        # Optionally show tick labels if Q/K are small enough
        # if K <= 20:
        #     ax.set_xticks(np.arange(K))
        #     ax.set_xticklabels(np.arange(K))
        # else:
        #     ax.set_xticks([]) # Hide if too many
        # if Q <= 20:
        #      ax.set_yticks(np.arange(Q))
        #      ax.set_yticklabels(np.arange(Q))
        # else:
        #      ax.set_yticks([]) # Hide if too many


    # Hide any unused subplots if num_plot_heads doesn't fill the grid
    for i in range(num_plot_heads, len(axes_flat)):
        axes_flat[i].axis('off')

    # Add a single colorbar for the entire figure
    if im is not None: # Only add colorbar if something was plotted
      # Position the colorbar appropriately - may need adjustment based on layout
      fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02, aspect=30) # Adjust shrink/pad/aspect
    else:
        print("No attention maps were plotted.")


    # Adjust layout to prevent titles/labels overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust rect for padding and suptitle
    # Optional: Add an overall title
    fig.suptitle(f"Attention Maps (Query vs Key) - Batch {batch_index}", fontsize=16, y=0.99)

    return fig
# --- Step 1: Build the Gradient Tree ---

def build_gradient_tree(
    model: torch.nn.Module,
    exclude_biases: bool = True,
) -> dict:
    """
    Builds a nested dictionary tree representing the model hierarchy
    and populates it with gradient norm information.

    Args:
        model: The PyTorch model (torch.nn.Module).
        exclude_biases: If True, bias parameters are excluded.

    Returns:
        A nested dictionary representing the hierarchy. Each node contains
        'sum_norm', 'param_count', and 'children'. Leaf nodes (parameters)
        also contain 'param_norm'. Returns None if no valid gradients found.
    """
    tree = {}
    found_valid_grads = False

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if exclude_biases and name.endswith(".bias"):
            continue

        # Calculate L2 norm for the parameter's gradient
        norm = torch.linalg.norm(param.grad.detach()).item()

        # Skip non-finite gradients
        if not math.isfinite(norm):
            print(f"Warning: Non-finite gradient norm ({norm}) detected for parameter {name}. Skipping.")
            continue

        found_valid_grads = True
        parts = name.split('.')
        current_level = tree

        # Traverse/create nodes for module path
        for i, part in enumerate(parts[:-1]): # Iterate through module parts
            if part not in current_level:
                current_level[part] = {
                    'sum_norm': 0.0,
                    'param_count': 0,
                    'children': {},
                    # 'mean_norm' will be calculated later
                }
            # Update aggregates as we descend
            current_level[part]['sum_norm'] += norm
            current_level[part]['param_count'] += 1
            current_level = current_level[part]['children']

        # Handle the leaf node (parameter itself)
        param_name = parts[-1]
        current_level[param_name] = {
            'param_norm': norm, # Store individual norm here
            'sum_norm': norm,   # Sum for this leaf is just its own norm
            'param_count': 1,
            'children': {}, # No children for a parameter leaf
        }

    return tree if found_valid_grads else None


# --- Step 2: Calculate Mean Norms in the Tree ---

def calculate_mean_norms(node: dict):
    """
    Recursively calculates the 'mean_norm' for each node in the tree.
    Assumes 'sum_norm' and 'param_count' are already populated.
    Modifies the tree in-place.
    """
    if not node: # Base case for empty nodes/children
        return

    count = node.get('param_count', 0)
    sum_n = node.get('sum_norm', 0.0)

    # Calculate mean_norm for the current node if it's a module (has children or count > 1)
    # We check 'children' to differentiate modules from parameters at leaves
    if node.get('children') or count > 1: # It's a module node
         node['mean_norm'] = sum_n / count if count > 0 else 0.0

    # Recursively process children
    children_dict = node.get('children', {})
    for child_key, child_node in children_dict.items():
        calculate_mean_norms(child_node)


# --- Step 3: Flatten the Tree for WandB Logging ---

def flatten_tree_for_wandb(
    node: dict,
    current_path: str,
    output_dict: dict,
    agg_prefix: str = "grads_ag",
    param_prefix: str = "grads_params"
):
    """
    Recursively flattens the processed tree into a dictionary for WandB.

    Args:
        node: The current node in the tree.
        current_path: The path string built up to this node.
        output_dict: The dictionary to store flattened results.
        agg_prefix: Prefix for aggregated mean norms (modules).
        param_prefix: Prefix for individual parameter norms.
    """
    if not node:
        return

    # Check if it's a parameter node (leaf)
    if 'param_norm' in node:
        param_key = f"{param_prefix}/{current_path}" if current_path else param_prefix
        output_dict[param_key] = node['param_norm']

    # Check if it's a module node (has children or pre-calculated mean_norm)
    # Log aggregate mean norm if it exists
    if 'mean_norm' in node:
        agg_key = f"{agg_prefix}/{current_path}" if current_path else agg_prefix
        output_dict[agg_key] = node['mean_norm']

    # Recursively process children
    children_dict = node.get('children', {})
    for child_key, child_node in children_dict.items():
        # Append child key to path, handling the root case
        new_path = f"{current_path}.{child_key}" if current_path else child_key
        flatten_tree_for_wandb(child_node, new_path, output_dict, agg_prefix, param_prefix)


# --- Step 4: Orchestrator Function for Direct Logging ---

def log_hierarchical_gradients_wandb(
    model: torch.nn.Module,
    step: int,
    agg_prefix: str = "grads_ag",    # Prefix for mean norms of modules
    param_prefix: str = "grads_params", # Prefix for individual param norms
    exclude_biases: bool = True
):
    """
    Builds a gradient norm tree, calculates mean norms, flattens it,
    and logs structured results (aggregates and parameters separately) to WandB.

    Args:
        model: The PyTorch model with gradients computed.
        step: The current training step number.
        agg_prefix: WandB prefix for aggregated mean norms (modules).
        param_prefix: WandB prefix for individual parameter norms.
        exclude_biases: If True, bias parameters are ignored.
    """
    if wandb.run is None:
        print("Warning: wandb.run is None. Initialize wandb with wandb.init() before logging.")
        return

    # 1. Build the raw tree with sums and counts
    gradient_tree = build_gradient_tree(
        model=model,
        exclude_biases=exclude_biases
    )

    if not gradient_tree:
        print(f"Step {step}: No valid gradients found or all were excluded, nothing logged.")
        # Optionally log a status message to WandB
        # wandb.log({f"{agg_prefix}/status": "no_gradients", f"{param_prefix}/status": "no_gradients"}, step=step)
        return

    # 2. Calculate mean norms recursively in-place
    calculate_mean_norms(gradient_tree) # Pass the root node(s) if tree is dict

    # 3. Flatten the processed tree for WandB logging
    wandb_log_dict = {}
    # Need to handle the possibility of multiple top-level keys if model has direct params and submodules
    if isinstance(gradient_tree, dict):
        for top_level_key, top_level_node in gradient_tree.items():
             flatten_tree_for_wandb(top_level_node, top_level_key, wandb_log_dict, agg_prefix, param_prefix)
    else:
         # Handle case where tree might be structured differently (e.g., a single root object)
         # Assuming the root is a dictionary for now based on build_gradient_tree
         print("Warning: Unexpected tree structure root.")


    # 4. Log to WandB
    if wandb_log_dict:
        wandb.log(wandb_log_dict, step=step)
    else:
        print(f"Step {step}: Flattened dictionary is empty, nothing logged.")


def log_gradients(model, step):
    """
    Logs mean gradient norm for each main component of the model.
    Should be called after loss.backward(), before optimizer.step().
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: {param.grad.norm().item()}")
        else:
            print(f"{name}: No grad")

    grad_tracker = {
        'encoder': [],
        'projector': [],
        'transformer': [],
        'decoder': [],
    }

    # Group parameters into major modules based on name
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if name.startswith('encoder.'):
            grad_tracker['encoder'].append(param.grad.norm().item())
        elif name.startswith('decoder.'):
            grad_tracker['decoder'].append(param.grad.norm().item())
        elif name.startswith('projector.') or name.startswith('pose_proj.'):
            grad_tracker['projector'].append(param.grad.norm().item())
        elif name.startswith('transformer.') or name.startswith('encoder.transformer_encoder'):
            grad_tracker['transformer'].append(param.grad.norm().item())

    # Now prepare a single dict for wandb
    wandb_log = {}
    for module_name, norms in grad_tracker.items():
        if norms:  # avoid empty modules
            wandb_log[f"grad_norm/{module_name}"] = sum(norms) / len(norms)  # mean grad norm per module

    # Log everything at once for this step
    wandb.log(wandb_log, step=step)


def training_loop(epochs=300, batch_size=4,
                  lr=4e-4, load=False):
    wandb.init(project="pointcloud-denoising-cameras",
               # id="e0abucsz",
               # resume="must",
               config={
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    })
    # During training
    dataset = SceneWiseDataLoader('./train_random_shapes/', device="mps")
    dataset_test = SceneWiseDataLoader('./test_random_shapes/', device="mps")
    model = MultiViewReconstructionModel().to("mps")
    params = []
    for name, param in model.named_parameters():
        if "translations" in name:
            params.append({"params": param, "lr": 1*lr})
        elif "encoder" in name:
            params.append({"params": param, "lr": 1*lr})
        elif "pose" in name:
            params.append({"params": param, "lr": 1*lr})
        else:
            params.append({"params": param, "lr": lr})
    optimizer = torch.optim.AdamW(params)
    scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=3e-6)
    start_epoch = 0
    if load:
        checkpoint = torch.load("checkpoint.pt", map_location="mps")

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']

        for g in optimizer.param_groups:
            g['lr'] = lr  # your desired new learning rate

    run_id = wandb.run.id
    print(f"🔑 Run ID: {run_id}")

    for epoch in range(start_epoch, epochs):
        scheduler.step()
        print(f"started epoch: {epoch}")
        for clean_points, labels in dataset.iterate_batches(batch_size=batch_size, shuffle=False):
            pred, raster, rgb_raster, decoder_attn_weights, centroid1, centroid2, tokens = model(clean_points)
            # loss = chamfer_l1_safe(pred, clean_points)
            encoder_attn_weights = model.encoder.transformer_encoder.last_attn_weights
            entropy = - (decoder_attn_weights * decoder_attn_weights.clamp(min=1e-8).log()).sum(dim=-1)  # shape (B, H, Q)
            #entropy = - (encoder_attn_weights * encoder_attn_weights.clamp(min=1e-8).log()).sum(dim=-1)  # shape (B, H, Q)
            decoder_entropy_loss = F.relu(entropy.mean()-0.5)
            diversity_loss = (token_diversity_loss(tokens) ** 2).mean()
            query_attn_diversity_loss = decoder_query_diversity(decoder_attn_weights)
            print("decoder entropy loss:", decoder_entropy_loss.item())
            print("token diversity loss:", diversity_loss.item())
            aux_loss = centroid_regression_loss(clean_points, labels, centroid1, centroid2)
            # loss = chamfer_safe(pred, clean_points) + 0.1 * decoder_entropy_loss + 0.1 * query_attn_diversity_loss#+ cos_an(epoch, 300, 0.2)*aux_loss  # + 0.1*aux_loss#+ 0.01*diversity_loss # ## + 0.1*entropy_loss + + 0.2*chamfer_safe(aux, clean_points) #+ 0.3*repulsion_loss(pred)
            loss = chamfer_safe(pred, clean_points) + cos_an(epoch, 300, 0.2) * decoder_entropy_loss + cos_an(epoch, 300, 0.2) * query_attn_diversity_loss
            optimizer.zero_grad()
            loss.backward()


            optimizer.step()
            wandb.log({"lr": scheduler.get_last_lr()[0]}, step=epoch)
            print(f"Loss: {loss.item():.6f}")


            wandb.log({"loss": loss.item()}, step=epoch)


        log_hierarchical_gradients_wandb(model, epoch)
        if epoch % 20 == 0:
            encoder_attn_fig = plot_attention_map(encoder_attn_weights)
            decoder_attn_fig1 = plot_attention_map(decoder_attn_weights)
            decoder_attn_fig2 = plot_attention_map(decoder_attn_weights, batch_index=2)
            wandb.log({"attention/encoder/query0_head0": wandb.Image(encoder_attn_fig)}, step=epoch)
            wandb.log({"attention/decoder/query0_head0": wandb.Image(decoder_attn_fig1)}, step=epoch)
            wandb.log({"attention/decoder/2/query0_head0": wandb.Image(decoder_attn_fig2)}, step=epoch)
            cam_rot_vecs = model.projector.rotation_vectors
            cam_translations = model.projector.translations
        
            #log_original_and_recon(clean_points[0], pred[0], epoch)
            log_camera_images(raster, rgb_raster, epoch)
            log_with_cameras(clean_points[0], pred[0], cam_rot_vecs, cam_translations, epoch)
        if epoch % 50 == 0:
            clean_points_test, labels_test = dataset_test.get_batch([0])
            pred, raster, rgb_raster, decoder_attn_weights, centroid1, centroid2, tokens = model(clean_points)
            cam_rot_vecs = model.projector.rotation_vectors
            cam_translations = model.projector.translations
            log_camera_images(raster, rgb_raster, epoch, prefix="test")
            log_with_cameras(clean_points[0], pred[0], cam_rot_vecs, cam_translations, epoch, prefix="test")
            
            #     with torch.no_grad():
            #         original_np = clean_points[0].cpu().numpy()
            #         noisy_np    = noisy_points[0].cpu().numpy()
            #         recon_np    = pred[0].detach().cpu().numpy()
            #
            #         log_pointcloud_comparison(
            #             original=original_np,
            #             noisy=noisy_np,
            #             reconstructed=recon_np,
            #             step=epoch
            #         )

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item()
        }, "checkpoint.pt")
    print(f"🔑 Run ID: {run_id}")


def cos_an(t, T, lambda_max):
    return lambda_max * 0.5 * (1 + math.cos(math.pi * t / T))


def decoder_query_diversity(attn_weights):
    """
        Penalizes similarity between decoder queries within each head.

            Args:
                    attn_weights: (B, H, Q, T) attention from queries to tokens
                        Returns:
                                Scalar loss (float) — higher when queries are more similar
                                    """
    B, H, Q, T = attn_weights.shape

        # Normalize attention maps over tokens
    attn = F.normalize(attn_weights, dim=-1)  # (B, H, Q, T)

        # Compute query-query cosine sim: (B, H, Q, Q)
    sim = torch.matmul(attn, attn.transpose(-2, -1))  # (B, H, Q, Q)

        # Mask diagonal — we only care about off-diagonal similarity
    eye = torch.eye(Q, device=attn.device).unsqueeze(0).unsqueeze(0)  # (1, 1, Q, Q)
    sim = sim.masked_fill(eye.bool(), 0.0)

    return sim.mean()
#     B, K, T, D = attn_weights.shape
#     attn_weights = attn_weights.reshape(B, K*T, D)
#     tokens = 


def token_diversity_loss(tokens):
    """
    Encourages encoder tokens within a scene to be different from each other.

    Args:
        tokens: (B, N, D) — where N = K * T tokens per scene
    Returns:
        Scalar loss value
    """
    # Cosine similarity between all token pairs within each sample
    tokens = F.normalize(tokens, dim=-1)       # (B, N, D)
    sim = torch.matmul(tokens, tokens.transpose(1, 2))  # (B, N, N)
    
    # Mask diagonal (self-similarity)
    B, N, _ = sim.shape
    mask = torch.eye(N, device=tokens.device).bool().unsqueeze(0)  # (1, N, N)
    sim = sim.masked_fill(mask, 0.0)
    
    # Penalize high similarity (i.e., encourage diversity)
    return sim.mean()




def main():
    training_loop()

if __name__ == "__main__":
    main()

