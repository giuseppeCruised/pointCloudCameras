import torch
import torch.nn.functional as F

from mode import MaskedReconstructionModule, Patchify3D
from util import chamfer_distance, mask_patches, chamfer_l1_safe, log_separate_pointclouds, log_original_and_recon
from dataset import SceneWiseDataLoader

import wandb


def training_loop(epochs=1500, batch_size=16,
                  lr=2e-4, feature_dim=128,
                  patch_size=32, num_patches=64, load=True):

    wandb.init(project="pointcloud-denoising",
               id="w5qody4u",
               resume="must",
               config={
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "feature_dim": feature_dim
                    })
    # During training
    dataset = SceneWiseDataLoader('./train/', device="mps")
    model = MaskedReconstructionModule(feature_dim, patch_size).to("mps")
    patchify = Patchify3D(num_patches, patch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
        print(f"started epoch: {epoch}")
        for clean_points, _ in dataset.iterate_batches(batch_size=batch_size, shuffle=False):
            patches = patchify(clean_points)  # (B, P, patch_size, 3)
            visible_patches, masked_patches, vis_idx, mask_idx = mask_patches(patches)

            pred = model(masked_patches, visible_patches)
            B, P_mask, k, _ = pred.shape
            pred_flat = pred.view(B*P_mask, patch_size, 3)
            masked_patches_flat = masked_patches.view(B*P_mask, patch_size, 3)
            loss = chamfer_l1_safe(pred_flat, masked_patches_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item():.6f}")
            wandb.log({"loss": loss.item()}, step=epoch)
        if epoch % 20 == 0:
             log_original_and_recon(patches[0].cpu(), mask_idx[0].cpu(), vis_idx[0].cpu(), pred[0].cpu(), epoch)
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

def main():
    training_loop()


if __name__ == "__main__":
    main()

