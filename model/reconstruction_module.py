import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.colorizer import LearnableColorizer
from model.Decoder import SceneDecoder
from model.differentiable_multi_view_projector import DifferentiableMultiViewProjector
from model.soft_rasterizer import SoftRasterizer
from model.view_aware_encoder import ViewAwareEncoder



class MultiViewReconstructionModel(nn.Module):
    """
    Combines Encoder, Decoder, and Renderer for Multi-View Point Cloud Reconstruction.

    Args:
        # Encoder args
        num_input_views (int): Number of input views (K).
        img_h (int): Height of input images.
        img_w (int): Width of input images.
        img_emb_dim (int): Dimension for CNN image features.
        pose_emb_dim (int): Hidden dimension for pose MLP.
        tokens_grid_size (int): Grid size for pooling in CNN (e.g., 2 for 2x2=4 tokens).
        intermediate_cnn_channels (tuple): Channels for intermediate CNN layers.
        transformer_nhead (int): Num heads for encoder transformer.
        transformer_layers (int): Num layers for encoder transformer.
        transformer_ffn_dim (int): FFN dim for encoder transformer.

        # Decoder args
        num_output_points (int): Number of points in the reconstructed cloud (N_pts).
        decoder_n_heads (int): Num heads for decoder cross-attention.
        num_fourier_bands (int): Num bands for decoder coord encoding.
        decoder_ffn_scale (int): Scale factor for decoder FFN.
        coord_range (tuple): Initialization range for decoder query points.

        # Renderer args (for training loss)
        num_render_views (int): Number of views for differentiable rendering (K').
        render_h (int): Height for rendered images.
        render_w (int): Width for rendered images.
        render_sigma (float): Sigma for soft rasterizer.
        render_inv_depth (bool): Use inverse depth weighting in rasterizer.
        render_cam_obj_radius (float): Object radius for render camera init.
        render_cam_distance (float): Camera distance for render camera init.
        render_cam_cluster_std (float): Std dev for render camera init.
        render_color (bool): Whether to use the LearnableColorizer.
    """
    def __init__(
        self,
        # Encoder
        num_input_views: int = 8,
        img_h: int = 128, # Example, adjust based on dataset
        img_w: int = 128, # Example, adjust based on dataset
        img_emb_dim: int = 256,
        pose_emb_dim: int = 128,
        tokens_grid_size: int = 2,
        intermediate_cnn_channels: tuple = (32, 64, 128),
        transformer_nhead: int = 8,
        transformer_layers: int = 4,
        transformer_ffn_dim: int = 256,
        tokens_per_view: int = 4,
        # Decoder
        num_output_points: int = 256,
        decoder_n_heads: int = 8,
        num_fourier_bands: int = 6,
        decoder_ffn_scale: int = 4,
        coord_range: tuple = (-1.0, 1.0),
        # Renderer
        num_render_views: int = 8, # Can be same or different from num_input_views
        render_h: int = 128,
        render_w: int = 128,
        render_sigma: float = 1.0,
        render_inv_depth: bool = False,
        render_cam_obj_radius: float = 0.9,
        render_cam_distance: float = 1.6,
        render_cam_cluster_std: float = 50.0,
        render_color: bool = True, # Use colorizer?
    ):
        super().__init__()

        self.num_input_views = num_input_views
        self.img_h = img_h
        self.img_w = img_w
        self.num_output_points = num_output_points
        self.num_render_views = num_render_views
        self.render_h = render_h
        self.render_w = render_w
        self.render_color = render_color

        self.rasterizer = SoftRasterizer(height=render_h, width=render_w)

        self.encoder = ViewAwareEncoder(
            img_emb_dim=img_emb_dim,
            pose_emb_dim=pose_emb_dim,
            tokens_per_view=tokens_per_view,
            transformer_nhead=transformer_nhead,
            transformer_layers=transformer_layers,
            transformer_ffn_dim=transformer_ffn_dim,
            tokens_grid_size=tokens_grid_size,
            intermediate_cnn_channels=intermediate_cnn_channels
        )

        self.decoder = SceneDecoder(
            num_points=num_output_points,
            memory_dim=img_emb_dim, # Memory dim must match encoder output dim
            n_heads=decoder_n_heads,
            num_fourier_bands=num_fourier_bands,
            ffn_scale=decoder_ffn_scale,
            coord_range=coord_range
        )

        # Setup rendering pipeline (used for loss calculation during training)
        # Need to manage device for projector parameters
        self.projector = DifferentiableMultiViewProjector(
            num_views=num_render_views,
            image_height=render_h,
            image_width=render_w,
            obj_radius=render_cam_obj_radius,
            cam_distance=render_cam_distance,
            cluster_std_deg=render_cam_cluster_std,
            device='cpu' # Placeholder, will be moved later
        )

        self.decoder2 = nn.Sequential(
                    nn.Linear(32*256, 32*3))

        self.rasterizer = SoftRasterizer(
            height=render_h,
            width=render_w,
            sigma=render_sigma,
            inv_depth_weighting=render_inv_depth
        )
        if self.render_color:
            self.colorizer = LearnableColorizer()
        else:
            self.colorizer = None # Or nn.Identity() if needed downstream

    def forward(self, input_points: torch.Tensor) -> torch.Tensor:
        """
        Performs the reconstruction from point cloud to point cloud

        Args:
            input_images (torch.Tensor): Batch of input images (B, K, 3, H, W).
            input_poses (torch.Tensor): Batch of corresponding input poses (B, K, 6).

        Returns:
            torch.Tensor: Reconstructed point cloud (B, N_pts, 3).
        """

        # Project input points on K cameras
        B, N, _ = input_points.shape
        point_projections, points_z = self.projector(input_points)

        B, K, N_prime, _ = point_projections.shape
        assert K == self.num_input_views, f"Expected {self.num_input_views} input views, got {K}"
        assert N == N_prime, f"{N}and {N_prime} do not match"
        poses = self.projector.get_camera_poses_6d().unsqueeze(0).repeat(B, 1, 1)

        input_images = self.rasterizer(point_projections, points_z)
        rgb_images = self.colorizer(input_images)
        # rgb_images = input_images.unsqueeze(2).repeat(1, 1, 3, 1, 1)


        # Input shape checks
        B, K, C, H, W = rgb_images.shape
        B_p, K_p, P = poses.shape
        assert K == self.num_input_views, f"Expected {self.num_input_views} input views, got {K}"
        assert H == self.img_h and W == self.img_w, f"Expected input image size {(self.img_h, self.img_w)}, got {(H, W)}"
        assert B == B_p and K == K_p, f"Batch size/Num views mismatch between images ({B, K}) and poses ({B_p, K_p})"
        assert C == 3, f"Expected 3 input channels, got {C}"
        assert P == 6, f"Expected 6D input poses, got {P}"

        # Move projector parameters to the correct device on the first forward pass
        # This is a common pattern to avoid hardcoding device in __init__ A
        if self.projector.rotation_vectors.device != input_images.device:
            self.projector.to(input_images.device)


        # 1. Encode images and poses -> memory tokens
        # (B, K*T, D)
        memory = self.encoder(rgb_images, poses)

        flat = memory.view(B, 32*256)
        points = self.decoder2(flat)
        rec = points.view(B, 32, 3)
        centroid1 = rec[:, 0, :]
        centroid2 = rec[:, 1, :]

        # 2. Decode memory to point cloud
        # (B, N_pts, 3)
        reconstructed_points, decoder_attn_weights = self.decoder(memory)

        return reconstructed_points, input_images, rgb_images, decoder_attn_weights, centroid1, centroid2, memory
