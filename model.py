import torch
import torch.nn as nn
import torch.nn.functional as F
import timm # Import the timm library
from util import init_clustered_cameras
# --- Differentiable Projector ---

class DifferentiableMultiViewProjector(nn.Module):
    """
    Projects a 3D point cloud onto multiple virtual camera planes using
    learnable camera intrinsics (focal length) and extrinsics (rotation, translation).
    Outputs coordinates suitable for rasterization (e.g., scaled NDC or pixel coordinates).

    Args:
        num_views (int): The number of virtual camera views (K).
        image_height (int): Target image height for coordinate scaling.
        image_width (int): Target image width for coordinate scaling.
    """
    def __init__(self, num_views: int, image_height: int, image_width: int):
        super().__init__()
        if not isinstance(num_views, int) or num_views <= 0:
            raise ValueError("num_views must be a positive integer")
        self.num_views = num_views
        self.image_height = image_height
        self.image_width = image_width

        rot_vecs, centres, f_init = init_clustered_cameras(num_views,
                                                           obj_radius=0.9,
                                                           cam_distance=1.6,
                                                           cluster_std_deg=50.0,
                                                           device="mps")

        self.rotation_vectors = nn.Parameter(rot_vecs)   # (N,3)
        self.translations     = nn.Parameter(centres)    # (N,3)
        self.focal_lengths    = nn.Parameter(f_init)     # (N,)

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
        """Returns learned camera poses as (K, 6) tensor (rot_vec, trans)."""
        return torch.cat([self.rotation_vectors, self.translations], dim=1)

    def forward(self, points_world: torch.Tensor) -> torch.Tensor:
        """
        Projects points and scales them to pixel coordinates for rasterization.

        Args:
            points_world (torch.Tensor): Input point cloud (B, N, 3).

        Returns:
            torch.Tensor: Projected 2D coordinates in pixel space (B, K, N, 2),
                          where origin (0,0) is top-left.
        """
        B, N, _ = points_world.shape
        points_expanded = points_world.unsqueeze(1).expand(B, self.num_views, N, 3)
        R = self._so3_exponential_map(self.rotation_vectors)
        R_expanded = R.view(1, self.num_views, 1, 3, 3)
        t_expanded = self.translations.view(1, self.num_views, 1, 3)

        points_camera_coords = torch.matmul(R_expanded, (points_expanded - t_expanded).unsqueeze(-1)).squeeze(-1)

        x_cam, y_cam = points_camera_coords[..., 0], points_camera_coords[..., 1]
        z_cam = points_camera_coords[..., 2].clamp(min=1e-5)
        f = self.focal_lengths.view(1, self.num_views, 1) # (1, K, 1)

        # Project to normalized plane (still centered at 0)
        x_proj_norm = f * (x_cam / z_cam)
        y_proj_norm = f * (y_cam / z_cam)

        # Convert to pixel coordinates (origin top-left)
        # Assumes principal point is at image center (W/2, H/2)
        x_pixel = x_proj_norm * (self.image_width / 2.0) + (self.image_width / 2.0)
        y_pixel = y_proj_norm * (self.image_height / 2.0) + (self.image_height / 2.0)

        projected_pixels = torch.stack([x_pixel, y_pixel], dim=-1) # (B, K, N, 2)
        return projected_pixels


# --- Differentiable Rasterizer ---

class SoftRasterizer(nn.Module):
    """
    Differentiable rasterization using Gaussian splatting.
    Converts projected 2D points into heatmaps.

    Args:
        height (int): Height of the output raster image.
        width (int): Width of the output raster image.
        sigma (float): Standard deviation for the Gaussian splat. Default: 1.0.
    """
    def __init__(self, height: int, width: int, sigma: float = 1.0):
        super().__init__()
        self.height = height
        self.width = width
        self.sigma = sigma

        # Create pixel grid buffers (persistent but not parameters)
        y_grid = torch.arange(height).float()
        x_grid = torch.arange(width).float()
        self.register_buffer("y_grid", y_grid.view(1, 1, 1, height, 1), persistent=False)
        self.register_buffer("x_grid", x_grid.view(1, 1, 1, 1, width), persistent=False)

    def forward(self, points_2d: torch.Tensor) -> torch.Tensor:
        """
        Rasterizes projected points.

        Args:
            points_2d (torch.Tensor): Projected 2D points (B, K, N, 2) in pixel coordinates.

        Returns:
            torch.Tensor: Rasterized heatmaps (B, K, H, W).
        """
        B, K, N, _ = points_2d.shape
        device = points_2d.device

        # Ensure grids are on the correct device
        y_grid = self.y_grid.to(device)
        x_grid = self.x_grid.to(device)

        # Extract x, y and reshape for broadcasting
        x = points_2d[..., 0].unsqueeze(-1).unsqueeze(-1)  # (B, K, N, 1, 1)
        y = points_2d[..., 1].unsqueeze(-1).unsqueeze(-1)  # (B, K, N, 1, 1)

        # Compute squared distances to each pixel center
        dist_sq = (x_grid - x)**2 + (y_grid - y)**2  # (B, K, N, H, W)

        # Apply Gaussian kernel weight = exp(-dist_sq / (2 * sigma^2))
        weights = torch.exp(-dist_sq / (2 * (self.sigma**2 + 1e-8)))  # (B, K, N, H, W)

        # Sum weights over the N points for each pixel
        raster_heatmap = weights.sum(dim=2)  # (B, K, H, W)

        return raster_heatmap

# --- Learnable Colorizer ---

class LearnableColorizer(nn.Module):
    """
    Applies a learnable pixel-wise transformation to convert a 1-channel heatmap
    to a 3-channel RGB image. Uses a 1x1 Convolution.

    Args:
        activation (nn.Module, optional): Activation function to apply after convolution.
                                         Defaults to nn.Sigmoid() to keep output in [0, 1].
    """
    def __init__(self, activation: nn.Module = nn.Sigmoid()):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        Converts heatmap to RGB.

        Args:
            heatmap (torch.Tensor): Input heatmap (B, K, H, W).

        Returns:
            torch.Tensor: Output RGB image (B, K, 3, H, W).
        """
        B, K, H, W = heatmap.shape
        heatmap_reshaped = heatmap.view(B * K, 1, H, W)
        rgb_reshaped = self.conv1x1(heatmap_reshaped)
        rgb_activated = self.activation(rgb_reshaped)
        rgb_output = rgb_activated.view(B, K, 3, H, W)
        return rgb_output

# --- View Encoder ---

from einops import rearrange

# ---------- Token‑Learner helper ----------
class TokenLearnerLite(nn.Module):
    def __init__(self, in_ch: int, num_tokens: int = 8):
        super().__init__()
        self.num_tokens = num_tokens
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, num_tokens, 1),
            nn.Sigmoid()
        )

    def forward(self, x):                  # x: (B,C,H,W)
        g = self.gate(x)                   # (B,M,H,W)
        x = x.unsqueeze(1) * g.unsqueeze(2)   # (B,M,C,H,W)
        tokens = x.flatten(-2).sum(-1)     # (B,M,C)
        return tokens                      # (B,M,C)

class SimpleViewEncoder(nn.Module):
    """
    Input : x  –  shape (B, K=8, 3, 124, 124)
    Output: tokens – shape (B, 8, 4, 256)   # 4 tokens = 2×2 grid per view
    """
    def __init__(self, dim=256):
        super().__init__()
        C = dim                        # final channel width
        self.conv1 = nn.Conv2d(3,   32, kernel_size=3, stride=2, padding=1)   # 124 → 62
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,  64, kernel_size=3, stride=2, padding=1)   # 62 → 31
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)   # 31 → 16
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, C, kernel_size=3, stride=2, padding=1)    # 16 →  8
        self.bn4   = nn.BatchNorm2d(C)

        # nothing learnable after this; we just pool to 2×2
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x):                          # x: (B, 8, 3, 124, 124)
        B, K, C_in, H, W = x.shape
        x = x.view(B * K, C_in, H, W)             # merge batch & view dims

        x = F.relu(self.bn1(self.conv1(x)))       # (B·K, 32, 62, 62)
        x = F.relu(self.bn2(self.conv2(x)))       # (B·K, 64, 31, 31)
        x = F.relu(self.bn3(self.conv3(x)))       # (B·K,128, 16, 16)
        x = F.relu(self.bn4(self.conv4(x)))       # (B·K,256,  8,  8)

        x = self.avgpool(x)                       # (B·K,256, 2, 2)

        # ---- reshape into token block ------------------------------------
        x = x.view(B, K, -1, 2, 2)                # (B, 8, 256, 2, 2)
        x = x.permute(0, 1, 3, 4, 2)              # (B, 8, 2, 2, 256)
        tokens = x.reshape(B, K, 4, -1)           # (B, 8, 4, 256)
        return tokens


class ViewAwareEncoder(nn.Module):
    def __init__(self, output_dim=256, tokens_per_view=4):
        super().__init__()
        self.tokens_per_view = tokens_per_view
        self.backbone = SimpleViewEncoder(output_dim)  # outputs (B, K, 4, 256)

        self.pose_proj = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim * tokens_per_view)  # (B, K, 4*256)
        )

        self.transformer_encoder = TokenTransformer(
            d_model=output_dim,
            nhead=8,
            num_layers=4,
            dim_feedforward=512
        )

    def forward(self, x, poses):  # x: (B, K, 3, H, W), poses: (B, K, 6)
        tokens = self.backbone(x)  # (B, K, 4, 256)
        B, K, T, D = tokens.shape   # T = 4, D = 256

        pose_enc = self.pose_proj(poses)            # (B, K, 4*256)
        pose_enc = pose_enc.view(B, K, T, D)        # (B, K, 4, 256)

        tokens = tokens + pose_enc                  # pose-aware token features

        tokens = tokens.view(B, K * T, D)           # (B, 32, 256)
        encoded = self.transformer_encoder(tokens)  # (B, 32, 256)

        return encoded


class TokenTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # important: ensures (B, seq, dim) instead of (seq, B, dim)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):  # x: (B, 32, 256)
        return self.encoder(x)  # output: (B, 32, 256)

class SceneDecoder(nn.Module):
    """
    Coordinate‑conditioned cross‑attention decoder.

    ‣ Input  : memory (B, 32, 256)  –– encoder tokens  
    ‣ Output : points  (B, 512, 3)  –– reconstructed point cloud
    """

    def __init__(
        self,
        num_points: int = 512,
        d_model: int = 256,
        n_heads: int = 8,
        num_fourier_bands: int = 6,
        ffn_scale: int = 4,
    ):
        super().__init__()
        self.num_points = num_points
        self.d_model = d_model
        self.num_fourier_bands = num_fourier_bands

        # --- learnable 3‑D query coordinates ---------------------------------
        self.coords = nn.Parameter(torch.rand(num_points, 3) * 2.0 - 1.0)  # in [‑1, 1]

        # --- positional / Fourier encoding -----------------------------------
        pe_dim = 3 + 3 * 2 * num_fourier_bands          # xyz  +  sin/cos
        self.input_proj = nn.Linear(pe_dim, d_model)

        # --- single cross‑attention + FFN block ------------------------------
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_scale * d_model),
            nn.GELU(),
            nn.Linear(ffn_scale * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # --- final projection to xyz -----------------------------------------
        self.out_proj = nn.Linear(d_model, 3)

    # -------------------------------------------------------------------------
    def _fourier_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, 3) → (B, N, 3 + 3·2·num_bands)
        """
        B, N, _ = x.shape
        device = x.device
        bands = torch.linspace(
            1.0,
            2.0 ** (self.num_fourier_bands - 1),
            self.num_fourier_bands,
            device=device,
        )                                            # (num_bands,)
        xb = x.unsqueeze(-1) * bands                 # (B, N, 3, num_bands)
        enc = torch.cat(
            (torch.sin(math.pi * xb), torch.cos(math.pi * xb)), dim=-1
        )                                            # (B, N, 3, 2·num_bands)
        enc = enc.flatten(start_dim=2)               # (B, N, 3·2·num_bands)
        return torch.cat((x, enc), dim=-1)           # (B, N, pe_dim)

    # -------------------------------------------------------------------------
    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """
        memory : (B, 32, 256)
        returns : reconstructed points (B, 512, 3)
        """
        B = memory.size(0)

        # 1. broadcast learnable queries to batch
        q_coords = self.coords.unsqueeze(0).expand(B, -1, -1)          # (B, 512, 3)

        # 2. encode coordinates + linear projection to d_model
        q = self.input_proj(self._fourier_encode(q_coords))            # (B, 512, 256)

        # 3. cross‑attention (queries = coord tokens, keys/values = memory)
        attn_out, _ = self.cross_attn(query=q, key=memory, value=memory)
        x = self.norm1(q + attn_out)

        # 4. feed‑forward refinement
        x = self.norm2(x + self.ffn(x))

        # 5. predict xyz
        return self.out_proj(x)                                        # (B, 512, 3)



#

# --- Main Reconstruction Model ---

class MultiViewPointCloudReconstructor(nn.Module):
    """
    Main model: Projects points, rasterizes to images, colorizes, encodes views,
    fuses features with pose info, and decodes back to a point cloud.

    Args:
        num_points (int): Number of points (N) in the output point cloud.
        latent_dim (int): Dimension (D) of the fused scene latent vector. Default: 256.
        num_views (int): Number of camera views (K). Default: 8.
        image_height (int): Height of the rasterized images. Default: 224.
        image_width (int): Width of the rasterized images. Default: 224.
        raster_sigma (float): Sigma for Gaussian splatting in rasterizer. Default: 1.0.
        view_encoder_output_dim (int): Output dimension of the view encoder. Default: 256.
        fusion_attention_dim (int): Attention dimension in the fusion module. Default: 256.
        pose_embedding_dim (int): Dimension of camera poses for fusion. Default: 6.
        colorizer_activation (nn.Module, optional): Activation for colorizer. Default: nn.Sigmoid().
        view_encoder_backbone (str): Name of timm model for view encoder. Default: 'mobilevit_s'.
    """
    def __init__(self, num_points: int, latent_dim: int = 256, num_views: int = 8,
                 image_height: int = 124, image_width: int = 124, raster_sigma: float = 1.0,
                 view_encoder_output_dim: int = 256, fusion_attention_dim: int = 256,
                 pose_embedding_dim: int = 6, colorizer_activation: nn.Module = nn.Sigmoid(),
                 view_encoder_backbone: str = 'mobilevit_s'): # Added backbone selection
        super().__init__()
        if not isinstance(num_points, int) or num_points <= 0:
            raise ValueError("num_points must be a positive integer")
        self.num_points = num_points
        self.latent_dim = latent_dim # Final latent dim feeding into decoder
        self.num_views = num_views

        # --- Components ---
        self.projector = DifferentiableMultiViewProjector(
            num_views=num_views,
            image_height=image_height,
            image_width=image_width
        )
        self.rasterizer = SoftRasterizer(
            height=image_height,
            width=image_width,
            sigma=raster_sigma
        )
        self.colorizer = LearnableColorizer(activation=colorizer_activation)
        self.view_encoder = ImageFeatureExtractor(
            output_dim=view_encoder_output_dim,
            img_size=image_height,
        )

        self.scene_fusion = MultiViewFeatureFusion(
            input_dim=view_encoder_output_dim,      # Matches view_encoder output
            attention_dim=fusion_attention_dim,     # Output dim of fusion
            pose_embedding_dim=pose_embedding_dim   # Matches pose input format
        )

        # Ensure latent_dim matches the output of the scene_fusion module
        if latent_dim != fusion_attention_dim:
            print(f"Warning: Provided latent_dim ({latent_dim}) doesn't match fusion output ({fusion_attention_dim}). Decoder input adjusted.")
            decoder_input_dim = fusion_attention_dim
            self.latent_dim = fusion_attention_dim # Update self.latent_dim to actual value
        else:
            decoder_input_dim = latent_dim

        # Decoder: MLP to reconstruct point cloud
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 512),
            nn.ReLU(inplace=True), # Use inplace ReLU where possible
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_points * 3) # Output N*3 coordinates
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: points -> projection -> raster -> colorize -> encode -> fuse -> decode -> points.

        Args:
            points (torch.Tensor): Input point cloud (B, N, 3).

        Returns:
            torch.Tensor: Reconstructed point cloud (B, N, 3).
        """
        B = points.shape[0]
        device = points.device

        # 1. Project points to K views (output in pixel coordinates)
        projected_pixels = self.projector(points) # (B, K, N, 2)

        # 2. Rasterize projected points into heatmaps
        raster_heatmaps = self.rasterizer(projected_pixels) # (B, K, H, W)

        # 3. Convert heatmaps to "RGB" images using learnable colorizer
        raster_rgb = self.colorizer(raster_heatmaps) # (B, K, 3, H, W)

        # print("raster_rgb", raster_rgb.mean().item(), raster_rgb.std().item())

        camera_poses_6d = self.projector.get_camera_poses_6d()     # (K,6)

        # Expand to (B,K,6) **without altering K or making it (K*B,1,6)**
        camera_poses_batch = camera_poses_6d.unsqueeze(0).expand(B, -1, -1)
        # Encode
        encoded_views = self.view_encoder(raster_rgb, camera_poses_batch)

        # print("encoded_views", encoded_views.mean().item(), encoded_views.std().item())

        # 5. Get camera poses from the projector
        # Expand poses for batch and move to correct device
        camera_poses_batch = camera_poses_6d.unsqueeze(0).expand(B, -1, -1).to(device) # (B, K, 6)

        # 6. Fuse view features using poses
        fused_scene = self.scene_fusion(encoded_views, camera_poses_batch) # (B, fusion_attention_dim)

        # 7. Decode the fused representation
        reconstructed_flat = self.decoder(fused_scene) # (B, N*3)

        # 8. Reshape decoder output: (B, N*3) -> (B, N, 3)
        reconstructed_points = reconstructed_flat.view(B, self.num_points, 3)

        # print("reconstructed_points", reconstructed_points.mean().item(), reconstructed_points.std().item())

        return reconstructed_points, raster_heatmaps, raster_rgb



# ---------- Plug‑and‑play View Encoder ----------
# class ImageFeatureExtractor(nn.Module):
#     def __init__(
#         self,
#         output_dim: int      = 256,
#         img_size:    int      = 128,         # <- can be 124, 96, …
#         backbone_name: str    = "mobilevit_xxs",
#         stage_idx:    int      = 2,
#         num_tokens:   int      = 8,
#     ):
#         super().__init__()
#         self.img_size  = img_size
#         self.stage_idx = stage_idx
#
#         # 1) domain adapter (3→16)
#         self.rgb_adapter = nn.Sequential(
#             nn.Conv2d(3, 16, 3, padding=1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.SiLU()
#         )
#
#         # 2) backbone (expects 16‑ch input)
#         self.backbone = timm.create_model(
#             backbone_name,
#             pretrained=True,
#             in_chans=16,
#             features_only=True,
#             img_size=img_size,
#         )
#         feat_ch = self.backbone.feature_info.channels()[stage_idx]
#
#         # 3a) 2‑D sine‑cos positional buffer
#         Hf = Wf = img_size // 16            # stage‑2 spatial size
#         grid_y, grid_x = torch.meshgrid(
#             torch.linspace(-1, 1, Hf),
#             torch.linspace(-1, 1, Wf),
#             indexing="ij"
#         )
#         pos = torch.stack([grid_x, grid_y], 0).unsqueeze(0)     # (1,2,Hf,Wf)
#         self.register_buffer("pos_enc", pos, persistent=False)
#         self.pos_proj = nn.Conv2d(2, feat_ch, 1, bias=False)
#
#         # 3b) pose MLP (6→feat_ch)
#         self.pose_mlp = nn.Sequential(
#             nn.Linear(6, feat_ch),
#             nn.SiLU(),
#             nn.Linear(feat_ch, feat_ch),
#         )
#
#         # 4) Token‑Learner
#         self.token_learner = TokenLearnerLite(feat_ch, num_tokens)
#
#         # 5) projection to output_dim
#         self.proj = nn.Linear(num_tokens * feat_ch, output_dim)
#
#
#         self.ln   = nn.LayerNorm(output_dim)
#
#     def forward(self, imgs: torch.Tensor, poses: torch.Tensor):
#         """
#         imgs : (B, K, 3, H, W)  – H,W must equal img_size
#         poses: (B, K, 6)        – axis‑angle(3) + T(3)
#         returns (B, K, output_dim)
#         """
#         B, K, C, H, W = imgs.shape
#         assert H == W == self.img_size, f"Expect {self.img_size}px, got {H}×{W}"
#
#         feat = imgs.reshape(B*K, C, H, W)
#         feat = self.rgb_adapter(feat)                      # (BK,16,H,W)
#         feat = self.backbone(feat)[self.stage_idx]         # (BK,C',Hf,Wf)
#
#         # add positional map (broadcast, no copy)
#         Hf, Wf = feat.shape[-2:]
#
#         # 1. create / resize positional grid on‑the‑fly
#         if (Hf, Wf) != self.pos_enc.shape[-2:]:
#             # make a new sine‑cos grid that exactly matches Hf×Wf
#             gy, gx = torch.meshgrid(
#                 torch.linspace(-1, 1, Hf, device=feat.device, dtype=feat.dtype),
#                 torch.linspace(-1, 1, Wf, device=feat.device, dtype=feat.dtype),
#                 indexing="ij",
#             )
#             pos_dynamic = torch.stack([gx, gy], 0).unsqueeze(0)      # (1,2,Hf,Wf)
#         else:
#             pos_dynamic = self.pos_enc.to(feat.dtype)                # (1,2,Hf,Wf)
#
#         # 2. project to feat_ch and add (broadcast over batch)
#         pos_emb = self.pos_proj(pos_dynamic)                         # (1,C',Hf,Wf)
#         feat    = feat + pos_emb                                     # (BK,C',Hf,Wf)
#
#         # pose embedding
#         pose_emb = self.pose_mlp(poses.reshape(B*K, 6))    # (BK,C')
#         feat = feat + pose_emb.view(B*K, -1, 1, 1)
#
#         # token learner
#         tokens = self.token_learner(feat)                  # (BK,M,C')
#         tokens = tokens.flatten(1)                         # (BK,M*C')
#
#         # tokens = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)
#         z = self.ln(self.proj(tokens))                     # (BK,output_dim)
#         return z.view(B, K, -1)                            # (B,K,output_dim)

# class ImageFeatureExtractor(nn.Module):
#     """
#     Encodes a batch of images into feature vectors using a backbone and attention pooling.
#
#     Args:
#         output_dim (int): Dimension of the output feature vector per view. Default: 256.
#         image_height (int): Expected input image height.
#         image_width (int): Expected input image width.
#         backbone_model_name (str): Name of the timm model. Default: 'mobilevit_s'.
#         feature_stage_index (int): Index of the backbone feature stage to use. Default: 3.
#         num_attention_heads (int): Number of attention heads. Default: 8.
#     """
#     def __init__(self, output_dim: int = 256, image_height: int = 224, image_width: int = 224, backbone_model_name: str = 'mobilevit_s', feature_stage_index: int = 3, num_attention_heads: int = 8):
#         super().__init__()
#         self.output_dim = output_dim
#         self.stage_index = feature_stage_index
#         self.image_height = image_height
#         self.image_width = image_width
#
#         try:
#             # Use timm.create_model instead of the placeholder
#             self.backbone = timm.create_model(
#                 backbone_model_name, pretrained=True, features_only=True
#             )
#             print(f"Successfully loaded timm model: {backbone_model_name}")
#         except ImportError:
#              print("Error: timm library not found. Please install it using 'pip install timm'")
#              raise
#         except Exception as e:
#             print(f"Failed to load timm model '{backbone_model_name}'. Error: {e}")
#             raise
#
#         # Determine num_channels dynamically
#         num_channels = self._get_backbone_feature_channels(backbone_model_name)
#
#         self.feature_projection = nn.Linear(num_channels, output_dim, bias=False)
#         self.view_query = nn.Parameter(torch.randn(1, 1, output_dim) / output_dim**0.5)
#         self.attention_pool = nn.MultiheadAttention(
#             embed_dim=output_dim, num_heads=num_attention_heads, batch_first=True
#         )
#
#     def _get_backbone_feature_channels(self, backbone_model_name):
#         """Helper to dynamically determine feature channels."""
#         try:
#             dummy_input = torch.randn(2, 3, self.image_height, self.image_width) # Use batch size > 1 for robustness
#             self.backbone.eval() # Set model to evaluation mode for dummy forward pass
#             with torch.no_grad():
#                  dummy_features = self.backbone(dummy_input)
#             self.backbone.train() # Set back to train mode
#             if not isinstance(dummy_features, (list, tuple)) or len(dummy_features) <= self.stage_index:
#                  raise ValueError(f"Backbone does not return features as list/tuple or stage index {self.stage_index} is out of bounds for model {backbone_model_name}.")
#             num_channels = dummy_features[self.stage_index].shape[1]
#             print(f"Determined {num_channels} channels for stage {self.stage_index} of {backbone_model_name}")
#             return num_channels
#         except Exception as e:
#             print(f"Warning: Could not dynamically determine feature channels for stage {self.stage_index} of {backbone_model_name}. Assuming 64. Error: {e}")
#             # You might want to raise an error here or have better fallback logic
#             return 64 # Fallback
#
#     def forward(self, view_images: torch.Tensor) -> torch.Tensor:
#         """Processes images to extract per-view feature vectors."""
#         if view_images.dim() != 5 or view_images.shape[2] != 3:
#              raise ValueError(f"Expected input view_images shape (B, K, 3, H, W), but got {view_images.shape}")
#         B, K, C, H, W = view_images.shape
#
#         bk_images = view_images.view(B * K, C, H, W)
#         all_feature_maps = self.backbone(bk_images)
#         image_features = all_feature_maps[self.stage_index]
#
#         BK, C_feat, H_feat, W_feat = image_features.shape
#         num_spatial_tokens = H_feat * W_feat
#
#         image_features = image_features.view(BK, C_feat, num_spatial_tokens).permute(0, 2, 1) # (BK, S, C_feat)
#         projected_features = self.feature_projection(image_features) # (BK, S, D)
#         query = self.view_query.expand(BK, -1, -1) # (BK, 1, D)
#         # Attention pooling: Query attends to projected features
#         pooled_features, _ = self.attention_pool(query, projected_features, projected_features) # (BK, 1, D)
#         encoded_view_feature = pooled_features.squeeze(1).view(B, K, self.output_dim) # (B, K, D)
#         return encoded_view_feature


# --- Scene Fusion ---


# class MultiViewFeatureFusion(nn.Module):
#     """
#     Fuses features from multiple camera views using pose information and cross-attention.
#
#     Args:
#         input_dim (int): Dimension of the input view features (D_in). Default: 256.
#         attention_dim (int): Dimension used within the attention mechanism (D_model). Default: 256.
#         pose_embedding_dim (int): Dimension of the raw camera pose input. Default: 6.
#         pose_mlp_hidden_dim (int): Hidden dimension for the pose MLP. Default: 128.
#         num_heads (int): Number of attention heads. Default: 8.
#     """
#     def __init__(
#         self,
#         input_dim: int = 256,
#         attention_dim: int = 256,
#         pose_embedding_dim: int = 6,
#         pose_mlp_hidden_dim: int = 128,
#         num_heads: int = 8,
#     ):
#         super().__init__()
#         self.pose_mlp = nn.Sequential(
#             nn.Linear(pose_embedding_dim, pose_mlp_hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(pose_mlp_hidden_dim, input_dim),
#         )
#         self.feature_projection = (
#             nn.Identity() if input_dim == attention_dim else nn.Linear(input_dim, attention_dim, bias=False)
#         )
#         self.scene_query = nn.Parameter(torch.randn(1, 1, attention_dim) / attention_dim**0.5)
#         self.cross_attention = nn.MultiheadAttention(
#             embed_dim=attention_dim, num_heads=num_heads, batch_first=True
#         )
#
#     def forward(self, view_features: torch.Tensor, camera_poses: torch.Tensor) -> torch.Tensor:
#         """
#         Fuses view features using pose information and cross-attention.
#
#         Args:
#             view_features (torch.Tensor): Feature vectors from K views (B, K, input_dim).
#             camera_poses (torch.Tensor): Camera poses for the K views (B, K, pose_embedding_dim).
#
#         Returns:
#             torch.Tensor: Aggregated scene feature vector (B, attention_dim).
#         """
#         B, K, _ = view_features.shape
#         pose_embeddings = self.pose_mlp(camera_poses)
#         fused_features = view_features + pose_embeddings # Add pose info to features
#         projected_features = self.feature_projection(fused_features) # Project to attention dim
#         scene_query_expanded = self.scene_query.expand(B, -1, -1) # Prepare scene query
#         # Cross-attend: scene query attends to view features
#         aggregated_scene_feature, _ = self.cross_attention(
#             scene_query_expanded, projected_features, projected_features
#         )
#         return aggregated_scene_feature.squeeze(1) # (B, attention_dim)
