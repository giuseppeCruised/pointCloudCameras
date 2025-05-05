import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
            heatmap (torch.Tensor): Input heatmap (B, K', H, W).

        Returns:
            torch.Tensor: Output RGB image (B, K', 3, H, W).
        """
        B, K_prime, H, W = heatmap.shape
        # (B*K', 1, H, W)
        heatmap_reshaped = heatmap.view(B * K_prime, 1, H, W)
        # (B*K', 3, H, W)
        rgb_reshaped = self.conv1x1(heatmap_reshaped)
        # (B*K', 3, H, W)
        rgb_activated = self.activation(rgb_reshaped)
        # (B, K', 3, H, W)
        rgb_output = rgb_activated.view(B, K_prime, 3, H, W)
        return rgb_output
