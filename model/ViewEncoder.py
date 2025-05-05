import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleViewEncoder(nn.Module):
    """
    Encodes views using a CNN backbone and pools features into tokens.
    Note: Assumes input size allows final feature map before pooling to be >= 2x2.
          Original comments implied 124x124 input -> 8x8 features -> 2x2 tokens.

    Args:
        output_dim (int): Dimension of output features (C). Default: 256.
        tokens_grid_size (int): The size of the grid for pooling (e.g., 2 for 2x2=4 tokens). Default: 2.
        intermediate_channels (tuple): Channels for intermediate conv layers. Default: (32, 64, 128).
    """
    def __init__(self, output_dim: int = 256, tokens_grid_size: int = 2, intermediate_channels: tuple = (32, 64, 128)):
        super().__init__()
        C = output_dim
        G = tokens_grid_size
        self.num_tokens = G * G

        c1, c2, c3 = intermediate_channels

        # Input: (B*K, 3, H, W)
        self.conv1 = nn.Conv2d(3,   c1, kernel_size=3, stride=2, padding=1) # H/2, W/2
        self.bn1   = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1,  c2, kernel_size=3, stride=2, padding=1) # H/4, W/4
        self.bn2   = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1) # H/8, W/8
        self.bn3   = nn.BatchNorm2d(c3)
        self.conv4 = nn.Conv2d(c3,  C, kernel_size=3, stride=2, padding=1) # H/16, W/16
        self.bn4   = nn.BatchNorm2d(C)

        # Pool to fixed grid size (e.g., 2x2)
        self.avgpool = nn.AdaptiveAvgPool2d((G, G))

    def forward(self, x): # x: (B, K, 3, H, W)
        B, K, C_in, H, W = x.shape
        x = x.view(B * K, C_in, H, W)         # (B*K, 3, H, W)

        x = F.relu(self.bn1(self.conv1(x)))   # (B*K, c1, H/2, W/2)
        x = F.relu(self.bn2(self.conv2(x)))   # (B*K, c2, H/4, W/4)
        x = F.relu(self.bn3(self.conv3(x)))   # (B*K, c3, H/8, W/8)
        x = F.relu(self.bn4(self.conv4(x)))   # (B*K, C, H/16, W/16)

        x = self.avgpool(x)                   # (B*K, C, G, G)

        # ---- reshape into token block ----
        # (B, K, C, G, G)
        x = x.view(B, K, -1, self.avgpool.output_size[0], self.avgpool.output_size[1])
        # (B, K, G, G, C)
        x = x.permute(0, 1, 3, 4, 2)
        # (B, K, G*G, C) = (B, K, num_tokens, C)
        tokens = x.reshape(B, K, self.num_tokens, -1)
        return tokens # (B, K, T, D) where T=num_tokens, D=output_dim

