"""Chapter 2: Independent ResNet-18 trajectory prediction model and masked loss."""
import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == channels
            else nn.Sequential(nn.Conv2d(in_channels, channels, 1, stride, bias=False), nn.BatchNorm2d(channels))
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.act(self.bn1(self.conv1(x)))
        return self.act(self.bn2(self.conv2(x)) + residual)


class ResNet18Trajectory(nn.Module):
    """From-scratch ResNet-18 backbone with adapted 4-channel input stem and MLP trajectory head.
    
    Total parameters: 20,116,648.
    Input:
      - bev: uint8 [B, 4, 128, 128]
      - ego: float32 [B, 2] (speed, yaw_rate)
    Output:
      - trajectory: float32 [B, 10, 4] normalized waypoints (x, y, v, heading)
    """

    def __init__(self, steps: int = 10, pooling_size: int = 4, mlp_hidden: int = 1024, latent_dim: int = 512):
        super().__init__()
        self.steps = steps
        self.stem = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        blocks = []
        previous = 64
        for stage, channels in enumerate([64, 128, 256, 512]):
            blocks.extend([BasicBlock(previous, channels, 1 if stage == 0 else 2), BasicBlock(channels, channels)])
            previous = channels
        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((pooling_size, pooling_size))
        self.head = nn.Sequential(
            nn.Linear(512 * pooling_size**2 + 2, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, steps * 4),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, bev: torch.Tensor, ego: torch.Tensor | None = None) -> torch.Tensor:
        if bev.dtype == torch.uint8:
            bev = bev.float() / 255.0
        if ego is None:
            ego = bev.new_zeros(len(bev), 2)
        features = self.pool(self.backbone(self.stem(bev))).flatten(1)
        return self.head(torch.cat([features, ego], dim=1)).reshape(-1, self.steps, 4)


def masked_trajectory_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked L1 loss on trajectory waypoints with 2.0x weight on x, y position coordinates.

    Args:
        prediction: [B, 10, 4]
        target: [B, 10, 4]
        valid: [B, 10] bool
        action_mask: [B] bool
    """
    error = (prediction - target).abs()
    # 2.0x priority weighting on spatial coordinates (x, y) relative to velocity and heading
    weighted = torch.cat([error[:, :, :2] * 2.0, error[:, :, 2:]], dim=-1)
    mask = valid.unsqueeze(-1) & action_mask.reshape(-1, 1, 1)
    return (weighted * mask).sum() / mask.sum().clamp_min(1)
