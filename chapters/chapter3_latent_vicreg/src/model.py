"""Chapter 3: Pre-norm SDPA Transformer encoder and one-step latent residual predictor."""
import torch
from torch import nn
import torch.nn.functional as F

from chapters.chapter3_latent_vicreg.src.config import Stage3Config


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int):
        super().__init__()
        if dim % heads:
            raise ValueError("Attention dimension must be divisible by heads")
        self.heads = heads
        self.norm_attention = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.projection = nn.Linear(dim, dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(self.norm_attention(x)).reshape(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x + self.projection(attended.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.norm_mlp(x))


class BEVEncoder(nn.Module):
    """Encodes [B, 4, 128, 128] uint8 BEV into [B, 64, 256] spatial latent tokens."""

    def __init__(self, config: Stage3Config):
        super().__init__()
        d = config.encoder_dim
        self.stem = nn.Sequential(
            nn.Conv2d(4, 64, 5, stride=2, padding=2), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
        )
        self.patch = nn.Conv2d(128, d, config.patch_stride, stride=config.patch_stride)
        self.token_count = (32 // config.patch_stride) ** 2  # (32 // 4)^2 = 64 tokens
        self.position = nn.Parameter(torch.empty(1, self.token_count, d))
        nn.init.normal_(self.position, std=0.02)
        self.blocks = nn.Sequential(*[
            TransformerBlock(d, config.encoder_heads, config.encoder_mlp)
            for _ in range(config.encoder_layers)
        ])
        self.output = nn.Linear(d, d)

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        if bev.ndim != 4 or bev.shape[1:] != (4, 128, 128) or bev.dtype != torch.uint8:
            raise ValueError(f"Encoder requires uint8 [B, 4, 128, 128], got {bev.shape} {bev.dtype}")
        x = self.stem(bev.float() / 255.0)  # [B, 128, 32, 32]
        tokens = self.patch(x).flatten(2).transpose(1, 2)  # [B, 64, 256]
        return self.output(self.blocks(tokens + self.position))


class OneStepPredictor(nn.Module):
    """Predicts next latent tokens conditioned on current motion (speed, yaw rate)."""

    def __init__(self, config: Stage3Config, token_count: int):
        super().__init__()
        d = config.predictor_dim
        self.input = nn.Linear(config.encoder_dim, d)
        self.ego = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.position = nn.Parameter(torch.empty(1, token_count, d))
        nn.init.normal_(self.position, std=0.02)
        self.blocks = nn.Sequential(*[
            TransformerBlock(d, config.predictor_heads, config.predictor_mlp)
            for _ in range(config.predictor_layers)
        ])
        self.output = nn.Linear(d, config.encoder_dim)
        # Initialize output projection to zero so training starts exactly at copy-current baseline
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, current: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        x = self.input(current) + self.ego(ego)[:, None] + self.position
        return current + self.output(self.blocks(x))


class LatentModel(nn.Module):
    """End-to-end self-supervised latent prediction model."""

    def __init__(self, config: Stage3Config):
        super().__init__()
        self.encoder = BEVEncoder(config)
        self.predictor = OneStepPredictor(config, self.encoder.token_count)

    def forward(self, current_bev: torch.Tensor, ego: torch.Tensor, future_bev: torch.Tensor = None):
        if future_bev is None:
            current = self.encoder(current_bev)
            return self.predictor(current, ego)
        current, target = self.encoder(torch.cat([current_bev, future_bev])).chunk(2)
        return self.predictor(current, ego), target, current
