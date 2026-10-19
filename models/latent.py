"""Original pre-norm SDPA transformers for deterministic stage 3 prediction."""
import torch
from torch import nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, hidden):
        super().__init__()
        if dim % heads:
            raise ValueError('Attention dimension must be divisible by heads')
        self.heads = heads
        self.norm_attention = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.projection = nn.Linear(dim, dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(self.norm_attention(x)).reshape(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        x = x + self.projection(attended.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.norm_mlp(x))


class BEVEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.encoder_dim
        self.stem = nn.Sequential(
            nn.Conv2d(4, 64, 5, stride=2, padding=2), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU())
        self.patch = nn.Conv2d(128, d, config.patch_stride, stride=config.patch_stride)
        self.token_count = (32 // config.patch_stride) ** 2
        self.position = nn.Parameter(torch.empty(1, self.token_count, d))
        nn.init.normal_(self.position, std=.02)
        self.blocks = nn.Sequential(*[TransformerBlock(d, config.encoder_heads, config.encoder_mlp)
                                      for _ in range(config.encoder_layers)])
        self.output = nn.Linear(d, d)

    def forward(self, bev):
        if bev.ndim != 4 or bev.shape[1:] != (4, 128, 128) or bev.dtype != torch.uint8:
            raise ValueError('Encoder requires uint8 [B,4,128,128]')
        x = self.stem(bev.float() / 255.)
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        # No final LayerNorm constraint on the channel covariance rank.
        return self.output(self.blocks(tokens + self.position))


class OneStepPredictor(nn.Module):
    def __init__(self, config, token_count):
        super().__init__()
        d = config.predictor_dim
        self.input = nn.Linear(config.encoder_dim, d)
        self.ego = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.position = nn.Parameter(torch.empty(1, token_count, d))
        nn.init.normal_(self.position, std=.02)
        self.blocks = nn.Sequential(*[TransformerBlock(d, config.predictor_heads, config.predictor_mlp)
                                      for _ in range(config.predictor_layers)])
        self.output = nn.Linear(d, config.encoder_dim)
        # Start exactly at the measured copy-current baseline, then learn its residual.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, current, ego):
        x = self.input(current) + self.ego(ego)[:, None] + self.position
        return current + self.output(self.blocks(x))


class LatentModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BEVEncoder(config)
        self.predictor = OneStepPredictor(config, self.encoder.token_count)

    def forward(self, current_bev, ego, future_bev=None):
        if future_bev is None:
            current = self.encoder(current_bev)
            return self.predictor(current, ego)
        # Both branches receive gradients: anti-collapse comes from explicit VICReg.
        current, target = self.encoder(torch.cat([current_bev, future_bev])).chunk(2)
        return self.predictor(current, ego), target, current

