"""Chapter 4 Models: Joint World & Action Model.

This module provides:
1. `constant_turn_trajectory`: Parameter-free kinematic prior (bicycle model / constant turn).
2. `BEVEncoder`: ConvStem + ViT token encoder mapping 4-channel BEV to 64 spatial tokens.
3. `OneStepPredictor`: Multi-horizon latent predictor forecasting 4 future token states.
4. `JointModel`: Spatially preserved Joint World/Action Model combining latent world foresight
   with residual trajectory planning.
"""
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from chapters.chapter4_joint_world_action.src.config import Stage4Config


def constant_turn_trajectory(ego: torch.Tensor, steps: int = 10, dt: float = 0.2) -> torch.Tensor:
    """Parameter-free constant-speed/yaw-rate kinematic prior from observed motion.

    Args:
        ego: Tensor of shape [B, 4, 2] or [B, 2] containing (normalized_speed, yaw_rate).
             Speed is in (m/s)/10, yaw_rate is in rad/s.
        steps: Number of waypoints to generate (default 10).
        dt: Time interval between waypoints in seconds (default 0.2s).

    Returns:
        Tensor of shape [B, steps, 4] containing (x, y, v, heading).
    """
    if ego.ndim == 3:
        current_ego = ego[:, -1]
    else:
        current_ego = ego

    t = torch.arange(1, steps + 1, device=ego.device, dtype=ego.dtype) * dt  # [steps]
    speed = current_ego[:, 0:1] * 10.0  # de-normalize to m/s, [B, 1]
    yaw = current_ego[:, 1:2]            # rad/s, [B, 1]

    angle = yaw * t                      # [B, steps]
    x = speed * t * torch.sinc(angle / torch.pi)
    y = speed * t * (angle / 2.0) * torch.sinc(angle / (2.0 * torch.pi)).square()
    v = speed.expand_as(x)

    return torch.stack([x, y, v, angle], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim)
        self.attn_proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        normed = self.norm1(x)
        qkv = self.attn_qkv(normed).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(b, n, d)
        x = x + self.attn_proj(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class BEVEncoder(nn.Module):
    """ConvStem + Transformer Tokenizer for 4-channel BEV maps."""

    def __init__(self, config: Stage4Config):
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv2d(config.bev_channels, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, config.encoder_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.encoder_dim),
            nn.GELU(),
            nn.Conv2d(config.encoder_dim, config.encoder_dim, kernel_size=4, stride=4, padding=0),
        )
        self.token_grid = config.bev_size // config.patch_size  # 128 // 16 = 8
        self.token_count = self.token_grid * self.token_grid   # 64
        self.pos_embed = nn.Parameter(torch.zeros(1, self.token_count, config.encoder_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.Sequential(*[
            TransformerBlock(config.encoder_dim, config.encoder_heads, config.mlp_ratio)
            for _ in range(config.encoder_depth)
        ])
        self.norm = nn.LayerNorm(config.encoder_dim)

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        x = bev.float() / 255.0
        x = self.stem(x)  # [B, 256, 8, 8]
        x = x.flatten(2).transpose(1, 2)  # [B, 64, 256]
        x = x + self.pos_embed
        x = self.blocks(x)
        return self.norm(x)


class OneStepPredictor(nn.Module):
    """Latent future token predictor conditioned on ego velocity and yaw rate."""

    def __init__(self, config: Stage4Config, token_count: int):
        super().__init__()
        self.input = nn.Linear(config.encoder_dim, config.predictor_dim)
        self.ego = nn.Linear(2, config.predictor_dim)
        self.position = nn.Parameter(torch.zeros(1, token_count, config.predictor_dim))
        nn.init.trunc_normal_(self.position, std=0.02)

        self.blocks = nn.Sequential(*[
            TransformerBlock(config.predictor_dim, config.predictor_heads, config.mlp_ratio)
            for _ in range(config.predictor_depth)
        ])
        # Forecast multiple future horizons in parallel
        self.output = nn.Linear(config.predictor_dim, config.future_frames * config.encoder_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, current_latent: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        b, n, _ = current_latent.shape
        x = self.input(current_latent) + self.ego(ego)[:, None, :] + self.position
        x = self.blocks(x)
        return self.output(x)


class JointModel(nn.Module):
    """Stage 4 Spatially Preserved Joint World and Action Model."""

    def __init__(self, config: Stage4Config, normalizer=None):
        super().__init__()
        self.config = config
        self.encoder = BEVEncoder(config)
        self.predictor = OneStepPredictor(config, self.encoder.token_count)

        # Zero-initialized history fusion
        self.history_fusion = nn.Linear(3 * config.encoder_dim + 8, config.encoder_dim)
        nn.init.zeros_(self.history_fusion.weight)
        nn.init.zeros_(self.history_fusion.bias)

        # Spatial token preservation: [B, 64, 5*256] -> [B, 64, 128]
        self.action_projection = nn.Sequential(
            nn.Linear(5 * config.encoder_dim, config.action_projection),
            nn.GELU(),
        )

        # Flattened spatial tokens (64 * 128 = 8192) + 8 ego inputs -> MLP -> 10 waypoints * 4 = 40
        action_in_dim = self.encoder.token_count * config.action_projection + 8
        self.action_head = nn.Sequential(
            nn.Linear(action_in_dim, config.action_hidden),
            nn.GELU(),
            nn.Linear(config.action_hidden, config.action_hidden),
            nn.GELU(),
            nn.Linear(config.action_hidden, config.trajectory_steps * 4),
        )
        nn.init.zeros_(self.action_head[-1].weight)
        nn.init.zeros_(self.action_head[-1].bias)

        # Trajectory normalization buffers
        mean = getattr(normalizer, "trajectory_mean", [7.325, -0.028, 6.843, -0.004])
        std = getattr(normalizer, "trajectory_std", [4.027, 0.801, 0.707, 0.117])
        scale = float(getattr(normalizer, "scale", 1.0))

        self.register_buffer("trajectory_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("trajectory_std", torch.tensor(std, dtype=torch.float32))
        self.register_buffer("trajectory_scale", torch.tensor(scale, dtype=torch.float32))

    def load_stage3_checkpoint(self, checkpoint_path: str | Path):
        """Warm-start weights from Stage 3 Latent World Model."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"[!] Stage 3 checkpoint {checkpoint_path} not found. Training from scratch.")
            return

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_dict = state.get("model", state.get("model_state_dict", state))

        # 1. Load encoder weights
        encoder_weights = {k[8:]: v for k, v in model_dict.items() if k.startswith("encoder.")}
        if encoder_weights:
            self.encoder.load_state_dict(encoder_weights)
            print(f"[*] Loaded {len(encoder_weights)} encoder tensors from Stage 3 checkpoint.")

        # 2. Load shared predictor weights
        shared_pred = {
            k[10:]: v for k, v in model_dict.items()
            if k.startswith("predictor.") and not k.startswith("predictor.output.")
        }
        if shared_pred:
            self.predictor.load_state_dict(shared_pred, strict=False)
            print(f"[*] Loaded {len(shared_pred)} predictor backbone tensors from Stage 3 checkpoint.")

        # 3. Initialize multi-horizon output weights by scaling the single-step projection
        if "predictor.output.weight" in model_dict:
            single_w = model_dict["predictor.output.weight"]
            single_b = model_dict["predictor.output.bias"]
            with torch.no_grad():
                for h in range(self.config.future_frames):
                    start = h * self.config.encoder_dim
                    stop = start + self.config.encoder_dim
                    self.predictor.output.weight[start:stop].copy_(single_w * (h + 1))
                    self.predictor.output.bias[start:stop].copy_(single_b * (h + 1))
            print(f"[*] Initialized multi-horizon output projection ({self.config.future_frames} horizons).")

    def forward_encoded(self, history: torch.Tensor, ego: torch.Tensor) -> dict:
        """Forward pass taking pre-encoded BEV history tokens.

        Args:
            history: [B, 4, N, D] encoded tokens for past 4 frames.
            ego: [B, 4, 2] observed motion for past 4 frames.
        """
        b, _, n, d = history.shape
        current = history[:, -1]  # [B, N, D]

        # Temporal differences relative to current frame
        differences = (history[:, :3] - current[:, None]).permute(0, 2, 1, 3).reshape(b, n, 3 * d)
        motion = ego.flatten(1)  # [B, 8]
        fused = current + self.history_fusion(torch.cat([differences, motion[:, None].expand(-1, n, -1)], dim=-1))

        # Predict future 4 steps in latent space
        hidden = self.predictor.input(fused) + self.predictor.ego(ego[:, -1])[:, None] + self.predictor.position
        hidden = self.predictor.blocks(hidden)
        delta = self.predictor.output(hidden).reshape(b, n, 4, d).permute(0, 2, 1, 3)  # [B, 4, N, D]
        future = current[:, None] + delta  # [B, 4, N, D]

        # Spatial token preservation: concatenate current and future tokens per location
        content = torch.cat([current, future.permute(0, 2, 1, 3).reshape(b, n, 4 * d)], dim=-1)  # [B, N, 5*D]
        spatial = self.action_projection(content).flatten(1)  # [B, N * action_proj]

        # Predict residual trajectory
        residual = self.action_head(torch.cat([spatial, motion], dim=-1)).reshape(b, self.config.trajectory_steps, 4)

        # Compute parameter-free kinematic prior
        raw_prior = constant_turn_trajectory(ego.float(), steps=self.config.trajectory_steps, dt=self.config.waypoint_dt_s)
        scale = torch.stack([self.trajectory_scale] * 3 + [self.trajectory_scale.new_tensor(1.0)])
        prior = (raw_prior * scale - self.trajectory_mean) / self.trajectory_std

        return {
            "trajectory": prior + residual.float(),
            "latent": future,
            "current": current,
            "prior": prior,
            "residual": residual,
        }

    def forward(self, history: torch.Tensor, ego: torch.Tensor, future_bev: torch.Tensor | None = None) -> dict:
        """Full forward pass from raw BEV images.

        Args:
            history: [B, 4, 4, 128, 128] uint8 BEV images.
            ego: [B, 4, 2] float32 observed motion.
            future_bev: Optional [B, 4, 4, 128, 128] uint8 BEV images for computing supervision loss.
        """
        b = len(history)
        observed = history.flatten(0, 1)  # [B*4, 4, 128, 128]
        frames = observed if future_bev is None else torch.cat([observed, future_bev.flatten(0, 1)], dim=0)

        encoded = self.encoder(frames)
        result = self.forward_encoded(encoded[:b * 4].reshape(b, 4, self.encoder.token_count, -1), ego)

        if future_bev is not None:
            result["target"] = encoded[b * 4:].reshape(b, 4, self.encoder.token_count, -1)

        return result

    def unnormalize_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Convert normalized trajectory back to physical coordinates (meters, m/s, radians)."""
        scale = torch.stack([self.trajectory_scale] * 3 + [self.trajectory_scale.new_tensor(1.0)])
        return (trajectory * self.trajectory_std + self.trajectory_mean) / scale
