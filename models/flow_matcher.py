"""Linear-interpolation Conditional Flow Matching trajectory head for multi-candidate generation."""
import math
import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_time_embedding(timesteps, dim):
    """Sinusoidal embedding for continuous time scalar t in [0, 1]."""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=timesteps.dtype) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ConditionalFlowMatcher(nn.Module):
    """Generates 10x4 trajectory from noise conditioned on spatiotemporal BEV tokens and ego motion.

    Uses linear-interpolation conditional flow matching (x_t = (1-t)x_0 + t*x_1, v = x_1 - x_0).
    """

    def __init__(self, token_count=64, token_dim=256, condition_dim=256, hidden_dim=256):
        super().__init__()
        self.token_count = token_count
        self.token_dim = token_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.time_emb_dim = 32

        # Temporal fusion over 4 history frames per spatial token
        self.history_proj = nn.Linear(4 * token_dim, condition_dim)
        self.ego_proj = nn.Linear(8, condition_dim)

        # Vector field network: noisy_x (40) + time_emb (32) + spatial_tokens (token_count * condition_dim)
        in_features = 40 + self.time_emb_dim + token_count * condition_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 40),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @property
    def model_arch(self):
        return dict(
            token_count=self.token_count,
            token_dim=self.token_dim,
            condition_dim=self.condition_dim,
            hidden_dim=self.hidden_dim,
            time_emb_dim=self.time_emb_dim,
        )

    def extract_condition(self, history, ego):
        """Extract condition representation preserving spatial token locations.

        history: [B, 4, N, D]
        ego: [B, 4, 2]
        """
        b, t, n, d = history.shape
        if t != 4 or n != self.token_count or d != self.token_dim:
            raise ValueError(f'Expected history of shape [B, 4, {self.token_count}, {self.token_dim}], got {history.shape}')
        temporal_flat = history.permute(0, 2, 1, 3).reshape(b, n, 4 * d)
        tokens = self.history_proj(temporal_flat)  # [B, N, condition_dim]
        ego_emb = self.ego_proj(ego.reshape(b, 8))[:, None, :]  # [B, 1, condition_dim]
        fused = tokens + ego_emb
        return fused.reshape(b, n * self.condition_dim)  # [B, N * condition_dim]

    def vector_field(self, x_t, t, condition):
        """Predict flow vector field v_t(x_t | condition).

        x_t: [B, 40] or [B, 10, 4]
        t: [B] in [0, 1]
        condition: [B, N * condition_dim]
        """
        b = len(x_t)
        x_flat = x_t.reshape(b, 40)
        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)
        inp = torch.cat([x_flat, t_emb, condition], dim=-1)
        return self.net(inp).reshape(b, 10, 4)

    def compute_loss(self, x_1, condition, action_mask, waypoint_valid, noise=None, t=None):
        """Compute Flow Matching MSE loss with sanitized invalid targets.

        Target: x_t = (1 - t) * x_0 + t * x_1
        Velocity field target: u_t = x_1 - x_0
        Invalid points are sanitized to x_0 so junk values never enter the network through x_t.
        """
        b = len(x_1)
        device = x_1.device
        dtype = x_1.dtype

        if noise is None:
            x_0 = torch.randn_like(x_1)
        else:
            x_0 = noise

        if t is None:
            t_sample = torch.rand(b, device=device, dtype=dtype)
        else:
            t_sample = t

        # Point mask: [B, 10, 1]
        point_mask = (action_mask[:, None] & waypoint_valid)[..., None]

        # Sanitize x_1 so invalid targets do not corrupt x_t
        x_1_clean = torch.where(point_mask, x_1, x_0)

        t_expand = t_sample[:, None, None]
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1_clean
        v_target = x_1_clean - x_0

        v_pred = self.vector_field(x_t, t_sample, condition)

        error_sq = (v_pred - v_target).square() * point_mask.float()
        # Normalization: true coordinate MSE across valid points (4 coordinates per point)
        denom = point_mask.float().sum() * 4.0
        loss = error_sq.sum() / denom.clamp_min(1.0)
        return loss, v_pred

    @torch.no_grad()
    def sample(self, condition, num_steps=10, num_candidates=1, seed=None, device=None):
        """Batched Euler ODE integration from Gaussian noise to trajectory: t from 0 to 1.

        Batches all K candidates across the batch into a single [B * K] tensor per Euler step.
        """
        b = len(condition)
        if device is None:
            device = condition.device

        k = num_candidates
        cond_rep = condition.repeat_interleave(k, dim=0)

        gen = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)

        x = torch.randn(b * k, 10, 4, generator=gen, device=device, dtype=condition.dtype)
        dt = 1.0 / num_steps

        for s in range(num_steps):
            t_val = s * dt
            t_tensor = torch.full((b * k,), t_val, device=device, dtype=condition.dtype)
            v = self.vector_field(x, t_tensor, cond_rep)
            x = x + v * dt

        x_out = x.reshape(b, k, 10, 4)
        if k == 1:
            return x_out[:, 0]
        return x_out
