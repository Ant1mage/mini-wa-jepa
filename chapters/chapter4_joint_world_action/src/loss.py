"""Chapter 4 Losses: Masked Joint VICReg & Action Loss."""
import torch
import torch.nn.functional as F

from chapters.chapter4_joint_world_action.src.config import Stage4Config


def masked_trajectory_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """L1 trajectory loss with positional weighting (2.0x on x, y) and valid masking."""
    error = (prediction - target).abs()
    # Double weight on x, y coordinates
    weighted = torch.cat([error[:, :, :2] * 2.0, error[:, :, 2:]], dim=-1)
    mask = valid.unsqueeze(-1) & action_mask.reshape(-1, 1, 1)
    return (weighted * mask).sum() / mask.sum().clamp_min(1)


def regularization(x: torch.Tensor, gamma: float = 1.0, epsilon: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg variance and covariance regularization.

    Args:
        x: Token tensor of shape [B, N, D] or [B, D].
        gamma: Target variance standard deviation (default 1.0).
        epsilon: Numerical stability constant.

    Returns:
        (var_loss, cov_loss)
    """
    if x.ndim == 3:
        # Average across tokens to get per-sample representation [B, D]
        x = x.mean(dim=1)

    b, d = x.shape
    if b < 2:
        zero = x.new_zeros(())
        return zero, zero

    # Variance loss: keep standard deviation >= gamma
    std = torch.sqrt(x.var(dim=0) + epsilon)
    var_loss = F.relu(gamma - std).mean()

    # Covariance loss: penalize off-diagonal correlations
    centered = x - x.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (b - 1)
    diag = torch.eye(d, device=x.device, dtype=torch.bool)
    cov_loss = cov.masked_select(~diag).pow(2).sum() / d

    return var_loss, cov_loss


def joint_loss(output: dict, batch: dict, config: Stage4Config) -> tuple[torch.Tensor, dict]:
    """Joint World and Action Loss function.

    Computes:
    1. Masked Action Trajectory Loss (L1 with positional priority).
    2. Multi-step Latent Prediction Invariance MSE (masked by world_valid).
    3. VICReg Variance & Covariance regularization to prevent representation collapse.

    Total Loss = latent_weight * (25 * Sim + 25 * Var + 1 * Cov) + beta_action * Action
    """
    action = masked_trajectory_loss(
        output["trajectory"],
        batch["trajectory"],
        batch["valid"],
        batch["action_mask"],
    )

    zero = action.new_zeros(())
    similarity = zero
    variance = zero
    covariance = zero

    if config.latent_weight > 0.0 and "target" in output and "world_valid" in batch:
        mask = batch["world_valid"]  # [B, 4]
        # MSE error between predicted future tokens and target future tokens: [B, 4]
        error = (output["latent"].float() - output["target"].float()).square().mean(dim=(-1, -2))
        similarity = (error * mask).sum() / mask.sum().clamp_min(1)

        # VICReg regularization across current tokens and valid future target tokens
        regs = [regularization(output["current"], config.variance_gamma, config.variance_epsilon)]
        for k in range(config.future_frames):
            selected = output["target"][:, k][mask[:, k]]
            if len(selected) >= 2:
                regs.append(regularization(selected, config.variance_gamma, config.variance_epsilon))

        variance = torch.stack([r[0] for r in regs]).mean()
        covariance = torch.stack([r[1] for r in regs]).mean()

    latent = (
        config.lambda_sim * similarity +
        config.lambda_var * variance +
        config.lambda_cov * covariance
    )

    total = config.latent_weight * latent + config.beta_action * action

    terms = {
        "action": action,
        "prediction": similarity,
        "variance": variance,
        "covariance": covariance,
        "latent": latent,
    }

    return total, terms
