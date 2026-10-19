"""Chapter 3: Self-contained VICReg (Variance-Invariance-Covariance) loss formulation."""
from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def variance_loss(x: torch.Tensor, gamma: float = 1.0, epsilon: float = 1e-4) -> torch.Tensor:
    """Encourages standard deviation of each feature across the batch to be at least gamma."""
    std = torch.sqrt(x.var(dim=0, unbiased=False) + epsilon)
    return F.relu(gamma - std).mean()


def covariance_loss(x: torch.Tensor) -> torch.Tensor:
    """Penalizes off-diagonal covariance terms across the feature dimensions to decorrelate features."""
    b, d = x.shape
    centered = x - x.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / max(b - 1, 1)
    # Mask out diagonal terms
    diag_mask = torch.eye(d, dtype=torch.bool, device=x.device)
    off_diag = cov[~diag_mask]
    return off_diag.square().sum() / d


def latent_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
    config: any,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Complete VICReg combined loss for self-supervised world prediction.
    
    Loss = lambda_sim * Invariance(MSE) + lambda_var * Variance + lambda_cov * Covariance
    """
    sim = F.mse_loss(prediction, target)

    # Flatten spatial tokens into batch dimension for statistical regularization
    # target: [B, N, D] -> [B*N, D]
    flat_target = target.reshape(-1, config.encoder_dim)
    var = variance_loss(flat_target, gamma=config.variance_gamma, epsilon=config.variance_epsilon)
    cov = covariance_loss(flat_target)

    total = config.lambda_sim * sim + config.lambda_var * var + config.lambda_cov * cov
    terms = {
        "prediction": sim.detach(),
        "variance": var.detach(),
        "covariance": cov.detach(),
        "total": total.detach(),
    }
    return total, terms
