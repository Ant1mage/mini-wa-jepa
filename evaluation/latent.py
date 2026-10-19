"""Float64 streaming covariance, content PCA and episode-cluster inference."""
import numpy as np
import torch


def spectrum_metrics(covariance):
    covariance = np.asarray(covariance, np.float64)
    covariance = (covariance + covariance.T) / 2
    eigenvalues, basis = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.)
    basis = basis[:, order]
    trace = float(eigenvalues.sum())
    if trace <= 1e-15:
        effective = participation = 0.
    else:
        p = eigenvalues[eigenvalues > 0] / trace
        effective = float(np.exp(-(p * np.log(p)).sum()))
        participation = float(1. / np.square(p).sum())
    std = np.sqrt(np.maximum(np.diag(covariance), 0.))
    metrics = dict(effective_rank=effective, effective_rank_ratio=effective / len(eigenvalues),
        participation_rank=participation, std_mean=float(std.mean()), std_min=float(std.min()),
        std_p10=float(np.percentile(std, 10)), covariance_trace=trace,
        eigenvalues=eigenvalues.tolist(), channel_std=std.tolist())
    # Fix PCA signs for reproducible plots; target/prediction share the same basis.
    for column in range(basis.shape[1]):
        pivot = np.argmax(np.abs(basis[:, column]))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    return metrics, basis


class LatentMoments:
    """Keep only sufficient statistics, never an entire validation set on the GPU."""
    def __init__(self, tokens, channels, device):
        self.count = 0
        self.tokens = tokens
        self.mean_position = torch.zeros(tokens, channels, dtype=torch.float64, device=device)
        self.cross = torch.zeros(channels, channels, dtype=torch.float64, device=device)
        self.pool_mean = torch.zeros(channels, dtype=torch.float64, device=device)
        self.pool_cross = self.cross.clone()

    @torch.no_grad()
    def update(self, z):
        z = z.detach().double()
        # Parallel Welford merge avoids cancellation of nearly constant features.
        previous, batch = self.count, len(z)
        self.count += batch
        mean = z.mean(0)
        delta = mean - self.mean_position
        flat = (z - mean).flatten(0, 1)
        self.cross += flat.T @ flat + delta.T @ delta * (previous * batch / self.count)
        self.mean_position += delta * (batch / self.count)
        pooled = z.mean(1)
        pool_mean = pooled.mean(0)
        pool_delta = pool_mean - self.pool_mean
        pool_centered = pooled - pool_mean
        self.pool_cross += (pool_centered.T @ pool_centered +
                           pool_delta[:, None] @ pool_delta[None] * (previous * batch / self.count))
        self.pool_mean += pool_delta * (batch / self.count)

    def finish(self):
        if self.count < 2:
            raise ValueError('At least two frames required for health evaluation')
        n, t = self.count, self.tokens
        cov = self.cross / (t * (n - 1))
        fixed = self.mean_position - self.mean_position.mean(0)
        raw_cov = (self.cross + n * fixed.T @ fixed) / (n * t - 1)
        pool_cov = self.pool_cross / (n - 1)
        health, basis = spectrum_metrics(cov.cpu().numpy())
        raw, _ = spectrum_metrics(raw_cov.cpu().numpy())
        pooled, _ = spectrum_metrics(pool_cov.cpu().numpy())
        health.update(frames=n, theoretical_channel_upper_bound=cov.shape[0],
                      raw_rank_ratio=raw['effective_rank_ratio'],
                      pooled_effective_rank_ratio=pooled['effective_rank_ratio'],
                      pooled_std_mean=pooled['std_mean'])
        return health, dict(covariance=cov.cpu().numpy(), pooled_covariance=pool_cov.cpu().numpy(),
                            position_mean=self.mean_position.cpu().numpy(), pca_basis=basis)


def paired_episode_bootstrap(prediction, copy, episodes, samples, seed):
    prediction, copy, episodes = np.asarray(prediction), np.asarray(copy), np.asarray(episodes)
    unique = np.unique(episodes)
    if len(unique) < 2:
        return dict(episode_count=len(unique), delta_ci95=None, ratio_ci95=None)
    stats = np.array([[prediction[episodes == ep].sum(), copy[episodes == ep].sum(),
                       (episodes == ep).sum()] for ep in unique], dtype=np.float64)
    rng = np.random.default_rng(seed)
    totals = stats[rng.integers(len(unique), size=(samples, len(unique)))].sum(1)
    delta = (totals[:, 0] - totals[:, 1]) / totals[:, 2]
    ratio = totals[:, 0] / np.maximum(totals[:, 1], 1e-15)
    return dict(episode_count=len(unique), bootstrap_samples=samples, seed=seed,
                delta_ci95=np.percentile(delta, [2.5, 97.5]).tolist(),
                ratio_ci95=np.percentile(ratio, [2.5, 97.5]).tolist())

