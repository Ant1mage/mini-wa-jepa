"""Chapter 2: Configuration for the ResNet-18 behavior cloning baseline."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineConfig:
    seed: int = 20261017
    epochs: int = 12
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    workers: int = 2
    cpu_threads: int = 6
    latent_dim: int = 512
    mlp_hidden: int = 256
    pooling_size: int = 4
    mixed_precision: str = "bf16"
    gradient_clip: float = 5.0
    trajectory_steps: int = 10
    waypoint_dt_s: float = 0.2


Stage2Config = BaselineConfig
