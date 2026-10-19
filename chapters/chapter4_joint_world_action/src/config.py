"""Chapter 4 Configuration: Stage 4 Joint World & Action Model."""
from dataclasses import dataclass


@dataclass
class Stage4Config:
    # Random seed
    seed: int = 20261018

    # BEV inputs
    bev_channels: int = 4
    bev_size: int = 128
    patch_size: int = 16  # -> 8x8 = 64 tokens

    # Architecture dimensions
    encoder_dim: int = 256
    encoder_depth: int = 4
    encoder_heads: int = 4
    predictor_dim: int = 256
    predictor_depth: int = 4
    predictor_heads: int = 4
    mlp_ratio: float = 4.0

    # Temporal horizons
    history_frames: int = 4
    future_frames: int = 4
    history_dt_s: float = 0.1
    trajectory_steps: int = 10
    waypoint_dt_s: float = 0.2

    # Action planning head
    action_projection: int = 128
    action_hidden: int = 512
    beta_action: float = 100.0
    latent_weight: float = 1.0  # Set to 0.0 for action-only ablation

    # VICReg hyperparameters
    variance_gamma: float = 1.0
    variance_epsilon: float = 1e-4
    lambda_sim: float = 25.0
    lambda_var: float = 25.0
    lambda_cov: float = 1.0

    # Optimization
    epochs: int = 12
    batch_size: int = 128
    learning_rate: float = 3e-4
    encoder_learning_rate: float = 3e-5
    predictor_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    workers: int = 4

    # Stage 3 pretrained checkpoint
    stage3_checkpoint: str = "artifacts/restart_stage3/vicreg_001/best.pt"

    # Hardware & Operator options
    sdpa_backend: str = "AUTO"  # 'AUTO', 'MATH', 'FLASH', 'EFFICIENT'
    device: str = "cuda"
    max_vram_gb: float | None = None
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    enable_oom_recovery: bool = True
    oom_max_retries: int = 3
