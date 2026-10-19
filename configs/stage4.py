"""Deterministic stage 4 settings; no degradation or generative action head."""
from dataclasses import dataclass
from configs.stage3 import Stage3Config


@dataclass(frozen=True)
class Stage4Config(Stage3Config):
    seed: int = 20261018
    epochs: int = 12
    batch_size: int = 128
    learning_rate: float = 3e-4
    encoder_learning_rate: float = 3e-5
    predictor_learning_rate: float = 1e-4
    history_frames: int = 4
    future_frames: int = 4
    history_dt_s: float = .1
    trajectory_steps: int = 10
    waypoint_dt_s: float = .2
    action_projection: int = 128
    action_hidden: int = 512
    beta_action: float = 100.
    latent_weight: float = 1.
    stage3_checkpoint: str = 'artifacts/restart_stage3/vicreg_001/best.pt'
    visualization_sample_ids: tuple = (0, 512, 5500, 8000)
    visualization_seed: int = 20261018
    # Hardware detection, operator selection, and calibration infrastructure
    sdpa_backend: str = 'MATH'  # 'MATH' (Stage 4 baseline), 'AUTO', 'FLASH', 'EFFICIENT', 'CUDNN'
    max_vram_gb: float | None = None  # None: dynamic headroom based on detected hardware; float: fixed ceiling
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    enable_oom_recovery: bool = False
    oom_max_retries: int = 3
    auto_calibrate: bool = False
    device: str = 'cuda'  # 'cuda' (supports ROCm/CUDA), 'cpu', 'mps'

