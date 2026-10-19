"""Stage 2 v2: frozen data, expert and baseline settings; test maps are held out."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BEVConfig:
    size: int = 128
    extent_m: float = 64.0
    height_m: float = 50.0
    line_width_m: float = 0.3  # legacy geometry renderer only, never used by v2


@dataclass(frozen=True)
class ExpertConfig:
    target_speed_m_s: float = 7.0
    min_lookahead_m: float = 4.0
    lookahead_time_s: float = 0.65
    max_lateral_accel_m_s2: float = 2.0
    speed_kp: float = 0.5
    speed_ki: float = 0.08
    speed_kd: float = 0.02
    headway_s: float = 1.8
    standstill_gap_m: float = 5.0


@dataclass(frozen=True)
class ActionNoiseConfig:
    enabled: bool = True
    event_probability: float = 0.10  # per eligible decision; not fraction of noisy frames
    steering_std: float = 0.18
    throttle_std: float = 0.12
    duration_min: int = 4
    duration_max: int = 8
    recovery_steps: int = 8  # full expert authority before another pulse is eligible


@dataclass(frozen=True)
class BEVNoiseConfig:
    enabled: bool = False
    missed_vehicle_probability: float = 0.15
    boundary_jitter_pixels: int = 1
    flicker_probability: float = 0.05


@dataclass(frozen=True)
class Stage2Config:
    seed: int = 20261016
    frames: int = 100000  # TRAIN frames; val/test quotas are additional
    validation_frames: int = 10000
    test_frames: int = 10000
    horizon: int = 1000
    dt_s: float = 0.1
    future_steps: int = 10
    waypoint_dt_s: float = 0.2
    history_frames: int = 4
    world_future_frames: int = 4
    shard_frames: int = 512
    traffic_density: float = 0.1
    scale: float = 1.0
    egl_device: int = 1
    offroad_grace_s: float = 1.0
    bev: BEVConfig = field(default_factory=BEVConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    action_noise: ActionNoiseConfig = field(default_factory=ActionNoiseConfig)

    @property
    def waypoint_stride(self):
        stride=round(self.waypoint_dt_s/self.dt_s)
        if abs(stride*self.dt_s-self.waypoint_dt_s)>1e-8:
            raise ValueError('Waypoint interval must be an integer number of decisions')
        return stride


@dataclass(frozen=True)
class BaselineConfig:
    seed: int = 20261016
    epochs: int = 12
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    workers: int = 2
    cpu_threads: int = 6
    mixed_precision: str = 'bf16'
    visualization_interval_epochs: int = 1
    visualization_samples: int = 8
    pooling_size: int = 4
    mlp_hidden: int = 1024
    latent_dim: int = 512


SPLITS = {
    'train': {'maps': ['S', 'C', 'SC', 'CS'], 'seed_start': 100000},
    'val': {'maps': ['CC', 'SS'], 'seed_start': 200000},
    'test': {'maps': ['SCS', 'CSC'], 'seed_start': 300000},
}
