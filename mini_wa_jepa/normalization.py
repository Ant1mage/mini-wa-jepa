"""One invertible scale/statistics contract shared by datasets and inference."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Normalizer:
    scale: float = 1.0
    trajectory_mean: np.ndarray = field(default_factory=lambda: np.zeros(4,dtype=np.float32))
    trajectory_std: np.ndarray = field(default_factory=lambda: np.ones(4,dtype=np.float32))
    speed_mean: float = 0.0
    speed_std: float = 1.0

    def __post_init__(self):
        if self.scale <= 0 or self.speed_std <= 0:
            raise ValueError("Scales must be positive")
        self.trajectory_mean=np.asarray(self.trajectory_mean,dtype=np.float32)
        self.trajectory_std=np.asarray(self.trajectory_std,dtype=np.float32)
        if np.any(self.trajectory_std <= 0): raise ValueError("Invalid trajectory std")

    def trajectory(self, raw):
        scaled=np.asarray(raw,dtype=np.float32).copy()
        scaled[..., :2] *= self.scale  # angles stay radians
        if scaled.shape[-1] == 4:scaled[..., 2] *= self.scale
        return (scaled-self.trajectory_mean)/self.trajectory_std

    def inverse_trajectory(self, normalized):
        raw=np.asarray(normalized,dtype=np.float32)*self.trajectory_std+self.trajectory_mean
        raw[..., :2] /= self.scale
        if raw.shape[-1] == 4:raw[..., 2] /= self.scale
        return raw

    def speed(self, raw):
        return (np.asarray(raw)*self.scale-self.speed_mean)/self.speed_std

    def inverse_speed(self, normalized):
        return (np.asarray(normalized)*self.speed_std+self.speed_mean)/self.scale

    def to_dict(self):
        return dict(scale=self.scale,trajectory_mean=self.trajectory_mean.tolist(),
                    trajectory_std=self.trajectory_std.tolist(),speed_mean=self.speed_mean,speed_std=self.speed_std)
