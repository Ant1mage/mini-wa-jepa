"""Stage 1 only: simulator and renderer verification, no training."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Stage1Config:
    episodes: int = 100
    start_seed: int = 0
    horizon: int = 500
    map: int = 3
    traffic_density: float = 0.1
    bev_size: int = 256
    bev_pixels_per_metre: float = 4.0
    film_size: int = 2048
    rgb_width: int = 256
    rgb_height: int = 144
    capture_count: int = 20

    def __post_init__(self):
        if min(self.episodes, self.horizon, self.bev_size, self.film_size,
               self.rgb_width, self.rgb_height) <= 0:
            raise ValueError("Counts and resolutions must be positive")
        if self.bev_pixels_per_metre <= 0 or self.capture_count < 0:
            raise ValueError("Invalid scaling or capture count")
