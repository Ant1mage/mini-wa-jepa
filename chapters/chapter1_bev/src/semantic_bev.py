"""Chapter 1: Standalone Semantic BEV extraction, channel specification, and coordinate conversion."""
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class SemanticBEVSpec:
    size: int = 128
    range_m: float = 64.0
    channels: int = 4

    @property
    def resolution(self) -> float:
        return self.range_m / self.size  # 0.5 m/pixel


def ego_to_pixels(points_ego, size: int = 128, range_m: float = 64.0) -> np.ndarray:
    """Convert points in ego coordinates (+X forward, +Y left) to pixel coordinates (u column, v row).
    
    像素索引 0~127 的几何中心为 (63.5, 63.5)，与原生相机标定一致。
    Forward (+X) maps to decreasing row index (upwards, -v).
    Left (+Y) maps to decreasing column index (leftwards, -u).
    """
    pts = np.asarray(points_ego)
    scale = size / range_m  # 2.0 pixels/meter
    x = pts[..., 0]
    y = pts[..., 1]
    u = (size - 1) / 2.0 - y * scale
    v = (size - 1) / 2.0 - x * scale
    return np.stack([u, v], axis=-1)


def pixels_to_ego(pixels, size: int = 128, range_m: float = 64.0) -> np.ndarray:
    """Convert pixel coordinates (u, v) back to ego physical coordinates (X forward, Y left)."""
    pts = np.asarray(pixels)
    scale = range_m / size  # 0.5 meters/pixel
    u = pts[..., 0]
    v = pts[..., 1]
    x = ((size - 1) / 2.0 - v) * scale
    y = ((size - 1) / 2.0 - u) * scale
    return np.stack([x, y], axis=-1)


def colorize_bev_4ch(bev_4ch: np.ndarray) -> np.ndarray:
    """Composite a 4-channel uint8 BEV array [4, H, W] into an RGB visualization image [H, W, 3].
    
    Channel semantics:
    - Channel 0: Road surface & lane markings (rendered in White/Gray)
    - Channel 1: 人行道、围栏、交通标志等静态边界要素（蓝色）
    - Channel 2: 其他交通参与者，不包含自车（红色）
    - Channel 3: 可见性；0=不可见，255=可见，不是自车或道路边界
    """
    assert bev_4ch.ndim == 3 and bev_4ch.shape[0] == 4
    h, w = bev_4ch.shape[1], bev_4ch.shape[2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    c0 = bev_4ch[0] > 0
    c1 = bev_4ch[1] > 0
    c2 = bev_4ch[2] > 0
    c3 = bev_4ch[3] > 0

    # Road surface: Gray/White
    rgb[c0] = [180, 180, 180]
    # 正式通道定义来自 env/semantic_bev.py；自车用叠加标记表示。
    rgb[c1] = [40, 120, 240]
    rgb[c2] = [230, 40, 40]
    # 遮挡区域统一压暗，避免把不可见区域误读为可信的空道路。
    rgb[~c3] = [25, 25, 25]

    return rgb
