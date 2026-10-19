"""Chapter 5: Perception Degradation Operators for Stress Testing.

Provides modular operators for simulating real-world sensor corruptions:
1. Spatial Occlusions: Front blindspot (truck occlusion), random rectangular cutout.
2. Semantic Boundary Loss: Faded lane markings, boundary erosion, road mask dropout.
3. Temporal Frame Dropout: Dropping intermediate or recent history frames (sensor freeze / packet loss).
"""
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch


def apply_front_blindspot(
    bev: np.ndarray | torch.Tensor,
    width_m: float = 16.0,
    depth_m: float = 20.0,
    offset_forward_m: float = 4.0,
    resolution_m: float = 0.5,
) -> np.ndarray | torch.Tensor:
    """遮挡自车前方矩形区域，四通道同时清零（含 visibility）。

    Args:
        bev: Array of shape [4, 128, 128] or [B, 4, 128, 128] (meters: 64m x 64m at 0.5m/px).
             自车中心为 (63.5, 63.5)，前向对应行号减小。
        width_m: Width of the blindspot in meters.
        depth_m: Depth of the blindspot in meters.
        offset_forward_m: 自车中心到矩形起点的前向距离（米）。
        resolution_m: Meters per pixel (default 0.5m/px).
    """
    is_torch = isinstance(bev, torch.Tensor)
    if is_torch:
        bev_np = bev.cpu().numpy().copy()
    else:
        bev_np = bev.copy()

    # Dimensions in pixels
    w_px = int(width_m / resolution_m)
    d_px = int(depth_m / resolution_m)
    offset_px = int(offset_forward_m / resolution_m)

    # 正式栅格中心为 (63.5,63.5)，自车前方映射到更小的行号。
    # ceil 将几何边界转换为数组切片端点，保持左右对称。
    cx, cy = (bev_np.shape[-1] - 1) / 2, (bev_np.shape[-2] - 1) / 2
    y_start = max(0, int(np.ceil(cy - offset_px - d_px)))
    y_end = max(0, int(np.ceil(cy - offset_px)))
    x_start = max(0, int(np.ceil(cx - w_px / 2)))
    x_end = min(bev_np.shape[-1], int(np.ceil(cx + w_px / 2)))

    if bev_np.ndim == 3:  # [C, H, W]
        bev_np[:, y_start:y_end, x_start:x_end] = 0
    elif bev_np.ndim == 4:  # [B, C, H, W]
        bev_np[:, :, y_start:y_end, x_start:x_end] = 0
    elif bev_np.ndim == 5:  # [B, T, C, H, W]
        bev_np[:, :, :, y_start:y_end, x_start:x_end] = 0

    return torch.from_numpy(bev_np).to(bev.device) if is_torch else bev_np


def apply_random_box_occlusion(
    bev: np.ndarray | torch.Tensor,
    num_boxes: int = 3,
    box_size_px: int = 24,
    seed: int | None = None,
) -> np.ndarray | torch.Tensor:
    """Applies random rectangular cutouts to simulate dirt, puddles, or camera blind patches."""
    is_torch = isinstance(bev, torch.Tensor)
    bev_np = bev.cpu().numpy().copy() if is_torch else bev.copy()
    rng = np.random.default_rng(seed)

    for _ in range(num_boxes):
        y0 = rng.integers(0, max(1, 128 - box_size_px))
        x0 = rng.integers(0, max(1, 128 - box_size_px))
        y1 = min(128, y0 + box_size_px)
        x1 = min(128, x0 + box_size_px)

        if bev_np.ndim == 3:
            bev_np[:, y0:y1, x0:x1] = 0
        elif bev_np.ndim == 4:
            bev_np[:, :, y0:y1, x0:x1] = 0
        elif bev_np.ndim == 5:
            bev_np[:, :, :, y0:y1, x0:x1] = 0

    return torch.from_numpy(bev_np).to(bev.device) if is_torch else bev_np


def apply_boundary_loss(
    bev: np.ndarray | torch.Tensor,
    drop_channels: Sequence[int] = (1,),  # 正式通道1：人行道、围栏、标志等静态边界要素
    drop_rate: float = 0.5,
    seed: int | None = None,
) -> np.ndarray | torch.Tensor:
    """按像素随机删除指定语义通道；并不代表独立的车道线磨损模型。"""
    is_torch = isinstance(bev, torch.Tensor)
    bev_np = bev.cpu().numpy().copy() if is_torch else bev.copy()
    rng = np.random.default_rng(seed)

    mask = rng.random(size=(128, 128)) > drop_rate  # True = keep, False = drop

    for ch in drop_channels:
        if bev_np.ndim == 3:
            bev_np[ch] = (bev_np[ch].astype(np.float32) * mask).astype(bev_np.dtype)
        elif bev_np.ndim == 4:
            bev_np[:, ch] = (bev_np[:, ch].astype(np.float32) * mask[None, :, :]).astype(bev_np.dtype)
        elif bev_np.ndim == 5:
            bev_np[:, :, ch] = (bev_np[:, :, ch].astype(np.float32) * mask[None, None, :, :]).astype(bev_np.dtype)

    return torch.from_numpy(bev_np).to(bev.device) if is_torch else bev_np


def apply_temporal_frame_dropout(
    history_bev: np.ndarray | torch.Tensor,
    drop_indices: Sequence[int] = (1, 2),
    strategy: str = "hold_previous",
) -> np.ndarray | torch.Tensor:
    """Simulates camera transmission lag or frame drops in multi-frame history.

    Args:
        history_bev: Tensor or array of shape [4, 4, 128, 128] or [B, 4, 4, 128, 128].
        drop_indices: Indices of history frames to drop (e.g. 1 and 2 out of 0, 1, 2, 3).
        strategy: 'hold_previous' (repeat prior frame) or 'zero_fill' (fill with zeros).
    """
    is_torch = isinstance(history_bev, torch.Tensor)
    h_np = history_bev.cpu().numpy().copy() if is_torch else history_bev.copy()

    # Time dimension is index 0 for [T, C, H, W] or index 1 for [B, T, C, H, W]
    time_dim = 1 if h_np.ndim == 5 else 0

    for idx in drop_indices:
        if strategy == "hold_previous":
            prev_idx = max(0, idx - 1)
            if time_dim == 0:
                h_np[idx] = h_np[prev_idx]
            else:
                h_np[:, idx] = h_np[:, prev_idx]
        elif strategy == "zero_fill":
            if time_dim == 0:
                h_np[idx] = 0
            else:
                h_np[:, idx] = 0

    return torch.from_numpy(h_np).to(history_bev.device) if is_torch else h_np


@dataclass
class DegradationConfig:
    enable_blindspot: bool = False
    blindspot_width_m: float = 16.0
    blindspot_depth_m: float = 20.0
    enable_random_boxes: bool = False
    num_boxes: int = 3
    box_size_px: int = 24
    enable_boundary_loss: bool = False
    boundary_drop_rate: float = 0.5
    enable_frame_dropout: bool = False
    dropped_frames: tuple[int, ...] = (1, 2)
    dropout_strategy: str = "hold_previous"


class DegradationPipeline:
    """Configurable pipeline for applying combinations of perception degradations."""

    def __init__(self, config: DegradationConfig):
        self.config = config

    def __call__(self, history_bev: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        out = history_bev
        if self.config.enable_frame_dropout:
            out = apply_temporal_frame_dropout(
                out,
                drop_indices=self.config.dropped_frames,
                strategy=self.config.dropout_strategy,
            )
        if self.config.enable_blindspot:
            out = apply_front_blindspot(
                out,
                width_m=self.config.blindspot_width_m,
                depth_m=self.config.blindspot_depth_m,
            )
        if self.config.enable_random_boxes:
            out = apply_random_box_occlusion(
                out,
                num_boxes=self.config.num_boxes,
                box_size_px=self.config.box_size_px,
            )
        if self.config.enable_boundary_loss:
            out = apply_boundary_loss(
                out,
                drop_rate=self.config.boundary_drop_rate,
            )
        return out
