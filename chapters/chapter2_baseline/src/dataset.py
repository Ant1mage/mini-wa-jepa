"""Chapter 2 Dataset: Single-frame BEV behavior cloning trajectory dataset."""
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from mini_wa_jepa.geometry import wrap_angle
from mini_wa_jepa.normalization import Normalizer


def validate_manifest(root: Path) -> dict:
    manifest_path = Path(root) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 2 or not manifest.get("completed"):
        raise ValueError("Only a completed v2 dataset schema is supported.")
    return manifest


class BaselineDataset(Dataset):
    """Single-frame BEV imitation learning dataset with memmap loading."""

    def __init__(self, root: str | Path, split: str = "train"):
        self.root = Path(root)
        self.split = split
        manifest = validate_manifest(self.root)

        cache_dir = self.root / "baseline_cache" / split
        if not cache_dir.exists():
            raise FileNotFoundError(f"Cache directory {cache_dir} not found.")

        self.cache_dir = cache_dir
        self.bev = np.load(cache_dir / "bev.npy", mmap_mode="r")
        self.states = np.load(cache_dir / "state.npy", mmap_mode="r")
        self.trajectory = np.load(cache_dir / "future_trajectory.npy", mmap_mode="r")
        self.valid = np.load(cache_dir / "future_valid.npy", mmap_mode="r")
        self.action_mask = np.load(cache_dir / "action_mask.npy", mmap_mode="r")
        self.episodes = np.load(cache_dir / "episode_id.npy", mmap_mode="r")
        self.steps = np.load(cache_dir / "step_id.npy", mmap_mode="r")

        self.anchors = np.flatnonzero(self.action_mask & self.valid.any(axis=1))
        norm_path = self.root / "normalization.json"
        if norm_path.exists():
            self.normalizer = Normalizer(**json.loads(norm_path.read_text()))
        else:
            self.normalizer = None
        self.dt_s = float(manifest.get("config", {}).get("dt_s", 0.1))

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        anchor = int(self.anchors[idx])
        bev = torch.from_numpy(self.bev[anchor].copy()).to(torch.uint8)

        st = self.states[anchor]
        speed = float(st["speed_mps"])
        yaw_rate = float(st["yaw_rate_rps"])
        ego_state = torch.tensor([speed / 10.0, yaw_rate], dtype=torch.float32)

        traj = self.trajectory[anchor]
        valid = torch.from_numpy(self.valid[anchor].copy()).to(torch.bool)
        action_mask = torch.tensor(bool(self.action_mask[anchor]), dtype=torch.bool)

        dx = traj["dx_m"]
        dy = traj["dy_m"]
        v = traj["speed_mps"]
        heading = traj["heading_rad"]
        raw_traj = np.stack([dx, dy, v, heading], axis=-1)

        if self.normalizer is not None:
            norm_traj = self.normalizer.normalize_trajectory(raw_traj)
            target = torch.from_numpy(norm_traj).to(torch.float32)
        else:
            target = torch.from_numpy(raw_traj).to(torch.float32)

        return {
            "bev": bev,
            "ego_state": ego_state,
            "trajectory": target,
            "valid": valid,
            "action_mask": action_mask,
        }
