"""Chapter 4 Dataset: JointSequencesDataset.

Extracts multi-frame sequences (4-frame history, 4-frame future, 10-step trajectory)
with episode-safe boundary handling and memory-mapped fast loading.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from mini_wa_jepa.geometry import wrap_angle
from mini_wa_jepa.normalization import Normalizer


class JointSequencesDataset(Dataset):
    """Multi-frame temporal dataset for Joint World and Action Model training."""

    _fields = {
        "bev": "bev",
        "states": "state",
        "episodes": "episode_id",
        "steps": "step_id",
        "trajectories": "future_trajectory",
        "valid": "future_valid",
        "action_mask": "action_mask",
        "noisy": "noise_active",
        "recovery": "recovery_active",
    }

    def __init__(self, root: str | Path, split: str = "train", sample_count: int | None = None, seed: int = 0):
        self.root = Path(root)
        self.split = split

        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        self.dt_s = float(manifest.get("config", {}).get("dt_s", 0.1))

        self.cache_dir = self.root / "baseline_cache" / split
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache not found at {self.cache_dir}")

        self._open()

        norm_path = self.root / "normalization.json"
        if norm_path.exists():
            self.normalizer = Normalizer(**json.loads(norm_path.read_text()))
        else:
            self.normalizer = None

        n = len(self.bev)
        positions = np.arange(n)
        self.history_indices = np.empty((n, 4), dtype=np.int64)
        self.future_indices = np.empty((n, 4), dtype=np.int64)
        self.future_valid = np.zeros((n, 4), dtype=bool)
        self.ego = np.zeros((n, 2), dtype=np.float32)
        self.ego[:, 0] = self.states[:, 3] / 10.0

        previous = np.maximum(positions - 1, 0)
        consecutive = (
            (positions > 0) &
            (self.episodes == self.episodes[previous]) &
            (self.steps - self.steps[previous] == 1)
        )
        self.ego[consecutive, 1] = wrap_angle(self.states[consecutive, 2] - self.states[previous[consecutive], 2]) / self.dt_s

        starts = np.maximum.accumulate(np.where(consecutive, 0, positions))
        for slot, lag in enumerate((3, 2, 1, 0)):
            self.history_indices[:, slot] = np.maximum(positions - lag, starts)

        for slot in range(4):
            candidate = np.minimum(positions + slot + 1, n - 1)
            valid = (
                (positions + slot + 1 < n) &
                (starts[candidate] == starts) &
                (self.steps[candidate] - self.steps == slot + 1)
            )
            self.future_indices[:, slot] = np.where(valid, candidate, positions)
            self.future_valid[:, slot] = valid

        self.anchors = positions
        if sample_count is not None and sample_count < n:
            rng = np.random.default_rng(seed)
            self.anchors = np.sort(rng.choice(n, sample_count, replace=False))

    def _open(self):
        for attr, fname in self._fields.items():
            setattr(self, attr, np.load(self.cache_dir / f"{fname}.npy", mmap_mode="r"))

    def __getstate__(self):
        state = self.__dict__.copy()
        for attr in self._fields:
            state.pop(attr, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index: int) -> dict:
        anchor = int(self.anchors[index])
        past = self.history_indices[anchor]

        raw_traj = self.trajectories[anchor]
        if self.normalizer is not None:
            traj = self.normalizer.trajectory(raw_traj)
        else:
            traj = raw_traj

        return {
            "bev": np.array(self.bev[past], dtype=np.uint8),
            "ego": np.array(self.ego[past], dtype=np.float32),
            "future": np.array(self.bev[self.future_indices[anchor]], dtype=np.uint8),
            "world_valid": self.future_valid[anchor].copy(),
            "trajectory": np.array(traj, dtype=np.float32),
            "valid": np.array(self.valid[anchor], dtype=bool),
            "action_mask": bool(self.action_mask[anchor]),
            "anchor": anchor,
            "episode": int(self.episodes[anchor]),
            "step": int(self.steps[anchor]),
            "noisy": bool(self.noisy[anchor]),
            "recovery": bool(self.recovery[anchor]),
        }
