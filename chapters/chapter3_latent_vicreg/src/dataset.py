"""Chapter 3 Dataset: LatentPairs for Self-Supervised World Modeling.

This dataset produces pairs of consecutive BEV frames (t and t+1) along with ego-vehicle
velocity/yaw rate states, used to train the Latent World Model under the VICReg objective.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from mini_wa_jepa.geometry import wrap_angle


def validate_manifest(root: Path) -> dict:
    manifest_path = Path(root) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 2 or not manifest.get("completed"):
        raise ValueError("Only a completed v2 dataset schema is supported.")
    return manifest


class LatentPairsDataset(Dataset):
    """Dataset providing (BEV_t, BEV_{t+1}, ego_action_t) pairs for self-supervised training."""

    def __init__(self, root: str | Path, split: str = "train", sample_count: int | None = None, seed: int = 0):
        self.root = Path(root)
        self.split = split
        manifest = validate_manifest(self.root)
        self.dt_s = float(manifest.get("config", {}).get("dt_s", 0.1))

        self.cache_dir = self.root / "baseline_cache" / split
        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"Cache directory {self.cache_dir} not found. Please ensure data cache is pre-built."
            )

        self._open()

        # Find consecutive frame pairs within the same episode
        consecutive = (
            (self.episodes[:-1] == self.episodes[1:]) &
            (self.steps[1:] - self.steps[:-1] == 1)
        )
        self.anchors = np.flatnonzero(consecutive)
        self.full_pair_count = len(self.anchors)

        if sample_count is not None and sample_count < len(self.anchors):
            rng = np.random.default_rng(seed)
            chosen = rng.choice(len(self.anchors), sample_count, replace=False)
            self.anchors = self.anchors[np.sort(chosen)]

    def _open(self):
        self.bev = np.load(self.cache_dir / "bev.npy", mmap_mode="r")
        self.states = np.load(self.cache_dir / "state.npy", mmap_mode="r")
        self.episodes = np.load(self.cache_dir / "episode_id.npy", mmap_mode="r")
        self.steps = np.load(self.cache_dir / "step_id.npy", mmap_mode="r")
        self.noisy = np.load(self.cache_dir / "noise_active.npy", mmap_mode="r")
        self.recovery = np.load(self.cache_dir / "recovery_active.npy", mmap_mode="r")

    def __getstate__(self):
        state = self.__dict__.copy()
        for field in ("bev", "states", "episodes", "steps", "noisy", "recovery"):
            state.pop(field, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index: int) -> dict:
        anchor = int(self.anchors[index])
        state = self.states[anchor]

        yaw_rate = 0.0
        if (anchor > 0 and self.episodes[anchor - 1] == self.episodes[anchor]
                and self.steps[anchor] - self.steps[anchor - 1] == 1):
            yaw_rate = float(wrap_angle(state[2] - self.states[anchor - 1, 2])) / self.dt_s

        # ego features: normalized speed (v / 10.0) and yaw_rate (rad/s)
        ego = np.array([state[3] / 10.0, yaw_rate], dtype=np.float32)

        return {
            "bev": np.array(self.bev[anchor], dtype=np.uint8),
            "future": np.array(self.bev[anchor + 1], dtype=np.uint8),
            "ego": ego,
            "anchor": anchor,
            "episode": int(self.episodes[anchor]),
            "step": int(self.steps[anchor]),
            "noisy": bool(self.noisy[anchor]),
            "recovery": bool(self.recovery[anchor]),
        }
