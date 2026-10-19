"""Dataset loading disk-cached BEV tokens and aligned trajectory targets for Flow Matching."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from mini_wa_jepa.normalization import Normalizer


class FlowFeaturesDataset(Dataset):
    """Loads frame-level mmap tokens and anchor-level index metadata."""

    def __init__(self, root, split='train'):
        self.root = Path(root)
        self.split = split

        manifest_path = self.root / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing cache manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        if not self.manifest.get('completed', False):
            raise ValueError(f"Cache manifest at {manifest_path} is not completed")

        norm_path = self.root / 'normalization.json'
        if not norm_path.exists():
            raise FileNotFoundError(f"Normalization file missing in cache directory: {norm_path}")
        self.normalizer = Normalizer(**json.loads(norm_path.read_text()))

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory {split_dir} does not exist")

        # Frame-level encoded tokens: [N_frames, 64, 256] in float16
        self.frame_tokens = np.load(split_dir / 'frame_tokens.npy', mmap_mode='r')

        # Anchor-level pre-filtered arrays (small metadata, safe to load into memory or copy)
        self.history_indices = np.load(split_dir / 'history_indices.npy')
        self.ego = np.load(split_dir / 'ego.npy')
        self.targets = np.load(split_dir / 'targets.npy')
        self.waypoint_valid = np.load(split_dir / 'waypoint_valid.npy')
        self.action_mask = np.load(split_dir / 'action_mask.npy')
        self.anchors = np.load(split_dir / 'anchors.npy')
        self.episodes = np.load(split_dir / 'episodes.npy')
        self.steps = np.load(split_dir / 'steps.npy')
        self.mse_predictions = np.load(split_dir / 'mse_predictions.npy')

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        hist_idx = self.history_indices[index]
        # Copy token slice from mmap to avoid read-only memory warnings
        tokens = np.array(self.frame_tokens[hist_idx], dtype=np.float32, copy=True)

        return dict(
            history=torch.from_numpy(tokens),
            ego=torch.from_numpy(np.array(self.ego[index], dtype=np.float32, copy=True)),
            target=torch.from_numpy(np.array(self.targets[index], dtype=np.float32, copy=True)),
            waypoint_valid=torch.from_numpy(np.array(self.waypoint_valid[index], dtype=bool, copy=True)),
            action_mask=torch.tensor(bool(self.action_mask[index]), dtype=torch.bool),
            mse_prediction=torch.from_numpy(np.array(self.mse_predictions[index], dtype=np.float32, copy=True)),
            anchor=int(self.anchors[index]),
            episode=int(self.episodes[index]),
            step=int(self.steps[index]),
        )
