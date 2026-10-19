"""All v2 anchors, episode-safe history padding and individually masked futures."""
import json
from pathlib import Path
import numpy as np

from dataset.baseline import build_cache, validate_manifest
from mini_wa_jepa.geometry import wrap_angle
from mini_wa_jepa.normalization import Normalizer


class JointSequences:
    _fields = dict(bev='bev', states='state', episodes='episode_id', steps='step_id',
                   trajectories='future_trajectory', valid='future_valid', action_mask='action_mask',
                   noisy='noise_active', recovery='recovery_active')

    def __init__(self, root, split='train', sample_count=None, seed=0):
        self.root = Path(root); self.split = split
        manifest = validate_manifest(root); self.dt_s = manifest['config']['dt_s']
        if abs(self.dt_s-.1) > 1e-8 or manifest['config']['future_steps'] != 10:
            raise ValueError('Joint model requires 0.1s observations and 10 trajectory waypoints')
        if abs(manifest['config']['waypoint_dt_s']-.2) > 1e-8:
            raise ValueError('Trajectory labels must have 0.2s spacing')
        self.cache = build_cache(root, split); self._open()
        self.normalizer = Normalizer(**json.loads((self.root/'normalization.json').read_text()))
        n = len(self.bev); positions = np.arange(n)
        self.history_indices = np.empty((n, 4), np.int64)
        self.future_indices = np.empty((n, 4), np.int64)
        self.future_valid = np.zeros((n, 4), bool)
        self.ego = np.zeros((n, 2), np.float32); self.ego[:, 0] = self.states[:, 3]/10
        previous = np.maximum(positions-1, 0)
        consecutive = ((positions > 0) & (self.episodes == self.episodes[previous]) &
                       (self.steps-self.steps[previous] == 1))
        self.ego[consecutive, 1] = wrap_angle(self.states[consecutive, 2]-self.states[previous[consecutive], 2])/self.dt_s
        # Segment starts also handle a missing step inside an episode, not only resets.
        starts = np.maximum.accumulate(np.where(consecutive, 0, positions))
        for slot, lag in enumerate((3, 2, 1, 0)):
            self.history_indices[:, slot] = np.maximum(positions-lag, starts)
        for slot in range(4):
            candidate = np.minimum(positions+slot+1, n-1)
            valid = ((positions+slot+1 < n) & (starts[candidate] == starts) &
                     (self.steps[candidate]-self.steps == slot+1))
            self.future_indices[:, slot] = np.where(valid, candidate, positions)
            self.future_valid[:, slot] = valid
        self.anchors = positions
        if sample_count is not None and sample_count < n:
            self.anchors = np.sort(np.random.default_rng(seed).choice(n, sample_count, replace=False))

    def _open(self):
        for field, name in self._fields.items():
            setattr(self, field, np.load(self.cache/f'{name}.npy', mmap_mode='r'))

    def __getstate__(self):
        state = self.__dict__.copy()
        for field in self._fields: state.pop(field)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state); self._open()

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        anchor = int(self.anchors[index]); past = self.history_indices[anchor]
        return dict(bev=np.array(self.bev[past]), ego=np.array(self.ego[past]),
            future=np.array(self.bev[self.future_indices[anchor]]),
            world_valid=self.future_valid[anchor].copy(),
            trajectory=self.normalizer.trajectory(self.trajectories[anchor]),
            valid=np.array(self.valid[anchor]), action_mask=bool(self.action_mask[anchor]),
            anchor=anchor, episode=int(self.episodes[anchor]), step=int(self.steps[anchor]),
            noisy=bool(self.noisy[anchor]), recovery=bool(self.recovery[anchor]),
            history_padding=int(np.sum(past == past[0])-1))

