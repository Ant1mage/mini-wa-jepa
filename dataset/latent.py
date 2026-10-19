"""Adjacent v2 frames for world supervision, without action-loss filtering."""
from pathlib import Path

import numpy as np

from dataset.baseline import build_cache, validate_manifest
from mini_wa_jepa.geometry import wrap_angle


class LatentPairs:
    def __init__(self, root, split='train', sample_count=None, seed=0):
        self.root = Path(root)
        manifest = validate_manifest(root)
        self.split = split
        self.dt_s = manifest['config']['dt_s']
        self.cache = build_cache(root, split)
        self._open()
        # Include noisy/recovery and terminal-failure targets, but never cross resets/gaps.
        consecutive = ((self.episodes[:-1] == self.episodes[1:]) &
                       (self.steps[1:] - self.steps[:-1] == 1))
        self.anchors = np.flatnonzero(consecutive)
        self.full_pair_count = len(self.anchors)
        if sample_count is not None and sample_count < len(self.anchors):
            chosen = np.random.default_rng(seed).choice(len(self.anchors), sample_count, replace=False)
            self.anchors = self.anchors[np.sort(chosen)]

    def _open(self):
        for field, name in [('bev', 'bev'), ('states', 'state'), ('episodes', 'episode_id'),
                            ('steps', 'step_id'), ('noisy', 'noise_active'), ('recovery', 'recovery_active')]:
            setattr(self, field, np.load(self.cache / f'{name}.npy', mmap_mode='r'))

    def __getstate__(self):
        state = self.__dict__.copy()
        for field in ('bev', 'states', 'episodes', 'steps', 'noisy', 'recovery'):
            state.pop(field)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        anchor = int(self.anchors[index])
        state = self.states[anchor]
        yaw_rate = 0.
        if (anchor > 0 and self.episodes[anchor - 1] == self.episodes[anchor]
                and self.steps[anchor] - self.steps[anchor - 1] == 1):
            yaw_rate = float(wrap_angle(state[2] - self.states[anchor - 1, 2])) / self.dt_s
        return dict(bev=np.array(self.bev[anchor]), future=np.array(self.bev[anchor + 1]),
                    ego=np.array([state[3] / 10., yaw_rate], np.float32),
                    anchor=anchor, episode=int(self.episodes[anchor]), step=int(self.steps[anchor]),
                    noisy=bool(self.noisy[anchor]), recovery=bool(self.recovery[anchor]))

