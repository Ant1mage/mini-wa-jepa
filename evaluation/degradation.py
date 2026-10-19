"""正式阶段5退化：因果的source映射和跨模型一致的episode栅格掩码。"""
from dataclasses import asdict
import numpy as np


class SpatialDegrader:
    def __init__(self, spec, seed=20261018):
        self.spec, self.seed, self.masks = spec, seed, {}

    def __call__(self, bev, episode):
        out = np.array(bev, copy=True)
        if self.spec.kind == 'occlusion':
            row, col = np.indices(out.shape[-2:])
            x = (63.5 - row) * .5; y = (63.5 - col) * .5
            mask = (x >= 4) & (x < 4 + self.spec.depth_m) & (np.abs(y) < self.spec.width_m / 2)
            out[..., mask] = 0
        elif self.spec.kind == 'boundary':
            if episode not in self.masks:
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(episode)]))
                self.masks[episode] = rng.random(out.shape[-2:]) < self.spec.boundary_drop
            out[..., 1, :, :] = np.where(self.masks[episode], 0, out[..., 1, :, :])
        return out


def temporal_sources(episodes, steps, dropped_frames):
    """以每秒最后N槽为丢失窗口，映射到同一连续episode最近收到的索引。"""
    positions = np.arange(len(steps)); previous = np.maximum(0, positions-1)
    consecutive = ((positions > 0) & (episodes == episodes[previous]) & (steps-steps[previous] == 1))
    starts = np.maximum.accumulate(np.where(consecutive, 0, positions))
    if dropped_frames == 0:
        return positions
    phase = np.asarray(steps) % 10
    lag = np.maximum(0, phase - (9-dropped_frames))
    return np.maximum(starts, positions-lag)


class DegradedSequences:
    def __init__(self, data, spec, seed=20261018):
        self.data, self.spec, self.spatial = data, spec, SpatialDegrader(spec, seed)
        self.sources = temporal_sources(data.episodes, data.steps, spec.dropped_frames)
        for field in ('root', 'split', 'normalizer', 'states', 'dt_s'):
            setattr(self, field, getattr(data, field))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        past = self.data.history_indices[item['anchor']]
        sources = self.sources[past]
        item['bev'] = self.spatial(self.data.bev[sources], item['episode'])
        item['observation_sources'] = sources.copy()
        return item


class OnlineDegrader:
    def __init__(self, spec, seed, episode, control_dt_s=.02):
        self.spatial = SpatialDegrader(spec, seed)
        self.spec, self.episode = spec, episode
        self.stride = round(.1/control_dt_s)
        if abs(self.stride * control_dt_s - .1) > 1e-8:
            raise ValueError('0.1s must be divisible by control dt')
        self.last_received = None; self.last_received_step = None

    def __call__(self, clean_bev, step):
        tick = step // self.stride
        dropped = self.spec.dropped_frames and tick % 10 >= 10-self.spec.dropped_frames
        if step % self.stride == 0 and not dropped:
            self.last_received = np.array(clean_bev, copy=True); self.last_received_step = step
        source_step = self.last_received_step if dropped else step
        received = self.last_received if dropped else clean_bev
        if received is None:
            raise ValueError('No past observation available; reset must start at step 0')
        return self.spatial(received, self.episode), source_step
