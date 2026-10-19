"""Map-style PyTorch-compatible dataset; deliberately imports no simulator."""
import json
from pathlib import Path
import os

import numpy as np

from configs.stage2 import BEVNoiseConfig
from dataset.augmentation import augment_bev
from dataset.storage import ShardReader
from mini_wa_jepa.geometry import world_to_ego, wrap_angle
from mini_wa_jepa.normalization import Normalizer


class DrivingSequences:
    def __init__(self, root, split='train', history=4, future=4, noise=BEVNoiseConfig(), seed=0):
        self.root=Path(root);self.reader=ShardReader(self.root/split)
        manifest=json.loads((self.root/'manifest.json').read_text())
        if manifest.get('schema_version')!=2:raise ValueError('DrivingSequences requires v2 data')
        self.history,self.future,self.noise=history,future,noise
        self.seed=seed;self._rng=None;self._pid=None
        self.normalizer=Normalizer(**json.loads((self.root/'normalization.json').read_text()))
        episodes=self.reader.metadata('episode_id')
        steps=self.reader.metadata('step_id')
        self.anchors=np.array([i for i in range(history-1,len(episodes)-future)
            if episodes[i-history+1]==episodes[i+future]
            and steps[i+future]-steps[i-history+1]==history+future-1],dtype=np.int64)

    def __len__(self):return len(self.anchors)

    def __getitem__(self,index):
        if self._rng is None or self._pid!=os.getpid():
            # Worker seeds are provided by DataLoader; avoid identical streams
            # after fork and keep epoch-dependent random draws in each worker.
            try:
                from torch.utils.data import get_worker_info
                worker=get_worker_info()
            except ImportError:worker=None
            self._rng=np.random.default_rng(self.seed+(worker.seed if worker else 0))
            self._pid=os.getpid()
        anchor=int(self.anchors[index])
        hist=[self.reader[i] for i in range(anchor-self.history+1,anchor+1)]
        row=hist[-1]
        clean=np.stack([r['bev'] for r in hist])
        observed=augment_bev(clean,self._rng,self.noise)
        states=np.stack([r['state'] for r in hist])
        history_xy=world_to_ego(states[:,:2],row['state'][:2],row['state'][2])
        history_traj=np.column_stack([history_xy,states[:,3],wrap_angle(states[:,2]-row['state'][2])])
        body_velocity=np.stack([world_to_ego(s[4:6],[0,0],s[2]) for s in states]).astype(np.float32)
        return dict(bev_history=observed.astype(np.float32)/255,
            bev_future=np.stack([self.reader[i]['bev'] for i in range(anchor+1,anchor+self.future+1)]).astype(np.float32)/255,
            history_trajectory=self.normalizer.trajectory(history_traj),
            speed=self.normalizer.speed(states[:,3]).astype(np.float32),
            velocity=body_velocity*self.normalizer.scale,
            trajectory=self.normalizer.trajectory(row['future_trajectory']),
            future_valid=row['future_valid'].copy(),action_mask=np.asarray(row['action_mask'],dtype=np.bool_),
            collision_target=np.asarray(row['future_collision'].any(),dtype=np.float32),
            out_of_road_target=np.asarray(row['future_out_of_road'].any(),dtype=np.float32),
            risk_target_valid=np.asarray(row['future_valid'].all() or row['future_collision'].any()
                                         or row['future_out_of_road'].any(),dtype=np.bool_))
