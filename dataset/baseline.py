"""Single-frame BEV baseline dataset with disk-backed uint8 caching."""
import json
from pathlib import Path

import numpy as np

from dataset.storage import ShardReader
from mini_wa_jepa.geometry import wrap_angle
from mini_wa_jepa.normalization import Normalizer


def validate_manifest(root):
    report=json.loads((Path(root)/'manifest.json').read_text())
    if report.get('schema_version')!=2 or not report.get('completed'):
        raise ValueError('Only a complete v2 dataset may be used for training/evaluation')
    return report


def build_cache(root,split):
    root=Path(root);validate_manifest(root)
    reader=ShardReader(root/split,cache_shards=1)
    cache=root/'baseline_cache'/split
    if (cache/'complete.json').exists():return cache
    cache.mkdir(parents=True,exist_ok=True)
    bev=np.lib.format.open_memmap(cache/'bev.npy',mode='w+',dtype=np.uint8,shape=(len(reader),4,128,128))
    fields={key:[] for key in ['state','episode_id','step_id','future_trajectory','future_valid','action_mask','noise_active','recovery_active']}
    offset=0
    for i in range(len(reader.ends)):
        shard=reader.shard(i);count=len(shard['state'])
        if shard['bev'].shape!=(count,4,128,128):raise ValueError('Invalid BEV data schema')
        bev[offset:offset+count]=shard['bev'];offset+=count
        for key in fields:fields[key].append(shard[key])
    bev.flush();del bev
    for key,parts in fields.items():np.save(cache/f'{key}.npy',np.concatenate(parts))
    (cache/'complete.json').write_text(json.dumps(dict(frames=len(reader),schema_version=2))+'\n')
    return cache


class BaselineDataset:
    def __init__(self,root,split='train'):
        self.root=Path(root);self.split=split
        manifest=validate_manifest(root)
        cache=build_cache(root,split)
        self.cache=cache
        self.bev=np.load(cache/'bev.npy',mmap_mode='r')
        self.states=np.load(cache/'state.npy',mmap_mode='r')
        self.trajectory=np.load(cache/'future_trajectory.npy',mmap_mode='r')
        self.valid=np.load(cache/'future_valid.npy',mmap_mode='r')
        self.action_mask=np.load(cache/'action_mask.npy',mmap_mode='r')
        self.episodes=np.load(cache/'episode_id.npy',mmap_mode='r')
        self.steps=np.load(cache/'step_id.npy',mmap_mode='r')
        self.anchors=np.flatnonzero(self.action_mask & self.valid.any(1))
        self.normalizer=Normalizer(**json.loads((self.root/'normalization.json').read_text()))
        self.dt_s=manifest['config']['dt_s']

    def __len__(self):return len(self.anchors)

    def __getstate__(self):
        state=self.__dict__.copy()
        for key in ('bev','states','trajectory','valid','action_mask','episodes','steps'):state.pop(key)
        return state

    def __setstate__(self,state):
        self.__dict__.update(state)
        for key,file in [('bev','bev'),('states','state'),('trajectory','future_trajectory'),
                         ('valid','future_valid'),('action_mask','action_mask'),('episodes','episode_id'),('steps','step_id')]:
            setattr(self,key,np.load(self.cache/f'{file}.npy',mmap_mode='r'))

    def __getitem__(self,index):
        anchor=int(self.anchors[index]);state=self.states[anchor]
        yaw_rate=0.
        if anchor>0 and self.episodes[anchor-1]==self.episodes[anchor] and self.steps[anchor]-self.steps[anchor-1]==1:
            yaw_rate=float(wrap_angle(state[2]-self.states[anchor-1,2]))/self.dt_s
        return dict(bev=np.array(self.bev[anchor]),ego=np.array([state[3]/10,yaw_rate],np.float32),
            trajectory=self.normalizer.trajectory(self.trajectory[anchor]),
            valid=np.array(self.valid[anchor]),action_mask=np.array(self.action_mask[anchor]),anchor=anchor)
