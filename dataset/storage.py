"""Compressed NPZ shards + a small bounded LRU for fast random access."""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np


class ShardWriter:
    def __init__(self, root, shard_frames=512, resume=False):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.shard_frames=shard_frames
        self.pending=[];self.pending_size=0;self.shards=[];self.total=0
        if resume:
            index=json.loads((self.root/'index.json').read_text())
            self.shards=index['shards'];self.total=index['frames']

    def add(self, arrays):
        start=0
        n=len(arrays['state'])
        while start<n:
            take=min(n-start,self.shard_frames-self.pending_size)
            self.pending.append({k:v[start:start+take] for k,v in arrays.items()})
            self.pending_size+=take;start+=take
            if self.pending_size==self.shard_frames:self.flush()

    def flush(self):
        if not self.pending:return
        arrays={k:np.concatenate([part[k] for part in self.pending]) for k in self.pending[0]}
        name=f"shard_{len(self.shards):04d}.npz"
        np.savez_compressed(self.root/name,**arrays)
        self.shards.append(dict(file=name,start=self.total,frames=self.pending_size))
        self.total+=self.pending_size;self.pending=[];self.pending_size=0

    def close(self):
        self.flush()
        (self.root/'index.json').write_text(json.dumps(dict(frames=self.total,shards=self.shards),indent=2)+'\n')


class ShardReader:
    def __init__(self, root, cache_shards=2):
        self.root=Path(root)
        self.index=json.loads((self.root/'index.json').read_text())
        self.cache=OrderedDict();self.cache_shards=cache_shards
        self.ends=np.cumsum([s['frames'] for s in self.index['shards']])

    def __len__(self):return self.index['frames']

    def shard(self, number):
        if number in self.cache:
            self.cache.move_to_end(number);return self.cache[number]
        with np.load(self.root/self.index['shards'][number]['file'],allow_pickle=False) as f:
            data={k:f[k] for k in f.files}
        self.cache[number]=data
        while len(self.cache)>self.cache_shards:self.cache.popitem(last=False)
        return data

    def __getitem__(self, index):
        if index<0 or index>=len(self):raise IndexError(index)
        shard=int(np.searchsorted(self.ends,index,side='right'))
        local=index-(int(self.ends[shard-1]) if shard else 0)
        return {k:v[local] for k,v in self.shard(shard).items()}

    def metadata(self, key):
        return np.concatenate([self.shard(i)[key] for i in range(len(self.ends))])

    def metadata_many(self, keys):
        """Decode each compressed BEV shard once when reading several small fields."""
        parts={key:[] for key in keys}
        for index in range(len(self.ends)):
            shard=self.shard(index)
            for key in keys:parts[key].append(shard[key])
        return {key:np.concatenate(values) for key,values in parts.items()}
