"""Memory-mapped immutable latent targets and causal commanded action inputs."""
import json
from pathlib import Path
import numpy as np


class BranchFeatures:
    fields=('history','ego','command','target','valid','group','candidate','seed','realized','waypoint_valid')

    def __init__(self,root,split):
        self.root=Path(root);self.split=split;self.folder=self.root/split
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        if not self.manifest['completed']:raise ValueError('Feature cache incomplete')
        self.rows=json.loads((self.folder/'rows.json').read_text());self._open()

    def _open(self):
        self.arrays={key:np.load(self.folder/f'{key}.npy',mmap_mode='r') for key in self.fields}

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):return {key:np.array(value[index]) for key,value in self.arrays.items()}

    def __getstate__(self):
        state=self.__dict__.copy();state.pop('arrays');return state

    def __setstate__(self,state):self.__dict__.update(state);self._open()
