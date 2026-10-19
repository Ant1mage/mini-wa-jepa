"""Read immutable causal world predictions; actual outcomes are targets only."""
import json
from pathlib import Path
import numpy as np


class RiskFeatures:
    fields = ('group', 'seed', 'command', 'prediction', 'target', 'valid', 'risk',
              'risk_mask', 'realized', 'waypoint_valid')

    def __init__(self, root, split):
        self.root = Path(root); self.split = split
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        if not self.manifest['completed']:
            raise ValueError('Incomplete risk cache')
        folder = self.root/split
        self.rows = json.loads((folder/'rows.json').read_text())
        self.arrays = {k: np.load(folder/f'{k}.npy', mmap_mode='r') for k in self.fields}
        self.histories = np.load(folder/'histories.npy', mmap_mode='r')
        self.egos = np.load(folder/'egos.npy', mmap_mode='r')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        group = int(self.arrays['group'][index])
        values = {k: np.array(v[index]) for k, v in self.arrays.items()}
        values.update(history=np.array(self.histories[group], np.float32),
                      ego=np.array(self.egos[group], np.float32))
        values['prediction'] = values['prediction'].astype(np.float32)
        return values
