"""阶段5共享的冻结正式模型加载、指标及证据接口。"""
from dataclasses import asdict
import hashlib
from pathlib import Path
import numpy as np
import torch
from configs.stage2 import BaselineConfig
from mini_wa_jepa.normalization import Normalizer
from scripts.evaluate_joint import load_joint


class FrozenPolicy:
    def __init__(self, name, checkpoint, device):
        self.name, self.device = name, device
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        if name == 'resnet':
            from training.baseline import make_model
            self.saved = torch.load(path, map_location='cpu', weights_only=False)
            if self.saved['data_schema_version'] != 2:
                raise ValueError('Only formal v2 checkpoint accepted')
            self.config = BaselineConfig(**self.saved['model_config'])
            self.model = make_model(self.config).to(device).eval()
            self.model.load_state_dict(self.saved['model'])
        else:
            self.model, self.config, self.saved = load_joint(path, device)
            if bool(self.config.latent_weight) != (name == 'joint'):
                raise ValueError('Joint/action-only checkpoint identity mismatch')
        self.normalizer = Normalizer(**self.saved['normalization'])
        self.metadata = dict(checkpoint=str(path), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             checkpoint_epoch=self.saved['epoch'], parameter_count=self.saved['parameter_count'],
                             model_config=asdict(self.config), training_source=self.saved['provenance'])

    def batch(self, raw, include_future=False):
        bev = torch.as_tensor(raw['bev'], device=self.device)
        ego = torch.as_tensor(raw['ego'], device=self.device)
        if self.name == 'resnet':
            return dict(trajectory=self.model(bev[:, -1], ego[:, -1]))
        future = torch.as_tensor(raw['future'], device=self.device) if include_future else None
        return self.model(bev, ego, future)

    def physical(self, tensor):
        return self.normalizer.inverse_trajectory(tensor.float().cpu().numpy())


def require_rocm():
    if not torch.version.hip or torch.version.cuda is not None or not torch.cuda.is_available():
        raise RuntimeError('Stage 5 formal evaluation requires native PyTorch ROCm')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cohort_data(root, smoke_limit=None):
    from dataset.joint import JointSequences
    data = JointSequences(root, 'test')
    frozen = np.load('artifacts/restart_stage4/test_eval/paired_cohort.npz')
    anchors = frozen['anchors']
    expected = np.flatnonzero(data.action_mask & data.valid.all(1))
    if not np.array_equal(anchors, expected) or len(anchors) != 9600:
        raise ValueError('Stage 4 matched cohort mismatch')
    if not np.array_equal(frozen['episodes'], data.episodes[anchors]):
        raise ValueError('Episode identity mismatch')
    data.anchors = anchors[:smoke_limit] if smoke_limit else anchors
    return data
