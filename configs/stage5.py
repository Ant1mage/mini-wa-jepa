"""阶段5冻结评测设计；无训练、无测试集选权重或调超参数。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Degradation:
    name: str = 'clean'
    kind: str = 'clean'
    level: int = 0
    depth_m: float = 0.
    width_m: float = 16.
    boundary_drop: float = 0.
    dropped_frames: int = 0


CONDITIONS = (Degradation(),) + tuple(
    Degradation(f'occlusion_l{i}', 'occlusion', i, depth_m=depth)
    for i, depth in enumerate((8., 16., 24.), 1)
) + tuple(
    Degradation(f'boundary_l{i}', 'boundary', i, boundary_drop=rate)
    for i, rate in enumerate((.25, .5, 1.), 1)
) + tuple(
    Degradation(f'dropout_l{i}', 'dropout', i, dropped_frames=i)
    for i in (1, 2, 3)
)


@dataclass(frozen=True)
class Stage5Config:
    seed: int = 20261018
    data: str = 'outputs/data/stage2_v2_100k'
    batch_size: int = 128
    workers: int = 2
    cpu_threads: int = 6
    sdpa_backend: str = 'MATH'
    control_dt_s: float = .02
    observation_dt_s: float = .1
    dropout_period_frames: int = 10
    closed_episodes_per_map: int = 3
    closed_conditions: tuple = ('clean', 'occlusion_l2', 'occlusion_l3', 'boundary_l2',
                                'boundary_l3', 'dropout_l2', 'dropout_l3')
    visualization_sample_ids: tuple = (64, 5500)
    bootstrap_samples: int = 2000
    egl_device: int = 1


CHECKPOINTS = {
    'joint': 'artifacts/restart_stage4/joint_001/best.pt',
    'action_only': 'artifacts/restart_stage4/action_only_001/best.pt',
    'resnet': 'artifacts/restart_stage2/baseline/best.pt',
}


def condition(name):
    return next(c for c in CONDITIONS if c.name == name)
