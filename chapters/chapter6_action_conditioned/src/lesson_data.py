"""阶段6真实分支与冻结模型输出的轻量只读入口。"""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'artifacts/restart_stage6'
NAMES=('A 保持路线','B 轻微左偏','C 轻微右偏','D 减速')
COLORS=('#16a6ef','#e7ad23','#cf5dd1','#ee6559')


def available():
    required=[DATA/'features/test/rows.json',DATA/'features/test/target.npy',
              DATA/'features/test/valid.npy',DATA/'branches/manifest.json']
    required += [DATA/f'test_eval/{kind}_20261019_raw.npz' for kind in ('conditioned','unconditional')]
    required += [DATA/f'features/test/{field}.npy' for field in ('group','seed')]
    return all(path.is_file() for path in required)


@lru_cache(maxsize=1)
def rows():
    return json.loads((DATA/'features/test/rows.json').read_text())


def scene_options():
    if not available():return {'尚无正式数据':0}
    road={'SCS':'直—弯—直','CSC':'弯—直—弯'}
    return {f'{i//4+1:02d}｜{road.get(r["map_spec"],r["map_spec"])}｜{r["anchor_step"]*.1:.1f}秒':i//4
            for i,r in enumerate(rows()) if i%4==0}


@lru_cache(maxsize=2)
def frozen_predictions(kind):
    if kind not in ('conditioned','unconditional'):raise ValueError('Unknown model')
    path=DATA/f'test_eval/{kind}_20261019_raw.npz'
    with np.load(path) as archive:
        result={key:archive[key] for key in ('prediction','wrong_prediction','groups','seeds')}
    # 防止不同样本顺序的输出被错误配到当前场景。
    for key,field in [('groups','group'),('seeds','seed')]:
        if not np.array_equal(result[key],np.load(DATA/f'features/test/{field}.npy')):
            raise ValueError('Frozen output/sample order mismatch')
    if len(result['prediction'])!=len(rows()):raise ValueError('Frozen output count mismatch')
    return result


@lru_cache(maxsize=1)
def targets():
    # 目标和有效掩码直接映射正式缓存，不在教学页重算或修改。
    return (np.load(DATA/'features/test/target.npy',mmap_mode='r'),
            np.load(DATA/'features/test/valid.npy',mmap_mode='r'))


@lru_cache(maxsize=8)
def context(scene):
    index=int(scene)*4
    if index<0:raise IndexError('Scene outside recorded test cohort')
    metadata=rows()[index:index+4]
    if len(metadata)!=4:raise IndexError('Scene outside recorded test cohort')
    if len({r['group'] for r in metadata})!=1:raise ValueError('Mixed-context candidates')
    with np.load(DATA/'branches'/metadata[0]['context']) as archive:
        initial={key:archive[key] for key in ('history','states','commands')}
    branches=[]
    for r in metadata:
        with np.load(DATA/'branches'/r['raw']) as archive:
            branches.append({key:archive[key] for key in ('future','valid','states','realized','waypoint_valid')})
    return dict(**initial,branches=branches,metadata=metadata,index=index)


def local_positions(states):
    # 将每条分支的真实世界位置变换到共同的起点坐标系。
    delta=states[:,:2]-states[0,:2]
    angle=states[0,2]
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    return delta@rotation


def closest_outcome(predicted, actual, valid):
    # 比较预测特征与四份真实后果的特征；不是从预测特征生成图片。
    distance=np.square(actual-predicted).mean(axis=(-1,-2))
    distance=np.where(valid,distance,np.inf)
    order=np.argsort(distance)
    # 无有效目标或并列最近时，不编造唯一答案。
    tied=not np.isfinite(distance[order[0]]) or np.isclose(distance[order[0]],distance[order[1]],atol=1e-8,rtol=0)
    return (None if tied else int(order[0])),distance


def compare(scene, candidate, kind, horizon, wrong=False):
    # 所有预测均读取已冻结的正式输出，不用实际未来重新进行推理。
    c=context(int(scene));index=c['index']+int(candidate);slot=round(float(horizon)/.5)-1
    predicted=frozen_predictions(kind)['wrong_prediction' if wrong else 'prediction'][index,slot]
    actual,valid=targets();actual=actual[c['index']:c['index']+4,slot];valid=valid[c['index']:c['index']+4,slot]
    matched,distance=closest_outcome(predicted,actual,valid)
    return c,slot,matched,distance
