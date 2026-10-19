"""One frozen, matched-cohort stage 4 test evaluation after both formal runs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from configs.stage2 import BaselineConfig
from configs.stage4 import Stage4Config
from dataset.baseline import BaselineDataset
from dataset.joint import JointSequences
from evaluation.latent import paired_episode_bootstrap
from mini_wa_jepa.normalization import Normalizer
from mini_wa_jepa.operators import sdpa_context
from mini_wa_jepa.provenance import provenance, write_json
from models.joint import JointModel, constant_turn_trajectory
from training.joint import evaluate, make_loader, visualize


def load_joint(path, device):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['data_schema_version']!=2:raise ValueError('Requires v2 checkpoint')
    config=Stage4Config(**saved['model_config']);normalizer=Normalizer(**saved['normalization'])
    model=JointModel(config,normalizer).to(device).eval();model.load_state_dict(saved['model'])
    return model,config,saved


def physical_metrics(arrays):
    error=arrays['prediction']-arrays['target'];distance=np.linalg.norm(error[...,:2],axis=-1)
    arrays.update(ade_m=distance.mean(1),fde_m=distance[:,-1],
        speed_mae_m_s=np.abs(error[...,2]).mean(1),
        heading_mae_rad=np.abs(np.arctan2(np.sin(error[...,3]),np.cos(error[...,3]))).mean(1))
    full=arrays['full_mask']
    return dict(full_horizon_frames=int(full.sum()),**{k:float(arrays[k][full].mean())
        for k in ('ade_m','fde_m','speed_mae_m_s','heading_mae_rad')})


@torch.no_grad()
def evaluate_baseline(path,root,device):
    from training.baseline import make_model,make_loader
    saved=torch.load(path,map_location='cpu',weights_only=False)
    config=BaselineConfig(**saved['model_config']);normalizer=Normalizer(**saved['normalization'])
    data=BaselineDataset(root,'test');model=make_model(config).to(device).eval();model.load_state_dict(saved['model'])
    plans=[];targets=[];anchors=[];full=[]
    for raw in make_loader(data,config):
        p=model(raw['bev'].to(device),raw['ego'].to(device)).float().cpu().numpy()
        plans.append(normalizer.inverse_trajectory(p));targets.append(normalizer.inverse_trajectory(raw['trajectory'].numpy()))
        anchors.extend(raw['anchor'].numpy());full.extend((raw['valid'].all(1)&raw['action_mask']).numpy())
    anchors=np.asarray(anchors)
    arrays=dict(anchors=anchors,episodes=np.asarray(data.episodes[anchors]),full_mask=np.asarray(full),
        prediction=np.concatenate(plans),target=np.concatenate(targets))
    return physical_metrics(arrays),arrays


def summarize_groups(arrays,root):
    manifest=json.loads((root/'manifest.json').read_text())
    maps={row['episode_id']:row['map_spec'] for row in manifest['episodes']}
    map_names=np.asarray([maps[int(ep)] for ep in arrays['episodes']])
    groups={f'map/{name}':map_names==name for name in np.unique(map_names)}
    for key in ('noisy','recovery'):
        if key in arrays:
            groups[key]=arrays[key].astype(bool);groups['without_'+key]=~arrays[key].astype(bool)
    result={}
    for name,mask in groups.items():
        selected=mask&arrays['full_mask']
        result[name]=dict(frames=int(selected.sum()),**{key:float(arrays[key][selected].mean()) if selected.any() else None
            for key in ('ade_m','fde_m')})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('artifacts/restart_stage4'))
    parser.add_argument('--output',type=Path,default=None,help='Output directory; defaults to root/test_eval')
    parser.add_argument('--data',type=Path,default=Path('outputs/data/stage2_v2_100k'))
    parser.add_argument('--sdpa-backend',type=str,default='MATH',choices=['MATH','AUTO','FLASH','EFFICIENT','CUDNN'],help='SDPA attention backend')
    args=parser.parse_args();output=args.output or (args.root/'test_eval');output.mkdir(parents=True,exist_ok=True)
    if not torch.version.hip or torch.version.cuda is not None:raise RuntimeError('ROCm required')
    torch.set_num_threads(6);device=torch.device('cuda');data=JointSequences(args.data,'test')
    reports={name:json.loads((args.root/name/'report.json').read_text()) for name in ('joint_001','action_only_001')}
    for run in reports.values():
        if not run['completed'] or run['max_batches'] is not None or len(run['epochs'])!=12:
            raise ValueError('Both full formal runs must finish before test')
    if reports['joint_001']['initial_model_sha256']!=reports['action_only_001']['initial_model_sha256']:
        raise ValueError('Ablation initialization mismatch')
    core=('configs/stage4.py','models/joint.py','dataset/joint.py','training/joint.py','training/joint_loss.py')
    for path in core:
        if len({r['source_files'][path] for r in reports.values()})!=1:raise ValueError(f'Core source changed: {path}')
    report=dict(**provenance(),stage=4,completed=False,models={},paired={},groups={},sdpa_backend=args.sdpa_backend,
        initial_model_sha256=reports['joint_001']['initial_model_sha256'],cohort='action-valid complete 2s horizons, identical raw anchors')
    raw={}
    with torch.inference_mode(),sdpa_context(args.sdpa_backend):
        for name,run in [('joint','joint_001'),('action_only','action_only_001')]:
            checkpoint=args.root/run/'best.pt';model,config,saved=load_joint(checkpoint,device)
            metrics,arrays=evaluate(model,make_loader(data,config),config,device)
            report['models'][name]=dict(**metrics,epoch=saved['epoch'],checkpoint=str(checkpoint),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            raw[name]=arrays;np.savez_compressed(output/f'{name}_raw.npz',**arrays)
            report['groups'][name]=summarize_groups(arrays,args.data)
            visualize(model,data,config,device,output/name,saved['global_step'],saved['epoch'])
            write_json(output/'report.json',report);del model
        metrics,arrays=evaluate_baseline(Path('artifacts/restart_stage2/baseline/best.pt'),args.data,device)
        report['models']['resnet']=metrics;raw['resnet']=arrays;np.savez_compressed(output/'resnet_raw.npz',**arrays)
        # No fit: the identical observed-motion prior from both learned policies.
        prior=constant_turn_trajectory(torch.from_numpy(np.asarray(data.ego[data.history_indices[data.anchors]])).to(device)).cpu().numpy()
        arrays=dict(anchors=data.anchors.copy(),episodes=np.asarray(data.episodes[data.anchors]),
            full_mask=np.asarray(data.valid[data.anchors]).all(1)&np.asarray(data.action_mask[data.anchors]),
            prediction=prior,target=np.asarray(data.trajectories[data.anchors]),
            noisy=np.asarray(data.noisy[data.anchors]),recovery=np.asarray(data.recovery[data.anchors]))
        report['models']['constant_turn']=physical_metrics(arrays);raw['constant_turn']=arrays
        np.savez_compressed(output/'constant_turn_raw.npz',**arrays)
    reference=raw['joint'];ids=reference['anchors'][reference['full_mask']];eps=reference['episodes'][reference['full_mask']]
    for name,arrays in raw.items():
        if not np.array_equal(ids,arrays['anchors'][arrays['full_mask']]):raise ValueError(f'Cohort mismatch: {name}')
        report['groups'][name]=summarize_groups(arrays,args.data)
    frozen=json.loads(Path('artifacts/restart_stage2/baseline_eval/report.json').read_text())['open_loop']
    for key in ('ade_m','fde_m'):
        if abs(report['models']['resnet'][key]-frozen[key])>2e-6:raise ValueError('Frozen ResNet score failed reproduction')
    for other in ('resnet','action_only','constant_turn'):
        report['paired']['joint_minus_'+other]={key:paired_episode_bootstrap(reference[key][reference['full_mask']],
            raw[other][key][raw[other]['full_mask']],eps,2000,20261018) for key in ('ade_m','fde_m')}
    np.savez_compressed(output/'paired_cohort.npz',anchors=ids,episodes=eps)
    report.update(completed=True,paired_frames=len(ids),paired_episodes=len(np.unique(eps)))
    write_json(output/'report.json',report);print(json.dumps(report['models'],indent=2))


if __name__=='__main__':main()
