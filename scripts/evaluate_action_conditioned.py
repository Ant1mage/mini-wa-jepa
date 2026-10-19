"""One frozen Stage 6 test evaluation after all matched formal runs finish."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from configs.stage6 import Stage6Config
from dataset.branch_features import BranchFeatures
from evaluation.action_conditioned import cluster_interval,consequence_metrics
from evaluation.latent import LatentMoments
from mini_wa_jepa.provenance import provenance,write_json
from training.action_conditioned import batch_gpu,evaluate,loader,make_model,visualize


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cfg=Stage6Config();torch.set_num_threads(cfg.cpu_threads)
    if not torch.version.hip or torch.version.cuda:raise RuntimeError('Native ROCm required')
    args.output.mkdir(parents=True,exist_ok=False);data=BranchFeatures(args.data,'test')
    target=np.asarray(data.arrays['target']);valid=np.asarray(data.arrays['valid']);groups=np.asarray(data.arrays['group']);seeds=np.asarray(data.arrays['seed'])
    report=dict(**provenance(cfg),stage=6,completed=False,models={},paired={},scientific_gates={},
        cohort=dict(samples=len(data),contexts=len(np.unique(groups)),seeds=len(np.unique(seeds)),future_times_s=list(cfg.future_times_s)))
    started=time.perf_counter();raws={};initials={}
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        moments=LatentMoments(64,256,'cuda')
        for i in range(0,len(data),16):
            z=torch.from_numpy(target[i:i+16][valid[i:i+16]]).to('cuda')
            moments.update(z)
        health,_=moments.finish();report['target_health']=health
        for seed in cfg.training_seeds:
            for conditioned in (True,False):
                name=('conditioned' if conditioned else 'unconditional')+f'_{seed}'
                run=args.runs/name;training=json.loads((run/'report.json').read_text())
                if not training['completed'] or training['smoke'] or len(training['epochs'])!=cfg.epochs:
                    raise ValueError(f'Incomplete/nonformal training {name}')
                initials[name]=training['initial_model_sha256']
                checkpoint=run/'best.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
                model,_=make_model(cfg,conditioned);model.load_state_dict(saved['model']);model=model.to('cuda').eval()
                _,arrays=evaluate(model,data,cfg,True);prediction=arrays['prediction']
                metrics,raw=consequence_metrics(prediction,target,valid,groups,seeds)
                wrong=[];timing=[]
                for batch_raw in loader(data,cfg):
                    batch=batch_gpu(batch_raw);commands=batch['command'].clone()
                    # Loader is ordered by whole groups of four; never permute across contexts.
                    if len(commands)%4:raise ValueError('Whole groups required for wrong-command control')
                    commands=commands.reshape(-1,4,10,4).roll(1,1).reshape(-1,10,4)
                    torch.cuda.synchronize();tick=time.perf_counter()
                    value=model(batch['history'],batch['ego'],commands)
                    torch.cuda.synchronize();timing.append((time.perf_counter()-tick)*1000/len(commands))
                    wrong.append(value.cpu().numpy())
                wrong=np.concatenate(wrong);wrong_error=np.square(wrong-target).mean((-1,-2))
                metrics['wrong_command_mse']=float(wrong_error[valid].mean())
                metrics['cached_prediction_ms_per_sample_mean']=float(np.mean(timing))
                prediction_moments=LatentMoments(64,256,'cuda')
                for i in range(0,len(data),16):prediction_moments.update(torch.from_numpy(prediction[i:i+16][valid[i:i+16]]).to('cuda'))
                prediction_health,_=prediction_moments.finish();metrics['prediction_health']=prediction_health
                row=dict(**metrics,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    best_epoch=saved['epoch'],seed=seed,conditioned=conditioned,parameter_count=saved['parameter_count'])
                raw.update(prediction=prediction,wrong_prediction=wrong,wrong_error=wrong_error,groups=groups,seeds=seeds)
                np.savez_compressed(args.output/f'{name}_raw.npz',**raw)
                visualize(model,data,cfg,args.output/name,saved['epoch'],saved['global_step'])
                raws[name]=raw;report['models'][name]=row;write_json(args.output/'report.json',report)
                del model,saved
            c,u=f'conditioned_{seed}',f'unconditional_{seed}'
            if initials[c]!=initials[u]:raise ValueError('Matched initialization mismatch')
            cr,ur=raws[c],raws[u];cm,um=report['models'][c],report['models'][u]
            paired=cluster_interval((cr['error']*valid).sum(1),(ur['error']*valid).sum(1),valid.sum(1),seeds,cfg.bootstrap_samples,cfg.seed)
            delta_ci=cluster_interval(cr['pair_error'],cr['target_pair_energy'],np.ones(len(cr['pair_error'])),cr['pair_seeds'],cfg.bootstrap_samples,cfg.seed)
            wrong_ci=cluster_interval((cr['error']*valid).sum(1),(cr['wrong_error']*valid).sum(1),valid.sum(1),seeds,cfg.bootstrap_samples,cfg.seed)
            gain=1-cm['latent_mse']/um['latent_mse'];delta_gain=1-cm['delta_mse']/cm['target_delta_mse']
            report['paired'][str(seed)]=dict(latent_gain=gain,delta_gain=delta_gain,latent=paired,consequence_delta=delta_ci,correct_vs_wrong=wrong_ci)
            report['scientific_gates'][str(seed)]=dict(
                latent_gain=gain>=cfg.minimum_latent_gain and paired['delta_ci95'][1]<0,
                delta_gain=delta_gain>=cfg.minimum_delta_gain,
                retrieval=cm['retrieval_accuracy']>=cfg.minimum_retrieval_accuracy,
                wrong_command=cm['wrong_command_mse']>cm['latent_mse'])
        copy=np.repeat(np.asarray(data.arrays['history'])[:,-1,None],4,axis=1)
        report['copy_current']=consequence_metrics(copy,target,valid,groups,seeds)[0]
        unadapted,_=make_model(cfg,False);unadapted=unadapted.to('cuda').eval()
        unadapted_metrics,unadapted_arrays=evaluate(unadapted,data,cfg,True)
        report['stage4_horizon_mismatch_context']=dict(**unadapted_metrics,
            trained_horizons_s=[.1,.2,.3,.4],evaluated_horizons_s=list(cfg.future_times_s),
            warning='Horizon-mismatched context only, not a matched baseline')
        np.savez_compressed(args.output/'stage4_horizon_mismatch_raw.npz',**unadapted_arrays)
    source=json.loads((Path(data.manifest['data'])/'manifest.json').read_text())
    branch_rows=[b for g in source['groups'] if g['split']=='test' for b in g['branches']]
    distance=np.linalg.norm(np.asarray(data.arrays['command'])[...,:2]-np.asarray(data.arrays['realized'])[...,:2],axis=-1)
    way_valid=np.asarray(data.arrays['waypoint_valid']);full=way_valid.all(1)
    report['actual_branch_outcomes']=dict(command_tracking_ade_m=float(distance[way_valid].mean()),
        command_tracking_fde_m=float(distance[full,-1].mean()) if full.any() else None,
        collision_branches=sum(r['collision'] for r in branch_rows),out_of_road_branches=sum(r['out_of_road'] for r in branch_rows),
        branches=len(branch_rows),complete_world_futures=int(valid.all(1).sum()),complete_trajectory_outcomes=int(full.sum()),
        predicted_trajectory_ade_m=None,predicted_trajectory_fde_m=None,
        autonomous_closed_loop_success_rate=None)
    noise_error=np.load(data.folder/'noise_error.npy');noise_valid=np.load(data.folder/'noise_valid.npy')
    noise=float(noise_error[noise_valid].mean()) if noise_valid.any() else None
    energy=report['models'][f'conditioned_{cfg.training_seeds[0]}']['target_delta_mse']
    report['replay_noise']=dict(replicate_contexts=len(noise_error),latent_mse=noise,
        actual_candidate_delta_mse=energy,delta_to_noise_ratio=energy/max(noise,1e-15) if noise is not None else None,
        passed=noise is not None and energy>10*noise)
    report.update(completed=True,scientific_passed=all(all(g.values()) for g in report['scientific_gates'].values()) and report['replay_noise']['passed'],elapsed_seconds=time.perf_counter()-started)
    write_json(args.output/'report.json',report)
    print(json.dumps(dict(completed=True,scientific_passed=report['scientific_passed'],paired=report['paired'])),flush=True)


if __name__=='__main__':main()
