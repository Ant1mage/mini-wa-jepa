"""阶段5：正式模型在匹配保留种子上接受在线感知退化。"""
import argparse
from dataclasses import asdict
import faulthandler
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.tensorboard import SummaryWriter

from configs.stage2 import Stage2Config, SPLITS
from configs.stage5 import Stage5Config, CHECKPOINTS, condition
from controller.trajectory_tracker import TrajectoryTracker
from env.data_v2 import NativeBEVProvider, configure_egl, make_env, third_person
from env.expert import NavigationExpert
from evaluation.degradation import OnlineDegrader
from evaluation.history import ObservedHistory
from evaluation.memory import memory_snapshot
from evaluation.stress import FrozenPolicy, require_rocm
from evaluation.stress_visualization import closed_recording
from mini_wa_jepa.geometry import wrap_angle
from mini_wa_jepa.provenance import provenance, write_json


def main():
    faulthandler.enable(); parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=list(CHECKPOINTS), required=True)
    parser.add_argument('--condition', default='clean', choices=Stage5Config.closed_conditions)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, help='Smoke only: one map/seed, never formal acceptance')
    parser.add_argument('--record', action='store_true')
    args=parser.parse_args();cfg=Stage5Config();spec=condition(args.condition)
    args.output.mkdir(parents=True,exist_ok=False);require_rocm()
    torch.set_num_threads(cfg.cpu_threads);torch.manual_seed(cfg.seed)
    policy=FrozenPolicy(args.model,CHECKPOINTS[args.model],torch.device('cuda'))
    simcfg=Stage2Config(egl_device=cfg.egl_device)
    report=dict(**provenance(cfg),stage=5,policy=args.model,condition=asdict(spec),split='test',
                completed=False,max_steps=args.max_steps,episodes=[],**policy.metadata,
                control_dt_s=cfg.control_dt_s,episodes_per_map=cfg.closed_episodes_per_map,
                inference_definition='degradation + transfer + FP32 MATH model + same tracker; ego uncorrupted',
                pipeline_definition='decision plus env.step/render/readback/conversion; first20/reset/recording excluded')
    writer=SummaryWriter(str(args.output/'tensorboard'))
    started=time.perf_counter();latencies=[];pipelines=[];sample_seeds=[];sample_steps=[];reset_seconds=[]
    write_json(args.output/'report.json',report)
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        raw=dict(bev=np.zeros((1,4,4,128,128),np.uint8),ego=np.zeros((1,4,2),np.float32))
        for _ in range(20):policy.batch(raw)
    torch.cuda.synchronize();_,restore=configure_egl(cfg.egl_device)
    try:
        with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
            maps=SPLITS['test']['maps'][:1] if args.max_steps else SPLITS['test']['maps']
            for map_index,map_spec in enumerate(maps):
                base=SPLITS['test']['seed_start']+map_index*10000
                sim=make_env(map_spec,base,simcfg,camera=args.record,num_scenarios=10000,control_dt_s=cfg.control_dt_s)
                try:
                    count=1 if args.max_steps else cfg.closed_episodes_per_map
                    for episode_index in range(count):
                        seed=base+9000+episode_index;tick=time.perf_counter()
                        observation,_=sim.reset(seed=seed);reset_seconds.append(time.perf_counter()-tick)
                        provider=NativeBEVProvider(sim,simcfg.bev);provider.reset(observation)
                        scorer=NavigationExpert(sim,simcfg.expert,cfg.control_dt_s)
                        tracker=TrajectoryTracker(sim.agent,cfg.control_dt_s)
                        history=ObservedHistory(cfg.control_dt_s,.1)
                        degrader=OnlineDegrader(spec,cfg.seed,seed,cfg.control_dt_s)
                        record_this=args.record and map_index==0 and episode_index==0
                        records=[];states=[];plans=[];actions=[];sources=[];laterals=[];routes=[];actor_pixels=[]
                        previous_heading=None;crash_seen=out_seen=False;latency_start=len(latencies)
                        limit=args.max_steps or round(simcfg.horizon*simcfg.dt_s/cfg.control_dt_s)
                        for step in range(limit):
                            record_sample=record_this and step%5==0
                            rgb=third_person(sim) if record_sample else None
                            tick=time.perf_counter();car=sim.agent;clean_bev=provider.capture().raster
                            bev,source_step=degrader(clean_bev,step)
                            state=np.array([*car.position,car.heading_theta,car.speed,*car.velocity],np.float32)
                            _,_,lateral,_=scorer.route(0.)
                            if args.model=='resnet':
                                yaw=0. if previous_heading is None else float(wrap_angle(car.heading_theta-previous_heading))/cfg.control_dt_s
                                motion=np.array([[car.speed/10,yaw]],np.float32)
                                result=dict(trajectory=policy.model(torch.from_numpy(bev[None]).to(policy.device),
                                                                   torch.from_numpy(motion).to(policy.device)))
                                indices=None
                            else:
                                encoded=policy.model.encoder(torch.from_numpy(bev[None]).to(policy.device))
                                observed,motion,indices=history.append(step,encoded,car.speed,car.heading_theta)
                                result=policy.model.forward_encoded(observed,torch.from_numpy(motion).to(policy.device))
                            plan=policy.physical(result['trajectory'])[0]
                            if not np.isfinite(plan).all():raise FloatingPointError('Nonfinite driving trajectory')
                            action,_=tracker.act(plan,car.speed);previous_heading=float(car.heading_theta)
                            decision_ms=(time.perf_counter()-tick)*1000
                            observation,_,tm,tc,info=sim.step(action);provider.update(observation)
                            pipeline_ms=(time.perf_counter()-tick)*1000
                            if step>=20:
                                latencies.append(decision_ms);pipelines.append(pipeline_ms);sample_seeds.append(seed);sample_steps.append(step)
                            states.append(state);plans.append(plan);actions.append(action);sources.append(source_step)
                            laterals.append(float(lateral));routes.append(float(info['route_completion']))
                            actor_pixels.append(int(np.count_nonzero(clean_bev[2])))
                            crash_seen|=bool(info['crash']);out_seen|=bool(info['out_of_road'])
                            if record_sample:
                                row=dict(step=step,bev=bev.copy(),rgb=rgb,plan=plan,action=np.asarray(action),
                                         source_step=source_step,history_indices=indices,lateral=float(lateral),route=routes[-1])
                                if args.model!='resnet':
                                    # 在计时段以外编码本时刻干净输入，事后作为其他时刻的真实未来目标。
                                    row['encoded']=policy.model.encoder(torch.from_numpy(clean_bev[None]).to(policy.device))[0].float().cpu().numpy()
                                    row['latent']=result['latent'][0].float().cpu().numpy()
                                records.append(row)
                            if step%250==0:
                                progress=dict(model=args.model,condition=spec.name,map_spec=map_spec,seed=seed,step=step,
                                              route_completion=routes[-1],speed_m_s=float(state[3]),decision_ms=decision_ms,pipeline_ms=pipeline_ms)
                                write_json(args.output/'progress.json',progress)
                                np.savez_compressed(args.output/'last_observation_mre.npz',bev=bev,clean_bev=clean_bev,
                                                    state=state,ego=motion,prediction=plan,seed=seed,step=step)
                                writer.add_scalar(f'rollout/{seed}/route_completion',routes[-1],step);writer.flush()
                            if tm or tc:break
                        final_state=np.array([*sim.agent.position,sim.agent.heading_theta,sim.agent.speed,*sim.agent.velocity],np.float32)
                        evidence=closed_recording(args.output,args.model,spec,map_spec,seed,records,states,final_state,cfg.control_dt_s) if record_this else {}
                        np.savez_compressed(args.output/f'episode_{seed}.npz',states=np.asarray(states+[final_state]),
                                            plans=np.asarray(plans),actions=np.asarray(actions),observation_source=np.asarray(sources),
                                            lateral_error=np.asarray(laterals),route_completion=np.asarray(routes),actor_pixels=np.asarray(actor_pixels))
                        row=dict(map_spec=map_spec,seed=seed,steps=step+1,simulated_seconds=(step+1)*cfg.control_dt_s,
                                 arrive_dest=bool(info['arrive_dest']),collision=crash_seen,out_of_road=out_seen,truncated=bool(tc),
                                 route_completion=routes[-1],lateral_error_m=float(np.mean(np.abs(laterals))),
                                 mean_speed_m_s=float(np.mean(np.asarray(states)[:,3])),memory=memory_snapshot(),evidence=evidence,
                                 inference_ms=float(np.mean(latencies[latency_start:])) if len(latencies)>latency_start else None,
                                 pipeline_ms_p95=float(np.percentile(pipelines[latency_start:],95)) if len(latencies)>latency_start else None)
                        report['episodes'].append(row);write_json(args.output/'report.json',report);print(json.dumps(row),flush=True)
                        for key,value in row.items():
                            if isinstance(value,(float,int)):writer.add_scalar(f'episode/{key}',value,len(report['episodes'])-1)
                finally:sim.close()
        rows=report['episodes']
        report.update(completed=True,metrics=dict(route_completion=float(np.mean([r['route_completion'] for r in rows])),
                      collision_rate=float(np.mean([r['collision'] for r in rows])),out_of_road_rate=float(np.mean([r['out_of_road'] for r in rows])),
                      success_rate=float(np.mean([r['arrive_dest'] for r in rows])),lateral_error_m=float(np.mean([r['lateral_error_m'] for r in rows])),
                      inference_ms=float(np.mean(latencies)) if latencies else None),
                      pipeline_ms_p50_p95_p99=np.percentile(pipelines,[50,95,99]).tolist() if pipelines else [],
                      latency_ms_p50_p95_p99=np.percentile(latencies,[50,95,99]).tolist() if latencies else [])
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        restore();report['elapsed_seconds']=time.perf_counter()-started;report['reset_seconds']=reset_seconds
        np.savez_compressed(args.output/'latency_samples_ms.npz',decision_ms=latencies,pipeline_ms=pipelines,seed=sample_seeds,step=sample_steps)
        write_json(args.output/'report.json',report);writer.close()


if __name__=='__main__':main()
