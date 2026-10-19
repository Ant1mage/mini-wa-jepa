"""Frozen stage 2 held-out closed loop, open loop, latency, video and Rerun."""
import argparse
from dataclasses import replace
import faulthandler
import json
from pathlib import Path
import time
import traceback

import numpy as np

from configs.stage2 import Stage2Config,SPLITS,BaselineConfig
from env.data_v2 import NativeBEVProvider,configure_egl,make_env,third_person
from env.expert import NavigationExpert
from evaluation.memory import memory_snapshot
from mini_wa_jepa.geometry import wrap_angle
from mini_wa_jepa.provenance import provenance,write_json


def main():
    faulthandler.enable()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',choices=['expert','baseline'],default='baseline')
    p.add_argument('--checkpoint',type=Path,default=Path('artifacts/restart_stage2/baseline/best.pt'))
    p.add_argument('--data',type=Path,default=Path('outputs/data/stage2_v2_100k'))
    p.add_argument('--output',type=Path,default=Path('artifacts/restart_stage2/baseline_eval'))
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--episodes-per-map',type=int,default=10)
    p.add_argument('--control-dt',type=float,default=.02)
    p.add_argument('--max-steps',type=int,help='Smoke only, otherwise full 100 simulated seconds')
    p.add_argument('--record',action='store_true')
    p.add_argument('--rerun',action='store_true')
    p.add_argument('--egl-device',type=int,default=1)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    cfg=Stage2Config(egl_device=args.egl_device)
    report=dict(**provenance(cfg),policy=args.policy,split=args.split,control_dt_s=args.control_dt,
        episodes_per_map=args.episodes_per_map,max_steps=args.max_steps,completed=False,episodes=[],
        inference_definition='CPU uint8 BEV + ego to HIP, model, inverse normalization and tracker; no env.step',
        pipeline_definition='inference/controller plus env.step, native semantic render/readback and four-channel conversion; visualization/disk separate')
    torch=model=normalizer=None
    if args.policy=='baseline':
        import torch
        from models.baseline import ResNet18Trajectory
        from mini_wa_jepa.normalization import Normalizer
        from configs.stage2 import BaselineConfig
        if not torch.version.hip or torch.version.cuda is not None:raise RuntimeError('ROCm required')
        torch.set_num_threads(6)
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if checkpoint['data_schema_version']!=2:raise ValueError('Checkpoint schema mismatch')
        model_cfg=BaselineConfig(**checkpoint['model_config'])
        model=ResNet18Trajectory(pooling_size=model_cfg.pooling_size,mlp_hidden=model_cfg.mlp_hidden,latent_dim=model_cfg.latent_dim).to('cuda').eval()
        model.load_state_dict(checkpoint['model']);normalizer=Normalizer(**checkpoint['normalization'])
        report.update(checkpoint=str(args.checkpoint),checkpoint_epoch=checkpoint['epoch'],
            parameter_count=checkpoint['parameter_count'],training_source=checkpoint['provenance'])
        # Open-loop evaluation is performed once, after checkpoint selection.
        from training.baseline import evaluate,make_loader
        from dataset.baseline import BaselineDataset
        if args.data.exists() and not args.max_steps:
            data=BaselineDataset(args.data,args.split)
            report['open_loop']=evaluate(model,make_loader(data,model_cfg),normalizer,torch.device('cuda'))
        # Warm inference before measuring steady-state controller latency.
        with torch.inference_mode():
            for _ in range(20):model(torch.zeros(1,4,128,128,dtype=torch.uint8,device='cuda'),torch.zeros(1,2,device='cuda'))
        torch.cuda.synchronize()
    write_json(args.output/'report.json',report)
    _,restore=configure_egl(cfg.egl_device)
    started=time.perf_counter();latencies=[];pipelines=[];sample_episode=[];sample_step=[];cold=[];logger=None;video=None
    try:
        for map_index,map_spec in enumerate(SPLITS[args.split]['maps']):
            base=SPLITS[args.split]['seed_start']+map_index*10000
            sim=make_env(map_spec,base,cfg,camera=args.record,num_scenarios=10000,control_dt_s=args.control_dt)
            try:
                for episode_index in range(args.episodes_per_map):
                    seed=base+9000+episode_index
                    t=time.perf_counter();observation,_=sim.reset(seed=seed);cold.append(time.perf_counter()-t)
                    provider=NativeBEVProvider(sim,cfg.bev);provider.reset(observation)
                    expert=NavigationExpert(sim,cfg.expert,args.control_dt)
                    if args.policy=='baseline':
                        from controller.trajectory_tracker import TrajectoryTracker
                        tracker=TrajectoryTracker(sim.agent,args.control_dt)
                    record_this=args.record and episode_index==0
                    if record_this:
                        from evaluation.visuals import VideoWriter
                        from mini_wa_jepa.visualization import RerunLogger
                        video=VideoWriter(map_spec,seed,stage=f'stage2_v2_{args.policy}',fps=10)
                        logger=RerunLogger(app_id=f'mini-wa-jepa-stage2-{args.policy}',spawn=args.rerun,
                                          save_path=args.output/f'map_{map_spec}_seed_{seed}.rrd')
                    states=[];plans=[];bevs=[];all_info={};speed=[];lateral=[];previous_heading=None
                    latency_start=len(latencies)
                    crash_seen=out_seen=False
                    max_steps=args.max_steps or round(cfg.horizon*cfg.dt_s/args.control_dt)
                    for step in range(max_steps):
                        record_sample=record_this and step%max(1,round(.1/args.control_dt))==0
                        rgb=third_person(sim) if record_sample else None
                        tick=time.perf_counter();car=sim.agent;bev=provider.capture().raster
                        state=np.array([*car.position,car.heading_theta,car.speed,*car.velocity],np.float32)
                        # Navigation geometry is used solely for scoring; the model sees no route data.
                        _,_,lat,_=expert.route(0.)
                        if args.policy=='expert':
                            action,detail=expert.act()
                            plan=None
                            if record_this:
                                from scripts.expert_drive import planned_trajectory
                                preview=planned_trajectory(expert,steps=10,dt=.2)
                                plan=np.column_stack([preview[:,:2],np.full(10,detail['target_speed_m_s']),preview[:,2]])
                        else:
                            yaw_rate=0. if previous_heading is None else float(wrap_angle(car.heading_theta-previous_heading))/args.control_dt
                            ego=np.array([[car.speed/10,yaw_rate]],np.float32)
                            with torch.inference_mode():
                                prediction=model(torch.from_numpy(bev[None]).to('cuda'),torch.from_numpy(ego).to('cuda'))
                                plan=normalizer.inverse_trajectory(prediction[0].float().cpu().numpy())
                            action,detail=tracker.act(plan,car.speed)
                        previous_heading=float(car.heading_theta)
                        decision_ms=(time.perf_counter()-tick)*1000
                        observation,_,tm,tc,info=sim.step(action);provider.update(observation)
                        pipeline_ms=(time.perf_counter()-tick)*1000
                        if step>=20:
                            latencies.append(decision_ms);pipelines.append(pipeline_ms)
                            sample_episode.append(seed);sample_step.append(step)
                        speed.append(float(state[3]));lateral.append(float(lat));states.append(state)
                        plans.append(plan)
                        crash_seen|=bool(info['crash']);out_seen|=bool(info['out_of_road'])
                        if record_sample:
                            from evaluation.visuals import compose
                            logger.set_step(step,step*args.control_dt)
                            logger.log_bev(bev);logger.log_state(state)
                            logger.log_telemetry(speed_m_s=state[3],steering=action[0],throttle_brake=action[1],
                                lateral_error_m=lat,route_completion=info['route_completion'])
                            if plan is not None:logger.log_trajectories(planned=plan)
                            logger.log_camera(rgb)
                            video.write(compose(bev,plan,state,action,
                                f'{args.policy} {map_spec} seed {seed} step {step} route {info["route_completion"]:.2f}',rgb,
                                caption='model prediction' if args.policy=='baseline' else 'expert preview'))
                        if tm or tc:break
                    # Backfill actual future trajectories at their original observation times.
                    if record_this:
                        from dataset.recording import future_labels_v2
                        final_state=np.array([*sim.agent.position,sim.agent.heading_theta,sim.agent.speed,*sim.agent.velocity],np.float32)
                        realized=np.asarray(states+[final_state])
                        gt,valid,_=future_labels_v2(realized,np.zeros(len(realized),bool),10,round(.2/args.control_dt))
                        for i in range(0,len(states),max(1,round(.1/args.control_dt))):
                            logger.set_step(i,i*args.control_dt);logger.log_trajectories(ground_truth=gt[i][valid[i]])
                        video_path=str(video.path);video.close();video=None;logger.close();logger=None
                    else:video_path=None
                    row=dict(map_spec=map_spec,seed=seed,steps=step+1,simulated_seconds=(step+1)*args.control_dt,
                        arrive_dest=bool(info['arrive_dest']),collision=crash_seen,out_of_road=out_seen,
                        truncated=bool(tc),route_completion=float(info['route_completion']),
                        lateral_error_m=float(np.mean(np.abs(lateral))),max_abs_lateral_m=float(np.max(np.abs(lateral))),
                        mean_speed_m_s=float(np.mean(speed)),video=video_path,memory=memory_snapshot(),
                        inference_ms=float(np.mean(latencies[latency_start:])) if len(latencies)>latency_start else None,
                        pipeline_ms_p95=float(np.percentile(pipelines[latency_start:],95)) if len(latencies)>latency_start else None)
                    report['episodes'].append(row);write_json(args.output/'report.json',report)
                    print(json.dumps(row),flush=True)
            finally:sim.close()
        rows=report['episodes']
        report.update(completed=True,metrics=dict(route_completion=float(np.mean([r['route_completion'] for r in rows])),
            collision_rate=float(np.mean([r['collision'] for r in rows])),
            out_of_road_rate=float(np.mean([r['out_of_road'] for r in rows])),
            success_rate=float(np.mean([r['arrive_dest'] for r in rows])),
            lateral_error_m=float(np.mean([r['lateral_error_m'] for r in rows])),
            inference_ms=float(np.mean(latencies)) if latencies else None),
            latency_ms_p50_p95_p99=np.percentile(latencies,[50,95,99]).tolist() if latencies else [],
            pipeline_ms_p50_p95_p99=np.percentile(pipelines,[50,95,99]).tolist() if pipelines else [],
            reset_seconds=cold)
        if 'open_loop' in report:
            report['metrics'].update({key:report['open_loop'][key] for key in ['ade_m','fde_m']})
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        if logger:logger.close()
        if video:video.close()
        restore();report['elapsed_seconds']=time.perf_counter()-started
        np.savez_compressed(args.output/'latency_samples_ms.npz',decision_ms=np.asarray(latencies),
            pipeline_ms=np.asarray(pipelines),seed=np.asarray(sample_episode),step=np.asarray(sample_step))
        report['raw_latency_samples']='latency_samples_ms.npz'
        write_json(args.output/'report.json',report)


if __name__=='__main__':main()
