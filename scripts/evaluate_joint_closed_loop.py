"""Frozen stage 4 policies on stage 2's exact closed-loop seeds and controller."""
import argparse
import faulthandler
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from configs.stage2 import Stage2Config,SPLITS
from controller.trajectory_tracker import TrajectoryTracker
from env.data_v2 import NativeBEVProvider,configure_egl,make_env,third_person
from env.expert import NavigationExpert
from evaluation.history import ObservedHistory
from evaluation.memory import memory_snapshot
from mini_wa_jepa.provenance import provenance,write_json
from scripts.evaluate_joint import load_joint


def delayed_latent_targets(records,dt):
    steps=[r['step'] for r in records]
    if len(set(steps))!=len(steps) or steps!=sorted(steps):raise ValueError('Recorded steps must be unique and ordered')
    stride=round(.1/dt)
    if abs(stride*dt-.1)>1e-8:raise ValueError('Latent observation interval must be an integer control stride')
    lookup={r['step']:r['encoded'] for r in records}
    predicted=np.stack([r['latent'] for r in records]);target=np.zeros_like(predicted)
    valid=np.zeros((len(records),4),bool)
    for index,row in enumerate(records):
        for horizon in range(4):
            future=lookup.get(row['step']+(horizon+1)*stride)
            if future is not None:target[index,horizon]=future;valid[index,horizon]=True
    return predicted,target,valid


def save_recording(output,model_name,map_spec,seed,records,states,final_state,dt,spawn=False):
    """Join predictions to subsequently observed targets; missing tails stay masked."""
    from dataset.recording import future_labels_v2
    from evaluation.visuals import VideoWriter,compose
    from mini_wa_jepa.geometry import ego_to_pixels
    from mini_wa_jepa.visualization import RerunLogger
    from PIL import Image,ImageDraw
    actual=np.asarray(states+[final_state]);steps=np.asarray([r['step'] for r in records])
    gt,valid,_=future_labels_v2(actual,np.zeros(len(actual),bool),10,round(.2/dt))
    predicted,target,world_valid=delayed_latent_targets(records,dt)
    rrd=output/f'map_{map_spec}_seed_{seed}.rrd'
    video=VideoWriter(map_spec,seed,stage=f'stage4_{model_name}',fps=10)
    snapshots=[]
    try:
        with RerunLogger(app_id=f'mini-wa-jepa-stage4-{model_name}',spawn=spawn,save_path=rrd) as logger:
            for index,row in enumerate(records):
                step=row['step'];logger.set_step(step,step*dt)
                mask=world_valid[index]
                logger.log_context(policy=model_name,map_spec=map_spec,seed=seed,control_step=step,
                    history_indices=row['history_indices'],world_valid=mask.tolist(),
                    latent_horizons_s=[.1,.2,.3,.4],waypoint_dt_s=.2,targets='delayed actual observations from this policy rollout')
                logger.log_bev(row['bev']);logger.log_camera(row['rgb']);logger.log_state(states[step])
                logger.log_trajectories(planned=row['plan'],ground_truth=gt[step][valid[step]])
                logger.log_telemetry(speed_m_s=states[step][3],steering=row['action'][0],throttle_brake=row['action'][1],
                    lateral_error_m=row['lateral'],route_completion=row['route'])
                if mask.any():logger.log_latents(predicted=predicted[index][mask],ground_truth=target[index][mask])
                canvas=Image.fromarray(compose(row['bev'],row['plan'],states[step],row['action'],
                    f'{model_name} {map_spec} seed {seed} step {step} route {row["route"]:.2f}',row['rgb'],
                    caption='cyan: model | orange: subsequently realized trajectory'))
                points=(ego_to_pixels(gt[step][valid[step],:2],128,64)+.5)*4+np.array([748,40])
                draw=ImageDraw.Draw(canvas)
                if len(points)>1:draw.line([tuple(p) for p in points],fill='orange',width=3)
                draw.rectangle((0,580,1280,640),fill=(17,21,29))
                draw.text((20,605),'Frozen policy rollout | delayed actual targets | offline acceptance video',fill=(225,230,240))
                video.write(np.asarray(canvas))
                if index in (0,len(records)//2,len(records)-1):
                    path=output/f'map_{map_spec}_seed_{seed}_step_{step:05d}.png';canvas.save(path);snapshots.append(str(path))
    finally:video.close()
    raw=output/f'map_{map_spec}_seed_{seed}_actual.npz'
    np.savez_compressed(raw,states=actual,steps=steps,trajectory_prediction=np.stack([r['plan'] for r in records]),
        trajectory_target=gt[steps],trajectory_valid=valid[steps],latent_prediction=predicted,
        latent_target=target,world_valid=world_valid,history_indices=np.asarray([r['history_indices'] for r in records]),
        actions=np.asarray([r['action'] for r in records]))
    return dict(video=str(video.path),rerun=str(rrd),raw=str(raw),snapshots=snapshots,
        recorded_frames=len(records),valid_latent_targets=int(world_valid.sum()),
        latent_mse=float(np.square(predicted-target).mean((-1,-2))[world_valid].mean()) if world_valid.any() else None)


def main():
    faulthandler.enable();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path('artifacts/restart_stage4/joint_001/best.pt'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/restart_stage4/joint_closed_loop'))
    parser.add_argument('--split',choices=['val','test'],default='test')
    parser.add_argument('--episodes-per-map',type=int,default=10)
    parser.add_argument('--control-dt',type=float,default=.02)
    parser.add_argument('--max-steps',type=int,help='Smoke only; no truncation in acceptance')
    parser.add_argument('--record',action='store_true');parser.add_argument('--rerun',action='store_true')
    parser.add_argument('--egl-device',type=int,default=1)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if not torch.version.hip or torch.version.cuda is not None:raise RuntimeError('ROCm required')
    torch.set_num_threads(6);device=torch.device('cuda');model,config,saved=load_joint(args.checkpoint,device)
    from mini_wa_jepa.normalization import Normalizer
    normalizer=Normalizer(**saved['normalization']);name='joint' if config.latent_weight else 'action_only'
    cfg=Stage2Config(egl_device=args.egl_device)
    report=dict(**provenance(cfg),stage=4,policy=name,split=args.split,control_dt_s=args.control_dt,
        episodes_per_map=args.episodes_per_map,max_steps=args.max_steps,completed=False,episodes=[],
        checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        checkpoint_epoch=saved['epoch'],parameter_count=saved['parameter_count'],model_config=saved['model_config'],
        training_source=saved['provenance'],history_dt_s=config.history_dt_s,
        inference_definition='current BEV CPU-to-HIP encoder, observed-history cache, FP32 SDPA MATH world/action, inverse normalization and same tracker',
        pipeline_definition='decision plus env.step/native semantic rendering/readback/4ch conversion; first20/reset/visualization/disk excluded')
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        dummy=torch.zeros(1,4,128,128,dtype=torch.uint8,device=device);ego=torch.zeros(1,4,2,device=device)
        for _ in range(20):
            z=model.encoder(dummy);model.forward_encoded(z[:,None].expand(-1,4,-1,-1),ego)
    torch.cuda.synchronize();write_json(args.output/'report.json',report)
    _,restore=configure_egl(cfg.egl_device)
    started=time.perf_counter();latencies=[];pipelines=[];sample_seed=[];sample_step=[];cold=[]
    try:
        with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
            for map_index,map_spec in enumerate(SPLITS[args.split]['maps']):
                base=SPLITS[args.split]['seed_start']+map_index*10000
                sim=make_env(map_spec,base,cfg,camera=args.record,num_scenarios=10000,control_dt_s=args.control_dt)
                try:
                    for episode_index in range(args.episodes_per_map):
                        seed=base+9000+episode_index;tick=time.perf_counter()
                        observation,_=sim.reset(seed=seed);cold.append(time.perf_counter()-tick)
                        provider=NativeBEVProvider(sim,cfg.bev);provider.reset(observation)
                        scorer=NavigationExpert(sim,cfg.expert,args.control_dt);tracker=TrajectoryTracker(sim.agent,args.control_dt)
                        history=ObservedHistory(args.control_dt,config.history_dt_s)
                        record_this=args.record and episode_index==0;records=[];states=[];speeds=[];laterals=[]
                        crash_seen=out_seen=False;latency_start=len(latencies)
                        max_steps=args.max_steps or round(cfg.horizon*cfg.dt_s/args.control_dt)
                        for step in range(max_steps):
                            record_sample=record_this and step%round(.1/args.control_dt)==0
                            rgb=third_person(sim) if record_sample else None
                            tick=time.perf_counter();car=sim.agent;bev=provider.capture().raster
                            state=np.array([*car.position,car.heading_theta,car.speed,*car.velocity],np.float32)
                            _,_,lateral,_=scorer.route(0.)
                            encoded=model.encoder(torch.from_numpy(bev[None]).to(device))
                            observed,motion,indices=history.append(step,encoded,car.speed,car.heading_theta)
                            result=model.forward_encoded(observed,torch.from_numpy(motion).to(device))
                            plan=normalizer.inverse_trajectory(result['trajectory'][0].float().cpu().numpy())
                            action,detail=tracker.act(plan,car.speed)
                            decision_ms=(time.perf_counter()-tick)*1000
                            observation,_,tm,tc,info=sim.step(action);provider.update(observation)
                            pipeline_ms=(time.perf_counter()-tick)*1000
                            if step>=20:
                                latencies.append(decision_ms);pipelines.append(pipeline_ms);sample_seed.append(seed);sample_step.append(step)
                            states.append(state);speeds.append(float(state[3]));laterals.append(float(lateral))
                            crash_seen|=bool(info['crash']);out_seen|=bool(info['out_of_road'])
                            if record_sample:
                                records.append(dict(step=step,bev=bev.copy(),rgb=rgb,plan=plan,action=np.asarray(action),
                                    encoded=encoded[0].float().cpu().numpy(),latent=result['latent'][0].float().cpu().numpy(),
                                    history_indices=indices,lateral=float(lateral),route=float(info['route_completion'])))
                            if step%250==0:
                                write_json(args.output/'progress.json',dict(map_spec=map_spec,seed=seed,step=step,
                                    route_completion=float(info['route_completion']),speed_m_s=float(state[3]),
                                    pipeline_ms=pipeline_ms,decision_ms=decision_ms))
                            if tm or tc:break
                        final_state=np.array([*sim.agent.position,sim.agent.heading_theta,sim.agent.speed,*sim.agent.velocity],np.float32)
                        evidence=save_recording(args.output,name,map_spec,seed,records,states,final_state,args.control_dt,args.rerun) if record_this else {}
                        row=dict(map_spec=map_spec,seed=seed,steps=step+1,simulated_seconds=(step+1)*args.control_dt,
                            arrive_dest=bool(info['arrive_dest']),collision=crash_seen,out_of_road=out_seen,truncated=bool(tc),
                            route_completion=float(info['route_completion']),lateral_error_m=float(np.mean(np.abs(laterals))),
                            max_abs_lateral_m=float(np.max(np.abs(laterals))),mean_speed_m_s=float(np.mean(speeds)),
                            memory=memory_snapshot(),video=evidence.get('video'),evidence=evidence,
                            inference_ms=float(np.mean(latencies[latency_start:])) if len(latencies)>latency_start else None,
                            pipeline_ms_p95=float(np.percentile(pipelines[latency_start:],95)) if len(latencies)>latency_start else None)
                        report['episodes'].append(row);write_json(args.output/'report.json',report);print(json.dumps(row),flush=True)
                finally:sim.close()
        rows=report['episodes']
        report.update(completed=True,metrics=dict(route_completion=float(np.mean([r['route_completion'] for r in rows])),
            collision_rate=float(np.mean([r['collision'] for r in rows])),out_of_road_rate=float(np.mean([r['out_of_road'] for r in rows])),
            success_rate=float(np.mean([r['arrive_dest'] for r in rows])),lateral_error_m=float(np.mean([r['lateral_error_m'] for r in rows])),
            inference_ms=float(np.mean(latencies)) if latencies else None),
            latency_ms_p50_p95_p99=np.percentile(latencies,[50,95,99]).tolist() if latencies else [],
            pipeline_ms_p50_p95_p99=np.percentile(pipelines,[50,95,99]).tolist() if pipelines else [],reset_seconds=cold)
    except BaseException:
        report['error']=traceback.format_exc()
        if 'bev' in locals():
            np.savez_compressed(args.output/'last_observation_mre.npz',bev=bev,state=state,ego=motion,
                encoded_history=observed.float().cpu().numpy(),seed=seed,step=step)
        raise
    finally:
        restore();report['elapsed_seconds']=time.perf_counter()-started
        np.savez_compressed(args.output/'latency_samples_ms.npz',decision_ms=np.asarray(latencies),pipeline_ms=np.asarray(pipelines),
            seed=np.asarray(sample_seed),step=np.asarray(sample_step))
        report['raw_latency_samples']='latency_samples_ms.npz';write_json(args.output/'report.json',report)


if __name__=='__main__':main()
