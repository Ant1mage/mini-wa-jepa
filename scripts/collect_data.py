"""Stage 2 v2 expert collection: native EGL BEV and explicit velocity waypoints."""
import argparse
from dataclasses import replace
import faulthandler
import json
from pathlib import Path
import time
import traceback

from configs.stage2 import SPLITS, Stage2Config
from dataset.recording import record_episode_v2
from dataset.storage import ShardWriter
from env.data_v2 import NativeBEVProvider, configure_egl, make_env
from evaluation.memory import memory_snapshot
from mini_wa_jepa.provenance import provenance, write_json
from mini_wa_jepa.visualization.streaming import add_rerun_arguments, make_rerun_logger, log_recorded_frame


def main():
    faulthandler.enable()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/data/stage2_v2_100k'))
    parser.add_argument('--frames',type=int,default=100000,help='Training-frame quota (validation/test additional)')
    parser.add_argument('--val-frames',type=int,default=10000)
    parser.add_argument('--test-frames',type=int,default=10000)
    parser.add_argument('--egl-device',type=int,default=1)
    parser.add_argument('--no-noise',action='store_true')
    parser.add_argument('--live',action='store_true',help='Alias for independent Rerun viewer')
    parser.add_argument('--headless',action='store_true',help='EGL simulator has no native window (always)')
    parser.add_argument('--camera',action='store_true',help='Add offscreen third-person RGB for visualization')
    parser.add_argument('--keyboard',action='store_true',help='Independent keyboard-control process')
    parser.add_argument('--only-split',choices=list(SPLITS))
    parser.add_argument('--only-map',help='Collect one map in an isolated worker')
    parser.add_argument('--map-frames',type=int,help='Quota for --only-map')
    parser.add_argument('--resume',action='store_true',help='Resume a flushed partial v2 run')
    add_rerun_arguments(parser)
    args=parser.parse_args()
    if min(args.frames,args.val_frames,args.test_frames)<4:parser.error('All split quotas must be >=4')
    if args.only_map and (not args.only_split or not args.map_frames):parser.error('--only-map requires --only-split and --map-frames')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    cfg=Stage2Config(frames=args.frames,validation_frames=args.val_frames,test_frames=args.test_frames,egl_device=args.egl_device)
    cfg=replace(cfg,action_noise=replace(cfg.action_noise,enabled=not args.no_noise))
    report=dict(schema_version=2,spec_status='v2',**provenance(cfg),completed=False,
        trajectory_fields=['x_forward_m','y_left_m','speed_m_s','heading_relative_rad'],
        future_times_s=[cfg.waypoint_dt_s*k for k in range(1,cfg.future_steps+1)],
        state_fields=['world_x_m','world_y_m','heading_rad','speed_m_s','world_vx_m_s','world_vy_m_s'],
        action_mask_rule='only irrecoverable terminal observation or structurally absent control; waypoint validity separate',
        splits={},episodes=[],memory_before=memory_snapshot())
    if args.resume:
        previous=json.loads((args.output/'manifest.json').read_text())
        if previous.get('schema_version')!=2 or previous.get('completed'):raise ValueError('Resume only an incomplete v2 run')
        previous_config=dict(previous['config'])
        if 'offroad_grace_s' not in previous_config:
            # Earlier partial runs never terminated off road: their trajectories
            # are unchanged under the stricter full-body/grace termination rule.
            if any(e['outcome']['out_of_road'] for e in previous['episodes']):
                raise ValueError('Old centre-off-lane terminals require recollection')
            previous_config['offroad_grace_s']=cfg.offroad_grace_s
            report['resume_compatibility_note']='No prior out-of-road events; full-body termination leaves retained trajectories unchanged'
        if previous_config!=report['config']:raise ValueError('Resume configuration mismatch')
        report['episodes']=previous['episodes']
        report['provenance_chain']=previous.get('provenance_chain',[])+[{key:previous[key] for key in
            ('git_commit','goal_sha256','source_tree_sha256','source_files')}]
    write_json(args.output/'manifest.json',report)
    logger=make_rerun_logger(args,'mini-wa-jepa-stage2-collection')
    controls=None
    if args.keyboard:
        from mini_wa_jepa.visualization.keyboard import KeyboardControls
        controls=KeyboardControls()
    adjustments,restore=configure_egl(cfg.egl_device)
    report['msaa_adjustments']=adjustments
    offset=sum(e['simulated_steps']+1 for e in report['episodes'])
    episode_id=len(report['episodes']);started=time.perf_counter();writer=None
    try:
        if logger:logger.log_context(static=True,schema_version=2,git_commit=report['git_commit'],
            goal_sha256=report['goal_sha256'],planned_trajectory_kind='expert_navigation_preview')
        for (split,spec),budget in zip(SPLITS.items(),[cfg.frames,cfg.validation_frames,cfg.test_frames]):
            if args.only_split and split!=args.only_split:continue
            writer=ShardWriter(args.output/split,cfg.shard_frames,resume=args.resume and (args.output/split/'index.json').exists())
            report['splits'][split]=dict(frames=budget,**spec)
            for map_index,map_spec in enumerate(spec['maps']):
                if args.only_map and map_spec!=args.only_map:continue
                quota=args.map_frames if args.only_map else budget//len(spec['maps'])+int(map_index<budget%len(spec['maps']))
                previous_episodes=[e for e in report['episodes'] if e['split']==split and e['map_spec']==map_spec]
                remaining=quota-sum(e['stored_frames'] for e in previous_episodes)
                if remaining<=0:continue
                base_seed=spec['seed_start']+map_index*10000
                sim=make_env(map_spec,base_seed,cfg,camera=args.camera,num_scenarios=10000)
                try:
                    for local_episode in range(len(previous_episodes),1000):
                        seed=base_seed+local_episode
                        observation,_=sim.reset(seed=seed)
                        provider=NativeBEVProvider(sim,cfg.bev);provider.reset(observation)
                        report['bev_metadata']=provider.capture().metadata
                        if 'graphics' not in report:
                            g=sim.engine.win.getGsg()
                            report['graphics']=dict(renderer=g.getDriverRenderer(),pipe=sim.engine.pipe.getType().getName())
                            if report['graphics']['pipe']!='eglGraphicsPipe':raise RuntimeError(report['graphics'])
                        def stream_step(row,planned):
                            log_recorded_frame(logger,row,dt_s=cfg.dt_s,timeline_step=offset+int(row['step_id']),
                                               extent_m=cfg.bev.extent_m,planned=planned)
                        data,episode=record_episode_v2(sim,provider,cfg,seed,episode_id,remaining,
                            on_step=stream_step if logger else None,controls=controls)
                        if data is not None:
                            writer.add(data);remaining-=len(data['state'])
                            if logger:
                                for i in range(len(data['state'])):
                                    logger.set_step(offset+int(data['step_id'][i]),(offset+int(data['step_id'][i]))*cfg.dt_s)
                                    logger.log_trajectories(ground_truth=data['future_trajectory'][i][data['future_valid'][i]])
                        episode.update(split=split,map_spec=map_spec,calibration=provider.calibration,
                            vehicle_dynamics=sim.agent.get_dynamics_parameters(),memory=memory_snapshot())
                        report['episodes'].append(episode)
                        offset+=episode['simulated_steps']+1;episode_id+=1
                        print(__import__('json').dumps(episode),flush=True)
                        write_json(args.output/'manifest.json',report)
                        if controls and controls.quit:break
                        if remaining<=0:break
                    if remaining>0 and not (controls and controls.quit):raise RuntimeError('Exceeded 1000 episodes/map; reserve evaluation seeds')
                finally:sim.close()
                if controls and controls.quit:break
            writer.close();writer=None
            if controls and controls.quit:break
        report['completed']=not (controls and controls.quit)
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        if writer:writer.close()
        if logger:logger.close()
        if controls:controls.close()
        restore()
        report.update(stored_frames=sum(e['stored_frames'] for e in report['episodes']),
            simulated_steps=sum(e['simulated_steps'] for e in report['episodes']),
            elapsed_seconds=time.perf_counter()-started,memory_after=memory_snapshot())
        write_json(args.output/'manifest.json',report)


if __name__=='__main__':main()
