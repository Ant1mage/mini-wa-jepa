"""Replay the current v2 dataset through the shared Rerun interface."""
import argparse
from dataclasses import fields
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from configs.stage2 import Stage2Config,BEVConfig,ExpertConfig,ActionNoiseConfig
from dataset.baseline import validate_manifest
from dataset.storage import ShardReader
from evaluation.visuals import VideoWriter,compose
from mini_wa_jepa.provenance import write_json
from mini_wa_jepa.visualization.streaming import add_rerun_arguments,make_rerun_logger,log_recorded_frame


def config_from_manifest(manifest):
    raw=dict(manifest['config'])
    for key,cls in [('bev',BEVConfig),('expert',ExpertConfig),('action_noise',ActionNoiseConfig)]:
        raw[key]=cls(**raw[key])
    return Stage2Config(**raw)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('outputs/data/stage2_v2_100k'))
    p.add_argument('--split',choices=['train','val','test'],default='train')
    p.add_argument('--episode',type=int)
    p.add_argument('--live',action='store_true',help='Alias for --rerun')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--max-frames',type=int,default=100000)
    p.add_argument('--reconstruct-camera',action='store_true',help='Explicit offscreen RGB reconstruction, without Pygame')
    p.add_argument('--video',action='store_true')
    p.add_argument('--output',type=Path,default=Path('artifacts/restart_stage2/replay'))
    add_rerun_arguments(p);args=p.parse_args()
    if args.max_frames<=0:p.error('--max-frames must be positive')
    manifest=validate_manifest(args.data);reader=ShardReader(args.data/args.split)
    args.output.mkdir(parents=True,exist_ok=False)
    episodes=reader.metadata('episode_id')
    episode_id=int(episodes[0]) if args.episode is None else args.episode
    positions=np.flatnonzero(episodes==episode_id)
    if not len(positions):raise ValueError('Episode not found in selected split')
    cfg=config_from_manifest(manifest)
    first=reader[int(positions[0])]
    logger=make_rerun_logger(args,'mini-wa-jepa-stage2-replay')
    video=sim=None;restore=lambda:None;errors=[];thumbnails=[]
    try:
        if args.reconstruct_camera:
            from env.data_v2 import configure_egl,make_env,third_person
            episode=next(e for e in manifest['episodes'] if e['episode_id']==episode_id)
            _,restore=configure_egl(cfg.egl_device)
            sim=make_env(episode['map_spec'],episode['seed'],cfg,camera=True)
            sim.reset(seed=episode['seed'])
        if logger:logger.log_context(static=True,data_path=str(args.data.resolve()),schema_version=2,
            git_commit=manifest['git_commit'],goal_sha256=manifest['goal_sha256'],split=args.split)
        if args.video:video=VideoWriter(str(first['map_id'])[:12],int(first['seed']),stage='stage2_v2_replay',fps=1/cfg.dt_s)
        preview_indices=set(np.linspace(0,min(len(positions),args.max_frames)-1,12,dtype=int).tolist())
        for index,position in enumerate(positions[:args.max_frames]):
            row=reader[int(position)]
            rgb=None
            if sim:
                rgb=third_person(sim)
                errors.append(float(np.linalg.norm(np.asarray(sim.agent.position)-row['state'][:2])))
                if errors[-1]>=.05:raise RuntimeError(f'Reconstruction diverged at {index}: {errors[-1]}m')
            row['camera_rgb']=rgb
            if logger:log_recorded_frame(logger,row,dt_s=cfg.dt_s,timeline_step=int(row['step_id']))
            frame=compose(row['bev'],row['future_trajectory'][row['future_valid']],row['state'],row['executed_action'],
                f"v2 ep {episode_id} step {row['step_id']} noise={row['noise_active']} recovery={row['recovery_active']} action={row['action_mask']}",
                rgb,caption='recorded actual future; 10 x (x,y,v,heading)')
            if video:video.write(frame)
            if index in preview_indices:
                Image.fromarray(frame).save(args.output/f'frame_{index:05d}.png')
                thumbnails.append(cv2.resize(frame,(640,320)))
            if sim and row['transition_valid']:sim.step(row['executed_action'])
        sheet=np.full((((len(thumbnails)+2)//3)*320,1920,3),20,np.uint8)
        for i,frame in enumerate(thumbnails):sheet[(i//3)*320:(i//3+1)*320,(i%3)*640:(i%3+1)*640]=frame
        Image.fromarray(sheet).save(args.output/'contact_sheet.png')
        report=dict(schema_version=2,episode_id=episode_id,seed=int(first['seed']),
            frames=min(len(positions),args.max_frames),video=str(video.path) if video else None,
            max_position_reconstruction_error_m=max(errors) if errors else None)
        write_json(args.output/'report.json',report);print(json.dumps(report),flush=True)
    finally:
        if logger:logger.close()
        if video:video.close()
        if sim:sim.close()
        restore()


if __name__=='__main__':main()
