"""Revised stage 1 only: 4-channel native BEV, calibrated EGL, VRAM and FPS.

Rendering/physics/readback/channel conversion are measured together. Disk/video
and resets have separately reported end-to-end wall time. No model or data work.
"""
import argparse
from dataclasses import asdict
import faulthandler
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import cv2
import numpy as np
from PIL import Image,ImageDraw

from env.semantic_bev import SemanticBEVSpec,make_semantic_env,native_rgb,semantic_channels,calibration
from evaluation.memory import memory_snapshot


def preview(rgb,bev,label):
    canvas=np.full((640,1280,3),22,np.uint8)
    canvas[64:576,16:528]=cv2.resize(rgb,(512,512),interpolation=cv2.INTER_NEAREST)
    cv2.arrowedLine(canvas,(272,336),(272,300),(255,255,255),2)
    cv2.putText(canvas,'Native EGL semantic BEV / ego up',(18,38),0,.65,(230,230,230),1,cv2.LINE_AA)
    for c,title in enumerate(['C0 drivable','C1 boundary/static','C2 other actors','C3 visibility (full GT)']):
        x=570+(c%2)*300;y=64+(c//2)*288
        image=np.repeat(bev[c,:,:,None],3,axis=-1)
        canvas[y:y+256,x:x+256]=cv2.resize(image,(256,256),interpolation=cv2.INTER_NEAREST)
        cv2.putText(canvas,title,(x,y-10),0,.5,(230,230,230),1,cv2.LINE_AA)
    cv2.putText(canvas,label,(18,618),0,.6,(230,230,230),1,cv2.LINE_AA)
    return canvas


def main():
    faulthandler.enable()
    p=argparse.ArgumentParser()
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--horizon',type=int,default=500)
    p.add_argument('--start-seed',type=int,default=0)
    p.add_argument('--egl-device',type=int,default=1)
    p.add_argument('--output',type=Path,default=Path('artifacts/restart_stage1/egl_100'))
    p.add_argument('--record',action='store_true')
    args=p.parse_args()
    if (args.output/'report.json').exists() or (args.output/'progress.json').exists():raise FileExistsError('Use a new run directory')
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ.pop('DISPLAY',None);os.environ.pop('WAYLAND_DISPLAY',None)
    from panda3d.core import loadPrcFileData
    loadPrcFileData('',f'load-display p3headlessgl\negl-device-index {args.egl_device}\naudio-library-name null\nsync-video false')
    from env.render_compat import limit_filter_msaa
    adjustments,restore=limit_filter_msaa(0)  # universally disable unneeded PBR AA
    spec=SemanticBEVSpec()
    report=dict(spec=asdict(spec),episodes=[],samples=[],completed=False,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        goal_sha256=hashlib.sha256(Path('GOAL.md').read_bytes()).hexdigest(),
        source_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD','--','*.py'])).hexdigest(),
        config=dict(episodes=args.episodes,horizon=args.horizon,start_seed=args.start_seed,egl_device=args.egl_device,
                    map_blocks=3,traffic_density=0.1,msaa=0,visibility_mode='full_ground_truth'),
        memory_before=memory_snapshot(),msaa_adjustments=adjustments,
        channels=['drivable','boundary_static','other_actors','visibility'],
        visibility_note='255 everywhere in un-degraded top-down ground truth. No ray-cast occlusion in stage 1.',
        timing_definition='env.step including native semantic render and readback + RGB unpack + four-channel extraction')
    sim=writer=None;total=0;seconds=0;durations=[];started=time.perf_counter()
    seeds=set(np.linspace(args.start_seed,args.start_seed+args.episodes-1,min(20,args.episodes),dtype=int).tolist())
    try:
        sim=make_semantic_env(spec,args.episodes,args.start_seed,args.horizon)
        for seed in range(args.start_seed,args.start_seed+args.episodes):
            tick=time.perf_counter();sim.reset(seed=seed);reset_s=time.perf_counter()-tick
            if 'graphics' not in report:
                g=sim.engine.win.getGsg()
                report['graphics']=dict(renderer=g.getDriverRenderer(),vendor=g.getDriverVendor(),
                    version=g.getDriverVersion(),pipe=sim.engine.pipe.getType().getName())
                assert report['graphics']['pipe']=='eglGraphicsPipe' and '7900 XT' in report['graphics']['renderer']
                if args.record:
                    from evaluation.visuals import VideoWriter
                    writer=VideoWriter(f'pg3-seed{seed}',seed,stage='restart_stage1',fps=10)
                    report['video']=str(writer.path)
            ep_seconds=0.0;worst_calibration=0.0;saved=False
            for step in range(1,args.horizon+1):
                tick=time.perf_counter()
                obs,_,tm,tc,info=sim.step([0,0])
                rgb=native_rgb(obs);bev=semantic_channels(rgb,sim.agent.LENGTH,sim.agent.WIDTH,spec)
                dt=time.perf_counter()-tick;seconds+=dt;ep_seconds+=dt;total+=1;durations.append(dt)
                assert bev.shape==(4,128,128) and bev.dtype==np.uint8
                if step%50==0 or tm or tc:
                    check=calibration(sim,spec)
                    worst_calibration=max(worst_calibration,max(c['max_error_pixel'] for c in check.values()))
                    assert worst_calibration<0.05,check
                if seed in seeds and not saved and (step>=100 or tm or tc):
                    check=calibration(sim,spec)
                    stem=f'seed_{seed:03d}_step_{step:03d}'
                    np.savez_compressed(args.output/f'{stem}.npz',bev=bev,native_rgb=rgb)
                    Image.fromarray(preview(rgb,bev,f'seed {seed} / step {step} / 0.5m per pixel / 64m square')).save(args.output/f'{stem}.png')
                    report['samples'].append(dict(seed=seed,step=step,file=stem+'.npz',calibration=check,
                        channel_pixels=(bev>0).sum((1,2)).tolist()))
                    saved=True
                if writer and seed==args.start_seed:
                    writer.write(preview(rgb,bev,f'seed {seed} step {step} speed {sim.agent.speed:.2f}m/s crash={info["crash"]} out={info["out_of_road"]}'))
                if tm or tc:break
            assert tm or tc
            if writer and seed==args.start_seed:writer.close();writer=None
            memory=memory_snapshot()
            row=dict(seed=seed,steps=step,reset_seconds=reset_s,pipeline_seconds=ep_seconds,
                     pipeline_fps=step/ep_seconds,max_calibration_error_pixel=worst_calibration,memory=memory,
                     **{k:info[k] for k in ['arrive_dest','crash','out_of_road','max_step']})
            report['episodes'].append(row)
            (args.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
            if len(report['episodes'])%5==0 or args.episodes<5:print(json.dumps(row),flush=True)
        report['completed']=True
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        if writer:writer.close()
        if sim:sim.close()
        restore();gc.collect()
        report.update(total_steps=total,pipeline_seconds=seconds,pipeline_fps=total/seconds if seconds else 0,
                      pipeline_ms_p50_p95=(np.percentile(durations,[50,95])*1000).tolist() if durations else [])
        report['elapsed_seconds']=time.perf_counter()-started
        report['wall_fps']=total/report['elapsed_seconds']
        report['memory_after_close']=memory_snapshot()
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    paths=sorted(args.output.glob('seed_*.png'))
    sheet=Image.new('RGB',(1280,((len(paths)+1)//2)*320),'#161616')
    for i,path in enumerate(paths):
        img=Image.open(path);img.thumbnail((640,320));sheet.paste(img,((i%2)*640,(i//2)*320))
    sheet.save(args.output/'contact_sheet.png')
    print(json.dumps({k:v for k,v in report.items() if k not in ['episodes','samples']},indent=2))


if __name__=='__main__':main()
