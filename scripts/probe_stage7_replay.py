"""Bounded native RGB/prefix determinism probe for a preserved failed context."""
import argparse
from collections import deque
import json
from pathlib import Path
import numpy as np
from configs.stage2 import Stage2Config
from dataset.recording import vehicle_state
from env.data_v2 import configure_egl,make_env,NativeBEVProvider
from env.semantic_bev import native_rgb
from mini_wa_jepa.provenance import provenance,write_json
from scripts.preflight_risk_collection import scene_state


def probe(context,map_spec,seed,output,repetitions=8,fresh=False,depth24=False,ordered=False):
    cfg=Stage2Config();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    raw=np.load(context);rows=[];sim=None;reference=None
    _,restore=configure_egl(1)
    overrides={'force_destroy':True}
    if depth24:
        from metadrive.component.sensors.semantic_camera import SemanticCamera
        from metadrive.component.sensors.base_camera import BaseCamera
        from panda3d.core import FrameBufferProperties
        class Depth24SemanticCamera(SemanticCamera):
            def __init__(self,width,height,engine,*,cuda=False):
                self.BUFFER_W,self.BUFFER_H=width,height
                p=FrameBufferProperties();p.setRgbaBits(8,8,8,8);p.setDepthBits(24)
                p.setForceHardware(True);p.setMultisamples(0);p.setSrgbColor(False);p.setStereo(False);p.setStencilBits(0)
                BaseCamera.__init__(self,engine,cuda,p)
        overrides['sensors']={'bev_semantic':(Depth24SemanticCamera,128,128)}
    try:
        for repeat in range(repetitions):
            if sim is None or fresh:
                if sim is not None:sim.close()
                sim=make_env(map_spec,seed,cfg,num_scenarios=1000,control_dt_s=.1,environment_overrides=overrides)
            obs,_=sim.reset(seed=seed);provider=NativeBEVProvider(sim,cfg.bev);provider.reset(obs)
            if ordered:
                from metadrive.constants import Semantics
                from panda3d.core import CullBinAttrib
                cam=sim.engine.get_sensor('bev_semantic').cam.node()
                for name in vars(Semantics):
                    if name.startswith('_'):continue
                    category=getattr(Semantics,name)
                    if not hasattr(category,'label'):continue
                    rank={'TERRAIN':0,'ROAD':10,'CROSSWALK':11,'LANE_LINE':12,'SIDEWALK':20}.get(name,30)
                    cam.setTagState(category.label,cam.getTagState(category.label).addAttrib(CullBinAttrib.make('fixed',rank)))
            histories=deque(maxlen=4);rgbs=deque(maxlen=4);states=deque(maxlen=5);poses=deque(maxlen=5)
            for action in raw['prefix_actions']:
                obs,_,tm,tc,info=sim.step(action);provider.update(obs)
                histories.append(provider.capture().raster.copy());rgbs.append(native_rgb(obs));states.append(vehicle_state(sim.agent))
                poses.append(np.asarray([*sim.agent.position,sim.agent.heading_theta,sim.agent.speed,*sim.agent.velocity],np.float64))
                if tm or tc:raise RuntimeError('Unexpected prefix termination')
            h=np.asarray(histories);rgb=np.asarray(rgbs);s=np.asarray(states);pose=np.asarray(poses)
            if reference is None:reference=(h,rgb,s,pose)
            olddiff=np.argwhere(h!=raw['history']);diff=np.argwhere(h!=reference[0])
            row=dict(repeat=repeat,vs_original_different=int(len(olddiff)),different=int(len(diff)),
                original_indices=olddiff.tolist(),indices=diff.tolist(),
                state_error=float(abs(s[-4:]-raw['states']).max()),
                float64_pose_error=float(abs(pose-reference[3]).max()),
                camera_matrix=list(sim.engine.get_sensor('bev_semantic').cam.getMat()),
                rgb_different_values=int(np.count_nonzero(rgb!=reference[1])))
            row['actual_depth_bits']=sim.engine.get_sensor('bev_semantic').buffer.getFbProperties().getDepthBits()
            row.pop('camera_matrix')  # Native Mat rows are captured as an actual numeric array below.
            np.savez_compressed(output/f'{repeat:02d}.npz',history=h,rgb=rgb,states=s,pose=pose,
                scene=scene_state(sim),camera=np.asarray(sim.engine.get_sensor('bev_semantic').cam.getMat()))
            rows.append(row);write_json(output/'report.json',dict(**provenance(),completed=False,fresh=fresh,depth24=depth24,ordered=ordered,rows=rows))
            print(json.dumps(row),flush=True)
        write_json(output/'report.json',dict(**provenance(),completed=True,fresh=fresh,depth24=depth24,ordered=ordered,rows=rows))
    finally:
        if sim is not None:sim.close()
        restore()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--context',required=True);p.add_argument('--map',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--repetitions',type=int,default=8)
    p.add_argument('--fresh',action='store_true');p.add_argument('--depth24',action='store_true')
    p.add_argument('--ordered',action='store_true')
    a=p.parse_args();probe(a.context,a.map,a.seed,a.output,a.repetitions,a.fresh,a.depth24,a.ordered)
