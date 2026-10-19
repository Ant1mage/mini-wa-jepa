"""Native-camera MRE before the new stage 1 acceptance run."""
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image


def main():
    os.environ.pop('DISPLAY',None)
    os.environ.pop('WAYLAND_DISPLAY',None)
    from panda3d.core import loadPrcFileData
    loadPrcFileData('', 'load-display p3headlessgl\negl-device-index 1\naudio-library-name null\nsync-video false')
    from env.render_compat import limit_filter_msaa
    changes, restore = limit_filter_msaa(0)
    from env.semantic_bev import make_semantic_env,native_rgb,semantic_channels,calibration
    root=Path('artifacts/restart_stage1/probe');root.mkdir(parents=True,exist_ok=True)
    sim=make_semantic_env(episodes=1,horizon=200)
    report={}
    try:
        obs,_=sim.reset(seed=0)
        report['calibration_initial']=calibration(sim)
        durations=[]
        for step in range(150):
            tick=time.perf_counter()
            obs,_,tm,tc,info=sim.step([0,0])
            rgb=native_rgb(obs)
            bev=semantic_channels(rgb,sim.agent.LENGTH,sim.agent.WIDTH)
            durations.append(time.perf_counter()-tick)
            if step==100:
                Image.fromarray(rgb).save(root/'native_semantic.png')
                np.save(root/'bev.npy',bev)
                report['palette']=[dict(rgb=c.tolist(),count=int(n)) for c,n in zip(*np.unique(rgb.reshape(-1,3),axis=0,return_counts=True))]
            if tm or tc:break
        g=sim.engine.win.getGsg()
        report.update(renderer=g.getDriverRenderer(),pipe=sim.engine.pipe.getType().getName(),
            calibration_final=calibration(sim),shape=list(bev.shape),channel_pixels=(bev>0).sum((1,2)).tolist(),
            complete_pipeline_fps=len(durations)/sum(durations),steps=len(durations),msaa_adjustments=changes,
            drm_fdinfo=[dict(fd=f.name,text=f.read_text()) for f in Path('/proc/self/fdinfo').iterdir()
                        if f.exists() and 'drm-' in f.read_text()])
        print(json.dumps(report,indent=2))
        (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:sim.close();restore()


if __name__=='__main__':main()
