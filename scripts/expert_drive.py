"""Interactive expert demo and one-episode recording implementation."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np

from configs.stage2 import ActionNoiseConfig, ExpertConfig
from env.expert import ActionNoise, NavigationExpert
from env.ground_truth_bev import SimulatorGroundTruthBEV
from env.stage2 import make_data_env
from evaluation.visuals import LiveViewer, VideoWriter, compose, main_view, prepare_graphics
from mini_wa_jepa.geometry import world_to_ego, wrap_angle


def ego_state(car):
    return np.array([*car.position, car.heading_theta, car.speed, *car.velocity],dtype=np.float32)


def planned_trajectory(expert, steps=25, dt=0.1):
    car = expert.sim.agent
    speed = max(2.0, min(expert.cfg.target_speed_m_s, car.speed))
    points = []
    for k in range(1, steps+1):
        point, _, _, heading = expert.route(speed * dt * k)
        xy = world_to_ego(point, car.position, car.heading_theta)
        points.append([*xy, float(wrap_angle(heading-car.heading_theta))])
    return np.asarray(points,dtype=np.float32)


def main(record_default=False):
    p = argparse.ArgumentParser()
    p.add_argument("--map", default="SC")
    p.add_argument("--seed", type=int, default=9000)
    p.add_argument("--traffic", type=float, default=0.1)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--live", action="store_true")
    group.add_argument("--headless", action="store_true")
    p.add_argument("--native-window",action="store_true",help="Use native Panda3D onscreen window as well; multi-reset stability is limited")
    p.add_argument("--noise", action="store_true")
    p.add_argument("--demonstrate-recovery", action="store_true",
                   help="Add a labeled +0.35 steering pulse for steps 80..87")
    p.add_argument("--record", action="store_true", default=record_default)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--stage", default="stage2")
    args = p.parse_args()
    visual = args.live or args.record
    _, restore = prepare_graphics(args.headless or not args.live, offscreen_egl=args.live and not args.native_window) if visual else ([], lambda:None)
    sim = make_data_env(args.map,args.seed,args.traffic,args.max_steps,
                        render_mode=("live" if args.native_window else "viewer") if args.live else "offscreen" if visual else "physics")
    viewer = writer = None
    rows, resets = [], 0
    try:
        sim.reset(seed=args.seed)
        provider=SimulatorGroundTruthBEV(sim); provider.reset()
        expert=NavigationExpert(sim)
        noise_config=replace(ActionNoiseConfig(),enabled=args.noise)
        noise=ActionNoise(args.seed+20261016,noise_config)
        if args.live: viewer=LiveViewer(sim=sim)
        if args.record: writer=VideoWriter(args.map,args.seed,args.stage)
        info={}
        for step in range(args.max_steps):
            if viewer:
                while not viewer.ready() and not viewer.quit and not viewer.reset_requested:
                    frame=compose(provider.capture().raster,planned_trajectory(expert),ego_state(sim.agent),
                        np.zeros(2),"PAUSED",main_view(sim))
                    viewer.draw(frame)
                if viewer.quit: break
                if viewer.reset_requested:
                    sim.reset(seed=args.seed);provider.reset();expert=NavigationExpert(sim)
                    noise=ActionNoise(args.seed+20261016+resets,noise_config)
                    resets+=1;viewer.reset_requested=False
            expert_action, detail=expert.act()
            action, noisy=noise.apply(expert_action)
            impulse=args.demonstrate_recovery and 80<=step<88
            if impulse:
                action[0]=np.clip(action[0]+0.35,-1,1);noisy=True
            manual=bool(viewer and viewer.manual)
            if manual: action=viewer.manual_action(); noisy=False
            state=ego_state(sim.agent)
            if visual:
                status=f"{'MANUAL' if manual else 'AUTO'} | {'NOISE ACTIVE' if noisy else 'tracking / recovery'} | lateral {detail['lateral_m']:+.2f} m | crash {info.get('crash',False)} out {info.get('out_of_road',False)}"
                frame=compose(provider.capture().raster,planned_trajectory(expert),state,action,status,main_view(sim))
                if viewer: viewer.draw(frame)
                if writer: writer.write(frame)
            _,_,tm,tc,info=sim.step(action)
            rows.append(dict(step=step,speed_m_s=float(state[3]),lateral_m=detail['lateral_m'],
                expert_action=expert_action.tolist(),executed_action=action.tolist(),noise_active=noisy,manual=manual,
                demonstration_impulse=impulse))
            if tm or tc: break
        # Include terminal flags in the last displayed/recorded frame.
        if visual and rows:
            frame=compose(provider.capture().raster,planned_trajectory(expert),ego_state(sim.agent),action,
                f"END arrive={info.get('arrive_dest')} crash={info.get('crash')} out={info.get('out_of_road')}",main_view(sim))
            if writer: writer.write(frame)
            if viewer: viewer.draw(frame)
        report=dict(map=args.map,seed=args.seed,noise_config=asdict(noise_config),steps=len(rows),
            video=str(writer.path) if writer else None,trace=rows,
            outcome={k:info.get(k) for k in ['arrive_dest','crash','out_of_road','max_step','route_completion']})
        if writer:
            writer.path.with_suffix('.json').write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({k:v for k,v in report.items() if k!='trace'},indent=2))
    finally:
        if writer: writer.close()
        if viewer: viewer.close()
        sim.close();restore()


if __name__ == "__main__":
    main()
