"""Small simulator experiment; development seeds never enter dataset splits."""
import argparse
import json
from pathlib import Path

import numpy as np

from configs.stage2 import ExpertConfig
from env.stage2 import make_data_env
from env.ground_truth_bev import SimulatorGroundTruthBEV
from env.expert import NavigationExpert
from mini_wa_jepa.geometry import wrap_angle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", default="C")
    p.add_argument("--seed", type=int, default=9000)
    p.add_argument("--traffic", type=float, default=0)
    p.add_argument("--output", type=Path, default=Path("artifacts/stage2/api_probe.json"))
    args = p.parse_args()
    sim = make_data_env(args.map, args.seed, traffic_density=args.traffic)
    report = {"map": args.map, "seed": args.seed, "traffic_density": args.traffic}
    try:
        for sign in [-1, 1]:
            sim.reset(seed=args.seed)
            sim.agent.set_velocity(sim.agent.heading, 6)
            before = sim.agent.heading_theta
            for _ in range(10):
                sim.step([sign * 0.2, 0])
            delta = float(wrap_angle(sim.agent.heading_theta - before))
            assert delta * sign > 0
            report[f"steer_{sign}_heading_delta_rad"] = delta
        sim.reset(seed=args.seed)
        report["dynamics"] = sim.agent.get_dynamics_parameters()
        report["dt_s"] = sim.config["physics_world_step_size"] * sim.config["decision_repeat"]
        assert report["dt_s"] == 0.1 and report["dynamics"]["max_engine_force"] == 800
        provider = SimulatorGroundTruthBEV(sim)
        provider.reset()
        b = provider.capture()
        report["bev"] = dict(shape=b.raster.shape, pixels=(b.raster>0).sum((1,2)).tolist(),
                             metadata=b.metadata, geometry_hash=provider.geometry_hash)
        expert = NavigationExpert(sim, ExpertConfig())
        laterals = []
        for step in range(1000):
            action, detail = expert.act()
            _, _, tm, tc, info = sim.step(action)
            laterals.append(detail["lateral_m"])
            if step % 200 == 0:
                print(step, sim.agent.speed, action.tolist(), detail, flush=True)
            if tm or tc:
                break
        report["expert"] = {k: info[k] for k in ["arrive_dest", "crash", "out_of_road", "max_step", "route_completion"]}
        report["expert"].update(steps=step+1, max_abs_lateral=float(np.max(np.abs(laterals))))
    finally:
        sim.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
