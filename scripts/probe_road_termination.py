"""Verify road footprint rays and grace timing against a real native map."""
from pathlib import Path
import numpy as np
from configs.stage2 import Stage2Config
from env.data_v2 import configure_egl,make_env,FullOffroadRule
from mini_wa_jepa.provenance import provenance,write_json


def main():
    cfg=Stage2Config();_,restore=configure_egl(cfg.egl_device)
    sim=make_env('S',500100,cfg)
    try:
        sim.reset(seed=500100);car=sim.agent;original=np.asarray(car.position).copy()
        rule=FullOffroadRule(.1,1.)
        car.set_position([1000.,1000.]);car.on_lane=False
        off=[]
        for step in range(1,11):
            sim.engine.episode_step=step
            first=rule(sim,car);second=rule(sim,car)
            assert first==second
            off.append(bool(first))
        assert off==[False]*9+[True],off
        car.set_position(original);car.on_lane=False
        sim.engine.episode_step=11
        assert not rule(sim,car)  # real lane hit resets the full-body off-road counter
        write_json(Path('artifacts/restart_stage2/road_termination_probe.json'),dict(**provenance(cfg),
            full_body_offroad_ten_decisions=off,repeated_query_does_not_advance_time=True,
            actual_native_map_rays=True,return_to_road_resets_counter=True,passed=True))
        print('Native road-footprint ray and 1s grace probe passed')
    finally:sim.close();restore()


if __name__=='__main__':main()
