"""Frozen expert held-out evaluation; no test-driven controller tuning."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from configs.stage2 import SPLITS,Stage2Config
from env.expert import NavigationExpert
from env.stage2 import make_data_env


def main():
    p=argparse.ArgumentParser();p.add_argument('--episodes-per-map',type=int,default=10)
    p.add_argument('--output',type=Path,default=Path('artifacts/stage2/expert_test.json'))
    args=p.parse_args();cfg=Stage2Config();rows=[];started=time.perf_counter()
    for map_index,map_spec in enumerate(SPLITS['test']['maps']):
        base_seed=SPLITS['test']['seed_start']+100*map_index
        sim=make_data_env(map_spec,base_seed,cfg.traffic_density,cfg.horizon)
        try:
            for seed in range(base_seed+50,base_seed+50+args.episodes_per_map):
                sim.reset(seed=seed);expert=NavigationExpert(sim,cfg.expert,cfg.dt_s);speeds=[];laterals=[]
                for step in range(cfg.horizon):
                    action,detail=expert.act();_,_,tm,tc,info=sim.step(action)
                    speeds.append(sim.agent.speed);laterals.append(detail['lateral_m'])
                    if tm or tc:break
                row=dict(map_spec=map_spec,seed=seed,steps=step+1,mean_speed_m_s=float(np.mean(speeds)),
                         max_abs_lateral_m=float(np.max(np.abs(laterals))),
                         **{k:info[k] for k in ['arrive_dest','crash','out_of_road','max_step','route_completion']})
                rows.append(row);print(json.dumps(row),flush=True)
        finally:sim.close()
    report=dict(config=asdict(cfg),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        episode_count=len(rows),episodes=rows,elapsed_seconds=time.perf_counter()-started,
        rates={k:sum(bool(r[k]) for r in rows)/len(rows) for k in ['arrive_dest','crash','out_of_road','max_step']},
        mean_speed_m_s=float(np.mean([r['mean_speed_m_s'] for r in rows])))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
