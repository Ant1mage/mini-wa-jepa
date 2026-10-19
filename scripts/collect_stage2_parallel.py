"""Two isolated EGL workers, deterministic map quotas, and verified shard merge."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from configs.stage2 import Stage2Config,SPLITS
from mini_wa_jepa.provenance import provenance,write_json


def merge(parts,output,cfg):
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema_version=2,spec_status='v2',**provenance(cfg),completed=False,
        episodes=[],splits={},parts=[],collection_mode='two isolated EGL processes, fixed per-map quotas')
    episode_offset=0
    try:
        for split,spec in SPLITS.items():
            destination=output/split;destination.mkdir()
            entries=[];frame_offset=0
            for map_spec in spec['maps']:
                source=parts/f'{split}_{map_spec}'
                manifest=json.loads((source/'manifest.json').read_text())
                if manifest.get('schema_version')!=2 or not manifest['completed']:raise RuntimeError(f'Incomplete worker: {source}')
                report['parts'].append(dict(path=str(source),source_tree_sha256=manifest['source_tree_sha256'],
                    goal_sha256=manifest['goal_sha256'],git_commit=manifest['git_commit'],
                    provenance_chain=manifest.get('provenance_chain',[])))
                for episode in manifest['episodes']:
                    report['episodes'].append(dict(episode,episode_id=episode['episode_id']+episode_offset))
                index=json.loads((source/split/'index.json').read_text())
                for shard in index['shards']:
                    with np.load(source/split/shard['file'],allow_pickle=False) as f:
                        arrays={key:f[key] for key in f.files}
                    arrays['episode_id']=arrays['episode_id'].astype(np.int32)+episode_offset
                    name=f'shard_{len(entries):04d}.npz'
                    np.savez_compressed(destination/name,**arrays)
                    entries.append(dict(file=name,start=frame_offset,frames=shard['frames']));frame_offset+=shard['frames']
                episode_offset+=len(manifest['episodes'])
                for key in ['bev_metadata','trajectory_fields','future_times_s','state_fields','action_mask_rule','graphics']:
                    if key in manifest:report[key]=manifest[key]
            write_json(destination/'index.json',dict(frames=frame_offset,shards=entries))
            report['splits'][split]=dict(frames=frame_offset,**spec)
        expected=cfg.frames+cfg.validation_frames+cfg.test_frames
        report['stored_frames']=sum(e['stored_frames'] for e in report['episodes'])
        report['simulated_steps']=sum(e['simulated_steps'] for e in report['episodes'])
        if report['stored_frames']!=expected:raise RuntimeError((report['stored_frames'],expected))
        report['completed']=True
    finally:write_json(output/'manifest.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parts',type=Path,default=Path('outputs/data/stage2_v2_parts'))
    p.add_argument('--output',type=Path,default=Path('outputs/data/stage2_v2_100k'))
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();args.parts.mkdir(parents=True,exist_ok=True)
    if not 1<=args.workers<=2:p.error('At most two EGL workers fit the measured host RAM budget')
    cfg=Stage2Config();jobs=[]
    for (split,spec),budget in zip(SPLITS.items(),[cfg.frames,cfg.validation_frames,cfg.test_frames]):
        for i,map_spec in enumerate(spec['maps']):
            quota=budget//len(spec['maps'])+int(i<budget%len(spec['maps']))
            path=args.parts/f'{split}_{map_spec}'
            complete=(path/'manifest.json').exists() and json.loads((path/'manifest.json').read_text()).get('completed')
            if not complete:jobs.append((split,map_spec,quota,path))
    active=[];log_root=Path('artifacts/restart_stage2/collection_workers');log_root.mkdir(exist_ok=True)
    started=time.perf_counter()
    try:
        while jobs or active:
            while jobs and len(active)<args.workers:
                split,map_spec,quota,path=jobs.pop(0)
                command=[sys.executable,'-m','scripts.collect_data','--output',str(path),'--only-split',split,
                         '--only-map',map_spec,'--map-frames',str(quota)]
                if path.exists():command.append('--resume')
                log=(log_root/f'{split}_{map_spec}.log').open('a')
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                active.append((process,log,split,map_spec));print(f'Start {split}/{map_spec} quota={quota} pid={process.pid}',flush=True)
            for item in active[:]:
                process,log,split,map_spec=item
                if process.poll() is not None:
                    log.close();active.remove(item)
                    if process.returncode:raise RuntimeError(f'{split}/{map_spec} failed ({process.returncode}); preserve log and manifest')
                    print(f'Completed {split}/{map_spec}',flush=True)
            time.sleep(.5)
        merge(args.parts,args.output,cfg)
        write_json(log_root/'supervisor.json',dict(**provenance(cfg),elapsed_seconds=time.perf_counter()-started,
            completed=True,workers=args.workers))
    finally:
        for process,log,_,_ in active:
            if process.poll() is None:process.send_signal(__import__('signal').SIGINT)
        for process,log,_,_ in active:
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.terminate()
            log.close()


if __name__=='__main__':main()
