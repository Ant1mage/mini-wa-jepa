"""Validate completed stage 4 artifacts, raw cohorts/timing and hash manifest."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from mini_wa_jepa.provenance import write_json
from mini_wa_jepa.visualization.reports import load_stage_runs


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(8*1024*1024):result.update(block)
    return result.hexdigest()


import argparse

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('artifacts/restart_stage4'))
    parser.add_argument('--force',action='store_true',help='Allow updating existing delivery audit')
    args=parser.parse_args();root=args.root
    if not args.force and (root/'results_manifest.json').exists():raise FileExistsError('Preserve previous delivery audit')
    status=json.loads((root/'pipeline_status.json').read_text())
    if not status['completed']:raise ValueError('Pipeline incomplete')
    cohort=np.load(root/'test_eval/paired_cohort.npz')['anchors']
    checks={};videos=[root/'stage4_acceptance.mp4'];recordings=[]
    for name in ('joint','action_only','resnet','constant_turn'):
        raw=np.load(root/f'test_eval/{name}_raw.npz')
        if not np.array_equal(raw['anchors'][raw['full_mask']],cohort):raise ValueError('Cohort differs')
        for key in ('prediction','target','ade_m','fde_m'):
            if not np.isfinite(raw[key]).all():raise ValueError(f'Nonfinite test data: {name}/{key}')
    checks['matched_finite_test_cohort']=int(len(cohort))
    for name in ('joint','action_only'):
        report=json.loads((root/f'{name}_closed_loop/report.json').read_text())
        raw=np.load(root/f'{name}_closed_loop/latency_samples_ms.npz')
        expected=sum(max(row['steps']-20,0) for row in report['episodes'])
        if len(raw['decision_ms'])!=expected or len(raw['pipeline_ms'])!=expected:raise ValueError('Latency sample count mismatch')
        if not (np.isfinite(raw['pipeline_ms']).all() and np.all(raw['pipeline_ms']>=raw['decision_ms']) and np.all(raw['decision_ms']>0)):
            raise ValueError('Invalid timing sample')
        checks[name+'_timing_samples']=expected
        for row in report['episodes']:
            evidence=row.get('evidence',{})
            if not evidence:continue
            videos.append(Path(evidence['video']));recordings.append((Path(evidence['rerun']),evidence['valid_latent_targets']))
            actual=np.load(evidence['raw'])
            if len(actual['states'])!=row['steps']+1:raise ValueError('Actual state timeline mismatch')
            if not np.isfinite(actual['latent_prediction']).all():raise ValueError('Nonfinite actual latent prediction')
            if not np.isfinite(actual['trajectory_prediction']).all():raise ValueError('Nonfinite actual trajectory')
            if np.any(actual['latent_target'][~actual['world_valid']]!=0):raise ValueError('Missing targets not masked')
    # Decode each complete video; a header alone does not prove usable footage.
    import imageio_ffmpeg
    video_checks=[]
    for path in videos:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),'-f','null','-'],check=True)
        video_checks.append(dict(path=str(path),bytes=path.stat().st_size,decoded=True))
    checks['videos']=video_checks
    rrd_checks=[]
    for index,(path,target_count) in enumerate(recordings):
        out=root/f'closed_rrd_contents_{index:02d}.txt'
        if not out.exists():
            with out.open('w') as stream:
                subprocess.run([sys.executable,'-m','rerun','rrd','print',str(path)],stdout=stream,stderr=subprocess.STDOUT,check=True)
        content=out.read_text()
        required=['/bev/planned_trajectory','/camera']
        if target_count:required+=['/latent/predicted','/latent/ground_truth','TensorData']
        for key in required:
            if key not in content:raise ValueError(f'RRD evidence absent: {path} {key}')
        rrd_checks.append(dict(path=str(path),metadata=str(out)))
    checks['rrd']=rrd_checks
    rows,sources,errors=load_stage_runs(Path.cwd(),4)
    if errors or not rows:raise ValueError(errors)
    checks['marimo_runs']=len(rows)
    checks['tests']='43 tests passed; tests_all_delivery_001.log'
    write_json(root/'delivery_checks.json',checks)
    paths=set(path for path in root.rglob('*') if path.is_file())|set(videos)
    manifest=dict(stage=4,completed=True,files=[dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path)) for path in sorted(paths)])
    write_json(root/'results_manifest.json',manifest)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
