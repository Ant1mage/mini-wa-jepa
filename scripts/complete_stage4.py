"""Continue the authorized stage 4 only, after the two sequential formal runs."""
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from mini_wa_jepa.provenance import write_json


def main():
    root=Path('artifacts/restart_stage4');status_path=root/'pipeline_status.json'
    if status_path.exists():raise FileExistsError(status_path)
    status=dict(stage=4,completed=False,current='waiting_for_both_formal_training_runs',steps=[])
    write_json(status_path,status)
    def run(name,arguments):
        status['current']=name;write_json(status_path,status);started=time.time()
        with (root/f'{name}.log').open('x') as stream:
            subprocess.run([sys.executable,*arguments],stdout=stream,stderr=subprocess.STDOUT,check=True)
        status['steps'].append(dict(name=name,elapsed_s=time.time()-started));write_json(status_path,status)
    try:
        while True:
            reports=[]
            for name in ('joint_001','action_only_001'):
                path=root/name/'report.json'
                if path.exists():
                    report=json.loads(path.read_text())
                    if report.get('error'):raise RuntimeError(f'{name} failed: {report["error"]}')
                    reports.append(report)
            if len(reports)==2 and all(r['completed'] for r in reports):break
            time.sleep(5)
        # Test is evaluated once for the frozen selected models, never by training.
        run('test_evaluation',['-m','scripts.evaluate_joint'])
        run('closed_loop_smoke',['-m','scripts.evaluate_joint_closed_loop','--split','val','--episodes-per-map','1',
            '--max-steps','50','--record','--output',str(root/'closed_loop_smoke_001')])
        for name,checkpoint in [('joint','joint_001'),('action_only','action_only_001')]:
            run(f'{name}_closed_loop',['-m','scripts.evaluate_joint_closed_loop','--checkpoint',str(root/checkpoint/'best.pt'),
                '--output',str(root/f'{name}_closed_loop'),'--record'])
        run('stage4_report',['-m','scripts.report_stage4'])
        run('dashboard_export',['-m','marimo','export','html','scripts/dashboard_marimo.py','-o',str(root/'dashboard.html')])
        status.update(completed=True,current='report_finished_await_user_stage5_confirmation');write_json(status_path,status)
    except BaseException:
        status['error']=traceback.format_exc();write_json(status_path,status);raise


if __name__=='__main__':main()
