"""串行推进阶段5：等待正式开环、执行原生smoke，再完成21组闭环。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

from configs.stage5 import Stage5Config, CHECKPOINTS
from mini_wa_jepa.provenance import provenance, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('artifacts/restart_stage5'))
    args=parser.parse_args();cfg=Stage5Config();root=args.root;root.mkdir(parents=True,exist_ok=True)
    status=dict(**provenance(cfg),stage=5,completed=False,jobs=[],phase='waiting_for_open_loop')
    started=time.time()
    try:
        while True:
            path=root/'open_loop/report.json'
            if path.exists():
                opened=json.loads(path.read_text())
                if opened.get('error'):raise RuntimeError('Open-loop failed; inspect original traceback')
                if opened.get('completed'):
                    if opened.get('smoke_samples') or opened['anchors']!=9600 or len(opened['models'])!=3:
                        raise ValueError('Formal matched open-loop results required')
                    break
            write_json(root/'suite_progress.json',status);time.sleep(5)
        jobs=[('joint','occlusion_l3',True),('resnet','dropout_l3',True)]
        jobs += [(model,case,False) for case in cfg.closed_conditions for model in CHECKPOINTS]
        for index,(model,case,smoke) in enumerate(jobs):
            parent=root/('smoke_closed' if smoke else 'closed_loop')
            stem=f'{model}__{case}';folder=parent/stem
            if (folder/'report.json').exists():
                old=json.loads((folder/'report.json').read_text())
                if old.get('completed'):
                    status['jobs'].append(dict(model=model,condition=case,smoke=smoke,output=str(folder),completed=True,reused=True));continue
                attempt=2
                while folder.exists():
                    folder=parent/f'{stem}_attempt_{attempt:03d}';attempt+=1
            parent.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-u','-m','scripts.evaluate_stress_closed_loop','--model',model,
                     '--condition',case,'--output',str(folder),'--record']
            if smoke:command+=['--max-steps','40']
            status.update(phase='native_smoke' if smoke else 'formal_closed_loop',current=dict(model=model,condition=case,output=str(folder)),job_index=index)
            write_json(root/'suite_progress.json',status)
            tick=time.time()
            with (parent/f'{folder.name}.log').open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            job=dict(model=model,condition=case,smoke=smoke,output=str(folder),returncode=result.returncode,
                     completed=result.returncode==0,seconds=time.time()-tick)
            status['jobs'].append(job);write_json(root/'suite_progress.json',status)
            print(json.dumps(job),flush=True)
            if result.returncode:raise RuntimeError(f'Closed-loop job failed: {folder}; original log preserved')
            report=json.loads((folder/'report.json').read_text())
            if not report['completed'] or (not smoke and len(report['episodes'])!=6):
                raise ValueError('Incomplete closed-loop cohort')
        status.update(completed=True,phase='evaluations_completed',elapsed_seconds=time.time()-started)
        write_json(root/'suite_progress.json',status)
        subprocess.run([sys.executable,'-m','scripts.report_stage5','--root',str(root)],check=True)
        status.update(phase='report_completed');write_json(root/'suite_progress.json',status)
    except BaseException:
        import traceback
        status.update(error=traceback.format_exc(),elapsed_seconds=time.time()-started)
        write_json(root/'suite_progress.json',status);raise


if __name__=='__main__':main()
