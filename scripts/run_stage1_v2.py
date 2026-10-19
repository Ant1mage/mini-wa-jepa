"""Run the GPU experiment in a child and measure device memory after it exits."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from evaluation.memory import memory_snapshot


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('artifacts/restart_stage1/egl_100'))
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--record',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    before=memory_snapshot()
    command=[sys.executable,'-m','scripts.verify_stage1_v2','--output',str(args.output),'--episodes',str(args.episodes)]
    if args.record:command.append('--record')
    with (args.output/'raw.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    time.sleep(0.5)
    report=dict(command=command,exit_code=result.returncode,memory_before_child=before,
                memory_after_child_exit=memory_snapshot())
    (args.output/'supervisor.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    sys.exit(result.returncode)


if __name__=='__main__':main()
