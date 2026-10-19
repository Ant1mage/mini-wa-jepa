"""Passive Linux process/DRM trend recording; does not initialize a GPU runtime."""
import argparse
import json
from pathlib import Path
import time
import psutil
from evaluation.memory import gpu_devices


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stop-file',type=Path,required=True);a=p.parse_args()
    with a.output.open('a') as stream:
        while not a.stop_file.exists():
            processes=[]
            for process in psutil.process_iter(['pid','cmdline','memory_info']):
                try:
                    argv=process.info['cmdline'] or [];command=' '.join(argv)
                    if argv and 'python' in Path(argv[0]).name and any(s in command for s in
                        ('scripts.collect_stage7_risk','scripts.train_stage7_risk','scripts.evaluate_stage7',
                         'scripts.train_flow_matching','scripts.cache_stage7')):
                        processes.append(dict(pid=process.pid,command=command,
                            rss_bytes=process.info['memory_info'].rss,pss_bytes=process.memory_full_info().pss))
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            devices=[]
            for d in gpu_devices():
                try:devices.append(dict(**d,used_bytes=int((Path(d['path'])/'mem_info_vram_used').read_text())))
                except OSError:pass
            stream.write(json.dumps(dict(timestamp=time.time(),available_ram_bytes=psutil.virtual_memory().available,
                processes=processes,gpu=devices))+'\n');stream.flush();time.sleep(5)
