"""Read-only process/DRM memory sampling while the two formal runs execute."""
import json
from pathlib import Path
import re
import time

import psutil


def main():
    output=Path('artifacts/restart_stage4/training_memory_samples.jsonl')
    with output.open('x') as stream:
        missed=0
        while True:
            found=False
            for process in psutil.process_iter(['pid','cmdline','memory_info']):
                command=process.info['cmdline'] or []
                if 'scripts.train_joint' not in command:continue
                found=True;clients={}
                try:
                    for path in Path(f'/proc/{process.pid}/fdinfo').iterdir():
                        try:fields=dict(line.split(':',1) for line in path.read_text().splitlines() if ':' in line)
                        except (FileNotFoundError,PermissionError):continue
                        if 'drm-client-id' not in fields:continue
                        match=re.search(r'(\d+)\s*(KiB|MiB|GiB|bytes)?',fields.get('drm-memory-vram',''))
                        if match:
                            unit={'KiB':1024,'MiB':1024**2,'GiB':1024**3,'bytes':1,None:1}[match.group(2)]
                            clients[(fields.get('drm-pdev','').strip(),fields['drm-client-id'].strip())]=int(match.group(1))*unit
                    row=dict(time_unix_s=time.time(),pid=process.pid,action_only='--action-only' in command,
                        rss_bytes=process.info['memory_info'].rss,process_vram_bytes=sum(clients.values()) if clients else None)
                    stream.write(json.dumps(row)+'\n');stream.flush()
                except (FileNotFoundError,psutil.NoSuchProcess):continue
            if not found:
                missed+=1
                # The shell transitions between the two sequential training processes.
                reports=[Path('artifacts/restart_stage4')/name/'report.json' for name in ('joint_001','action_only_001')]
                if all(path.exists() and (json.loads(path.read_text()).get('completed') or json.loads(path.read_text()).get('error')) for path in reports):break
                if missed>=3:break
            else:missed=0
            time.sleep(5)


if __name__=='__main__':main()
