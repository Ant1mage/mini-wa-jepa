"""Linux/DRM measurements; no vendor CLI or PyTorch allocator side effects."""
from pathlib import Path
import re

import psutil


def gpu_devices():
    result=[]
    for device in Path('/sys/class/drm').glob('card[0-9]*/device'):
        total=device/'mem_info_vram_total'
        if not total.exists():continue
        result.append(dict(path=str(device),pci=device.resolve().name,total_bytes=int(total.read_text())))
    return sorted(result,key=lambda d:d['total_bytes'],reverse=True)


def memory_snapshot():
    clients={}
    for entry in Path('/proc/self/fdinfo').iterdir():
        try:text=entry.read_text()
        except (FileNotFoundError,PermissionError):continue
        fields=dict(line.split(':',1) for line in text.splitlines() if ':' in line)
        if 'drm-client-id' not in fields:continue
        pci=fields.get('drm-pdev','').strip()
        client=fields['drm-client-id'].strip()
        match=re.search(r'(\d+)\s*(KiB|MiB|GiB|bytes)?',fields.get('drm-memory-vram',''))
        if match:
            unit={'KiB':1024,'MiB':1024**2,'GiB':1024**3,'bytes':1,None:1}[match.group(2)]
            clients[pci,client]=int(match.group(1))*unit
    devices=[]
    for device in gpu_devices():
        try:used=int((Path(device['path'])/'mem_info_vram_used').read_text())
        except OSError:used=None
        devices.append(dict(**device,device_used_bytes=used,
            process_vram_bytes=sum(value for (pci,_),value in clients.items() if pci==device['pci']) if clients else None))
    return dict(rss_bytes=psutil.Process().memory_info().rss,devices=devices,
                drm_client_count=len(clients))
