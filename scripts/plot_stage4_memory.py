"""Plot measured memory with decimal process GB and allocator GiB kept separate."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('artifacts/restart_stage4');path=root/'memory_trends.png'
    if path.exists():raise FileExistsError(path)
    samples=[json.loads(line) for line in (root/'training_memory_samples.jsonl').read_text().splitlines()]
    start=samples[0]['time_unix_s'];fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name,action in [('joint',False),('action_only',True)]:
        rows=[row for row in samples if row['action_only']==action]
        x=[(row['time_unix_s']-start)/60 for row in rows]
        axes[0].plot(x,[row['rss_bytes']/1e9 for row in rows],label=name)
        axes[1].plot(x,[row['process_vram_bytes']/1e9 if row['process_vram_bytes'] is not None else float('nan') for row in rows],label=name)
        report=json.loads((root/('action_only_001' if action else 'joint_001')/'report.json').read_text())
        epochs=report['epochs'];epoch=[row['epoch'] for row in epochs]
        axes[2].plot(epoch,[row['peak_allocated_gib'] for row in epochs],label=name+' allocated')
        axes[2].plot(epoch,[row['peak_reserved_gib'] for row in epochs],linestyle='--',label=name+' reserved')
    axes[0].set_ylabel('Process RSS (decimal GB)');axes[1].set_ylabel('Process DRM VRAM (decimal GB)')
    axes[1].axhline(14,color='red',linestyle=':',label='GOAL strict 14GB')
    for ax in axes[:2]:ax.set_xlabel('Minutes since external sampling began')
    axes[2].set_xlabel('Epoch');axes[2].set_ylabel('PyTorch peak (GiB)')
    for ax in axes:ax.legend(fontsize=7)
    fig.suptitle('Actual measurements; external sampling begins during joint epoch 2. GB and GiB are separate units.')
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


if __name__=='__main__':main()
