"""Train a new stage 4 joint or action-only fine-tuning run."""
import argparse
from dataclasses import replace
import faulthandler
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from configs.stage4 import Stage4Config
from training.joint import train


def main():
    faulthandler.enable();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('outputs/data/stage2_v2_100k'))
    p.add_argument('--output',type=Path,default=Path('artifacts/restart_stage4/joint_001'))
    p.add_argument('--action-only',action='store_true')
    p.add_argument('--epochs',type=int);p.add_argument('--batch-size',type=int);p.add_argument('--validation-pairs',type=int)
    p.add_argument('--max-batches',type=int,help='Smoke only; never formal acceptance')
    p.add_argument('--sdpa-backend',type=str,choices=['MATH','AUTO','FLASH','EFFICIENT','CUDNN'],help='SDPA attention backend')
    p.add_argument('--max-vram-gb',type=float,help='VRAM ceiling in GiB; default is dynamic based on hardware')
    p.add_argument('--auto-calibrate',action='store_true',help='Display hardware profile and parameter recommendations')
    p.add_argument('--enable-oom-recovery',action='store_true',help='Enable finite-retry OOM recovery with emergency checkpoints')
    p.add_argument('--gradient-accumulation-steps',type=int,help='Gradient accumulation steps')
    args=p.parse_args()
    param_keys=['epochs','batch_size','validation_pairs','sdpa_backend','max_vram_gb','auto_calibrate','enable_oom_recovery','gradient_accumulation_steps']
    changes={k:getattr(args,k) for k in param_keys if getattr(args,k) is not None}
    if args.action_only:changes['latent_weight']=0.
    train(args.data,args.output,replace(Stage4Config(),**changes),args.max_batches)


if __name__=='__main__':main()
