"""Train the stage 2 baseline; no world-model or later-stage code."""
import argparse
from dataclasses import replace
from pathlib import Path
from configs.stage2 import BaselineConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('outputs/data/stage2_v2_100k'))
    p.add_argument('--output',type=Path,default=Path('artifacts/restart_stage2/baseline'))
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--max-batches',type=int,help='Smoke only; never report as full training')
    args=p.parse_args()
    from training.baseline import train
    train(args.data,args.output,replace(BaselineConfig(),epochs=args.epochs,batch_size=args.batch_size,workers=args.workers),args.max_batches)


if __name__=='__main__':main()
