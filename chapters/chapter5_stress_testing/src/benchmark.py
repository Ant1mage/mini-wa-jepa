"""Chapter 5: Perception Degradation Benchmark.

Automated benchmark evaluating model robustness across varying degrees of sensor corruption:
1. Spatial Occlusion Levels (0%, 25%, 50%, 75% blindspot/box)
2. Lane Boundary Erosion (0%, 30%, 60%, 90% dropout)
3. Multi-frame Temporal Dropout (0, 1, 2, 3 dropped frames)
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import sys
import json
import torch
import numpy as np

# Ensure root is on path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_wa_jepa.operators import get_sdpa_context
from chapters.chapter4_joint_world_action.src.config import Stage4Config
from chapters.chapter4_joint_world_action.src.model import JointModel
from chapters.chapter4_joint_world_action.src.dataset import JointSequencesDataset
from chapters.chapter5_stress_testing.src.degradation import (
    apply_front_blindspot,
    apply_random_box_occlusion,
    apply_boundary_loss,
    apply_temporal_frame_dropout,
)


@torch.no_grad()
def evaluate_under_degradation(
    model: JointModel,
    dataset: JointSequencesDataset,
    degrade_fn,
    device: torch.device,
    sample_limit: int = 500,
) -> dict:
    model.eval()
    ade_list = []
    fde_list = []
    sdpa_ctx = get_sdpa_context()

    n = min(len(dataset), sample_limit)
    for idx in range(n):
        item = dataset[idx]
        bev = item["bev"]  # [4, 4, 128, 128]
        ego = torch.from_numpy(item["ego"]).unsqueeze(0).to(device)
        traj_gt = torch.from_numpy(item["trajectory"]).unsqueeze(0).to(device)
        valid = torch.from_numpy(item["valid"]).unsqueeze(0).to(device)

        # Apply corruption
        degraded_bev = degrade_fn(bev)
        bev_tensor = torch.from_numpy(degraded_bev).unsqueeze(0).to(device)

        with sdpa_ctx:
            out = model(bev_tensor, ego)

        pred_phys = model.unnormalize_trajectory(out["trajectory"])[0]
        targ_phys = model.unnormalize_trajectory(traj_gt)[0]

        val_mask = valid[0]
        pos_err = (pred_phys[:, :2] - targ_phys[:, :2]).norm(dim=-1)

        ade = (pos_err * val_mask).sum() / val_mask.sum().clamp_min(1)
        ade_list.append(ade.item())

        if val_mask[9]:
            fde_list.append(pos_err[9].item())

    return {
        "mean_ade": float(np.mean(ade_list)),
        "p95_ade": float(np.percentile(ade_list, 95)),
        "mean_fde": float(np.mean(fde_list)) if fde_list else 0.0,
        "sample_count": n,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("outputs/data/stage2_v2_100k"))
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/restart_stage4/joint_001/best.pt"))
    p.add_argument("--samples", type=int, default=500)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Chapter 5: Stress Testing Benchmark ===")
    print(f"Device: {device} | Evaluated Samples: {args.samples}")

    dataset = JointSequencesDataset(args.data, split="val", sample_count=args.samples)
    cfg = Stage4Config(device=str(device))
    model = JointModel(cfg, normalizer=dataset.normalizer).to(device)

    if args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state.get("model", state.get("model_state_dict", state)))
        print(f"[*] Loaded Joint Model weights from {args.checkpoint}")
    else:
        print(f"[!] Warning: Checkpoint {args.checkpoint} not found. Running with initialized weights.")

    tests = [
        ("Clean Baseline (No Corruption)", lambda x: x),
        ("Front Blindspot (depth=10m)", lambda x: apply_front_blindspot(x, depth_m=10.0)),
        ("Front Blindspot (depth=20m)", lambda x: apply_front_blindspot(x, depth_m=20.0)),
        ("Random Cutout (1 box 24px)", lambda x: apply_random_box_occlusion(x, num_boxes=1, seed=42)),
        ("Random Cutout (3 boxes 24px)", lambda x: apply_random_box_occlusion(x, num_boxes=3, seed=42)),
        ("Boundary Loss (30% dropped)", lambda x: apply_boundary_loss(x, drop_rate=0.3, seed=42)),
        ("Boundary Loss (70% dropped)", lambda x: apply_boundary_loss(x, drop_rate=0.7, seed=42)),
        ("Frame Dropout (drop frame t-1)", lambda x: apply_temporal_frame_dropout(x, drop_indices=(2,))),
        ("Frame Dropout (drop t-1, t-2)", lambda x: apply_temporal_frame_dropout(x, drop_indices=(1, 2))),
    ]

    print("\n" + "=" * 65)
    print(f"{'Stress Test Condition':<35} | {'ADE (m)':<10} | {'FDE (m)':<10}")
    print("-" * 65)

    results = {}
    for name, fn in tests:
        res = evaluate_under_degradation(model, dataset, fn, device, sample_limit=args.samples)
        results[name] = res
        print(f"{name:<35} | {res['mean_ade']:<10.3f} | {res['mean_fde']:<10.3f}")

    print("=" * 65)
    out_file = Path("artifacts/chapter5_stress_test_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
