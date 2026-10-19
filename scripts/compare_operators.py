"""Compare inference and training fidelity, throughput, and memory between SDPA backends.

Uses the frozen Stage 4 checkpoint to verify:
- Output numerical equivalence between MATH and AUTO backends.
- Inference latency, throughput, and peak memory.
- Backward pass / optimizer update dynamics under both backends.
- Saves structured comparison report without modifying Stage 4 artifacts.
"""
import argparse
import json
from pathlib import Path
import sys
import time

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch
import torch.nn.functional as F

from configs.stage4 import Stage4Config
from dataset.joint import JointSequences
from mini_wa_jepa.normalization import Normalizer
from mini_wa_jepa.operators import sdpa_context
from models.joint import JointModel
from training.joint import make_loader, to_device


def load_model(checkpoint_path: Path, device: torch.device):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = Stage4Config(**saved["model_config"])
    normalizer = Normalizer(**saved["normalization"])
    model = JointModel(config, normalizer).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, config, normalizer, saved


@torch.no_grad()
def benchmark_inference(model, loader, device, backend_name: str, max_batches: int = 25):
    """Benchmark inference latency, memory, and output predictions."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    predictions = []
    latents = []
    step_times = []
    total_samples = 0

    with torch.inference_mode(), sdpa_context(backend_name):
        # Warmup 2 batches
        for i, raw in enumerate(loader):
            if i >= 2:
                break
            batch = to_device(raw, device)
            _ = model(batch["bev"], batch["ego"])

        torch.cuda.synchronize()

        for i, raw in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            batch = to_device(raw, device)
            t0 = time.perf_counter()
            out = model(batch["bev"], batch["ego"])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            step_times.append(elapsed)
            predictions.append(out["trajectory"].float().cpu().numpy())
            latents.append(out["latent"].float().cpu().numpy())
            total_samples += len(raw["bev"])

    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    avg_step_ms = float(np.mean(step_times) * 1000)
    p50_step_ms = float(np.percentile(step_times, 50) * 1000)
    p95_step_ms = float(np.percentile(step_times, 95) * 1000)
    throughput_fps = float(total_samples / sum(step_times))

    return {
        "backend": backend_name,
        "peak_vram_gib": peak_vram,
        "avg_step_ms": avg_step_ms,
        "p50_step_ms": p50_step_ms,
        "p95_step_ms": p95_step_ms,
        "throughput_fps": throughput_fps,
        "total_samples": total_samples,
        "predictions": np.concatenate(predictions, axis=0),
        "latents": np.concatenate(latents, axis=0),
    }


def benchmark_training_step(model, loader, device, backend_name: str, steps: int = 10):
    """Benchmark training step (forward + backward + optimizer) latency and peak memory."""
    from training.joint_loss import joint_loss

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    config = model.config

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    step_times = []
    total_samples = 0

    with sdpa_context(backend_name):
        for i, raw in enumerate(loader):
            if i >= steps:
                break
            batch = to_device(raw, device)
            optimizer.zero_grad()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch["bev"], batch["ego"], batch["future"])
            loss, _ = joint_loss(out, batch, config)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            step_times.append(elapsed)
            total_samples += len(raw["bev"])

    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    avg_step_ms = float(np.mean(step_times) * 1000)
    throughput = float(total_samples / sum(step_times))

    model.eval()
    return {
        "backend": backend_name,
        "training_peak_vram_gib": peak_vram,
        "training_avg_step_ms": avg_step_ms,
        "training_throughput_fps": throughput,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/restart_stage4/joint_001/best.pt"))
    p.add_argument("--data", type=Path, default=Path("outputs/data/stage2_v2_100k"))
    p.add_argument("--output", type=Path, default=Path("artifacts/hardware_upgrade/operator_comparison.json"))
    p.add_argument("--max-batches", type=int, default=25, help="Number of batches for benchmark")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Stage 4 checkpoint from: {args.checkpoint}")
    model, config, normalizer, saved = load_model(args.checkpoint, device)
    test_data = JointSequences(args.data, "test")
    loader = make_loader(test_data, config, shuffle=False)

    print("\n--- 1. Evaluating MATH Backend (Stage 4 Baseline) ---")
    math_results = benchmark_inference(model, loader, device, "MATH", max_batches=args.max_batches)
    print(f"  Peak VRAM: {math_results['peak_vram_gib']:.2f} GiB | Avg Latency: {math_results['avg_step_ms']:.2f} ms | Throughput: {math_results['throughput_fps']:.1f} FPS")

    print("\n--- 2. Evaluating AUTO Backend (Vendor Optimized) ---")
    auto_results = benchmark_inference(model, loader, device, "AUTO", max_batches=args.max_batches)
    print(f"  Peak VRAM: {auto_results['peak_vram_gib']:.2f} GiB | Avg Latency: {auto_results['avg_step_ms']:.2f} ms | Throughput: {auto_results['throughput_fps']:.1f} FPS")

    print("\n--- 3. Numerical Error Analysis: MATH vs AUTO ---")
    traj_diff = np.abs(math_results["predictions"] - auto_results["predictions"])
    traj_max_abs = float(traj_diff.max())
    traj_mean_abs = float(traj_diff.mean())
    traj_rel_l2 = float(np.linalg.norm(traj_diff) / max(np.linalg.norm(math_results["predictions"]), 1e-12))

    latent_diff = np.abs(math_results["latents"] - auto_results["latents"])
    latent_max_abs = float(latent_diff.max())
    latent_mean_abs = float(latent_diff.mean())
    latent_rel_l2 = float(np.linalg.norm(latent_diff) / max(np.linalg.norm(math_results["latents"]), 1e-12))

    print(f"  Trajectory Max Absolute Error: {traj_max_abs:.2e} m")
    print(f"  Trajectory Mean Absolute Error: {traj_mean_abs:.2e} m")
    print(f"  Trajectory Relative L2 Error: {traj_rel_l2:.2e}")
    print(f"  Latent Max Absolute Error: {latent_max_abs:.2e}")
    print(f"  Latent Relative L2 Error: {latent_rel_l2:.2e}")

    # ADE / FDE on physical trajectory units
    phys_math = normalizer.inverse_trajectory(math_results["predictions"])
    phys_auto = normalizer.inverse_trajectory(auto_results["predictions"])
    phys_diff = np.abs(phys_math - phys_auto)
    phys_max_xy_err_m = float(phys_diff[..., :2].max())
    print(f"  Physical Waypoint Max (X,Y) Error: {phys_max_xy_err_m:.2e} meters")

    print("\n--- 4. Benchmarking Training Steps (Forward + Backward + Optimizer) ---")
    math_train = benchmark_training_step(model, loader, device, "MATH", steps=10)
    print(f"  MATH Train Step: {math_train['training_avg_step_ms']:.1f} ms | Peak VRAM: {math_train['training_peak_vram_gib']:.2f} GiB | {math_train['training_throughput_fps']:.1f} FPS")

    auto_train = benchmark_training_step(model, loader, device, "AUTO", steps=10)
    print(f"  AUTO Train Step: {auto_train['training_avg_step_ms']:.1f} ms | Peak VRAM: {auto_train['training_peak_vram_gib']:.2f} GiB | {auto_train['training_throughput_fps']:.1f} FPS")

    report = {
        "checkpoint": str(args.checkpoint),
        "evaluated_samples": int(math_results["total_samples"]),
        "inference": {
            "math": {
                "peak_vram_gib": math_results["peak_vram_gib"],
                "avg_step_ms": math_results["avg_step_ms"],
                "p50_step_ms": math_results["p50_step_ms"],
                "p95_step_ms": math_results["p95_step_ms"],
                "throughput_fps": math_results["throughput_fps"],
            },
            "auto": {
                "peak_vram_gib": auto_results["peak_vram_gib"],
                "avg_step_ms": auto_results["avg_step_ms"],
                "p50_step_ms": auto_results["p50_step_ms"],
                "p95_step_ms": auto_results["p95_step_ms"],
                "throughput_fps": auto_results["throughput_fps"],
            },
            "speedup_ratio": math_results["avg_step_ms"] / max(auto_results["avg_step_ms"], 1e-9),
        },
        "training": {
            "math": math_train,
            "auto": auto_train,
            "training_speedup_ratio": math_train["training_avg_step_ms"] / max(auto_train["training_avg_step_ms"], 1e-9),
        },
        "fidelity": {
            "trajectory_max_abs_error": traj_max_abs,
            "trajectory_mean_abs_error": traj_mean_abs,
            "trajectory_relative_l2_error": traj_rel_l2,
            "physical_max_xy_err_m": phys_max_xy_err_m,
            "latent_max_abs_error": latent_max_abs,
            "latent_relative_l2_error": latent_rel_l2,
            "allclose_fp32_tol": bool(traj_max_abs < 1e-4),
        },
    }

    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nComparison report saved to: {args.output}")


if __name__ == "__main__":
    main()
