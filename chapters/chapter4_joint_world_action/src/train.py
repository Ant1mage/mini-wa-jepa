"""Chapter 4: Joint World and Action Model Training.

Supports:
1. Joint Model training: Jointly optimizing Latent World foresight and Action planning.
2. Action-Only ablation: Setting latent_weight=0 to isolate the pure Behavior Cloning baseline.
3. Differential learning rates for fine-tuning pre-trained backbone vs action head.
4. Integration with vendor-optimized SDPA operators.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_wa_jepa.operators import get_sdpa_context
from chapters.chapter4_joint_world_action.src.config import Stage4Config
from chapters.chapter4_joint_world_action.src.model import JointModel
from chapters.chapter4_joint_world_action.src.loss import joint_loss
from chapters.chapter4_joint_world_action.src.dataset import JointSequencesDataset


def make_loader(dataset, config, shuffle=False):
    kwargs = {
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "drop_last": shuffle,
        "num_workers": config.workers,
        "pin_memory": True,
        "generator": torch.Generator().manual_seed(config.seed),
    }
    if config.workers > 0:
        kwargs.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def evaluate(model, loader, config, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    terms_accum = {"action": 0.0, "prediction": 0.0, "variance": 0.0, "covariance": 0.0, "latent": 0.0}
    ade_accum = 0.0
    fde_accum = 0.0
    count = 0
    valid_count = 0

    sdpa_ctx = get_sdpa_context()

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        bev = batch["bev"].to(device, non_blocking=True)
        ego = batch["ego"].to(device, non_blocking=True)
        future = batch["future"].to(device, non_blocking=True)
        world_valid = batch["world_valid"].to(device, non_blocking=True)
        trajectory = batch["trajectory"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        action_mask = batch["action_mask"].to(device, non_blocking=True)

        batch_device = {
            "trajectory": trajectory,
            "valid": valid,
            "action_mask": action_mask,
            "world_valid": world_valid,
        }

        with sdpa_ctx:
            output = model(bev, ego, future if config.latent_weight > 0 else None)
            loss, terms = joint_loss(output, batch_device, config)

        b = bev.shape[0]
        total_loss += loss.item() * b
        for k in terms_accum:
            terms_accum[k] += terms[k].item() * b
        count += b

        # Compute physical ADE / FDE
        pred_phys = model.unnormalize_trajectory(output["trajectory"])
        targ_phys = model.unnormalize_trajectory(trajectory)
        pos_err = (pred_phys[:, :, :2] - targ_phys[:, :, :2]).norm(dim=-1)  # [B, 10]

        valid_mask = valid & action_mask.unsqueeze(1)
        ade_accum += (pos_err * valid_mask).sum().item()
        # FDE at waypoint 9 (index 9)
        fde_valid = valid_mask[:, 9]
        if fde_valid.any():
            fde_accum += pos_err[:, 9][fde_valid].sum().item()
        valid_count += valid_mask.sum().item()

    avg_loss = total_loss / max(count, 1)
    avg_terms = {k: v / max(count, 1) for k, v in terms_accum.items()}
    avg_ade = ade_accum / max(valid_count, 1)
    avg_fde = fde_accum / max(count, 1)

    return avg_loss, avg_terms, avg_ade, avg_fde


def build_optimizer(model: JointModel, config: Stage4Config) -> torch.optim.Optimizer:
    """Build AdamW optimizer with layer-specific differential learning rates."""
    param_groups = [
        {"params": model.encoder.parameters(), "lr": config.encoder_learning_rate},
        {"params": model.predictor.parameters(), "lr": config.predictor_learning_rate},
        {"params": model.history_fusion.parameters(), "lr": config.learning_rate},
        {"params": model.action_projection.parameters(), "lr": config.learning_rate},
        {"params": model.action_head.parameters(), "lr": config.learning_rate},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=config.weight_decay, betas=(0.9, 0.95))


def train(data_dir: Path, output_dir: Path, config: Stage4Config, max_batches: int | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"=== Chapter 4: Joint World & Action Model Training ===")
    print(f"Mode: {'Joint (World + Action)' if config.latent_weight > 0 else 'Action-Only Ablation'}")
    print(f"Device: {device} | Output: {output_dir}")

    # Datasets
    train_dataset = JointSequencesDataset(data_dir, split="train", seed=config.seed)
    val_dataset = JointSequencesDataset(data_dir, split="val", sample_count=5000, seed=config.seed)

    train_loader = make_loader(train_dataset, config, shuffle=True)
    val_loader = make_loader(val_dataset, config, shuffle=False)

    # Initialize model
    model = JointModel(config, normalizer=train_dataset.normalizer).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {param_count:,} ({param_count / 1e6:.2f}M)")

    # Warm start from stage 3 if available
    if config.stage3_checkpoint and Path(config.stage3_checkpoint).exists():
        model.load_stage3_checkpoint(config.stage3_checkpoint)

    optimizer = build_optimizer(model, config)
    sdpa_ctx = get_sdpa_context()
    best_val_ade = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        n_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            if max_batches and batch_idx >= max_batches:
                break

            bev = batch["bev"].to(device, non_blocking=True)
            ego = batch["ego"].to(device, non_blocking=True)
            future = batch["future"].to(device, non_blocking=True)
            world_valid = batch["world_valid"].to(device, non_blocking=True)
            trajectory = batch["trajectory"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            action_mask = batch["action_mask"].to(device, non_blocking=True)

            batch_device = {
                "trajectory": trajectory,
                "valid": valid,
                "action_mask": action_mask,
                "world_valid": world_valid,
            }

            optimizer.zero_grad(set_to_none=True)

            with sdpa_ctx:
                output = model(bev, ego, future if config.latent_weight > 0 else None)
                loss, terms = joint_loss(output, batch_device, config)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            b = bev.shape[0]
            running_loss += loss.item() * b
            n_samples += b

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch}/{config.epochs}] Step [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} (Act: {terms['action']:.4f}, Lat: {terms['latent']:.4f})")

        train_loss = running_loss / max(n_samples, 1)
        val_loss, val_terms, val_ade, val_fde = evaluate(model, val_loader, config, device, max_batches=max_batches)
        elapsed = time.time() - epoch_start

        print(f"--- Epoch {epoch} Finished ({elapsed:.1f}s) ---")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Action Loss: {val_terms['action']:.4f} | Val ADE: {val_ade:.3f}m | Val FDE: {val_fde:.3f}m")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "val_loss": val_loss,
            "val_ade": val_ade,
        }, output_dir / "latest.pt")

        if val_ade < best_val_ade:
            best_val_ade = val_ade
            torch.save(model.state_dict(), output_dir / "best.pt")
            print(f"[*] New best validation ADE: {best_val_ade:.3f}m (saved best.pt)")

    print(f"Training completed. Checkpoints saved to {output_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("outputs/data/stage2_v2_100k"))
    p.add_argument("--output", type=Path, default=Path("artifacts/chapter4_run"))
    p.add_argument("--action-only", action="store_true", help="Run action-only ablation (latent_weight=0)")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--max-batches", type=int, help="Limit batches for fast testing")
    args = p.parse_args()

    cfg = Stage4Config()
    if args.action_only:
        cfg.latent_weight = 0.0
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    train(args.data, args.output, cfg, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
