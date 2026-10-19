"""Chapter 3: Self-Supervised Latent World Model Training with VICReg.

Standalone training script demonstrating:
1. ConvStem + 4-layer Transformer BEV Token Encoder
2. Action-conditioned OneStepPredictor with zero-initialized residual projection
3. VICReg self-supervised regularization (Variance + Covariance + Invariance)
4. Integration with vendor-optimized SDPA operators
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_wa_jepa.operators import get_sdpa_context
from chapters.chapter3_latent_vicreg.src.config import Stage3Config
from chapters.chapter3_latent_vicreg.src.model import LatentModel
from chapters.chapter3_latent_vicreg.src.loss import latent_loss
from chapters.chapter3_latent_vicreg.src.dataset import LatentPairsDataset


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
    terms_accum = {"prediction": 0.0, "variance": 0.0, "covariance": 0.0}
    count = 0

    sdpa_ctx = get_sdpa_context()

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        bev = batch["bev"].to(device, non_blocking=True)
        ego = batch["ego"].to(device, non_blocking=True)
        future = batch["future"].to(device, non_blocking=True)

        with sdpa_ctx:
            pred_latent, target_latent, cur_latent = model(bev, ego, future)
            loss, terms = latent_loss(pred_latent, target_latent, cur_latent, config)

        b = bev.shape[0]
        total_loss += loss.item() * b
        for k in terms_accum:
            terms_accum[k] += terms[k].item() * b
        count += b

    if count == 0:
        return 0.0, {k: 0.0 for k in terms_accum}

    return total_loss / count, {k: v / count for k, v in terms_accum.items()}


def train(data_dir: Path, output_dir: Path, config: Stage3Config, max_batches: int | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"=== Chapter 3: Latent World Model Training ===")
    print(f"Device: {device} | Output: {output_dir}")
    print(f"Config: {asdict(config)}")

    # Initialize model
    model = LatentModel(config).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {param_count:,} ({param_count / 1e6:.2f}M)")

    # Datasets
    train_dataset = LatentPairsDataset(data_dir, split="train", seed=config.seed)
    val_dataset = LatentPairsDataset(data_dir, split="val", sample_count=config.validation_pairs, seed=config.seed)

    train_loader = make_loader(train_dataset, config, shuffle=True)
    val_loader = make_loader(val_dataset, config, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    sdpa_ctx = get_sdpa_context()
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_terms = {"prediction": 0.0, "variance": 0.0, "covariance": 0.0}
        n_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            if max_batches and batch_idx >= max_batches:
                break
            bev = batch["bev"].to(device, non_blocking=True)
            ego = batch["ego"].to(device, non_blocking=True)
            future = batch["future"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with sdpa_ctx:
                pred_latent, target_latent, cur_latent = model(bev, ego, future)
                loss, terms = latent_loss(pred_latent, target_latent, cur_latent, config)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            b = bev.shape[0]
            running_loss += loss.item() * b
            for k in running_terms:
                running_terms[k] += terms[k].item() * b
            n_samples += b

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch}/{config.epochs}] Step [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} (Sim: {terms['prediction']:.4f}, "
                      f"Var: {terms['variance']:.4f}, Cov: {terms['covariance']:.4f})")

        train_loss = running_loss / max(n_samples, 1)
        val_loss, val_terms = evaluate(model, val_loader, config, device, max_batches=max_batches)
        elapsed = time.time() - epoch_start

        print(f"--- Epoch {epoch} Finished ({elapsed:.1f}s) ---")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Terms -> Sim: {val_terms['prediction']:.4f}, Var: {val_terms['variance']:.4f}, Cov: {val_terms['covariance']:.4f}")

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, output_dir / "latest.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best.pt")
            print(f"[*] New best validation loss: {best_val_loss:.4f} (saved best.pt)")

    print(f"Training completed. Checkpoints saved to {output_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("outputs/data/stage2_v2_100k"))
    p.add_argument("--output", type=Path, default=Path("artifacts/chapter3_run"))
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--validation-pairs", type=int)
    p.add_argument("--max-batches", type=int, help="Limit batches for fast testing")
    args = p.parse_args()

    cfg = Stage3Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.validation_pairs is not None:
        cfg.validation_pairs = args.validation_pairs

    train(args.data, args.output, cfg, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
