"""Chapter 2: Single-frame Behavior Cloning ResNet-18 Trajectory Training."""
import argparse
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chapters.chapter2_baseline.src.config import Stage2Config
from chapters.chapter2_baseline.src.dataset import BaselineDataset
from chapters.chapter2_baseline.src.model import ResNet18Trajectory, masked_trajectory_loss


def make_loader(dataset, config, shuffle=False):
    kwargs = {
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "drop_last": shuffle,
        "num_workers": config.workers,
        "pin_memory": True,
    }
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    count = 0

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        bev = batch["bev"].to(device, non_blocking=True)
        ego = batch["ego_state"].to(device, non_blocking=True)
        traj = batch["trajectory"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        mask = batch["action_mask"].to(device, non_blocking=True)

        pred = model(bev, ego)
        loss = masked_trajectory_loss(pred, traj, valid, mask)
        total_loss += float(loss.item())
        count += 1

    return total_loss / max(count, 1)


def train(config=None):
    if config is None:
        config = Stage2Config()

    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    print(f"Training on device: {device}")

    model = ResNet18Trajectory(
        pooling_size=config.pooling_size,
        mlp_hidden=config.mlp_hidden,
        latent_dim=config.latent_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    try:
        train_ds = BaselineDataset(config.dataset_root, split="train")
        val_ds = BaselineDataset(config.dataset_root, split="val")
        train_loader = make_loader(train_ds, config, shuffle=True)
        val_loader = make_loader(val_ds, config, shuffle=False)
        print(f"Loaded {len(train_ds)} train samples, {len(val_ds)} val samples.")
    except Exception as e:
        print(f"Skipping dataset training: {e}")
        return model

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        count = 0
        for i, batch in enumerate(train_loader):
            bev = batch["bev"].to(device, non_blocking=True)
            ego = batch["ego_state"].to(device, non_blocking=True)
            traj = batch["trajectory"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            mask = batch["action_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(bev, ego)
            loss = masked_trajectory_loss(pred, traj, valid, mask)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            count += 1

        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{config.epochs}: Train Loss = {epoch_loss/max(count,1):.4f}, Val Loss = {val_loss:.4f}")

    return model


if __name__ == "__main__":
    train()
