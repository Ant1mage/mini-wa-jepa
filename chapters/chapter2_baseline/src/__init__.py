"""Chapter 2 Source Package: Single-frame ResNet-18 behavior cloning baseline."""
from chapters.chapter2_baseline.src.config import Stage2Config
from chapters.chapter2_baseline.src.model import ResNet18Trajectory, masked_trajectory_loss
from chapters.chapter2_baseline.src.dataset import BaselineDataset
from chapters.chapter2_baseline.src.train import train, evaluate

__all__ = [
    "Stage2Config",
    "ResNet18Trajectory",
    "masked_trajectory_loss",
    "BaselineDataset",
    "train",
    "evaluate",
]
