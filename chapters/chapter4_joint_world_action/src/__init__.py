"""Chapter 4 Source Package: Joint World and Action Model with Kinematic Prior."""
from chapters.chapter4_joint_world_action.src.config import Stage4Config
from chapters.chapter4_joint_world_action.src.model import (
    JointModel,
    constant_turn_trajectory,
)
from chapters.chapter4_joint_world_action.src.loss import joint_loss
from chapters.chapter4_joint_world_action.src.dataset import JointSequencesDataset
from chapters.chapter4_joint_world_action.src.train import train, evaluate

__all__ = [
    "Stage4Config",
    "JointModel",
    "constant_turn_trajectory",
    "joint_loss",
    "JointSequencesDataset",
    "train",
    "evaluate",
]
