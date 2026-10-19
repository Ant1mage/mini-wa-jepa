"""Chapter 3 Source Package: Latent World Model and VICReg regularization."""
from chapters.chapter3_latent_vicreg.src.config import Stage3Config
from chapters.chapter3_latent_vicreg.src.model import (
    BEVEncoder,
    LatentModel,
    OneStepPredictor,
    TransformerBlock,
)
from chapters.chapter3_latent_vicreg.src.loss import (
    covariance_loss,
    latent_loss,
    variance_loss,
)
from chapters.chapter3_latent_vicreg.src.dataset import LatentPairsDataset
from chapters.chapter3_latent_vicreg.src.train import train, evaluate

__all__ = [
    "Stage3Config",
    "BEVEncoder",
    "LatentModel",
    "OneStepPredictor",
    "TransformerBlock",
    "covariance_loss",
    "latent_loss",
    "variance_loss",
    "LatentPairsDataset",
    "train",
    "evaluate",
]
