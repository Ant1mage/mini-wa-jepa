"""Simulator-independent BEV contract; no route or ground-truth state in inputs."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BEVFrame:
    raster: np.ndarray
    metadata: dict

    def __post_init__(self):
        channels=self.metadata.get('channels',[])
        if self.raster.dtype != np.uint8 or self.raster.ndim!=3 or self.raster.shape[0]!=len(channels):
            raise ValueError("BEV must be uint8 CHW with an explicit matching channel schema")
        if min(self.raster.shape)<=0:
            raise ValueError("BEV dimensions must be positive")


class BEVProvider(ABC):
    @abstractmethod
    def reset(self):
        """Reset provider state after an environment/scene reset."""

    @abstractmethod
    def capture(self) -> BEVFrame:
        """Return a raster and generic geometry/channel metadata."""
