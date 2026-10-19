"""Chapter 0: Automatic training parameter calibration and hardware-adaptive recommendation."""
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from chapters.chapter0_hardware.src.hardware import HardwareProfile, detect_hardware

logger = logging.getLogger(__name__)


@dataclass
class CalibrationRecommendation:
    batch_size: int = 128
    mixed_precision: str = "bf16"
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    workers: int = 2
    cpu_threads: int = 4
    vram_target_budget_gib: float = 17.0
    vram_headroom_gib: float = 3.0
    vicreg_batch_note: str = ""
    reasoning: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            "================ Training Parameter Recommendations ================",
            f"Batch Size: {self.batch_size} (Grad Accum Steps: {self.gradient_accumulation_steps})",
            f"Effective Batch Size: {self.batch_size * self.gradient_accumulation_steps}",
            f"Mixed Precision: {self.mixed_precision.upper()}",
            f"Gradient Checkpointing: {'Enabled' if self.gradient_checkpointing else 'Disabled'}",
            f"DataLoader Workers: {self.workers} | CPU Threads: {self.cpu_threads}",
            f"VRAM Budget: {self.vram_target_budget_gib:.2f} GiB (Safety Headroom: {self.vram_headroom_gib:.2f} GiB)",
            f"VICReg Statistic Note: {self.vicreg_batch_note}",
            "===================================================================",
        ]
        return "\n".join(lines)


def recommend_training_parameters(
    hardware: Optional[HardwareProfile] = None,
    current_config: Optional[Any] = None,
    safety_headroom_fraction: float = 0.15,
) -> CalibrationRecommendation:
    if hardware is None:
        hardware = detect_hardware()

    total_vram = hardware.gpu.total_memory_gib
    headroom = total_vram * safety_headroom_fraction
    target_budget = max(0.0, total_vram - headroom)

    if hardware.runtime.bf16_supported:
        rec_precision = "bf16"
    elif hardware.runtime.fp16_supported:
        rec_precision = "fp16"
    else:
        rec_precision = "fp32"

    vicreg_note = ""
    grad_checkpointing = False

    if total_vram >= 16.0:
        rec_batch = 128
        rec_accum = 1
        vicreg_note = "Batch size 128 provides optimal VICReg covariance sample statistics without gradient accumulation."
    elif total_vram >= 10.0:
        rec_batch = 64
        rec_accum = 2
        vicreg_note = "Batch size 64 used with 2-step gradient accumulation (effective batch 128). " \
                      "VICReg covariance is computed across 64 samples per step (empirically stable)."
    elif total_vram >= 6.0:
        rec_batch = 32
        rec_accum = 4
        vicreg_note = "CAUTION: Batch size 32 used with 4-step gradient accumulation (effective batch 128). " \
                      "Lower per-step sample count may reduce VICReg covariance matrix rank stability; monitor rank."
    else:
        rec_batch = 16
        rec_accum = 8
        grad_checkpointing = True
        vicreg_note = "WARNING: Constrained VRAM. Batch size 16 with checkpointing. " \
                      "Severe risk of VICReg representation collapse due to small mini-batch covariance estimation."

    phys_cores = hardware.cpu.physical_cores
    ram_gib = hardware.cpu.available_ram_gib

    if ram_gib >= 16.0 and phys_cores >= 8:
        rec_workers = 2
    elif ram_gib >= 8.0 and phys_cores >= 4:
        rec_workers = 2
    else:
        rec_workers = 0

    rec_threads = min(max(2, phys_cores // 2), 6)

    reasoning = {
        "gpu_vendor": hardware.gpu.vendor,
        "gpu_device": hardware.gpu.device_name,
        "total_vram_gib": total_vram,
        "safety_headroom_gib": headroom,
        "physical_cores": phys_cores,
        "system_ram_gib": hardware.cpu.total_ram_gib,
    }

    return CalibrationRecommendation(
        batch_size=rec_batch,
        mixed_precision=rec_precision,
        gradient_accumulation_steps=rec_accum,
        gradient_checkpointing=grad_checkpointing,
        workers=rec_workers,
        cpu_threads=rec_threads,
        vram_target_budget_gib=target_budget,
        vram_headroom_gib=headroom,
        vicreg_batch_note=vicreg_note,
        reasoning=reasoning,
    )


if __name__ == "__main__":
    profile = detect_hardware()
    print(profile.summary())
    rec = recommend_training_parameters(profile)
    print(rec.summary())
