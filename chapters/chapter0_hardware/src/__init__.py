"""Chapter 0 Source Package: Hardware detection, SDPA operators, calibration, and OOM recovery."""
from chapters.chapter0_hardware.src.hardware import (
    AttentionCapabilities,
    CPUInfo,
    GPUInfo,
    HardwareProfile,
    RuntimeInfo,
    detect_hardware,
)
from chapters.chapter0_hardware.src.operators import (
    configure_operator_optimizations,
    get_sdpa_context,
    sdpa_context,
    verify_attention_fidelity,
)
from chapters.chapter0_hardware.src.calibration import (
    CalibrationRecommendation,
    recommend_training_parameters,
)
from chapters.chapter0_hardware.src.oom_recovery import OOMRecoveryManager

__all__ = [
    "detect_hardware",
    "HardwareProfile",
    "GPUInfo",
    "CPUInfo",
    "RuntimeInfo",
    "AttentionCapabilities",
    "configure_operator_optimizations",
    "get_sdpa_context",
    "sdpa_context",
    "verify_attention_fidelity",
    "CalibrationRecommendation",
    "recommend_training_parameters",
    "OOMRecoveryManager",
]
