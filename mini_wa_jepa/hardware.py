"""Hardware auto-detection and environment capability profiling.

Provides structured profiling of:
- GPU vendor, device model, total and available VRAM.
- CPU topology (physical/logical cores) and system RAM.
- PyTorch execution backend (CUDA / ROCm / MPS / CPU).
- Precision capabilities (FP32, BF16, FP16, TF32).
- Scaled Dot-Product Attention (SDPA) backends and vendor-optimized operator support.
"""
from dataclasses import asdict, dataclass, field
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional

import psutil
import torch


@dataclass
class GPUInfo:
    available: bool = False
    vendor: str = "None"  # "AMD", "NVIDIA", "Apple", "Intel", "CPU"
    device_name: str = "None"
    device_index: int = 0
    total_memory_bytes: int = 0
    total_memory_gib: float = 0.0
    free_memory_bytes: int = 0
    free_memory_gib: float = 0.0
    multi_processor_count: int = 0
    compute_capability: str = "None"
    gcn_arch_name: str = "None"  # ROCm specific (e.g. gfx1100)


@dataclass
class CPUInfo:
    processor: str = "Unknown"
    architecture: str = "Unknown"
    physical_cores: int = 1
    logical_cores: int = 1
    total_ram_bytes: int = 0
    total_ram_gib: float = 0.0
    available_ram_bytes: int = 0
    available_ram_gib: float = 0.0


@dataclass
class RuntimeInfo:
    python_version: str = ""
    torch_version: str = ""
    backend: str = "cpu"  # "rocm", "cuda", "mps", "cpu"
    rocm_version: Optional[str] = None
    cuda_version: Optional[str] = None
    cudnn_version: Optional[int] = None
    miopen_version: Optional[str] = None
    tf32_supported: bool = False
    bf16_supported: bool = False
    fp16_supported: bool = False


@dataclass
class AttentionCapabilities:
    math_sdpa: bool = True
    flash_sdpa: bool = False
    mem_efficient_sdpa: bool = False
    cudnn_sdpa: bool = False
    available_backends: List[str] = field(default_factory=list)
    preferred_backend: str = "MATH"


@dataclass
class HardwareProfile:
    gpu: GPUInfo = field(default_factory=GPUInfo)
    cpu: CPUInfo = field(default_factory=CPUInfo)
    runtime: RuntimeInfo = field(default_factory=RuntimeInfo)
    attention: AttentionCapabilities = field(default_factory=AttentionCapabilities)

    @property
    def gpu_total_vram_gib(self) -> float:
        return self.gpu.total_memory_gib

    @property
    def gpu_free_vram_gib(self) -> float:
        return self.gpu.free_memory_gib

    @property
    def cpu_physical_cores(self) -> int:
        return self.cpu.physical_cores

    @property
    def supports_bf16(self) -> bool:
        return self.runtime.bf16_supported

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            "================ Hardware Environment Profile ================",
            f"GPU: {self.gpu.device_name} (Vendor: {self.gpu.vendor})",
            f"VRAM: {self.gpu.total_memory_gib:.2f} GiB Total, {self.gpu.free_memory_gib:.2f} GiB Free",
            f"CPU: {self.cpu.processor} ({self.cpu.physical_cores} physical / {self.cpu.logical_cores} logical cores)",
            f"RAM: {self.cpu.total_ram_gib:.2f} GiB Total, {self.cpu.available_ram_gib:.2f} GiB Available",
            f"PyTorch: {self.runtime.torch_version} | Backend: {self.runtime.backend.upper()}",
            f"ROCm: {self.runtime.rocm_version} | CUDA: {self.runtime.cuda_version}",
            f"Precision: BF16={'YES' if self.runtime.bf16_supported else 'NO'}, "
            f"FP16={'YES' if self.runtime.fp16_supported else 'NO'}, "
            f"TF32={'YES' if self.runtime.tf32_supported else 'NO'}",
            f"SDPA Backends: {', '.join(self.attention.available_backends)} "
            f"(Preferred: {self.attention.preferred_backend})",
            "=============================================================",
        ]
        return "\n".join(lines)


def _detect_gpu(device_idx: int = 0) -> GPUInfo:
    info = GPUInfo(device_index=device_idx)
    if not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info.available = True
            info.vendor = "Apple"
            info.device_name = "Apple Silicon GPU (MPS)"
        else:
            info.available = False
            info.vendor = "CPU"
            info.device_name = "CPU Only"
        return info

    info.available = True
    info.device_name = torch.cuda.get_device_name(device_idx)
    props = torch.cuda.get_device_properties(device_idx)
    info.total_memory_bytes = props.total_memory
    info.total_memory_gib = props.total_memory / (1024 ** 3)
    info.multi_processor_count = getattr(props, "multi_processor_count", 0)

    # Determine free memory without allocating
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
        info.free_memory_bytes = free_bytes
        info.free_memory_gib = free_bytes / (1024 ** 3)
    except Exception:
        info.free_memory_bytes = info.total_memory_bytes
        info.free_memory_gib = info.total_memory_gib

    # Determine vendor and architecture
    if getattr(torch.version, "hip", None) is not None:
        info.vendor = "AMD"
        info.gcn_arch_name = getattr(props, "gcnArchName", "unknown")
    elif getattr(torch.version, "cuda", None) is not None:
        info.vendor = "NVIDIA"
        major = getattr(props, "major", 0)
        minor = getattr(props, "minor", 0)
        info.compute_capability = f"{major}.{minor}"
    else:
        info.vendor = "Generic"

    return info


def _detect_cpu() -> CPUInfo:
    mem = psutil.virtual_memory()
    physical_cores = psutil.cpu_count(logical=False) or 1
    logical_cores = psutil.cpu_count(logical=True) or 1
    
    # Try reading CPU model name from /proc/cpuinfo or platform
    processor = platform.processor() or "Unknown"
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        processor = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    return CPUInfo(
        processor=processor,
        architecture=platform.machine(),
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        total_ram_bytes=mem.total,
        total_ram_gib=mem.total / (1024 ** 3),
        available_ram_bytes=mem.available,
        available_ram_gib=mem.available / (1024 ** 3),
    )


def _detect_runtime(gpu: GPUInfo) -> RuntimeInfo:
    backend = "cpu"
    rocm_ver = getattr(torch.version, "hip", None)
    cuda_ver = getattr(torch.version, "cuda", None)
    if rocm_ver is not None:
        backend = "rocm"
    elif cuda_ver is not None:
        backend = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        backend = "mps"

    bf16_ok = False
    if gpu.available and backend in ("cuda", "rocm"):
        try:
            bf16_ok = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_ok = False

    tf32_ok = False
    if backend == "cuda" and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        tf32_ok = True

    return RuntimeInfo(
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        backend=backend,
        rocm_version=rocm_ver,
        cuda_version=cuda_ver,
        cudnn_version=torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available() else None,
        tf32_supported=tf32_ok,
        bf16_supported=bf16_ok,
        fp16_supported=gpu.available,
    )


def _detect_attention_capabilities(runtime: RuntimeInfo, gpu: GPUInfo) -> AttentionCapabilities:
    caps = AttentionCapabilities()
    backends = ["MATH"]

    if gpu.available and runtime.backend in ("cuda", "rocm"):
        # Check PyTorch native SDPA backend availability flags
        try:
            flash_enabled = torch.backends.cuda.flash_sdp_enabled()
        except Exception:
            flash_enabled = False
        try:
            mem_eff_enabled = torch.backends.cuda.mem_efficient_sdp_enabled()
        except Exception:
            mem_eff_enabled = False
        try:
            cudnn_enabled = torch.backends.cuda.cudnn_sdp_enabled()
        except Exception:
            cudnn_enabled = False

        caps.flash_sdpa = flash_enabled
        caps.mem_efficient_sdpa = mem_eff_enabled
        caps.cudnn_sdpa = cudnn_enabled

        if flash_enabled:
            backends.append("FLASH_ATTENTION")
        if mem_eff_enabled:
            backends.append("EFFICIENT_ATTENTION")
        if cudnn_enabled and runtime.backend == "cuda":
            backends.append("CUDNN_ATTENTION")

        # Preferred vendor backend:
        # NVIDIA Ampere/Ada/Hopper -> FLASH > EFFICIENT > MATH
        # AMD ROCm RDNA/CDNA -> FLASH > EFFICIENT > MATH
        if "FLASH_ATTENTION" in backends:
            caps.preferred_backend = "FLASH_ATTENTION"
        elif "EFFICIENT_ATTENTION" in backends:
            caps.preferred_backend = "EFFICIENT_ATTENTION"
        else:
            caps.preferred_backend = "MATH"
    else:
        caps.preferred_backend = "MATH"

    caps.available_backends = backends
    return caps


def detect_hardware(device_idx: int = 0) -> HardwareProfile:
    """Run non-intrusive static hardware detection without executing forward/backward passes."""
    gpu = _detect_gpu(device_idx)
    cpu = _detect_cpu()
    runtime = _detect_runtime(gpu)
    attention = _detect_attention_capabilities(runtime, gpu)
    return HardwareProfile(gpu=gpu, cpu=cpu, runtime=runtime, attention=attention)
