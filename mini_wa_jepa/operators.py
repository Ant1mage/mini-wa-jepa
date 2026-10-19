"""Vendor-optimized operator selection, SDPA backend routing, and fallback management.

Provides:
- Explicit Attention backend context management (`sdpa_context`).
- Support for MATH (baseline), FLASH_ATTENTION, EFFICIENT_ATTENTION, CUDNN_ATTENTION, and AUTO.
- Safe automatic fallback to PyTorch native MATH backend if vendor kernels are unavailable or fail.
- Precision and vendor matrix math configuration (TF32, cudnn benchmark).
- Verification utility for comparing output fidelity and numerical error across backends.
"""
from contextlib import contextmanager
import logging
from typing import Any, Dict, Generator, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

logger = logging.getLogger(__name__)

# Backend name aliases mapping to SDPBackend enum values
BACKEND_NAME_MAP = {
    "MATH": SDPBackend.MATH,
    "FLASH": SDPBackend.FLASH_ATTENTION,
    "FLASH_ATTENTION": SDPBackend.FLASH_ATTENTION,
    "EFFICIENT": SDPBackend.EFFICIENT_ATTENTION,
    "EFFICIENT_ATTENTION": SDPBackend.EFFICIENT_ATTENTION,
    "CUDNN": SDPBackend.CUDNN_ATTENTION,
    "CUDNN_ATTENTION": SDPBackend.CUDNN_ATTENTION,
}


def resolve_sdpa_backends(backend: str, vendor: str = "AMD") -> List[SDPBackend]:
    """Resolve a backend string identifier into an ordered list of SDPBackend candidates.
    
    Args:
        backend: One of 'MATH', 'FLASH', 'EFFICIENT', 'CUDNN', 'AUTO'.
        vendor: Detected GPU vendor ('AMD', 'NVIDIA', etc.).
    """
    normalized = backend.upper().strip()

    if normalized == "MATH":
        return [SDPBackend.MATH]

    if normalized in BACKEND_NAME_MAP:
        # User explicitly requested a single backend, but include MATH as last-resort fallback
        return [BACKEND_NAME_MAP[normalized], SDPBackend.MATH]

    if normalized == "AUTO":
        if vendor == "NVIDIA":
            # NVIDIA CUDA: FlashAttention -> MemoryEfficient -> cuDNN -> Math
            return [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.MATH,
            ]
        elif vendor == "AMD":
            # AMD ROCm: FlashAttention -> MemoryEfficient -> Math
            import os
            os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
            return [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        else:
            return [SDPBackend.MATH]

    logger.warning("Unrecognized SDPA backend '%s', falling back to MATH.", backend)
    return [SDPBackend.MATH]


@contextmanager
def sdpa_context(
    backend: str = "MATH",
    vendor: str = "AMD",
    allow_fallback: bool = True,
) -> Generator[List[SDPBackend], None, None]:
    """Context manager controlling PyTorch Scaled Dot-Product Attention backends.
    
    Guarantees:
    - If backend == 'MATH', activates ONLY SDPBackend.MATH (100% Stage 4 reproducible).
    - If backend != 'MATH', tries the requested or vendor-optimized backends in priority order.
    - If entering or using the optimized backend fails, catches the error and falls back to MATH.
    """
    backends = resolve_sdpa_backends(backend, vendor=vendor)

    # Fast path for Stage 4 regression baseline
    if len(backends) == 1 and backends[0] == SDPBackend.MATH:
        with sdpa_kernel(SDPBackend.MATH):
            yield backends
        return

    # Multi-backend / optimized path with fallback
    try:
        with sdpa_kernel(backends):
            yield backends
    except Exception as e:
        if allow_fallback and SDPBackend.MATH not in backends[:1]:
            logger.warning(
                "SDPA backend %s failed (%s). Falling back to SDPBackend.MATH.",
                [b.name for b in backends],
                str(e),
            )
            with sdpa_kernel(SDPBackend.MATH):
                yield [SDPBackend.MATH]
        else:
            raise


get_sdpa_context = sdpa_context


def configure_operator_optimizations(
    vendor: str = "AMD",
    allow_tf32: bool = True,
    cudnn_benchmark: bool = False,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Configure system-wide PyTorch operator execution flags.
    
    Preserves determinism while allowing vendor-specific matrix multiplication optimizations.
    """
    applied = {}

    # TF32 configuration (primarily NVIDIA Ampere+)
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        applied["cuda_matmul_allow_tf32"] = allow_tf32
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = allow_tf32
        applied["cudnn_allow_tf32"] = allow_tf32

    # Benchmark & determinism
    torch.backends.cudnn.benchmark = cudnn_benchmark
    applied["cudnn_benchmark"] = cudnn_benchmark

    torch.backends.cudnn.deterministic = deterministic
    applied["cudnn_deterministic"] = deterministic

    return applied


def verify_attention_fidelity(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    candidate_backend: str = "AUTO",
    vendor: str = "AMD",
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> Dict[str, Any]:
    """Compare attention output of candidate backend against the reference MATH backend.
    
    NOTE: Designed to be called during controlled verification stages, NOT on module import.
    """
    with sdpa_kernel(SDPBackend.MATH):
        ref_out = F.scaled_dot_product_attention(q, k, v)

    with sdpa_context(candidate_backend, vendor=vendor):
        cand_out = F.scaled_dot_product_attention(q, k, v)

    diff = (cand_out.float() - ref_out.float()).abs()
    max_abs_err = float(diff.max())
    rel_l2_err = float((cand_out.float() - ref_out.float()).norm() / max(float(ref_out.float().norm()), 1e-12))
    close = bool(torch.allclose(cand_out, ref_out, atol=atol, rtol=rtol))

    return {
        "candidate_backend": candidate_backend,
        "max_abs_error": max_abs_err,
        "relative_l2_error": rel_l2_err,
        "is_allclose": close,
        "shapes": list(q.shape),
        "dtype": str(q.dtype),
    }
