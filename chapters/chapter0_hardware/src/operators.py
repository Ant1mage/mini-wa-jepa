"""Chapter 0: Operator selection, SDPA backend routing, and fallback management."""
from contextlib import contextmanager
import logging
from typing import Any, Dict, Generator, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

logger = logging.getLogger(__name__)

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
    normalized = backend.upper().strip()
    if normalized == "MATH":
        return [SDPBackend.MATH]

    if normalized in BACKEND_NAME_MAP:
        return [BACKEND_NAME_MAP[normalized], SDPBackend.MATH]

    if normalized == "AUTO":
        if vendor == "NVIDIA":
            return [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.MATH,
            ]
        elif vendor == "AMD":
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
    backends = resolve_sdpa_backends(backend, vendor=vendor)

    if len(backends) == 1 and backends[0] == SDPBackend.MATH:
        with sdpa_kernel(SDPBackend.MATH):
            yield backends
        return

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


def configure_operator_optimizations(
    vendor: str = "AMD",
    allow_tf32: bool = True,
    cudnn_benchmark: bool = False,
    deterministic: bool = True,
) -> Dict[str, Any]:
    applied = {}
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        applied["cuda_matmul_allow_tf32"] = allow_tf32
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = allow_tf32
        applied["cudnn_allow_tf32"] = allow_tf32
    torch.backends.cudnn.benchmark = cudnn_benchmark
    applied["cudnn_benchmark"] = cudnn_benchmark
    torch.backends.cudnn.deterministic = deterministic
    applied["cudnn_deterministic"] = deterministic
    return applied


get_sdpa_context = sdpa_context


def verify_attention_fidelity(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    candidate_backend: str = "AUTO",
    vendor: str = "AMD",
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> Dict[str, Any]:
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
