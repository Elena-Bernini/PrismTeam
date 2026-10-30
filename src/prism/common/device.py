"""Device selection and platform-specific settings.

Supported targets:

- Windows + NVIDIA GPU (Pier)        -> ``cuda``
- macOS + Apple Silicon M1 (Elena)   -> ``mps``
- AImageLab-HPC cluster nodes        -> ``cuda``
- anything else                      -> ``cpu``

``configure_platform()`` must run before ``torch`` / ``faiss`` are imported so
that the environment variables below take effect.
"""

from __future__ import annotations

import os
import platform
from typing import Any, Optional


def configure_platform() -> None:
    """Set environment variables that avoid known platform pitfalls.

    - ``PYTORCH_ENABLE_MPS_FALLBACK=1``: ops not implemented on Apple MPS fall
      back to CPU instead of raising.
    - ``KMP_DUPLICATE_LIB_OK=TRUE`` (macOS only): torch and faiss each ship an
      OpenMP runtime; loading both otherwise aborts with "OMP: Error #15".
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if platform.system() == "Darwin":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def torch_available() -> bool:
    """Return ``True`` when PyTorch can be imported."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_device(requested: str = "auto") -> str:
    """Translate ``auto|cpu|cuda|mps`` into a concrete torch device string.

    ``auto`` prefers CUDA, then Apple MPS, then CPU. Requesting an unavailable
    accelerator raises a clear error instead of silently falling back.
    """
    configure_platform()
    requested = (requested or "auto").lower()
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if requested in {"cuda", "mps"}:
            raise RuntimeError(f"Device '{requested}' requested but PyTorch is not installed.") from None
        return "cpu"

    cuda_ok = torch.cuda.is_available()
    mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if requested == "auto":
        if cuda_ok:
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"
    if requested == "cuda":
        if not cuda_ok:
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False. "
                               "Install a CUDA build of PyTorch (see README).")
        return "cuda"
    if requested == "mps":
        if not mps_ok:
            raise RuntimeError("MPS requested but not available (needs macOS 12.3+ and Apple Silicon).")
        return "mps"
    raise ValueError(f"Unknown device '{requested}'")


def resolve_num_workers(requested: Any) -> int:
    """Return the DataLoader worker count.

    ``auto`` uses 0 workers on Windows and macOS (process spawning is slow and
    error-prone there for small per-item work) and 4 on Linux.
    """
    if requested is None or str(requested).lower() == "auto":
        return 0 if platform.system() in {"Windows", "Darwin"} else 4
    return max(0, int(requested))


def describe_device(device: str) -> str:
    """Return a human-readable description of the selected device."""
    if device == "cuda":
        try:
            import torch
            return f"cuda ({torch.cuda.get_device_name(0)})"
        except Exception:  # pragma: no cover - informative only
            return "cuda"
    if device == "mps":
        return "mps (Apple Silicon GPU)"
    return f"cpu ({platform.processor() or platform.machine()})"


def empty_cache(device: Optional[str]) -> None:
    """Release cached accelerator memory (no-op on CPU)."""
    try:
        import torch
    except ImportError:
        return
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
