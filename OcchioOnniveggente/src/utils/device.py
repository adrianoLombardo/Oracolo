from __future__ import annotations

import os
from typing import Literal

from ..metrics import read_system_metrics

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch may be missing
    class _CudaStub:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def get_device_properties(_index: int):
            class _Props:
                total_memory = 0

            return _Props()

    class _TorchStub:
        cuda = _CudaStub()

    torch = _TorchStub()  # type: ignore

_MIN_CUDA_GB = int(os.getenv("ORACOLO_MIN_CUDA_GB", "4"))
_GPU_UTIL_THRESHOLD = float(os.getenv("ORACOLO_GPU_UTIL_THRESHOLD", "90"))


def resolve_device(preferred: Literal["auto", "cpu", "cuda"]) -> str:
    """Return the compute device to use.

    The decision can be forced via the ``ORACOLO_DEVICE`` environment
    variable. When ``preferred`` (or ``ORACOLO_DEVICE``) is ``"auto"``, a GPU
    is selected only if available *and* it has at least ``_MIN_CUDA_GB`` of
    VRAM (default 4 GB).
    """

    pref = os.getenv("ORACOLO_DEVICE", preferred)
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            util = read_system_metrics().get("gpu_util", 0.0)
            if util < _GPU_UTIL_THRESHOLD and total >= _MIN_CUDA_GB * 1024 ** 3:
                return "cuda"
        except Exception:  # pragma: no cover - very defensive
            return "cuda"
    return "cpu"
