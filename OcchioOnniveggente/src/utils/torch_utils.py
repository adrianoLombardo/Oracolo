from __future__ import annotations

"""Helpers for working with ``torch`` without a hard dependency."""

try:  # pragma: no cover - torch may be unavailable
    import torch  # type: ignore
except Exception:  # pragma: no cover - fallback stub
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

__all__ = ["torch"]
