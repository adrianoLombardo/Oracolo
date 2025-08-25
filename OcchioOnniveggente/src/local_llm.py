from __future__ import annotations

"""Placeholder helpers for local LLM inference.

These functions are intentionally minimal and serve as stubs for future
implementations.  They expose a ``device`` parameter so that callers can
select the preferred compute backend (``auto``, ``cpu`` or ``cuda``).
"""

from typing import Any, Dict, Literal


def llm_local(
    prompt: str,
    *,
    device: Literal["auto", "cpu", "cuda"] = "auto",
    **_: Dict[str, Any],
) -> str:
    """Generate a response using a local LLM.

    Current implementation is a stub and should be replaced with an actual
    model invocation.  The ``device`` parameter is included to keep the API
    compatible with future backends.
    """

    raise NotImplementedError("Local LLM not implemented")
