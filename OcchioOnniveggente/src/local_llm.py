from __future__ import annotations

"""Minimal local LLM wrapper used as a fallback backend.

This module lazily loads models via ``transformers`` to avoid importing heavy
dependencies when the local backend is not used. It provides a single
:func:`generate` function which accepts a list of OpenAI-style messages and
returns the generated text.
"""

from typing import Dict, List

from .service_container import container


def generate(
    messages: List[Dict[str, str]],
    *,
    model_path: str,
    device: str = "cpu",
    max_new_tokens: int = 256,
) -> str:
    """Generate a response using a local model.

    Parameters
    ----------
    messages:
        Conversation in OpenAI chat format ``[{"role": ..., "content": ...}, ...]``.
    model_path:
        Path or model identifier compatible with ``transformers``.
    device:
        Device where the model should run (e.g. ``"cpu"``, ``"cuda"``).
    max_new_tokens:
        Number of tokens to generate.
    """
    tokenizer, model = container.load_llm(model_path, device)

    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Strip the prompt from the generated sequence if it is echoed back
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()

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

