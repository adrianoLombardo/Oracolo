from __future__ import annotations

"""Minimal local LLM wrapper used as a fallback backend.

This module lazily loads models via ``transformers`` to avoid importing heavy
dependencies when the local backend is not used. It provides a single
:func:`generate` function which accepts a list of OpenAI-style messages and
returns the generated text.
"""

from typing import Dict, List, Tuple

# simple in-memory cache so that the model is loaded only once
_MODEL_CACHE: dict[Tuple[str, str], Tuple[object, object]] = {}


def _load_model(model_path: str, device: str) -> Tuple[object, object]:
    """Load the model/tokenizer pair for the given path and device."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception as exc:  # pragma: no cover - import error handled at runtime
        raise RuntimeError(
            "transformers and torch are required for the local LLM backend"
        ) from exc

    key = (model_path, device)
    if key not in _MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.to(device)
        model.eval()
        _MODEL_CACHE[key] = (tokenizer, model)
    return _MODEL_CACHE[key]


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
    tokenizer, model = _load_model(model_path, device)

    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Strip the prompt from the generated sequence if it is echoed back
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()
