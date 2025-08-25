from __future__ import annotations

"""Minimal local LLM wrapper used as a fallback backend.

This module lazily loads models via ``transformers`` to avoid importing heavy
dependencies when the local backend is not used. It provides a single
:func:`generate` function which accepts a list of OpenAI-style messages and
returns the generated text.
"""

from typing import Dict, List, Tuple, Literal, Iterator

# simple in-memory cache so that the model is loaded only once
_MODEL_CACHE: dict[Tuple[str, str, str], Tuple[object, object]] = {}


def _load_model(
    model_path: str,
    device: str,
    precision: Literal["fp32", "fp16", "bf16", "int4"],
) -> Tuple[object, object]:
    """Load the model/tokenizer pair for the given path, device and precision."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception as exc:  # pragma: no cover - import error handled at runtime
        raise RuntimeError(
            "transformers and torch are required for the local LLM backend"
        ) from exc

    key = (model_path, device, precision)
    if key not in _MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if precision == "int4":
            try:
                AutoModelArgs = {
                    "load_in_4bit": True,
                    "device_map": "auto" if device != "cpu" else {"": "cpu"},
                }
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, **AutoModelArgs
                )
            except Exception as exc:  # pragma: no cover - bitsandbytes runtime
                raise RuntimeError(
                    "bitsandbytes is required for int4 precision"
                ) from exc
        else:
            dtype = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }[precision]
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype
            )
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
    precision: Literal["fp32", "fp16", "bf16", "int4"] = "fp32",
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
    tokenizer, model = _load_model(model_path, device, precision)

    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Strip the prompt from the generated sequence if it is echoed back
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()


def stream_generate(
    messages: List[Dict[str, str]],
    *,
    model_path: str,
    device: str = "cpu",
    max_new_tokens: int = 256,
    precision: Literal["fp32", "fp16", "bf16", "int4"] = "fp32",
) -> Iterator[str]:
    """Yield generated tokens incrementally using :class:`TextIteratorStreamer`.

    This helper mirrors :func:`generate` but returns an iterator that yields
    partial text as soon as it is produced by the model.  It is designed for
    interactive applications where the caller wants to update the UI token by
    token.
    """

    from threading import Thread
    from transformers import TextIteratorStreamer

    tokenizer, model = _load_model(model_path, device, precision)

    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = dict(
        **inputs, max_new_tokens=max_new_tokens, streamer=streamer
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    try:
        for text in streamer:
            yield text
    finally:
        thread.join()

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

