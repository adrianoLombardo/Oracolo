from __future__ import annotations

"""Minimal local LLM wrapper used as a fallback backend.

This module lazily loads models via ``transformers`` to avoid importing heavy
dependencies when the local backend is not used. It provides a single
:func:`generate` function which accepts a list of OpenAI-style messages and
returns the generated text.
"""

from typing import Dict, List, Tuple, Literal

# simple in-memory cache so that the model is loaded only once
_MODEL_CACHE: dict[Tuple[str, str, str, bool], Tuple[object, object]] = {}


def _load_model(
    model_path: str,
    device: str,
    precision: Literal["fp32", "fp16", "bf16", "int4"],
    use_onnx: bool = False,
) -> Tuple[object, object]:
    """Load the model/tokenizer pair for the given path, device and precision."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception as exc:  # pragma: no cover - import error handled at runtime
        raise RuntimeError(
            "transformers and torch are required for the local LLM backend"
        ) from exc

    key = (model_path, device, precision, use_onnx)
    if key not in _MODEL_CACHE:
        if use_onnx and device == "cpu":
            try:
                import onnxruntime as ort
                from pathlib import Path
            except Exception as exc:  # pragma: no cover - runtime dep
                raise RuntimeError(
                    "onnxruntime is required for ONNX models"
                ) from exc
            # ``model_path`` can be either a direct ONNX file or a directory
            onnx_path = Path(model_path)
            if onnx_path.is_dir():
                onnx_path = onnx_path / "model.onnx"
            tokenizer = AutoTokenizer.from_pretrained(onnx_path.parent.as_posix())
            session = ort.InferenceSession(onnx_path.as_posix())
            _MODEL_CACHE[key] = (tokenizer, session)
        else:
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
    use_onnx: bool = False,
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
    tokenizer, model = _load_model(model_path, device, precision, use_onnx)

    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    if use_onnx and device == "cpu":
        import numpy as np

        inputs = tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        attention = inputs["attention_mask"]
        for _ in range(max_new_tokens):
            outputs = model.run(None, {"input_ids": input_ids, "attention_mask": attention})
            logits = outputs[0]
            next_id = int(logits[:, -1, :].argmax(axis=-1))
            input_ids = np.concatenate([input_ids, [[next_id]]], axis=1)
            attention = np.concatenate([attention, [[1]]], axis=1)
            if next_id == tokenizer.eos_token_id:
                break
        text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

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

