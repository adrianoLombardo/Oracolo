"""Utility per convertire modelli in formato ONNX.

Questo script supporta due tipi di modello:

* ``llm`` – modelli di linguaggio gestiti da ``transformers``;
* ``whisper`` – modelli di trascrizione ``openai/whisper``.

Esempi d'uso::

    # Esporta un modello transformers in ONNX
    python scripts/convert_to_onnx.py --model gpt2 --output gpt2.onnx --type llm

    # Esporta un modello Whisper base in ONNX
    python scripts/convert_to_onnx.py --model base --output whisper-base.onnx --type whisper
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _convert_transformer(model: str, output: Path) -> None:
    """Convert a ``transformers`` causal LM to ONNX."""

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.onnx import FeaturesManager, export

    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model)
    model_obj = AutoModelForCausalLM.from_pretrained(model)
    onnx_config = FeaturesManager.get_config("causal-lm", model)
    export(tokenizer, model_obj, onnx_config, output)


def _convert_whisper(model: str, output: Path) -> None:
    """Convert a Whisper model to ONNX."""

    import torch
    import whisper

    output.parent.mkdir(parents=True, exist_ok=True)
    model_obj = whisper.load_model(model)
    dummy = torch.randn(1, 80, 3000)
    torch.onnx.export(model_obj, dummy, output.as_posix(), opset_version=17)


def main() -> None:
    parser = argparse.ArgumentParser(description="Converti modelli in ONNX")
    parser.add_argument("--model", required=True, help="Nome o percorso del modello")
    parser.add_argument("--output", required=True, help="File ONNX di destinazione")
    parser.add_argument(
        "--type", choices=["llm", "whisper"], default="llm", help="Tipo di modello"
    )
    args = parser.parse_args()

    out = Path(args.output)
    if args.type == "llm":
        _convert_transformer(args.model, out)
    else:
        _convert_whisper(args.model, out)


if __name__ == "__main__":  # pragma: no cover - script utility
    main()

