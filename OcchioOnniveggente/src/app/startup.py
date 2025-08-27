from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import ValidationError
import logging

from src.config import Settings, get_openai_api_key
from src.cli import _ensure_utf8_stdout, say
from src.logging_utils import setup_logging
from src.profile_utils import get_active_profile

logger = logging.getLogger(__name__)


@dataclass
class StartupData:
    args: argparse.Namespace
    session_id: str
    listener: Any
    cfg_path: Path
    raw_settings: dict[str, Any]
    settings: Settings
    client: AsyncOpenAI
    session_profile_name: str
    session_persona_name: str
    debug: bool
    tone: str


def startup() -> StartupData:
    """Parse command line arguments and bootstrap the application."""
    _ensure_utf8_stdout()

    session_id = uuid.uuid4().hex

    parser = argparse.ArgumentParser(description="Occhio Onniveggente · Oracolo")
    parser.add_argument("--autostart", action="store_true", help="Avvia direttamente senza prompt input()")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Nasconde i log dalla console (vista conversazione pulita)",
    )
    parser.add_argument("--persona", type=str, default=None, help="Personalità iniziale")
    args = parser.parse_args()

    listener = setup_logging(
        Path("data/logs/oracolo.log"), session_id=session_id, console=not args.quiet
    )

    say("Occhio Onniveggente · Oracolo ✨", role="system")

    load_dotenv()

    cfg_path = Path("settings.yaml")
    raw_settings: dict[str, Any] = {}
    if cfg_path.exists():
        try:
            raw_settings = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            settings = Settings.model_validate(raw_settings)
        except (ValidationError, yaml.YAMLError) as e:
            logger.warning("⚠️ Configurazione non valida: %s", e)
            logger.warning("Uso impostazioni di default.")
            raw_settings = {}
            settings = Settings()
    else:
        logger.warning("⚠️ settings.yaml non trovato, uso impostazioni di default.")
        raw_settings = {}
        settings = Settings()

    api_key = get_openai_api_key(settings)
    if not api_key:
        logger.error(
            "❌ È necessaria una API key OpenAI.\n"
            "   Imposta la variabile d'ambiente OPENAI_API_KEY"
            " oppure aggiungi openai.api_key a settings.yaml."
        )
        raise SystemExit(1)

    client = AsyncOpenAI(api_key=api_key)

    session_profile_name, _ = get_active_profile(raw_settings)
    session_persona_name = (
        args.persona
        or raw_settings.get("persona", {}).get("current")
        or getattr(getattr(settings, "persona", None), "current", "standard")
    )

    debug = bool(settings.debug) and (not args.quiet)
    tone = getattr(settings.chat, "tone", "informal")

    return StartupData(
        args=args,
        session_id=session_id,
        listener=listener,
        cfg_path=cfg_path,
        raw_settings=raw_settings,
        settings=settings,
        client=client,
        session_profile_name=session_profile_name,
        session_persona_name=session_persona_name,
        debug=debug,
        tone=tone,
    )
