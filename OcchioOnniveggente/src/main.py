from src.app.startup import startup
from src.app.audio_setup import audio_setup
from src.app.run import run



import asyncio
import numpy as np
import sounddevice as sd
import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import ValidationError
import logging

from src.config import Settings, get_openai_api_key
from src.filters import ProfanityFilter
from src.audio.processing import AudioPreprocessor
from src.audio.recording import record_until_silence, play_and_pulse
from src.lights import SacnLight, WledLight, color_from_text
from src.oracle import (
    transcribe,
    fast_transcribe,
    oracle_answer,
    synthesize,
    append_log,
    extract_summary,
    detect_language,
)
from src.domain import validate_question
from src.audio.hotword import is_wake, matches_hotword_text, strip_hotword_prefix
from src.chat import ChatState
from src.conversation import ConversationManager, DialogState
from src.logging_utils import setup_logging
from src.conversation import update_language
from src.cli import _ensure_utf8_stdout, say, oracle_greeting, default_response
from src.audio.audio_device import pick_device, debug_print_devices
from src.profile_utils import get_active_profile, make_domain_settings

logger = logging.getLogger(__name__)



# --------------------------- main ------------------------------------- #

def main() -> None:
    start = startup()
    audio = audio_setup(start.settings, start.debug, start.args)
    run(start, audio)


if __name__ == "__main__":
    main()
