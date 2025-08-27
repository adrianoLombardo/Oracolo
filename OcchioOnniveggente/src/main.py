"""Entry point for the OcchioOnniveggente application."""

from src.app.startup import startup
from src.app.audio_setup import audio_setup
from src.app.run import run


def main() -> None:
    """Initialize and run the application."""
    start = startup()
    audio = audio_setup(start.settings, start.debug, start.args)
    run(start, audio)


if __name__ == "__main__":
    main()

