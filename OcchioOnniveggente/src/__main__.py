"""Entry point dell'applicazione."""

import argparse

from . import main as offline
from . import realtime_oracolo as realtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Occhio Onniveggente")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Esegui la versione realtime via WebSocket",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.realtime:
        realtime.main()
    else:
        offline.main()


if __name__ == "__main__":
    main()

