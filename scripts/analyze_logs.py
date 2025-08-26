#!/usr/bin/env python3
"""Utility script to analyze interaction logs.

The script expects a path to a JSON Lines file generated via
``log_interaction`` and prints simple statistics such as the most common
questions and follow-up prompts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze interaction logs")
    parser.add_argument("logfile", type=Path, help="Path to JSONL log file")
    args = parser.parse_args()

    questions: Counter[str] = Counter()
    followups: Counter[str] = Counter()

    if not args.logfile.exists():
        raise SystemExit(f"Log file {args.logfile} does not exist")

    for line in args.logfile.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if q := entry.get("question"):
            questions[q] += 1
        if f := entry.get("follow_up"):
            followups[f] += 1

    print("Top questions:")
    for q, count in questions.most_common(5):
        print(f"{count:3} {q}")

    print("\nTop follow-ups:")
    for f, count in followups.most_common(5):
        print(f"{count:3} {f}")


if __name__ == "__main__":  # pragma: no cover - convenience script
    main()
