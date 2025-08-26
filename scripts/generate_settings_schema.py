"""Generate JSON schema for the configuration models."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "OcchioOnniveggente"))
from src.config import Settings


def main() -> None:
    schema = Settings.model_json_schema()
    out = Path(__file__).resolve().parents[1] / "docs" / "settings.schema.json"
    out.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
