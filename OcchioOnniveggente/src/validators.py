import logging
from typing import Any, Dict

import sounddevice as sd

logger = logging.getLogger(__name__)

def validate_device_config(cfg: Dict[str, Any]) -> None:
    """Validate audio device configuration.

    Raises:
        ValueError: if configuration is invalid.
    """
    try:
        devices = sd.query_devices()
    except Exception as exc:
        logger.error("Unable to query audio devices: %s", exc)
        raise ValueError(f"Impossibile leggere i device audio: {exc}") from exc

    errors: list[str] = []
    for key, kind in (("input_device", "input"), ("output_device", "output")):
        dev = cfg.get(key)
        if dev is None:
            errors.append(f"{kind} device non configurato")
            continue
        if isinstance(dev, int):
            if dev < 0 or dev >= len(devices):
                errors.append(f"indice {kind} {dev} non valido")
        else:
            if not any(dev == d.get("name") for d in devices):
                errors.append(f"{kind} device '{dev}' non trovato")
    if errors:
        msg = "; ".join(errors)
        logger.error("Device config validation failed: %s", msg)
        raise ValueError(msg)
