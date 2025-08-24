import logging
from typing import Any, Dict

import sounddevice as sd

logger = logging.getLogger(__name__)


def validate_device_config(cfg: Dict[str, Any]) -> None:
    """Validate audio device configuration.

    Non-`None` values are checked against the list of available devices. When
    both devices are ``None`` the function returns early, allowing
    ``sounddevice`` to use the system defaults.

    Raises:
        ValueError: if configuration is invalid.
    """

    input_dev = cfg.get("input_device")
    output_dev = cfg.get("output_device")
    if input_dev is None and output_dev is None:
        logger.info(
            "Input e output device non configurati: verranno usati quelli di default"
        )
        return

    try:
        devices = sd.query_devices()
    except Exception as exc:
        logger.error("Unable to query audio devices: %s", exc)
        raise ValueError(f"Impossibile leggere i device audio: {exc}") from exc

    errors: list[str] = []
    for key, kind in (("input_device", "input"), ("output_device", "output")):
        dev = cfg.get(key)
        if dev is None:
            logger.info("%s device non configurato: uso del device predefinito", kind)
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
