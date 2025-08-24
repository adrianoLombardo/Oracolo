"""Utility functions for audio device selection and debugging."""

from __future__ import annotations

from typing import Any

import sounddevice as sd


def pick_device(spec: Any, kind: str) -> Any:
    """Return the index of the audio device matching ``spec``.

    ``spec`` can be ``None``/empty string, an integer index or a substring of
    the device name. If ``spec`` is ``None`` the system default is used when
    valid. The ``kind`` parameter determines whether to search for an input or
    output device.
    """
    devices = sd.query_devices()

    def _valid(info: dict) -> bool:
        ch_key = "max_input_channels" if kind == "input" else "max_output_channels"
        return info.get(ch_key, 0) > 0

    if spec in (None, ""):
        try:
            idx = sd.default.device[0 if kind == "input" else 1]
            if idx is not None and _valid(sd.query_devices(idx)):
                return None  # use system default
        except Exception:
            pass
    else:
        if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
            idx = int(spec)
            if 0 <= idx < len(devices) and _valid(devices[idx]):
                return idx
        spec_str = str(spec).lower()
        for i, info in enumerate(devices):
            if spec_str in info.get("name", "").lower() and _valid(info):
                return i

    for i, info in enumerate(devices):
        if _valid(info):
            return i
    return None


def debug_print_devices() -> None:
    """Print a formatted table with the available audio devices."""
    try:
        devices = sd.query_devices()
    except Exception as e:  # pragma: no cover - execution environment dependent
        print(f"⚠️ Unable to query audio devices: {e}")
        return
    header = f"{'Idx':>3}  {'Device Name':<40}  {'In/Out'}"
    print(header)
    print("-" * len(header))
    for idx, info in enumerate(devices):
        name = info.get("name", "")
        in_ch = info.get("max_input_channels", 0)
        out_ch = info.get("max_output_channels", 0)
        print(f"{idx:>3}  {name:<40}  {in_ch}/{out_ch}")
