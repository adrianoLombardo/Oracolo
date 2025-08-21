from __future__ import annotations

"""Utilities for hotword detection.

This module exposes :func:`listen_for_wakeword` which blocks until the given
wake word is detected on the microphone.  The implementation attempts to use
`pvporcupine` (Picovoice Porcupine) together with ``pyaudio``.  When these
optional dependencies are missing, the function falls back to a no-op so that
missing packages do not crash the application.
"""

from typing import Optional
import struct


def listen_for_wakeword(wakeword: str, device_id: Optional[int] = None) -> None:
    """Block until *wakeword* is detected on the microphone.

    Parameters
    ----------
    wakeword:
        Name of the keyword model to detect.  This should match one of the
        keywords supported by the underlying engine (e.g. ``"picovoice"`` or
        ``"porcupine"``).
    device_id:
        Optional index of the input device to use.  ``None`` lets the backend
        choose the default system device.

    Notes
    -----
    If ``pvporcupine`` or ``pyaudio`` are not installed the function prints a
    warning and returns immediately, effectively disabling wake-word detection.
    """

    try:
        import pvporcupine  # type: ignore
        import pyaudio  # type: ignore
    except Exception:
        print("⚠️ Hotword detection unavailable (missing dependencies).", flush=True)
        return

    porcupine = pvporcupine.create(keywords=[wakeword])
    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length,
        input_device_index=device_id,
    )

    try:
        while True:
            pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            if porcupine.process(pcm) >= 0:
                break
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        porcupine.delete()
