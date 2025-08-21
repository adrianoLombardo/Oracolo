from __future__ import annotations

"""Utilities for hotword detection.

This module provides a helper function :func:`listen_for_wakeword` that blocks
until a configured wake word is detected using a wake word engine such as
Porcupine or Snowboy.  If the required third party dependencies are not
available, the function falls back to a no-op so that the application can still
run without hotword support.
"""

from typing import Optional



def listen_for_wakeword(wakeword: str, device_id: Optional[int] = None) -> None:
    """Block until *wakeword* is detected on the microphone.

    Parameters
    ----------
    wakeword:
        Name of the keyword model to detect.  This should match one of the
        keywords supported by the underlying engine (e.g. "picovoice",
        "porcupine", ...).
    device_id:
        Optional index of the input device to use.  ``None`` lets the backend
        choose the default system device.

    Notes
    -----
    The function attempts to use `pvporcupine` (Picovoice Porcupine) for
    detection.  If the dependency or ``pyaudio`` is missing, a warning is

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
