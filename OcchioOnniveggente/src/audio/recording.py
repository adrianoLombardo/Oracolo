from __future__ import annotations

import collections
import threading
from pathlib import Path
from typing import Any, Dict

import numpy as np
import soundfile as sf

try:  # pragma: no cover - optional dependency
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None

try:  # pragma: no cover - optional dependency
    import webrtcvad  # type: ignore
except Exception:  # pragma: no cover
    webrtcvad = None

from .processing import (
    AudioPreprocessor,
    apply_agc,
    apply_limiter,
    load_audio_as_float,
)


def record_wav(path: Path, seconds: int, sr: int) -> None:
    """Record audio for a fixed amount of seconds."""
    if sd is None:  # pragma: no cover - runtime check
        raise RuntimeError("sounddevice is required for recording; install sounddevice")

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n🎤 Registra (max {seconds}s)…")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    sf.write(path.as_posix(), apply_agc(audio[:, 0] if audio.ndim > 1 else audio), sr)
    print("✅ Fatto.")


def _record_with_webrtcvad(
    path: Path,
    sr: int,
    recording_conf: Dict[str, Any],
    input_device_id: Any | None,
    tts_playing: threading.Event | None = None,
    preprocessor: AudioPreprocessor | None = None,
) -> bool:
    frame_ms = 20
    frame_samples = int(sr * frame_ms / 1000)
    START_MS = int(recording_conf.get("start_ms", 200))
    END_MS = int(recording_conf.get("end_ms", 800))
    MAX_MS = int(recording_conf.get("max_ms", 15000))
    MIN_SPEECH = float(recording_conf.get("min_speech_level", 0.01))
    mode = int(recording_conf.get("vad_sensitivity", 2))
    mode = max(0, min(3, mode))
    vad = webrtcvad.Vad(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n🎤 Parla pure (VAD webrtc, max {MAX_MS/1000:.1f}s)…")
    stream_kwargs = {"samplerate": sr, "blocksize": frame_samples, "channels": 1, "dtype": "int16"}
    if input_device_id is not None:
        stream_kwargs["device"] = input_device_id
    voiced = bytearray()
    started = False
    speech_ms = 0
    silence_ms = 0
    total_ms = 0
    try:
        with sd.RawInputStream(**stream_kwargs) as stream:
            while total_ms < MAX_MS:
                block, _ = stream.read(frame_samples)
                block_bytes = bytes(block)
                if tts_playing is not None and tts_playing.is_set():
                    continue
                total_ms += frame_ms
                if not started:
                    if vad.is_speech(block_bytes, sr):
                        speech_ms += frame_ms
                        if speech_ms >= START_MS:
                            started = True
                            voiced.extend(block_bytes)
                    else:
                        speech_ms = 0
                else:
                    voiced.extend(block_bytes)
                    if vad.is_speech(block_bytes, sr):
                        silence_ms = 0
                    else:
                        silence_ms += frame_ms
                        if silence_ms >= END_MS:
                            break
    except sd.PortAudioError as e:
        print(f"⚠️ Microfono non disponibile: {e}")
        return False
    if not voiced:
        print("😴 Silenzio rilevato. Resto in attesa.")
        return False
    y = np.frombuffer(bytes(voiced), dtype=np.int16).astype(np.float32) / 32768.0
    if preprocessor is not None:
        y = preprocessor.process(y)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < MIN_SPEECH:
        print("😴 Non ho colto parole distinte. Resto in attesa.")
        return False
    sf.write(path.as_posix(), apply_agc(y), sr)
    print(f"✅ Registrazione completata ({len(y)/sr:.2f}s).")
    return True


def record_until_silence(
    path: Path,
    sr: int,
    vad_conf: Dict[str, Any],
    recording_conf: Dict[str, Any],
    *,
    debug: bool = False,
    input_device_id: Any | None = None,
    tts_playing: threading.Event | None = None,
    preprocessor: AudioPreprocessor | None = None,
) -> bool:
    """VAD basato sull'energia che termina quando rileva silenzio."""
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass

    if recording_conf.get("use_webrtcvad", True) and webrtcvad is not None:
        if sr not in (8000, 16000, 32000, 48000):
            print(
                f"⚠️ WebRTC VAD richiede 8/16/32/48 kHz. Rilevati {sr} Hz, uso VAD a energia."
            )
        else:
            return _record_with_webrtcvad(
                path, sr, recording_conf, input_device_id, tts_playing, preprocessor
            )

    FRAME_MS = int(vad_conf.get("frame_ms", 30))
    START_MS = int(vad_conf.get("start_ms", 150))
    END_MS = int(vad_conf.get("end_ms", 800))
    MAX_MS = int(vad_conf.get("max_ms", 15000))
    PREROLL_MS = int(vad_conf.get("preroll_ms", 300))
    NOISE_WIN_MS = int(vad_conf.get("noise_window_ms", 800))
    START_MULT = float(vad_conf.get("start_mult", 1.8))
    END_MULT = float(vad_conf.get("end_mult", 1.3))
    BASE_START = float(vad_conf.get("base_start", 0.006))
    BASE_END = float(vad_conf.get("base_end", 0.0035))

    FALLBACK_TIMED = bool(recording_conf.get("fallback_to_timed", False))
    TIMED_SECONDS = int(recording_conf.get("timed_seconds", 10))
    MIN_SPEECH_LEVEL = float(recording_conf.get("min_speech_level", 0.01))

    assert FRAME_MS in (10, 20, 30)
    frame_samples = int(sr * FRAME_MS / 1000)

    print(f"\n🎤 Parla pure (VAD energia, max {MAX_MS/1000:.1f}s)…")

    pre_frames = PREROLL_MS // FRAME_MS
    preroll = collections.deque(maxlen=pre_frames)
    noise_frames_needed = max(1, NOISE_WIN_MS // FRAME_MS)
    noise_rms_history = collections.deque(maxlen=noise_frames_needed)

    started = False
    speech_streak = 0
    silence_streak = 0
    total_ms = 0
    recorded_blocks = []

    def rms(block_f32: np.ndarray) -> float:
        if block_f32.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(block_f32 ** 2)))

    stream_kwargs = {"samplerate": sr, "blocksize": frame_samples, "channels": 1, "dtype": "float32"}
    if input_device_id is not None:
        stream_kwargs["device"] = input_device_id

    try:
        with sd.InputStream(**stream_kwargs) as stream:
            while total_ms < MAX_MS:
                audio_block, _ = stream.read(frame_samples)
                if tts_playing is not None and tts_playing.is_set():
                    continue
                block = audio_block[:, 0]
                total_ms += FRAME_MS

                if not started and len(noise_rms_history) < noise_frames_needed:
                    noise_rms_history.append(rms(block))

                noise_floor = np.median(noise_rms_history) if noise_rms_history else 0.003
                start_thresh = max(noise_floor * START_MULT, BASE_START)
                end_thresh = max(noise_floor * END_MULT, BASE_END)

                level = rms(block)
                if debug and total_ms % 300 == 0 and not started:
                    print(
                        f"   [DBG] level={level:.4f} start_th={start_thresh:.4f} noise={noise_floor:.4f}"
                    )

                if not started:
                    preroll.append(block.copy())
                    if level >= start_thresh:
                        speech_streak += FRAME_MS
                        if speech_streak >= START_MS:
                            started = True
                            if preroll:
                                recorded_blocks.extend([b.copy() for b in preroll])
                            recorded_blocks.append(block.copy())
                            silence_streak = 0
                    else:
                        speech_streak = 0
                else:
                    recorded_blocks.append(block.copy())
                    if level < end_thresh:
                        silence_streak += FRAME_MS
                        if silence_streak >= END_MS:
                            break
                    else:
                        silence_streak = 0
    except sd.PortAudioError as e:
        print(f"⚠️ Microfono non disponibile: {e}")
        return False

    if not recorded_blocks:
        print("😴 Silenzio rilevato. Resto in attesa.")
        if FALLBACK_TIMED:
            print("↻ Fallback: registrazione temporizzata.")
            record_wav(path, seconds=TIMED_SECONDS, sr=sr)
            return True
        return False

    y = np.concatenate(recorded_blocks, axis=0).astype(np.float32)
    if preprocessor is not None:
        y = preprocessor.process(y)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < MIN_SPEECH_LEVEL:
        if debug:
            print(
                f"   [DBG] Registrazione scartata: peak={peak:.4f} < min={MIN_SPEECH_LEVEL:.4f}"
            )
        print("😴 Non ho colto parole distinte. Resto in attesa.")
        return False

    sf.write(path.as_posix(), apply_agc(y), sr)
    dur = len(y) / sr
    print(f"✅ Registrazione completata ({dur:.2f}s).")
    return True


def play_and_pulse(
    path: Path,
    light: Any,
    sr: int,
    lighting_conf: Dict[str, Any],
    output_device_id: Any | None = None,
    duck_event: threading.Event | None = None,
    tts_event: threading.Event | None = None,
) -> None:
    """Play audio and drive lights, with optional volume ducking."""
    y, sr = load_audio_as_float(path, sr)
    y = apply_limiter(y)
    stop = False

    if tts_event is not None:
        tts_event.set()

    def worker() -> None:
        win = int(0.02 * sr)
        pos = 0
        cur = 0.1
        while not stop and pos < len(y):
            seg = y[pos : pos + win]
            rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
            level = max(0.0, min(1.0, rms * 6.0))
            cur = 0.7 * cur + 0.3 * level
            if hasattr(light, "set_rgb"):
                idle = lighting_conf["sacn"]["idle_level"]
                peak = lighting_conf["sacn"]["peak_level"]
                v = int(idle + (peak - idle) * cur)
                light.set_rgb(v // 3, v // 3, v)
            else:
                light.pulse(cur)
            time.sleep(win / sr)
            pos += win

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        frame = int(0.02 * sr)
        pos = 0
        volume = 1.0
        with sd.OutputStream(
            samplerate=sr, channels=1, dtype="float32", device=output_device_id
        ) as stream:
            while pos < len(y):
                seg = y[pos : pos + frame]
                if duck_event is not None and duck_event.is_set():
                    volume *= 0.7
                    if volume <= 0.05:
                        break
                stream.write(seg * volume)
                pos += frame
    except sd.PortAudioError as e:
        print(f"⚠️ Impossibile riprodurre audio: {e}")
    finally:
        stop = True
        t.join()
        if tts_event is not None:
            tts_event.clear()
