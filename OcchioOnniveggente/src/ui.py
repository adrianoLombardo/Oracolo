# src/ui.py
"""
UI Tkinter per Occhio Onniveggente.

- Carica settings.yaml + settings.local.yaml (overlay).
- Salvataggio "split": debug e audio.(input/output)_device -> settings.local.yaml,
  tutto il resto -> settings.yaml.
- Menu Documenti: Gestione documenti (indice, aggiungi/rimuovi/reindicizza, test retrieval) e Libreria.
- Avvio/Stop di src.main in modalità --autostart con log in tempo reale.
- Controllo client Realtime WebSocket (start/stop) con streaming audio.
"""

from __future__ import annotations

import json
import csv
import os
import queue
import logging
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk
from typing import Any, Callable

from src.retrieval import retrieve
from src.chat import ChatState
from src.conversation import ConversationManager
from src.oracle import (
    oracle_answer,
    oracle_answer_async,
    synthesize,
    synthesize_async,
)

from src.domain import validate_question
from src.validators import validate_device_config
from src.config import get_openai_api_key
from src.ui_state import UIState
from src.ui_controller import UIController
from src.oracle import synthesize
from src.ui_controller import UiController, _REASON_RE


import asyncio
import numpy as np
import sounddevice as sd
import websockets
import yaml
try:  # markdown2 è opzionale
    from markdown2 import markdown
except ImportError:  # pragma: no cover - fallback se non installato
    def markdown(text: str, *_, **__):
        """Ritorna il testo originale se markdown2 non è disponibile."""
        return text
from openai import OpenAI
import re
import webbrowser

try:  # opzionale: drag&drop documenti
    import tkinterdnd2 as tkdnd  # type: ignore
except Exception:  # pragma: no cover - fallback se non disponibile
    tkdnd = None

WS_URL = os.getenv("ORACOLO_WS_URL", "ws://localhost:8765")
SR = 24_000


def _highlight_terms(text: str, query: str) -> str:
    """Evidenzia i termini della query all'interno del testo."""
    tokens = set(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", query.lower()))
    for t in sorted(tokens, key=len, reverse=True):
        pattern = re.compile(re.escape(t), re.IGNORECASE)
        text = pattern.sub(lambda m: f"[{m.group(0)}]", text)
    return text

# opzionale: compatibilità icona PNG via Pillow
try:
    from PIL import Image, ImageTk  # pip install pillow
except Exception:
    Image = None
    ImageTk = None


# ------------------------- Realtime WS client ---------------------------- #
class RealtimeWSClient:
    """Gestisce la connessione WebSocket realtime con audio bidirezionale."""

    def __init__(
        self,
        url: str,
        sr: int,
        on_partial=lambda text, final: None,
        on_answer=lambda text: None,
        *,
        barge_threshold: float = 500.0,
        ping_interval: int = 20,
        ping_timeout: int = 20,
        auto_reconnect: bool = False,
        on_input_level=lambda level: None,
        on_output_level=lambda level: None,
        on_event=lambda evt: None,
        on_ping=lambda ms: None,
        profile_name: str = "museo",
    ) -> None:
        self.url = url
        self.sr = sr
        self.frame_samples = int(self.sr * 0.02)
        self.on_partial = on_partial
        self.on_answer = on_answer
        self.barge_threshold = barge_threshold
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.auto_reconnect = auto_reconnect
        self.on_input_level = on_input_level
        self.on_output_level = on_output_level
        self.on_event = on_event
        self.on_ping = on_ping
        self.profile_name = profile_name
        self.send_q: "queue.Queue[bytes]" = queue.Queue()
        self.audio_q: "queue.Queue[bytes]" = queue.Queue()
        self.state: dict[str, Any] = {
            "tts_playing": False,
            "barge_sent": False,
            "barge_threshold": barge_threshold,
        }
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.ws = None
        self.stop_event: asyncio.Event | None = None

    async def _mic_worker(self) -> None:
        assert self.ws is not None and self.stop_event is not None
        loop = asyncio.get_running_loop()

        def callback(indata, frames, time_info, status) -> None:  # type: ignore[override]
            data = bytes(indata)  # CFFI -> bytes

            if not self.state.get("tts_playing"):
                self.send_q.put_nowait(data)

            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            level = float(np.sqrt(np.mean(samples ** 2)))
            self.on_input_level(level / 32768.0)

            if self.state.get("tts_playing"):
                if level > self.state.get("barge_threshold", 500.0):
                    self.state["hot_frames"] = self.state.get("hot_frames", 0) + 1
                else:
                    self.state["hot_frames"] = 0
                now = time.monotonic()
                if (
                    self.state.get("hot_frames", 0) >= 10
                    and not self.state.get("barge_sent", False)
                    and now - self.state.get("last_barge_ts", 0) > 0.6
                ):
                    self.state["barge_sent"] = True
                    self.state["last_barge_ts"] = now
                    asyncio.run_coroutine_threadsafe(
                        self.ws.send(json.dumps({"type": "barge_in"})), loop
                    )

        with sd.RawInputStream(
            samplerate=self.sr,
            blocksize=self.frame_samples,
            channels=1,
            dtype="int16",
            callback=callback,
            latency="low",
        ):
            while not self.stop_event.is_set():
                await asyncio.sleep(0.1)

    async def _sender(self) -> None:
        assert self.ws is not None and self.stop_event is not None
        while not self.stop_event.is_set():
            data = await asyncio.get_running_loop().run_in_executor(None, self.send_q.get)
            await self.ws.send(data)

    async def _receiver(self) -> None:
        assert self.ws is not None and self.stop_event is not None
        async for msg in self.ws:
            if isinstance(msg, bytes):
                self.audio_q.put_nowait(msg)
                continue
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue
            kind = data.get("type")
            text = data.get("text", "")
            if kind == "partial":
                self.on_partial(text, bool(data.get("final")))
            elif kind in ("final", "transcript"):
                self.on_partial(text, True)
            elif kind == "answer":
                self.on_answer(text)
            if self.stop_event.is_set():
                break

    async def _player(self) -> None:
        assert self.stop_event is not None

        def callback(outdata, frames, time_info, status) -> None:  # type: ignore[override]
            try:
                chunk = self.audio_q.get_nowait()
                n = len(outdata)
                vol = 0.3 if self.state.get("barge_sent") else 1.0
                data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                data = np.clip(data * vol, -32768, 32767).astype(np.int16).tobytes()
                if len(data) >= n:
                    outdata[:] = data[:n]
                else:
                    outdata[:len(data)] = data
                    outdata[len(data):] = b"\x00" * (n - len(data))
                samples = np.frombuffer(outdata, dtype=np.int16).astype(np.float32)
                level = float(np.sqrt(np.mean(samples ** 2)))
                self.on_output_level(level / 32768.0)
                self.state["tts_playing"] = True
            except queue.Empty:
                outdata[:] = b"\x00" * len(outdata)
                self.state["tts_playing"] = False
                self.state["barge_sent"] = False
                self.on_output_level(0.0)

        with sd.RawOutputStream(
            samplerate=self.sr,
            blocksize=self.frame_samples,
            channels=1,
            dtype="int16",
            callback=callback,
        ):
            while not self.stop_event.is_set():
                await asyncio.sleep(0.1)

    async def _pinger(self) -> None:
        assert self.ws is not None and self.stop_event is not None
        while not self.stop_event.is_set():
            start = time.perf_counter()
            try:
                pong = await self.ws.ping()
                await pong
                self.on_ping((time.perf_counter() - start) * 1000)
            except Exception:
                break
            await asyncio.sleep(self.ping_interval)

    async def _run(self) -> None:
        self.stop_event = asyncio.Event()
        while not self.stop_event.is_set():
            try:
                async with websockets.connect(
                    self.url, ping_interval=None, ping_timeout=self.ping_timeout
                ) as ws:
                    self.ws = ws
                    self.on_event("connected")
                    await ws.send(
                        json.dumps(
                            {"type": "hello", "sr": self.sr, "format": "pcm_s16le", "channels": 1}
                        )
                    )
                    try:
                        ready_raw = await asyncio.wait_for(ws.recv(), timeout=10)
                        data = json.loads(ready_raw) if isinstance(ready_raw, str) else {}
                        if data.get("type") != "ready":
                            raise RuntimeError("handshake")
                    except Exception as e:
                        self.on_event(f"handshake_error:{e}")
                        if not self.auto_reconnect:
                            return
                        await asyncio.sleep(2)
                        continue

                    self.on_event("handshake_ok")
                    try:
                        await ws.send(json.dumps({"type": "profile", "value": self.profile_name}))
                    except Exception:
                        pass

                    tasks = [
                        asyncio.create_task(self._mic_worker()),
                        asyncio.create_task(self._sender()),
                        asyncio.create_task(self._receiver()),
                        asyncio.create_task(self._player()),
                        asyncio.create_task(self._pinger()),
                    ]
                    await self.stop_event.wait()
                    for t in tasks:
                        t.cancel()
            except Exception as e:
                self.on_event(f"error:{e}")
                if not self.auto_reconnect:
                    break
                await asyncio.sleep(2)
            finally:
                self.ws = None
            if not self.auto_reconnect:
                break
        self.on_event("disconnected")

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_until_complete, args=(self._run(),), daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if not self.loop:
            return
        if self.stop_event is not None:
            self.loop.call_soon_threadsafe(self.stop_event.set)
        if self.ws is not None:
            asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        self.thread = None
        self.loop = None


# ------------------------------- UI class -------------------------------- #
class OracoloUI(tk.Tk):
    """Interfaccia grafica per l'Oracolo."""

    def __init__(
        self,
        state: UIState | None = None,
        controller: UIController | None = None,
    ) -> None:
        super().__init__()
        self.state = state or UIState()
        self.controller = controller
        self.root_dir = Path(__file__).resolve().parents[1]
        self.title("Occhio Onniveggente")

        # icona con fallback
        try:
            logo_png = self.root_dir / "img" / "logo.png"
            if logo_png.exists():
                try:
                    self.iconphoto(False, tk.PhotoImage(file=str(logo_png)))
                except tk.TclError:
                    if Image is not None and ImageTk is not None:
                        try:
                            img = Image.open(str(logo_png))
                            icon = ImageTk.PhotoImage(img)
                            self.iconphoto(False, icon)
                        except Exception:
                            pass
            logo_ico = self.root_dir / "img" / "logo.ico"
            if sys.platform.startswith("win") and logo_ico.exists():
                try:
                    self.iconbitmap(default=str(logo_ico))
                except Exception:
                    pass
        except Exception:
            pass

        self.geometry("900x600")

        # tema
        self._bg = "#0f0f0f"
        self._fg = "#00ffe1"
        self._mid = "#161616"
        self.configure(bg=self._bg)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "TButton",
            background="#1e1e1e",
            foreground=self._fg,
            borderwidth=1,
            focusthickness=3,
            focuscolor="none",
            padding=8,
        )
        style.map("TButton", background=[("active", self._fg)], foreground=[("active", self._bg)])
        style.configure("TLabel", background=self._bg, foreground=self._fg)
        style.configure("TFrame", background=self._bg)

        # settings and controller
        self.controller = UiController(self.root_dir)
        self.base_settings = self.controller.base_settings
        self.local_settings = self.controller.local_settings
        self.settings = self.controller.settings
        self.conv = self.controller.conv
        self.chat_state = self.controller.chat_state

        # quick toggles state
        self.lang_map = {"Auto": "auto", "IT": "it", "EN": "en"}
        self.mode_map = {"Dettagliata": "detailed", "Concisa": "concise"}
        self.lang_choice = tk.StringVar(value="Auto")
        default_mode = (
            "Dettagliata" if self.settings.get("answer_mode", "detailed") == "detailed" else "Concisa"
        )
        self.mode_choice = tk.StringVar(value=default_mode)
        self.style_var = tk.BooleanVar(value=True)
        self.use_mic_var = tk.BooleanVar(value=True)
        chat_conf = self.settings.get("chat", {})
        self.remember_var = tk.BooleanVar(value=bool(chat_conf.get("enabled", True)))
        self.turns_var = tk.IntVar(value=int(chat_conf.get("max_turns", 10)))
        dom = self.settings.get("domain", {})
        profiles = dom.get("profiles", {}) or {}
        self.profile_names = list(profiles.keys())
        prof_val = dom.get("profile", {})
        if isinstance(prof_val, dict):
            current_profile = prof_val.get(
                "current", self.profile_names[0] if self.profile_names else ""
            )
        else:
            current_profile = prof_val or (
                self.profile_names[0] if self.profile_names else ""
            )
        self.profile_var = tk.StringVar(value=current_profile)
        self.tts_muted = False
        self.last_answer = ""
        self.last_sources: list[dict[str, str]] = []
        self.last_activity = time.time()
        self.last_test_result: dict[str, Any] | None = None
        self.topic_threshold = tk.DoubleVar(value=float(self.settings.get("topic_threshold", 0.65)))
        self.keywords: list[str] = list(self.settings.get("keywords", []))

        # sandbox and log state
        self.sandbox_var = tk.BooleanVar(value=False)
        self.log_filters = {
            c: tk.BooleanVar(value=True)
            for c in ["STT", "LLM", "TTS", "WS", "DOMAIN", "DOCS", "WAKE"]
        }
        self.log_entries: list[tuple[str, str]] = []

        self.profile_cb: ttk.Combobox | None = None

        # process & logs
        self.proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop_reader = threading.Event()

        # websocket server
        self.ws_server_proc: subprocess.Popen | None = None
        self._ws_server_thread: threading.Thread | None = None
        self._ws_stop_reader = threading.Event()

        # realtime client
        self.ws_client: RealtimeWSClient | None = None

        # background asyncio loop for blocking operations
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(
            target=self._async_loop.run_forever, daemon=True
        )
        self._async_thread.start()

        # UI
        self._build_menubar()
        self._build_layout()
        if self.profile_var.get():
            self._apply_profile(self.profile_var.get())

        # poll
        self.after(500, self._poll_process)
        self.after(500, self._poll_status)
        self.after(1000, self._poll_idle)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        logging.getLogger().addHandler(UILogHandler(self))

    # ------------------------------ state props ---------------------------- #

    @property
    def settings(self) -> dict[str, Any]:
        return self.state.settings

    @settings.setter
    def settings(self, value: dict[str, Any]) -> None:
        self.state.settings = value

    @property
    def conv(self) -> ConversationManager | None:
        return self.state.conversation

    @conv.setter
    def conv(self, value: ConversationManager) -> None:
        self.state.conversation = value

    @property
    def audio(self) -> Any | None:
        return self.state.audio

    @audio.setter
    def audio(self, value: Any | None) -> None:
        self.state.audio = value

    # --------------------------- Menubar & dialogs ------------------------ #
    def _build_menubar(self) -> None:
        menubar = tk.Menu(self)

        # Documenti
        docs_menu = tk.Menu(menubar, tearoff=0)
        docs_menu.add_command(label="Gestione & Regole…", command=self._open_doc_manager_dialog)
        docs_menu.add_command(label="Libreria…", command=self._open_library_dialog)
        menubar.add_cascade(label="Documenti", menu=docs_menu)

        # Impostazioni
        settings_menu = tk.Menu(menubar, tearoff=0)
        self.debug_var = tk.BooleanVar(value=bool(self.settings.get("debug", False)))
        settings_menu.add_checkbutton(label="Debug", variable=self.debug_var, command=self._update_debug)
        settings_menu.add_command(label="Audio…", command=self._open_audio_dialog)
        settings_menu.add_command(label="Recording…", command=self._open_recording_dialog)
        settings_menu.add_command(label="Luci…", command=self._open_lighting_dialog)
        settings_menu.add_command(label="OpenAI…", command=self._open_openai_dialog)
        settings_menu.add_command(label="Wake…", command=self._open_wake_dialog)
        settings_menu.add_separator()
        settings_menu.add_command(label="Importa profili…", command=self._import_profiles)
        settings_menu.add_command(label="Esporta profili…", command=self._export_profiles)
        settings_menu.add_separator()
        settings_menu.add_command(label="Salva", command=self.save_settings)
        menubar.add_cascade(label="Impostazioni", menu=settings_menu)

        # Strumenti
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Prompt di sistema…", command=self._open_system_prompt_dialog)
        tools_menu.add_command(label="Pannello Test…", command=self._open_test_dialog)
        tools_menu.add_checkbutton(label="Sandbox", variable=self.sandbox_var, command=self._update_sandbox)
        tools_menu.add_command(label="Esporta conversazione…", command=self._export_chat)
        tools_menu.add_command(label="Risposta in audio…", command=self._export_audio)
        tools_menu.add_command(label="Limiti OpenAI…", command=self._open_quota_dialog)
        tools_menu.add_command(label="Salva log…", command=self._export_log)
        menubar.add_cascade(label="Strumenti", menu=tools_menu)

        # Server WS
        server_menu = tk.Menu(menubar, tearoff=0)
        server_menu.add_command(label="Avvia server WS", command=self.start_ws_server)
        self._srv_start_idx = server_menu.index("end")
        server_menu.add_command(label="Ferma server WS", command=self.stop_ws_server, state="disabled")
        self._srv_stop_idx = server_menu.index("end")
        menubar.add_cascade(label="Server", menu=server_menu)
        self.server_menu = server_menu

        self.config(menu=menubar)

    # --------------------------- Layout / Widgets ------------------------- #
    def _build_layout(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill="x", padx=16, pady=(12, 8))
        ttk.Label(header, text="Occhio Onniveggente", font=("Helvetica", 18, "bold")).pack(side="left")
        if self.profile_names:
            ttk.Label(header, text="Profilo:").pack(side="left", padx=(16, 4))
            cb = ttk.Combobox(header, textvariable=self.profile_var, values=self.profile_names, state="readonly", width=12)
            cb.pack(side="left")
            cb.bind("<<ComboboxSelected>>", self._on_profile_change)
            self.profile_cb = cb

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=16, pady=(0, 8))
        self.start_btn = ttk.Button(bar, text="Avvia", command=self.start_oracolo)
        self.stop_btn = ttk.Button(bar, text="Ferma", command=self.stop_oracolo, state="disabled")
        self.ws_start_btn = ttk.Button(bar, text="Avvia WS", command=self.start_realtime)
        self.ws_stop_btn = ttk.Button(bar, text="Ferma WS", command=self.stop_realtime, state="disabled")
        self.start_btn.pack(side="left", padx=(0, 8))
        self.stop_btn.pack(side="left")
        self.ws_start_btn.pack(side="left", padx=(8, 8))
        self.ws_stop_btn.pack(side="left")
        self.in_level = tk.DoubleVar(value=0.0)
        self.out_level = tk.DoubleVar(value=0.0)

        self.status_var = tk.StringVar(value="🟡 In attesa")
        ttk.Label(bar, textvariable=self.status_var).pack(side="right")

        opts = ttk.Frame(self)
        opts.pack(fill="x", padx=16, pady=(0, 8))
        ttk.Checkbutton(opts, text="Stile poetico", variable=self.style_var).pack(side="left")
        ttk.Label(opts, text="Lingua:").pack(side="left", padx=(8, 4))
        self.lang_cb = ttk.Combobox(
            opts,
            textvariable=self.lang_choice,
            values=list(self.lang_map.keys()),
            state="readonly",
            width=7,
        )
        self.lang_cb.pack(side="left")
        ttk.Checkbutton(opts, text="Ricorda", variable=self.remember_var, command=self._update_memory).pack(side="left", padx=(8, 4))
        self.turns_spin = tk.Spinbox(opts, from_=1, to=20, width=3, textvariable=self.turns_var, command=self._update_memory)
        self.turns_spin.pack(side="left")
        ttk.Label(opts, text="Risposta:").pack(side="left", padx=(8, 4))
        self.mode_cb = ttk.Combobox(
            opts,
            textvariable=self.mode_choice,
            values=list(self.mode_map.keys()),
            state="readonly",
            width=11,
        )
        self.mode_cb.pack(side="left")
        ttk.Label(opts, text="Voce TTS:").pack(side="left", padx=(8, 4))
        self.tts_voice = tk.StringVar(value=self.settings.get("tts_voice", "alloy"))
        self.tts_cb = ttk.Combobox(
            opts,
            textvariable=self.tts_voice,
            values=["alloy", "verse", "aria"],
            state="readonly",
            width=10,
        )
        self.tts_cb.pack(side="left")
        ttk.Button(opts, text="▶ Prova", command=self._test_voice).pack(side="left", padx=(4, 4))
        ttk.Label(opts, text="Velocità:").pack(side="left", padx=(8, 4))
        self.tts_speed = tk.DoubleVar(value=1.0)
        ttk.Scale(opts, from_=0.5, to=2.0, variable=self.tts_speed, orient="horizontal", length=100).pack(side="left")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        chat_frame = ttk.Frame(notebook)
        notebook.add(chat_frame, text="Chat")

        self.chat_view = scrolledtext.ScrolledText(
            chat_frame,
            wrap="word",
            state="disabled",
            height=15,
            bg=self._mid,
            fg="#d7fff9",
            insertbackground=self._fg,
            relief="flat",
            font=("Consolas", 10),
        )
        self.chat_view.pack(fill="both", expand=True, padx=4, pady=(4, 0))

        # Configurazione dei tag per simulare le bolle della chat
        self.chat_view.tag_config(
            "user_msg",
            background="#2d3e4e",
            foreground="#d7fff9",
            borderwidth=2,
            relief="solid",
            padx=6,
            pady=4,
        )
        self.chat_view.tag_config(
            "assistant_msg",
            background="#394b59",
            foreground="#ffffff",
            borderwidth=2,
            relief="solid",
            padx=6,
            pady=4,
        )

        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill="x", padx=4, pady=(4, 0))
        ttk.Checkbutton(input_frame, text="Usa microfono", variable=self.use_mic_var).pack(side="left")
        self.chat_entry = tk.Entry(input_frame)
        self.chat_entry.pack(side="left", fill="x", expand=True, padx=(8, 8))
        self.chat_entry.bind("<Return>", self._on_chat_enter)

        btns = ttk.Frame(chat_frame)
        btns.pack(fill="x", padx=4, pady=(4, 4))
        ttk.Button(btns, text="/reset", command=lambda: self._handle_command("/reset")).pack(side="left")
        ttk.Button(btns, text="/mute TTS", command=self._mute_tts).pack(side="left", padx=(4, 0))
        ttk.Button(btns, text="/stop TTS", command=self._stop_tts).pack(side="left", padx=(4, 0))
        ttk.Button(btns, text="Copia risposta", command=self._copy_last).pack(side="left", padx=(4, 0))
        ttk.Button(btns, text="Citazioni", command=self._copy_citations).pack(side="left", padx=(4, 0))
        ttk.Button(btns, text="Audio", command=self._export_audio).pack(side="left", padx=(4, 0))
        ttk.Button(btns, text="Esporta", command=self._export_chat).pack(side="left", padx=(4, 0))
        ttk.Button(btns, text="Pin messaggio", command=self._pin_last).pack(side="left", padx=(4, 0))

        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="Log")
        filt = ttk.Frame(log_frame)
        filt.pack(fill="x")
        for name, var in self.log_filters.items():
            ttk.Checkbutton(filt, text=name, variable=var, command=self._refresh_log).pack(side="left")
        ttk.Button(filt, text="Salva", command=self._export_log).pack(side="right")
        self.log = scrolledtext.ScrolledText(
            log_frame,
            wrap="word",
            state="disabled",
            height=15,
            bg=self._mid,
            fg="#d7fff9",
            insertbackground=self._fg,
            relief="flat",
            font=("Consolas", 10),
        )
        self.log.pack(fill="both", expand=True)

        topics_frame = ttk.Frame(notebook)
        notebook.add(topics_frame, text="Argomenti & Regole")
        ttk.Label(topics_frame, text="Soglia cambio argomento:").pack(anchor="w", padx=8, pady=(8, 0))
        ttk.Scale(
            topics_frame,
            from_=0.0,
            to=1.0,
            variable=self.topic_threshold,
            orient="horizontal",
            length=200,
        ).pack(fill="x", padx=8)
        kw_frame = ttk.Frame(topics_frame)
        kw_frame.pack(fill="both", expand=True, padx=8, pady=8)
        self.keyword_entry = tk.Entry(kw_frame)
        self.keyword_entry.pack(fill="x")
        ttk.Button(kw_frame, text="Aggiungi", command=self._add_keyword).pack(pady=4)
        self.keyword_listbox = tk.Listbox(kw_frame, height=5)
        self.keyword_listbox.pack(fill="both", expand=True)
        ttk.Button(kw_frame, text="Rimuovi selezionati", command=self._remove_keyword).pack(pady=4)
        for kw in self.keywords:
            self.keyword_listbox.insert("end", kw)

        status = ttk.Frame(self)
        status.pack(fill="x", padx=16, pady=(0, 4))
        ttk.Progressbar(
            status, orient="horizontal", length=80, mode="determinate", variable=self.in_level, maximum=1.0
        ).pack(side="left", padx=(0, 2))
        ttk.Progressbar(
            status, orient="horizontal", length=80, mode="determinate", variable=self.out_level, maximum=1.0
        ).pack(side="left", padx=(2, 8))
        self.ping_var = tk.StringVar(value="Ping: -")
        self.tts_queue = tk.StringVar(value="TTS q=0")
        self.doc_index = tk.StringVar(value="Index: OK")
        self.ws_reconnect_var = tk.BooleanVar(value=True)
        ttk.Label(status, textvariable=self.ping_var).pack(side="left")
        ttk.Label(status, textvariable=self.tts_queue).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(status, text="Auto-reconnect", variable=self.ws_reconnect_var).pack(side="left", padx=(8, 0))
        ttk.Label(status, textvariable=self.doc_index).pack(side="right")

        footer = ttk.Frame(self)
        footer.pack(fill="x", padx=16, pady=(0, 12))
        ttk.Button(footer, text="Esci", command=self._on_close).pack(side="right")

    # --------------------------- Chat helpers ------------------------------ #
    def _append_chat(self, role: str, text: str) -> None:
        html = markdown(text, extras=["fenced-code-blocks"]) if text else ""
        clean = re.sub("<[^<]+?>", "", html)
        self.chat_view.configure(state="normal")
        prefix = "👤" if role == "user" else ("🔮" if role == "assistant" else "ℹ️")
        tag = "user_msg" if role == "user" else ("assistant_msg" if role == "assistant" else None)
        if tag:
            self.chat_view.insert("end", f"{prefix} {clean}\n", tag)
        else:
            self.chat_view.insert("end", f"{prefix} {clean}\n")
        self.chat_view.see("end")
        self.chat_view.configure(state="disabled")
        if role == "assistant":
            self.last_answer = clean

    def _append_citations(self, sources: list[dict[str, str]]) -> None:
        if not sources:
            return
        self.chat_view.configure(state="normal")
        for i, src in enumerate(sources, 1):
            cid = src.get("id", f"doc{i}")
            self.chat_view.insert("end", f"[{i}] {cid} ")
            btn = ttk.Button(self.chat_view, text="apri documento", command=lambda s=src: self._open_citation(s))
            self.chat_view.window_create("end", window=btn)
            self.chat_view.insert("end", "\n")
        self.chat_view.configure(state="disabled")
        self.chat_view.see("end")

    def _open_citation(self, src: dict[str, str]) -> None:
        path = src.get("path")
        if path and Path(path).exists():
            webbrowser.open(Path(path).as_uri())
            return
        win = tk.Toplevel(self)
        win.title(src.get("title") or src.get("id", "Documento"))
        box = scrolledtext.ScrolledText(win, wrap="word", width=60, height=20)
        box.pack(fill="both", expand=True)
        box.insert("1.0", src.get("text", ""))
        box.configure(state="disabled")

    def _update_memory(self) -> None:
        if self.remember_var.get():
            self.chat_state.max_turns = int(self.turns_var.get())
        else:
            self.chat_state.max_turns = 0
        chat_conf = self.settings.setdefault("chat", {})
        chat_conf["enabled"] = bool(self.remember_var.get())
        chat_conf["max_turns"] = int(self.turns_var.get())

    def _add_keyword(self) -> None:
        kw = self.keyword_entry.get().strip()
        if kw and kw not in self.keywords:
            self.keywords.append(kw)
            self.keyword_listbox.insert("end", kw)
            self.settings["keywords"] = self.keywords
            self.controller.save_settings(self._append_log)
            self.base_settings = self.controller.base_settings
            self.local_settings = self.controller.local_settings
            self.settings = self.controller.settings
            self._append_log(f"Aggiunta keyword: {kw}", "DOMAIN")
        self.keyword_entry.delete(0, "end")

    def _remove_keyword(self) -> None:
        sel = list(self.keyword_listbox.curselection())
        removed = False
        for idx in reversed(sel):
            kw = self.keyword_listbox.get(idx)
            self._append_log(f"Rimossa keyword: {kw}", "DOMAIN")
            self.keywords.remove(kw)
            self.keyword_listbox.delete(idx)
            removed = True
        if removed:
            self.settings["keywords"] = self.keywords
            self.controller.save_settings(self._append_log)
            self.base_settings = self.controller.base_settings
            self.local_settings = self.controller.local_settings
            self.settings = self.controller.settings

    def _pin_last(self) -> None:
        self.chat_state.pin_last_user()
        self._append_chat("system", "Messaggio fissato.")

    def _on_profile_change(self, event=None) -> None:
        self._apply_profile(self.profile_var.get())
        self._append_chat("system", f"Profilo cambiato: {self.profile_var.get()}")

    def _apply_profile(self, name: str) -> None:
        self.settings.setdefault("domain", {})["profile"] = name
        self.controller.save_settings(self._append_log)
        self.base_settings = self.controller.base_settings
        self.local_settings = self.controller.local_settings
        self.settings = self.controller.settings
        self._append_log(f"Profilo attivo: {name}", "DOMAIN")
        if self.ws_client is not None:
            self.ws_client.profile_name = name
            if self.ws_client.ws is not None:
                asyncio.run_coroutine_threadsafe(
                    self.ws_client.ws.send(json.dumps({"type": "profile", "value": name})),
                    self.ws_client.loop,
                )
        self.chat_state.reset()

    def _on_chat_enter(self, event) -> str:
        text = self.chat_entry.get().strip()
        if not text:
            return "break"
        self.chat_entry.delete(0, "end")
        if text.startswith("/"):
            self._handle_command(text)
        else:
            self._send_chat(text)
        return "break"

    def _handle_command(self, cmd: str) -> None:
        parts = cmd.strip().split()
        if not parts:
            return
        base = parts[0].lower()
        args = parts[1:]
        if base == "/reset":
            self.chat_state.reset()
            self.chat_view.configure(state="normal")
            self.chat_view.delete("1.0", "end")
            self.chat_view.configure(state="disabled")
        elif base == "/profile" and args:
            self._append_chat("system", f"Profilo cambiato: {args[0]}")
        elif base == "/topic" and args:
            self.chat_state.topic_text = " ".join(args)
            self._append_chat("system", f"Topic: {' '.join(args)}")
        elif base == "/docs" and args:
            if args[0] == "add":
                dom = self.settings.get("domain", {}) or {}
                prof_val = dom.get("profile", {})
                if isinstance(prof_val, dict):
                    current_profile = prof_val.get("current", "")
                else:
                    current_profile = prof_val or ""
                profiles = dom.get("profiles", {}) or {}
                docstore_path = profiles.get(current_profile, {}).get(
                    "docstore_path", self.settings.get("docstore_path")
                )
                self._add_documents(current_profile, docstore_path)
            elif args[0] == "reindex":
                self._reindex_documents()
        elif base == "/realtime" and args:
            if args[0].lower() == "on":
                self.start_realtime()
            else:
                self.stop_realtime()
        else:
            self._append_chat("system", "Comando sconosciuto")

    
    def _send_chat(self, text: str) -> None:
        """Wrapper that schedules the async chat handler."""
        self.last_activity = time.time()
        self._append_chat("user", text)
        self.conv.push_user(text)
        self.last_sources = []
        openai_conf = self.settings.get("openai", {})
        api_key = get_openai_api_key(self.settings)
        if not api_key:
            if messagebox.askyesno(
                "OpenAI", "È necessaria una API key OpenAI. Aprire le impostazioni?"
            ):
                self._open_openai_dialog()
                api_key = get_openai_api_key(self.settings)
            if not api_key:
                self.chat_entry.configure(state="disabled")
                self.status_var.set("🟡 In attesa")
                return
            else:
                self.chat_entry.configure(state="normal")
        self.status_var.set("⏳ In corso…")
        self._async_loop.call_soon_threadsafe(
            asyncio.create_task, self._send_chat_async(text, api_key, openai_conf)
        )
    async def _send_chat_async(
        self, text: str, api_key: str, openai_conf: dict
    ) -> None:
        try:
            client = OpenAI(api_key=api_key)
            style_prompt = (
                self.settings.get("style_prompt", "") if self.style_var.get() else ""
            )
            lang = self.lang_map.get(self.lang_choice.get(), "auto")
            mode = self.mode_map.get(self.mode_choice.get(), "detailed")
            docstore_path = self.settings.get("docstore_path")
            top_k = int(self.settings.get("retrieval_top_k", 3))
            ok, ctx, needs_clar, reason, _ = await asyncio.to_thread(
                validate_question,
                text,
                settings=self.settings,
                client=client,
                docstore_path=docstore_path,
                top_k=top_k,
                embed_model=openai_conf.get("embed_model", "text-embedding-3-small"),
                topic=self.chat_state.topic_text,
                history=self.chat_state.history,
            )
            if not ok:
                if needs_clar:
                    ans = "Potresti fornire maggiori dettagli o chiarire la tua domanda?"
                else:
                    m = _REASON_RE.search(reason)
                    if m:
                        score = float(m.group("score"))
                        thr = float(m.group("thr"))
                        ans = f"Richiesta fuori dominio (score {score:.2f} < {thr:.2f})."
                    else:
                        ans = "Richiesta fuori dominio."
                self.conv.push_assistant(ans)
                self.after(0, self._append_chat, "assistant", ans)
                self.after(0, self._append_citations, [])
                self.last_activity = time.time()
                self.after(0, self.status_var.set, "🟢 Attivo")
                return
            if self.sandbox_var.get():
                ctx = []
            else:
                dom = self.settings.get("domain", {}) or {}
                prof = dom.get("profile", "")
                if isinstance(prof, dict):
                    prof = prof.get("current", "")
                ctx = await asyncio.to_thread(
                    retrieve,
                    text,
                    self.settings.get("docstore_path", ""),
                    top_k=int(self.settings.get("retrieval_top_k", 3)),
                    topic=prof,
                )
            pin_ctx = [{"id": f"pin{i}", "text": t} for i, t in enumerate(self.chat_state.pinned)]
            ctx = pin_ctx + ctx
            ans, used_ctx = await oracle_answer_async(
                text,
                lang,
                client,
                self.settings.get("llm_model", "gpt-4o"),
                style_prompt,
                context=ctx,
                history=self.chat_state.history,
                topic=self.chat_state.topic_text,
                mode=mode,
            )
            try:
                ans, used_ctx = self.controller.send_chat(
                    text,
                    self.lang_map.get(self.lang_choice.get(), "auto"),
                    self.mode_map.get(self.mode_choice.get(), "detailed"),
                    self.style_var.get(),
                    self.sandbox_var.get(),

                )
                self.last_sources = used_ctx
            except Exception as e:
                ans = f"Errore: {e}"
                self.last_sources = []
            self.conv.push_assistant(ans)

            self._append_chat("assistant", ans)
            self._append_citations(self.last_sources)
            self.last_activity = time.time()
            self.after(0, self._append_chat, "assistant", ans)
            self.after(0, self._append_citations, self.last_sources)
            self.after(0, self.status_var.set, "🟢 Attivo")
        except Exception as e:
            ans = f"Errore: {e}"
            self.last_sources = []
            self.conv.push_assistant(ans)
            self.after(0, self._append_chat, "assistant", ans)
            self.after(0, self._append_citations, [])
            self.after(0, self.status_var.set, "🟢 Attivo")

    def _mute_tts(self) -> None:
        self.tts_muted = not self.tts_muted

    def _stop_tts(self) -> None:
        if (
            self.ws_client is not None
            and self.ws_client.ws is not None
            and self.ws_client.loop is not None
        ):
            try:
                asyncio.run_coroutine_threadsafe(
                    self.ws_client.ws.send(json.dumps({"type": "barge_in"})),
                    self.ws_client.loop,
                )
            except Exception:
                pass

    def _copy_last(self) -> None:
        if self.last_answer:
            self.clipboard_clear()
            self.clipboard_append(self.last_answer)

    def _export_chat(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("Markdown", "*.md"), ("JSON", "*.json")],
            parent=self,
        )
        if not path:
            return
        text = self.chat_view.get("1.0", "end")
        if path.lower().endswith(".json"):
            data = self.chat_state.history
            Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            Path(path).write_text(text, encoding="utf-8")

    def _export_audio(self) -> None:
        if not self.last_answer:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV", "*.wav"), ("MP3", "*.mp3")],
            parent=self,
        )
        if not path:
            return
        self.status_var.set("⏳ Esportazione audio…")
        self._async_loop.call_soon_threadsafe(
            asyncio.create_task, self._export_audio_async(Path(path))
        )

    async def _export_audio_async(self, path: Path) -> None:
        try:
            openai_conf = self.settings.get("openai", {})
            api_key = get_openai_api_key(self.settings)
            client = OpenAI(api_key=api_key) if api_key else OpenAI()
            tts_model = openai_conf.get("tts_model", "gpt-4o-mini-tts")
            tts_voice = openai_conf.get("tts_voice", "alloy")
            await synthesize_async(self.last_answer, Path(path), client, tts_model, tts_voice)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Audio", f"Errore esportando audio: {e}"))
        finally:
            self.after(0, self.status_var.set, "🟢 Attivo")

    def _copy_citations(self) -> None:
        if not self.last_sources:
            return
        cites = ", ".join(s.get("id", "") for s in self.last_sources)
        self.clipboard_clear()
        self.clipboard_append(cites)

    def _export_log(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("CSV", "*.csv")],
            parent=self,
        )
        if not path:
            return
        if path.lower().endswith(".csv"):
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["category", "text"])
                for cat, txt in self.log_entries:
                    w.writerow([cat, txt.strip()])
        else:
            data = [{"category": c, "text": t.strip()} for c, t in self.log_entries]
            Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _test_voice(self) -> None:
        self._append_log(f"Voce test: {self.tts_voice.get()}\n", "TTS")

    def _poll_status(self) -> None:
        if self.ws_client is not None:
            self.tts_queue.set(f"TTS q={self.ws_client.audio_q.qsize()}")
        else:
            self.tts_queue.set("TTS q=0")
            self.ping_var.set("Ping: -")
            self.in_level.set(0.0)
            self.out_level.set(0.0)
        self.after(500, self._poll_status)

    def _poll_idle(self) -> None:
        now = time.time()
        timeout = self.settings.get("wake", {}).get("idle_timeout", 50)
        if now - self.last_activity > timeout:
            self.status_var.set("😴 Dormiente — dì Ciao Oracolo per riattivarmi")
        elif self.status_var.get().startswith("😴"):
            self.status_var.set("🟡 In attesa")
        self.after(1000, self._poll_idle)

    # --------------------------- Log helpers ------------------------------ #
    class UILogHandler(logging.Handler):
        """Inoltra i log della libreria standard al widget della UI."""

        _LEVEL_MAP = {
            "INFO": "MISC",
            "ERROR": "ERR",
            "WARNING": "WARN",
            "DEBUG": "DBG",
        }

        def __init__(self, ui: "OracoloUI") -> None:
            super().__init__()
            self.ui = ui

        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - UI side effect
            msg = record.getMessage()
            if not msg.endswith("\n"):
                msg += "\n"
            cat = self._LEVEL_MAP.get(record.levelname, record.name)
            self.ui._append_log(msg, cat)

    def _append_log(self, text: str, category: str = "MISC") -> None:
        if self.sandbox_var.get():
            return
        cat = category.upper()
        self.log_entries.append((cat, text))
        if cat in self.log_filters and not self.log_filters[cat].get():
            return
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_entries.clear()
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _refresh_log(self) -> None:
        if self.sandbox_var.get():
            return
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        for cat, txt in self.log_entries:
            if cat in self.log_filters and not self.log_filters[cat].get():
                continue
            self.log.insert("end", txt)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _open_test_dialog(self) -> None:
        win = tk.Toplevel(self)
        win.title("Pannello Test")
        win.configure(bg=self._bg)

        q_var = tk.StringVar()
        lang_var = tk.StringVar(value=self.lang_choice.get())
        topk_var = tk.StringVar(value=str(self.settings.get("retrieval_top_k", 3)))

        tk.Label(win, text="Domanda", fg=self._fg, bg=self._bg).grid(row=0, column=0, padx=6, pady=6, sticky="e")
        tk.Entry(win, textvariable=q_var, width=50).grid(row=0, column=1, columnspan=2, padx=6, pady=6)

        tk.Label(win, text="Lingua", fg=self._fg, bg=self._bg).grid(row=1, column=0, padx=6, pady=6, sticky="e")
        ttk.Combobox(win, textvariable=lang_var, values=list(self.lang_map.keys()), state="readonly", width=10).grid(
            row=1, column=1, padx=6, pady=6, sticky="w"
        )

        tk.Label(win, text="Top-K", fg=self._fg, bg=self._bg).grid(row=1, column=2, padx=6, pady=6, sticky="e")
        tk.Entry(win, textvariable=topk_var, width=4).grid(row=1, column=3, padx=6, pady=6, sticky="w")

        result_box = scrolledtext.ScrolledText(win, width=70, height=20)
        result_box.grid(row=3, column=0, columnspan=4, padx=6, pady=6)

        def run_test() -> None:
            question = q_var.get().strip()
            if not question:
                return
            try:
                k = int(topk_var.get() or 3)
            except Exception:
                k = 3
            lang = self.lang_map.get(lang_var.get(), "auto")
            start = time.time()
            try:
                openai_conf = self.settings.get("openai", {})
                api_key = get_openai_api_key(self.settings)
                client = OpenAI(api_key=api_key) if api_key else OpenAI()
            except Exception:
                client = None
            dom = self.settings.get("domain", {}) or {}
            prof = dom.get("profile", "")
            if isinstance(prof, dict):
                prof = prof.get("current", "")
            ok, ctx, _clar, reason, _ = validate_question(
                question,
                lang,
                settings=self.settings,
                client=client,
                docstore_path=self.settings.get("docstore_path"),
                top_k=k,
                embed_model=openai_conf.get("embed_model", "text-embedding-3-small")
                if openai_conf
                else None,
                topic=prof,
            )
            end = time.time()
            m = _REASON_RE.search(reason)
            kw = emb = rag = score = thr = 0.0
            if m:
                kw = float(m.group("kw"))
                emb = float(m.group("emb"))
                rag = float(m.group("rag"))
                score = float(m.group("score"))
                thr = float(m.group("thr"))
            result_box.delete("1.0", "end")
            result_box.insert(
                "end",
                f"kw_overlap={kw:.2f} emb_sim={emb:.2f} rag_score={rag:.2f}\nscore={score:.2f} thr={thr:.2f} → {'OK' if ok else 'NO'}\n\n",
            )
            toks = q_var.get()
            for i, ch in enumerate(ctx[:k], start=1):
                text = _highlight_terms(str(ch.get("text", "")), toks)
                result_box.insert("end", f"[{i}] {text}\n\n")
            self.last_test_result = {
                "input": question,
                "lang": lang,
                "messages": [{"role": "user", "content": question}],
                "context": ctx,
                "citations": [c.get("id", "") for c in ctx],
                "timings": {"start": start, "end": end, "elapsed": end - start},
                "metrics": {
                    "kw_overlap": kw,
                    "emb_sim": emb,
                    "rag_score": rag,
                    "score": score,
                    "threshold": thr,
                    "decision": ok,
                },
            }

        def export_session() -> None:
            if not self.last_test_result:
                return
            path = filedialog.asksaveasfilename(
                parent=win,
                defaultextension=".json",
                filetypes=[("JSON", "*.json"), ("Tutti", "*.*")],
            )
            if not path:
                return
            Path(path).write_text(
                json.dumps(self.last_test_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        ttk.Button(win, text="Prova", command=run_test).grid(row=0, column=3, padx=6, pady=6)
        ttk.Button(win, text="Esporta", command=export_session).grid(row=4, column=0, columnspan=4, pady=(0, 6))

    def _open_system_prompt_dialog(self) -> None:
        win = tk.Toplevel(self)
        win.title("Prompt di sistema")
        txt = scrolledtext.ScrolledText(win, width=60, height=15)
        txt.pack(padx=8, pady=8)
        txt.insert("1.0", self.settings.get("style_prompt", ""))
        dom = self.settings.get("domain", {})
        presets = list(dom.get("profiles", {}).keys())
        if presets:
            var = tk.StringVar(value="")
            combo = ttk.Combobox(win, textvariable=var, values=presets, state="readonly")
            combo.pack(padx=8, pady=(0, 8))

            def on_sel(event=None):
                name = var.get()
                prof = dom.get("profiles", {}).get(name, {})
                sp = prof.get("oracle_system", "")
                txt.delete("1.0", "end")
                txt.insert("1.0", sp)

            combo.bind("<<ComboboxSelected>>", on_sel)

        btns = ttk.Frame(win)
        btns.pack(pady=8)

        def reset() -> None:
            txt.delete("1.0", "end")

        def save() -> None:
            self.settings["style_prompt"] = txt.get("1.0", "end").strip()
            win.destroy()

        ttk.Button(btns, text="Ripristina", command=reset).pack(side="left", padx=4)
        ttk.Button(btns, text="Salva", command=save).pack(side="left", padx=4)
        ttk.Button(btns, text="Chiudi", command=win.destroy).pack(side="left", padx=4)

    def _open_quota_dialog(self) -> None:
        win = tk.Toplevel(self)
        win.title("Limiti OpenAI")
        try:
            openai_conf = self.settings.get("openai", {})
            api_key = get_openai_api_key(self.settings)
            client = OpenAI(api_key=api_key) if api_key else OpenAI()
            model = self.settings.get("llm_model", "gpt-4o")
            info = client.models.retrieve(model)
            ttk.Label(win, text=f"Modello: {info.id}").pack(anchor="w", padx=8, pady=4)
        except Exception as e:
            ttk.Label(win, text=f"Errore: {e}").pack(anchor="w", padx=8, pady=4)
        ttk.Button(win, text="Chiudi", command=win.destroy).pack(pady=8)

    def _update_sandbox(self) -> None:
        if self.sandbox_var.get():
            self.tts_muted = True
            self.chat_state.persist_jsonl = None
        else:
            self.tts_muted = False
            pj = self.settings.get("chat", {}).get("persist_jsonl")
            if pj:
                self.chat_state.persist_jsonl = Path(pj)

    def _import_profiles(self) -> None:
        path = filedialog.askdirectory(parent=self)
        if not path:
            return
        prof_dir = Path(path)
        dom = self.settings.setdefault("domain", {})
        profiles = dom.setdefault("profiles", {})
        for p in prof_dir.glob("*.yaml"):
            profiles[p.stem] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        self.profile_names = list(profiles.keys())
        if self.profile_cb:
            self.profile_cb["values"] = self.profile_names
        if self.profile_names and self.profile_var.get() not in self.profile_names:
            self.profile_var.set(self.profile_names[0])

    def _export_profiles(self) -> None:
        path = filedialog.askdirectory(parent=self)
        if not path:
            return
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, conf in self.settings.get("domain", {}).get("profiles", {}).items():
            (out_dir / f"{name}.yaml").write_text(
                yaml.safe_dump(conf, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )

    # --------------------------- Document actions ------------------------- #
    def _find_ingest_script(self) -> Path | None:
        p = self.root_dir / "scripts" / "ingest_docs.py"
        return p if p.exists() else None

    def _run_ingest_process(self, cmd: list[str]) -> tuple[bool, bool, str]:
        """Esegue un comando di ingest mostrando output in tempo reale.

        Ritorna una tupla ``(successo, annullato, output)``.
        """
        top = tk.Toplevel(self)
        top.title("Ingest documenti")
        pb = ttk.Progressbar(top, mode="indeterminate")
        pb.pack(fill="x", padx=5, pady=5)
        txt = scrolledtext.ScrolledText(top, width=80, height=20)
        txt.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        pb.start(10)

        output: list[str] = []
        cancelled = threading.Event()
        proc_holder: dict[str, subprocess.Popen | None] = {"proc": None}
        rc_holder: dict[str, int | None] = {"rc": None}

        def append(line: str) -> None:
            txt.insert("end", line)
            txt.see("end")
            output.append(line)
            self._append_log(line, "DOCS")

        def worker() -> None:
            proc = subprocess.Popen(
                cmd,
                cwd=self.root_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            proc_holder["proc"] = proc

            def reader(stream: Any) -> None:
                for line in stream:  # type: ignore[assignment]
                    self.after(0, append, line)
                stream.close()

            threads = [
                threading.Thread(target=reader, args=(proc.stdout,)),
                threading.Thread(target=reader, args=(proc.stderr,)),
            ]
            for t in threads:
                t.daemon = True
                t.start()
            proc.wait()
            rc_holder["rc"] = proc.returncode
            if top.winfo_exists():
                self.after(0, pb.stop)
                self.after(0, top.destroy)

        threading.Thread(target=worker, daemon=True).start()

        def cancel() -> None:
            cancelled.set()
            p = proc_holder.get("proc")
            if p and p.poll() is None:
                p.terminate()
            pb.stop()
            if top.winfo_exists():
                top.destroy()
            self._append_log("Operazione annullata\n", "DOCS")

        ttk.Button(top, text="Annulla", command=cancel).pack(pady=5)

        self.wait_window(top)
        return rc_holder.get("rc", 1) == 0, cancelled.is_set(), "".join(output)

    def _add_documents(self, topic: str = "", docstore_path: str | None = None) -> None:
        dom = self.settings.get("domain", {}) or {}
        prof_val = dom.get("profile", {})
        if isinstance(prof_val, dict):
            current_profile = prof_val.get("current", "")
        else:
            current_profile = prof_val or ""
        if not topic:
            topic = current_profile
        if docstore_path is None:
            profiles = dom.get("profiles", {}) or {}
            docstore_path = profiles.get(current_profile, {}).get(
                "docstore_path", self.settings.get("docstore_path")
            )

        paths = list(filedialog.askopenfilenames(parent=self))
        if not paths:
            directory = filedialog.askdirectory(parent=self)
            if directory:
                paths = [directory]
        if not paths:
            return
        script = self._find_ingest_script()
        if not script:
            messagebox.showwarning("Documento", "Script di ingest non trovato (scripts/ingest_docs.py).")
            return
        self._append_log("Aggiunta documenti:\n" + "\n".join(paths) + "\n", "DOCS")
        cmd = [sys.executable, "-m", "scripts.ingest_docs", "--add", *paths]
        if docstore_path:
            cmd.extend(["--path", docstore_path])
        if topic:
            cmd.extend(["--topic", topic])
        ok, cancelled, _ = self._run_ingest_process(cmd)
        if cancelled:
            return
        if ok:
            messagebox.showinfo("Successo", "Documenti aggiunti.")
        else:
            messagebox.showerror("Errore", "Ingest fallito.")

    def _remove_documents(self, docstore_path: str | None = None) -> None:
        paths = list(filedialog.askopenfilenames(parent=self))
        if not paths:
            directory = filedialog.askdirectory(parent=self)
            if directory:
                paths = [directory]
        if not paths:
            return
        script = self._find_ingest_script()
        if not script:
            messagebox.showwarning("Documento", "Script di ingest non trovato (scripts/ingest_docs.py).")
            return
        self._append_log("Rimozione documenti:\n" + "\n".join(paths) + "\n", "DOCS")
        cmd = [sys.executable, "-m", "scripts.ingest_docs", "--remove", *paths]
        target = docstore_path or self.settings.get("docstore_path")
        if target:
            cmd.extend(["--path", target])
        ok, cancelled, _ = self._run_ingest_process(cmd)
        if cancelled:
            return
        if ok:
            messagebox.showinfo("Successo", "Documenti rimossi.")
        else:
            messagebox.showerror("Errore", "Rimozione fallita.")

    def _reindex_documents(self, docstore_path: str | None = None) -> None:
        script = self._find_ingest_script()
        if not script:
            messagebox.showwarning("Indice", "Script di ingest non trovato (scripts/ingest_docs.py).")
            return
        tried = (["--reindex"], ["--rebuild"], ["--refresh"])
        for args in tried:
            self._append_log(f"Reindex: tentativo {' '.join(args)}\n", "DOCS")
            self._append_log(
                f"Aggiorna indice: {' '.join(args)} in {self.root_dir}\n",
                "DOCS",
            )
            cmd = [sys.executable, "-m", "scripts.ingest_docs", *args]
            target = docstore_path or self.settings.get("docstore_path")
            if target:
                cmd.extend(["--path", target])
            ok, cancelled, output = self._run_ingest_process(cmd)
            if cancelled:
                return
            if ok:
                match = re.search(r"(\d+)\s+doc", output, re.IGNORECASE)
                if match:
                    self._append_log(
                        f"Indice aggiornato ({match.group(1)} documenti)\n", "DOCS"
                    )
                else:
                    self._append_log("Indice aggiornato\n", "DOCS")
                messagebox.showinfo("Indice", f"Indice aggiornato ({' '.join(args)}).")
                return
        self._append_log("Reindex: fallito\n", "DOCS")
        messagebox.showerror(
            "Indice", "Impossibile aggiornare l'indice (nessuna delle opzioni supportata)."
        )

    def _open_doc_manager_dialog(self) -> None:
        if tkdnd is not None:
            win = tkdnd.TkinterDnD.Toplevel(self)  # type: ignore[attr-defined]
        else:
            win = tk.Toplevel(self)
        win.title("Gestione Documenti & Regole")
        win.configure(bg=self._bg)

        notebook = ttk.Notebook(win)
        notebook.pack(fill="both", expand=True)

        # -------------------------- Tab Documenti ------------------------- #
        doc_tab = ttk.Frame(notebook)
        notebook.add(doc_tab, text="Documenti")

        dom = self.settings.get("domain", {})
        profiles = dom.get("profiles", {}) if isinstance(dom, dict) else {}
        current_profile = self.profile_var.get()
        default_path = profiles.get(current_profile, {}).get(
            "docstore_path", self.settings.get("docstore_path", "")
        )
        doc_var = tk.StringVar(value=default_path)
        topic_var = tk.StringVar(value=current_profile)
        topk_var = tk.StringVar(value=str(self.settings.get("retrieval_top_k", 3)))
        query_var = tk.StringVar()

        tk.Label(doc_tab, text="Indice", fg=self._fg, bg=self._bg).grid(row=0, column=0, padx=6, pady=6, sticky="e")
        tk.Entry(doc_tab, textvariable=doc_var, width=40).grid(row=0, column=1, padx=6, pady=6, sticky="w")

        def browse() -> None:
            p = filedialog.askopenfilename(
                title="Indice", filetypes=[("JSON", "*.json"), ("Tutti", "*.*")]
            )
            if p:
                doc_var.set(p)
                refresh_tree()

        ttk.Button(doc_tab, text="Sfoglia", command=browse).grid(row=0, column=2, padx=6, pady=6)

        tk.Label(doc_tab, text="Sezione", fg=self._fg, bg=self._bg).grid(row=1, column=0, padx=6, pady=6, sticky="e")
        opts = [p for p in self.profile_names if p != current_profile]
        tk.OptionMenu(doc_tab, topic_var, current_profile, *opts).grid(
            row=1, column=1, padx=6, pady=6, sticky="w"
        )

        def _on_topic_change(*_: Any) -> None:
            path = profiles.get(topic_var.get(), {}).get("docstore_path", "")
            if path:
                doc_var.set(path)
                refresh_tree()

        topic_var.trace_add("write", _on_topic_change)

        btn_frame = ttk.Frame(doc_tab)
        btn_frame.grid(row=1, column=2, rowspan=2, padx=6, pady=6, sticky="n")

        def _add_from_topic() -> None:
            path = profiles.get(topic_var.get(), {}).get("docstore_path", doc_var.get())
            self._add_documents(topic_var.get(), path)
            doc_var.set(path)
            refresh_tree()

        ttk.Button(btn_frame, text="Aggiungi", command=_add_from_topic).pack(fill="x")
        ttk.Button(
            btn_frame,
            text="Rimuovi",
            command=lambda: (self._remove_documents(doc_var.get()), refresh_tree()),
        ).pack(fill="x")
        ttk.Button(
            btn_frame,
            text="Reindicizza",
            command=lambda: (self._reindex_documents(doc_var.get()), refresh_tree()),
        ).pack(fill="x")

        tree = ttk.Treeview(doc_tab, columns=("topic", "date", "size"), show="headings")
        tree.heading("topic", text="Sezione")
        tree.heading("date", text="Indicizzazione")
        tree.heading("size", text="Dim.")
        tree.grid(row=2, column=0, columnspan=3, padx=6, pady=6, sticky="nsew")
        doc_tab.grid_rowconfigure(2, weight=1)
        doc_tab.grid_columnconfigure(1, weight=1)

        def refresh_tree() -> None:
            for i in tree.get_children():
                tree.delete(i)
            path = doc_var.get().strip() or "DataBase/index.json"
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:
                data = []
            docs = data.get("documents") if isinstance(data, dict) else (data if isinstance(data, list) else [])
            for d in docs:
                name = d.get("title") or d.get("id") or ""
                tag = ", ".join(d.get("tags", [])) or d.get("topic", "")
                date = d.get("date", "")
                size = len(d.get("text", ""))
                tree.insert("", "end", iid=str(d.get("id", name)), values=(tag, date, size), text=name)

        refresh_tree()

        tk.Label(doc_tab, text="Top-K", fg=self._fg, bg=self._bg).grid(row=3, column=0, padx=6, pady=6, sticky="e")
        tk.Entry(doc_tab, textvariable=topk_var, width=6).grid(row=3, column=1, padx=6, pady=6, sticky="w")

        tk.Label(doc_tab, text="Test query", fg=self._fg, bg=self._bg).grid(row=4, column=0, padx=6, pady=6, sticky="e")
        tk.Entry(doc_tab, textvariable=query_var, width=40).grid(row=4, column=1, padx=6, pady=6, sticky="w")

        result_box = scrolledtext.ScrolledText(doc_tab, width=50, height=8)
        result_box.grid(row=5, column=0, columnspan=3, padx=6, pady=6, sticky="nsew")
        doc_tab.grid_rowconfigure(5, weight=1)

        def run_test() -> None:
            result_box.delete("1.0", "end")
            try:
                k = int(topk_var.get() or 3)
            except Exception:
                k = 3
            try:
                res = retrieve(query_var.get(), doc_var.get(), top_k=k)
            except Exception as e:
                result_box.insert("end", f"Errore: {e}\n")
                return
            for item in res:
                text = str(item.get("text", "")).replace("\n", " ")[:200]
                title = str(item.get("title", ""))
                if title:
                    result_box.insert("end", f"- {title}: {text}\n")
                else:
                    result_box.insert("end", f"- {text}\n")

        ttk.Button(doc_tab, text="Prova", command=run_test).grid(row=4, column=2, padx=6, pady=6)

        # --------------------- Tab Argomenti & Regole --------------------- #
        domain_tab = ttk.Frame(notebook)
        notebook.add(domain_tab, text="Argomenti & Regole")

        dom = self.settings.setdefault("domain", {})

        topic_box = scrolledtext.ScrolledText(domain_tab, width=40, height=4)
        topic_box.insert("1.0", dom.get("topic", ""))
        tk.Label(domain_tab, text="Topic", fg=self._fg, bg=self._bg).grid(row=0, column=0, padx=6, pady=6, sticky="ne")
        topic_box.grid(row=0, column=1, padx=6, pady=6, sticky="w")

        kw_var = tk.StringVar(value=", ".join(dom.get("keywords", [])))
        tk.Label(domain_tab, text="Keywords", fg=self._fg, bg=self._bg).grid(row=1, column=0, padx=6, pady=6, sticky="e")
        tk.Entry(domain_tab, textvariable=kw_var, width=40).grid(row=1, column=1, padx=6, pady=6, sticky="w")

        weights = dom.get("weights", {})
        ov_var = tk.DoubleVar(value=weights.get("kw", 0.4))
        emb_var = tk.DoubleVar(value=weights.get("emb", 0.3))
        rag_var = tk.DoubleVar(value=weights.get("rag", 0.3))
        tk.Label(domain_tab, text="Peso Overlap", fg=self._fg, bg=self._bg).grid(row=2, column=0, padx=6, pady=6, sticky="e")
        ttk.Scale(domain_tab, from_=0, to=1, variable=ov_var, orient="horizontal", length=180).grid(row=2, column=1, padx=6, pady=6, sticky="w")
        tk.Label(domain_tab, text="Peso Embedding", fg=self._fg, bg=self._bg).grid(row=3, column=0, padx=6, pady=6, sticky="e")
        ttk.Scale(domain_tab, from_=0, to=1, variable=emb_var, orient="horizontal", length=180).grid(row=3, column=1, padx=6, pady=6, sticky="w")
        tk.Label(domain_tab, text="Peso RAG", fg=self._fg, bg=self._bg).grid(row=4, column=0, padx=6, pady=6, sticky="e")
        ttk.Scale(domain_tab, from_=0, to=1, variable=rag_var, orient="horizontal", length=180).grid(row=4, column=1, padx=6, pady=6, sticky="w")

        acc_var = tk.DoubleVar(value=dom.get("accept_threshold", 0.5))
        clar_var = tk.DoubleVar(value=dom.get("clarify_margin", 0.15))
        tk.Label(domain_tab, text="Accept threshold", fg=self._fg, bg=self._bg).grid(row=5, column=0, padx=6, pady=6, sticky="e")
        ttk.Scale(domain_tab, from_=0, to=1, variable=acc_var, orient="horizontal", length=180).grid(row=5, column=1, padx=6, pady=6, sticky="w")
        tk.Label(domain_tab, text="Clarify margin", fg=self._fg, bg=self._bg).grid(row=6, column=0, padx=6, pady=6, sticky="e")
        ttk.Scale(domain_tab, from_=0, to=1, variable=clar_var, orient="horizontal", length=180).grid(row=6, column=1, padx=6, pady=6, sticky="w")

        wake_var = tk.BooleanVar(value=dom.get("always_accept_wake", True))
        tk.Checkbutton(
            domain_tab,
            text="Accetta sempre saluti/wake words",
            variable=wake_var,
            bg=self._bg,
            fg=self._fg,
            selectcolor=self._bg,
        ).grid(row=7, column=0, columnspan=2, padx=6, pady=6, sticky="w")

        # ----------------------------- Salvataggio ------------------------ #
        def on_ok() -> None:
            self.settings["docstore_path"] = doc_var.get().strip()
            try:
                self.settings["retrieval_top_k"] = int(topk_var.get())
            except Exception:
                pass
            dom["topic"] = topic_box.get("1.0", "end").strip()
            dom["keywords"] = [k.strip() for k in kw_var.get().split(",") if k.strip()]
            self._append_log(
                "Keywords aggiornate: " + ", ".join(dom["keywords"]),
                "DOMAIN",
            )
            dom["weights"] = {"kw": float(ov_var.get()), "emb": float(emb_var.get()), "rag": float(rag_var.get())}
            dom["accept_threshold"] = float(acc_var.get())
            dom["clarify_margin"] = float(clar_var.get())
            dom["always_accept_wake"] = bool(wake_var.get())
            self._append_log(
                f"Gestione Documenti: index={self.settings['docstore_path']} top_k={self.settings.get('retrieval_top_k')} topic={topic_var.get()}\n",
                "MISC",
            )
            self._append_log(
                f"Domain: topic={dom['topic']} kw={dom['keywords']} weights={dom['weights']} acc={dom['accept_threshold']} clar={dom['clarify_margin']} wake={dom['always_accept_wake']}\n",
                "DOMAIN",
            )
            self.controller.save_settings(self._append_log)
            self.base_settings = self.controller.base_settings
            self.local_settings = self.controller.local_settings
            self.settings = self.controller.settings
            win.destroy()

        ttk.Button(win, text="OK", command=on_ok).pack(pady=10)
    def _open_library_dialog(self) -> None:
        win = tk.Toplevel(self)
        win.title("Libreria")
        win.configure(bg=self._bg)

        tree = ttk.Treeview(win, columns=("tag", "date", "size"), show="headings")
        tree.heading("tag", text="Tag")
        tree.heading("date", text="Indicizzazione")
        tree.heading("size", text="Dim.")
        tree.pack(fill="both", expand=True, padx=6, pady=6)

        docs = self._load_doc_index()
        for d in docs:
            name = d.get("title") or d.get("id") or ""
            tag = ", ".join(d.get("tags", [])) or d.get("topic", "")
            date = d.get("date", "")
            size = len(d.get("text", ""))
            tree.insert("", "end", iid=str(d.get("id", name)), values=(tag, date, size), text=name)

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill="x", padx=6, pady=4)

        def make_collection() -> None:
            sel = tree.selection()
            if not sel:
                messagebox.showinfo("Collezione", "Seleziona almeno un documento")
                return
            names = [tree.item(i, "text") for i in sel]
            dom = self.settings.setdefault("domain", {})
            dom["topic"] = "\n".join(names)
            messagebox.showinfo("Collezione", "Topic aggiornato dalla selezione.")

        def clear_library() -> None:
            script = self._find_ingest_script()
            if not script:
                messagebox.showwarning(
                    "Libreria", "Script di ingest non trovato (scripts/ingest_docs.py)."
                )
                return
            path = self.settings.get("docstore_path", "DataBase/index.json")
            cmd = [sys.executable, "-m", "scripts.ingest_docs", "--clear"]
            if path:
                cmd.extend(["--path", path])
            try:
                proc = subprocess.run(
                    cmd,
                    check=True,
                    cwd=self.root_dir,
                    capture_output=True,
                    text=True,
                )
                self._append_log(proc.stdout, "DOCS")
                if proc.stderr:
                    self._append_log(proc.stderr, "DOCS")
                for i in tree.get_children():
                    tree.delete(i)
                messagebox.showinfo("Libreria", "Libreria svuotata.")
            except subprocess.CalledProcessError as exc:
                if exc.stdout:
                    self._append_log(exc.stdout, "DOCS")
                if exc.stderr:
                    self._append_log(exc.stderr, "DOCS")
                self._append_log(str(exc) + "\n", "DOCS")
                messagebox.showerror(
                    "Libreria", f"Impossibile svuotare la libreria: {exc}"
                )

        ttk.Button(btn_frame, text="Crea Collezione da selezione", command=make_collection).pack(side="left")
        ttk.Button(btn_frame, text="Svuota libreria", command=clear_library).pack(side="right")
        ttk.Button(btn_frame, text="Verifica indice", command=self._open_doc_manager_dialog).pack(side="right")

    def _load_doc_index(self) -> list[dict]:
        path = self.settings.get("docstore_path", "DataBase/index.json")
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return []
        if isinstance(data, dict) and "documents" in data:
            return list(data.get("documents", []))
        if isinstance(data, list):
            return data
        return []

    # --------------------------- Settings dialogs ------------------------- #
    def _update_debug(self) -> None:
        self.settings["debug"] = bool(self.debug_var.get())

    def _open_audio_dialog(self) -> None:
        win = tk.Toplevel(self)
        win.title("Dispositivi audio")
        win.configure(bg=self._bg)
        try:
            devs = sd.query_devices()
        except Exception as e:
            messagebox.showerror("Audio", f"Impossibile leggere i device: {e}")
            return

        options = []
        for idx, d in enumerate(devs):
            options.append(f"[{idx}] {d['name']}  (in:{d['max_input_channels']} out:{d['max_output_channels']})")

        audio = self.settings.setdefault("audio", {})
        cur_in = audio.get("input_device", None)
        cur_out = audio.get("output_device", None)
        rt_conf = self.settings.setdefault("realtime", {})
        rec_conf = self.settings.setdefault("recording", {})
        barge_var = tk.DoubleVar(value=float(rt_conf.get("barge_rms_threshold", 500.0)))
        vad_var = tk.DoubleVar(value=float(rec_conf.get("min_speech_level", 0.01)))

        def find_label_for_index(i: int | None) -> str:
            if i is None:
                return "(default di sistema)"
            if isinstance(i, int) and 0 <= i < len(options):
                return options[i]
            if isinstance(i, str):
                for s in options:
                    if i.lower() in s.lower():
                        return s
            return "(default di sistema)"

        in_var = tk.StringVar(value=find_label_for_index(cur_in))
        out_var = tk.StringVar(value=find_label_for_index(cur_out))

        tk.Label(win, text="Input", fg=self._fg, bg=self._bg).grid(row=0, column=0, padx=6, pady=6, sticky="e")
        tk.OptionMenu(win, in_var, "(default di sistema)", *options).grid(row=0, column=1, padx=6, pady=6, sticky="w")

        tk.Label(win, text="Output", fg=self._fg, bg=self._bg).grid(row=1, column=0, padx=6, pady=6, sticky="e")
        tk.OptionMenu(win, out_var, "(default di sistema)", *options).grid(row=1, column=1, padx=6, pady=6, sticky="w")

        tk.Label(win, text="Barge soglia", fg=self._fg, bg=self._bg).grid(row=2, column=0, padx=6, pady=6, sticky="e")
        ttk.Scale(win, from_=100, to=5000, orient="horizontal", length=200, variable=barge_var).grid(row=2, column=1, padx=6, pady=6, sticky="w")
        tk.Label(win, text="Sens. VAD", fg=self._fg, bg=self._bg).grid(row=3, column=0, padx=6, pady=6, sticky="e")
        ttk.Scale(win, from_=0.001, to=0.1, orient="horizontal", length=200, variable=vad_var).grid(row=3, column=1, padx=6, pady=6, sticky="w")
        ttk.Button(win, text="Test microfono", command=lambda: self._test_mic(label_to_index(in_var.get()))).grid(row=4, column=0, padx=6, pady=6)
        ttk.Button(win, text="Test altoparlanti", command=lambda: self._test_speakers(label_to_index(out_var.get()))).grid(row=4, column=1, padx=6, pady=6)

        def label_to_index(lbl: str) -> int | None:
            if not lbl or lbl.startswith("(default"):
                return None
            try:
                num = lbl.split("]", 1)[0].strip("[")
                return int(num)
            except Exception:
                return None

        def on_ok() -> None:
            audio["input_device"] = label_to_index(in_var.get())
            audio["output_device"] = label_to_index(out_var.get())
            self._append_log(
                f"Input: {in_var.get()}  Output: {out_var.get()}  "
                f"Barge={barge_var.get():.1f}  VAD={vad_var.get():.3f}",
                "AUDIO",
            )
            rt_conf["barge_rms_threshold"] = float(barge_var.get())
            rec_conf["min_speech_level"] = float(vad_var.get())
            # applica immediatamente
            sd.default.device = (audio["input_device"], audio["output_device"])
            self._append_log(
                f"Audio: in={audio['input_device']} out={audio['output_device']} barge={rt_conf['barge_rms_threshold']} vad={rec_conf['min_speech_level']}\n",
                "MISC",
            )
            win.destroy()

        ttk.Button(win, text="OK", command=on_ok).grid(row=5, column=0, columnspan=2, pady=10)

    def _test_mic(self, device_index: int | None) -> None:
        sr = int(self.settings.get("realtime", {}).get("sample_rate", SR))
        win = tk.Toplevel(self)
        win.title("Test microfono")
        win.configure(bg=self._bg)
        level_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(win, orient="horizontal", length=200, mode="determinate", maximum=1.0, variable=level_var).pack(
            padx=10, pady=10
        )
        running = True

        def callback(indata, frames, time_info, status) -> None:  # type: ignore[override]
            if not running:
                return
            samples = np.frombuffer(indata, dtype=np.int16).astype(np.float32)
            lvl = float(np.sqrt(np.mean(samples ** 2))) / 32768.0
            self.after(0, level_var.set, lvl)

        try:
            stream = sd.RawInputStream(
                samplerate=sr,
                blocksize=1024,
                channels=1,
                dtype="int16",
                callback=callback,
                device=device_index,
            )
            stream.start()
        except Exception as e:
            messagebox.showerror("Audio", f"Errore microfono: {e}")
            win.destroy()
            return

        def on_close() -> None:
            nonlocal running
            running = False
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

    def _test_speakers(self, device_index: int | None) -> None:
        sr = int(self.settings.get("realtime", {}).get("sample_rate", SR))
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        try:
            sd.play(tone, sr, device=device_index)
            sd.wait()
        except Exception as e:
            messagebox.showerror("Audio", f"Errore altoparlanti: {e}")

    def _open_recording_dialog(self) -> None:
        win = tk.Toplevel(self)
        win.title("Recording")
        win.configure(bg=self._bg)

        rec = self.settings.setdefault("recording", {})
        vad = self.settings.setdefault("vad", {})

        mode_var = tk.StringVar(value=rec.get("mode", "vad"))
        timed_var = tk.StringVar(value=str(rec.get("timed_seconds", 10)))
        fallback_var = tk.BooleanVar(value=bool(rec.get("fallback_to_timed", False)))
        minlevel_var = tk.StringVar(value=str(rec.get("min_speech_level", 0.01)))

        frame_ms = tk.StringVar(value=str(vad.get("frame_ms", 30)))
        start_ms = tk.StringVar(value=str(vad.get("start_ms", 150)))
        end_ms = tk.StringVar(value=str(vad.get("end_ms", 800)))
        max_ms = tk.StringVar(value=str(vad.get("max_ms", 15000)))
        preroll_ms = tk.StringVar(value=str(vad.get("preroll_ms", 300)))
        noise_win = tk.StringVar(value=str(vad.get("noise_window_ms", 800)))
        start_mult = tk.StringVar(value=str(vad.get("start_mult", 1.8)))
        end_mult = tk.StringVar(value=str(vad.get("end_mult", 1.3)))
        base_start = tk.StringVar(value=str(vad.get("base_start", 0.006)))
        base_end = tk.StringVar(value=str(vad.get("base_end", 0.0035)))

        r = 0

        def add_row(label, var):
            nonlocal r
            tk.Label(win, text=label, fg=self._fg, bg=self._bg).grid(row=r, column=0, padx=6, pady=4, sticky="e")
            tk.Entry(win, textvariable=var, width=10).grid(row=r, column=1, padx=6, pady=4, sticky="w")
            r += 1

        tk.Label(win, text="Mode (vad/timed)", fg=self._fg, bg=self._bg).grid(row=r, column=0, padx=6, pady=4, sticky="e")
        tk.OptionMenu(win, mode_var, "vad", "timed").grid(row=r, column=1, padx=6, pady=4, sticky="w")
        r += 1
        add_row("Timed seconds", timed_var)
        tk.Checkbutton(
            win,
            text="Fallback to timed in silenzio",
            variable=fallback_var,
            bg=self._bg,
            fg=self._fg,
            selectcolor="#222",
        ).grid(row=r, column=0, columnspan=2, padx=6, pady=4, sticky="w")
        r += 1
        add_row("Min speech level", minlevel_var)

        ttk.Separator(win).grid(row=r, column=0, columnspan=2, sticky="ew", pady=8)
        r += 1
        tk.Label(win, text="Parametri VAD", fg=self._fg, bg=self._bg, font=("Helvetica", 11, "bold")).grid(
            row=r, column=0, columnspan=2, pady=(0, 6)
        )
        r += 1

        for lab, var in [
            ("frame_ms", frame_ms),
            ("start_ms", start_ms),
            ("end_ms", end_ms),
            ("max_ms", max_ms),
            ("preroll_ms", preroll_ms),
            ("noise_window_ms", noise_win),
            ("start_mult", start_mult),
            ("end_mult", end_mult),
            ("base_start", base_start),
            ("base_end", base_end),
        ]:
            add_row(lab, var)

        def on_ok() -> None:
            rec["mode"] = mode_var.get().strip().lower()
            try:
                rec["timed_seconds"] = int(timed_var.get())
            except Exception:
                rec["timed_seconds"] = 10
            rec["fallback_to_timed"] = bool(fallback_var.get())
            try:
                rec["min_speech_level"] = float(minlevel_var.get())
            except Exception:
                rec["min_speech_level"] = 0.01

            for var, key, cast, default in [
                (frame_ms, "frame_ms", int, 30),
                (start_ms, "start_ms", int, 150),
                (end_ms, "end_ms", int, 800),
                (max_ms, "max_ms", int, 15000),
                (preroll_ms, "preroll_ms", int, 300),
                (noise_win, "noise_window_ms", int, 800),
                (start_mult, "start_mult", float, 1.8),
                (end_mult, "end_mult", float, 1.3),
                (base_start, "base_start", float, 0.006),
                (base_end, "base_end", float, 0.0035),
            ]:
                try:
                    vad[key] = cast(var.get())
                except Exception:
                    vad[key] = default
            self._append_log(
                f"Recording: mode={rec['mode']} timed={rec['timed_seconds']} fallback={rec['fallback_to_timed']} min_level={rec['min_speech_level']}\n",
                "MISC",
            )
            win.destroy()

        ttk.Button(win, text="OK", command=on_ok).grid(row=r, column=0, columnspan=2, pady=10)

    def _open_openai_dialog(self) -> None:
        win = tk.Toplevel(self)
        win.title("OpenAI")
        win.configure(bg=self._bg)

        openai_conf = self.settings.setdefault("openai", {})
        api_var = tk.StringVar(value=openai_conf.get("api_key", ""))
        stt_var = tk.StringVar(value=openai_conf.get("stt_model", ""))
        llm_var = tk.StringVar(value=openai_conf.get("llm_model", ""))
        tts_model_var = tk.StringVar(value=openai_conf.get("tts_model", ""))
        tts_voice_var = tk.StringVar(value=openai_conf.get("tts_voice", ""))

        rows = [
            ("API key", api_var),
            ("STT model", stt_var),
            ("LLM model", llm_var),
            ("TTS model", tts_model_var),
            ("TTS voice", tts_voice_var),
        ]
        for i, (lab, var) in enumerate(rows):
            tk.Label(win, text=lab, fg=self._fg, bg=self._bg).grid(row=i, column=0, padx=6, pady=6, sticky="e")
            show = "*" if lab == "API key" else None
            tk.Entry(win, textvariable=var, show=show, width=30).grid(row=i, column=1, padx=6, pady=6, sticky="w")

        def on_ok() -> None:
            openai_conf["api_key"] = api_var.get().strip()
            openai_conf["stt_model"] = stt_var.get().strip()
            openai_conf["llm_model"] = llm_var.get().strip()
            openai_conf["tts_model"] = tts_model_var.get().strip()
            openai_conf["tts_voice"] = tts_voice_var.get().strip()
            self._append_log("Chiave OpenAI aggiornata\n", "MISC")
            self._append_log(
                f"Modelli aggiornati: STT={stt_var.get()}, LLM={llm_var.get()}, "
                f"TTS={tts_model_var.get()} ({tts_voice_var.get()})\n",
                "MISC",
            )
            if openai_conf["api_key"]:
                self.chat_entry.configure(state="normal")
            win.destroy()

        ttk.Button(win, text="OK", command=on_ok).grid(row=len(rows), column=0, columnspan=2, pady=10)

    def _open_domain_dialog(self) -> None:
        win = tk.Toplevel(self)
        win.title("Argomenti & Regole")
        win.configure(bg=self._bg)

        dom = self.settings.setdefault("domain", {})

        topic_box = scrolledtext.ScrolledText(win, width=40, height=4)
        topic_box.insert("1.0", dom.get("topic", ""))
        tk.Label(win, text="Topic", fg=self._fg, bg=self._bg).grid(row=0, column=0, padx=6, pady=6, sticky="ne")
        topic_box.grid(row=0, column=1, padx=6, pady=6, sticky="w")

        kw_var = tk.StringVar(value=", ".join(dom.get("keywords", [])))
        tk.Label(win, text="Keywords", fg=self._fg, bg=self._bg).grid(row=1, column=0, padx=6, pady=6, sticky="e")
        tk.Entry(win, textvariable=kw_var, width=40).grid(row=1, column=1, padx=6, pady=6, sticky="w")

        weights = dom.get("weights", {})
        ov_var = tk.DoubleVar(value=weights.get("kw", 0.4))
        emb_var = tk.DoubleVar(value=weights.get("emb", 0.3))
        rag_var = tk.DoubleVar(value=weights.get("rag", 0.3))
        tk.Label(win, text="Peso Overlap", fg=self._fg, bg=self._bg).grid(row=2, column=0, padx=6, pady=6, sticky="e")
        ttk.Scale(win, from_=0, to=1, variable=ov_var, orient="horizontal", length=180).grid(row=2, column=1, padx=6, pady=6, sticky="w")
        tk.Label(win, text="Peso Embedding", fg=self._fg, bg=self._bg).grid(row=3, column=0, padx=6, pady=6, sticky="e")
        ttk.Scale(win, from_=0, to=1, variable=emb_var, orient="horizontal", length=180).grid(row=3, column=1, padx=6, pady=6, sticky="w")
        tk.Label(win, text="Peso RAG", fg=self._fg, bg=self._bg).grid(row=4, column=0, padx=6, pady=6, sticky="e")
        ttk.Scale(win, from_=0, to=1, variable=rag_var, orient="horizontal", length=180).grid(row=4, column=1, padx=6, pady=6, sticky="w")

        acc_var = tk.DoubleVar(value=dom.get("accept_threshold", 0.5))
        clar_var = tk.DoubleVar(value=dom.get("clarify_margin", 0.15))
        tk.Label(win, text="Accept threshold", fg=self._fg, bg=self._bg).grid(row=5, column=0, padx=6, pady=6, sticky="e")
        ttk.Scale(win, from_=0, to=1, variable=acc_var, orient="horizontal", length=180).grid(row=5, column=1, padx=6, pady=6, sticky="w")
        tk.Label(win, text="Clarify margin", fg=self._fg, bg=self._bg).grid(row=6, column=0, padx=6, pady=6, sticky="e")
        ttk.Scale(win, from_=0, to=1, variable=clar_var, orient="horizontal", length=180).grid(row=6, column=1, padx=6, pady=6, sticky="w")

        wake_var = tk.BooleanVar(value=dom.get("always_accept_wake", True))
        tk.Checkbutton(
            win,
            text="Accetta sempre saluti/wake words",
            variable=wake_var,
            bg=self._bg,
            fg=self._fg,
            selectcolor=self._bg,
        ).grid(row=7, column=0, columnspan=2, padx=6, pady=6, sticky="w")

        def on_ok() -> None:
            dom["topic"] = topic_box.get("1.0", "end").strip()
            dom["keywords"] = [k.strip() for k in kw_var.get().split(",") if k.strip()]
            self._append_log(
                "Keywords aggiornate: " + ", ".join(dom["keywords"]),
                "DOMAIN",
            )
            dom["weights"] = {"kw": float(ov_var.get()), "emb": float(emb_var.get()), "rag": float(rag_var.get())}
            dom["accept_threshold"] = float(acc_var.get())
            dom["clarify_margin"] = float(clar_var.get())
            dom["always_accept_wake"] = bool(wake_var.get())
            self._append_log(
                f"Domain: topic={dom['topic']} kw={dom['keywords']} weights={dom['weights']} acc={dom['accept_threshold']} clar={dom['clarify_margin']} wake={dom['always_accept_wake']}\n",
                "DOMAIN",
            )
            self.controller.save_settings(self._append_log)
            self.base_settings = self.controller.base_settings
            self.local_settings = self.controller.local_settings
            self.settings = self.controller.settings
            win.destroy()

        ttk.Button(win, text="OK", command=on_ok).grid(row=8, column=0, columnspan=2, pady=10)

    def _open_wake_dialog(self) -> None:
        win = tk.Toplevel(self)
        win.title("Inattività & Wake")
        win.configure(bg=self._bg)

        wake = self.settings.setdefault("wake", {})
        timeout_var = tk.IntVar(value=int(wake.get("idle_timeout", 50)))
        it_var = tk.StringVar(value=", ".join(wake.get("it_phrases", [])))
        en_var = tk.StringVar(value=", ".join(wake.get("en_phrases", [])))

        tk.Label(win, text="Timeout (s)", fg=self._fg, bg=self._bg).grid(row=0, column=0, padx=6, pady=6, sticky="e")
        tk.Scale(win, from_=10, to=300, orient="horizontal", variable=timeout_var, length=200).grid(row=0, column=1, padx=6, pady=6, sticky="w")
        tk.Label(win, text="Wake IT", fg=self._fg, bg=self._bg).grid(row=1, column=0, padx=6, pady=6, sticky="e")
        tk.Entry(win, textvariable=it_var, width=40).grid(row=1, column=1, padx=6, pady=6, sticky="w")
        tk.Label(win, text="Wake EN", fg=self._fg, bg=self._bg).grid(row=2, column=0, padx=6, pady=6, sticky="e")
        tk.Entry(win, textvariable=en_var, width=40).grid(row=2, column=1, padx=6, pady=6, sticky="w")

        def on_ok() -> None:
            wake["idle_timeout"] = int(timeout_var.get())
            wake["it_phrases"] = [p.strip() for p in it_var.get().split(",") if p.strip()]
            wake["en_phrases"] = [p.strip() for p in en_var.get().split(",") if p.strip()]
            self._append_log(
                f"Wake timeout={timeout_var.get()} IT={it_var.get()} EN={en_var.get()}",
                "WAKE",
            )
            win.destroy()

        ttk.Button(win, text="OK", command=on_ok).grid(row=3, column=0, columnspan=2, pady=10)

    def _open_lighting_dialog(self) -> None:
        win = tk.Toplevel(self)
        win.title("Luci")
        win.configure(bg=self._bg)

        lighting = self.settings.setdefault("lighting", {})
        mode_var = tk.StringVar(value=lighting.get("mode", "sacn"))

        tk.Label(win, text="Modalità", fg=self._fg, bg=self._bg).grid(row=0, column=0, padx=6, pady=6, sticky="e")
        tk.OptionMenu(win, mode_var, "sacn", "wled").grid(row=0, column=1, padx=6, pady=6, sticky="w")

        sacn_frame = tk.Frame(win, bg=self._bg)
        sacn_conf = lighting.setdefault("sacn", {})
        sacn_ip = tk.StringVar(value=sacn_conf.get("destination_ip", ""))
        sacn_uni = tk.StringVar(value=str(sacn_conf.get("universe", 1)))
        sacn_idle = tk.StringVar(value=str(sacn_conf.get("idle_level", 10)))
        sacn_peak = tk.StringVar(value=str(sacn_conf.get("peak_level", 255)))

        for i, (lab, var) in enumerate(
            [("IP", sacn_ip), ("Universe", sacn_uni), ("Idle", sacn_idle), ("Peak", sacn_peak)]
        ):
            tk.Label(sacn_frame, text=lab, fg=self._fg, bg=self._bg).grid(row=i, column=0, padx=6, pady=4, sticky="e")
            tk.Entry(sacn_frame, textvariable=var, width=18).grid(row=i, column=1, padx=6, pady=4, sticky="w")

        wled_frame = tk.Frame(win, bg=self._bg)
        wled_conf = lighting.setdefault("wled", {})
        wled_host = tk.StringVar(value=wled_conf.get("host", ""))

        tk.Label(wled_frame, text="Host", fg=self._fg, bg=self._bg).grid(row=0, column=0, padx=6, pady=6, sticky="e")
        tk.Entry(wled_frame, textvariable=wled_host, width=22).grid(row=0, column=1, padx=6, pady=6, sticky="w")

        def update_mode(*_):
            sacn_frame.grid_remove()
            wled_frame.grid_remove()
            if mode_var.get() == "sacn":
                sacn_frame.grid(row=1, column=0, columnspan=2, padx=6, pady=6, sticky="w")
            else:
                wled_frame.grid(row=1, column=0, columnspan=2, padx=6, pady=6, sticky="w")

        update_mode()
        mode_var.trace_add("write", update_mode)

        def on_ok() -> None:
            lighting["mode"] = mode_var.get()
            sacn_conf["destination_ip"] = sacn_ip.get().strip()
            try:
                sacn_conf["universe"] = int(sacn_uni.get())
            except Exception:
                sacn_conf["universe"] = 1
            try:
                sacn_conf["idle_level"] = int(sacn_idle.get())
            except Exception:
                pass
            try:
                sacn_conf["peak_level"] = int(sacn_peak.get())
            except Exception:
                pass
            wled_conf["host"] = wled_host.get().strip()
            self._append_log(
                f"Lighting: mode={lighting['mode']} sacn={sacn_conf} wled={wled_conf}\n",
                "MISC",
            )
            win.destroy()

        ttk.Button(win, text="OK", command=on_ok).grid(row=3, column=0, columnspan=2, pady=10)

    # ------------------------------ Save ----------------------------------- #
    def save_settings(self) -> None:
        self.settings["debug"] = bool(self.debug_var.get())
        self.settings["topic_threshold"] = float(self.topic_threshold.get())
        self.settings["keywords"] = self.keywords
        try:
            self.controller.save_settings(self._append_log)
            self.base_settings = self.controller.base_settings
            self.local_settings = self.controller.local_settings
            self.settings = self.controller.settings
            self.keywords = list(self.settings.get("keywords", []))
            self.keyword_listbox.delete(0, "end")
            for kw in self.keywords:
                self.keyword_listbox.insert("end", kw)
            self.topic_threshold.set(float(self.settings.get("topic_threshold", self.topic_threshold.get())))
            messagebox.showinfo("Impostazioni", "Salvate correttamente.")
            self._append_log(
                f"Impostazioni salvate: topic_threshold={self.settings.get('topic_threshold')} debug={self.settings.get('debug')}\n",
                "MISC",
            )
        except Exception as e:
            messagebox.showerror("Impostazioni", f"Errore nel salvataggio: {e}")

    # --------------------------- Start / Stop + logs ----------------------- #
    def start_oracolo(self) -> None:
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Oracolo", "È già in esecuzione.")
            return
        audio_cfg = self.settings.get("audio", {})
        try:
            validate_device_config(audio_cfg)
        except ValueError as e:
            messagebox.showerror("Audio", str(e))
            logging.error("Invalid device configuration: %s", e)
            return
        try:
            self._clear_log()
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"

            env["ORACOLO_STYLE"] = "poetic" if self.style_var.get() else "plain"
            env["ORACOLO_LANG"] = self.lang_map.get(self.lang_choice.get(), "auto")
            env["ORACOLO_ANSWER_MODE"] = self.mode_map.get(self.mode_choice.get(), "detailed")
            self.settings["answer_mode"] = env["ORACOLO_ANSWER_MODE"]

            use_quiet = not bool(self.settings.get("debug", False))

            args = [sys.executable, "-u", "-m", "src.main", "--autostart"]
            if use_quiet:
                args.append("--quiet")

            self.proc = subprocess.Popen(
                args,
                cwd=self.root_dir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0,
                env=env,
            )
            self.status_var.set("🟢 In esecuzione")
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")

            self._stop_reader.clear()
            self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
            self._reader_thread.start()
        except Exception as e:
            messagebox.showerror("Avvio", f"Impossibile avviare l'oracolo: {e}")

    def _read_stdout(self) -> None:
        if not self.proc or not self.proc.stdout:
            return
        f = self.proc.stdout
        while not self._stop_reader.is_set():
            line = f.readline()
            if not line:
                if self.proc and self.proc.poll() is not None:
                    break
                time.sleep(0.05)
                continue
            text = line.rstrip("\n")
            try:
                data = json.loads(text)
                text = str(data.get("text", ""))
            except Exception:
                pass
            self.after(
                0,
                lambda t=text: self._append_log(
                    t + ("\n" if not t.endswith("\n") else ""), "LLM"
                ),
            )
            if text.startswith("…"):
                msg = text.lstrip("…").strip()
                self.conv.push_user(msg)
                self.after(0, lambda m=msg: self._append_chat("user", m))
            elif text.startswith("🔮"):
                msg = text.lstrip("🔮").strip()
                self.conv.push_assistant(msg)
                self.after(0, lambda m=msg: self._append_chat("assistant", m))
        rest = f.read()
        if rest:
            self.after(0, lambda rest=rest: self._append_log(rest, "LLM"))

    def stop_oracolo(self) -> None:
        if not self.proc or self.proc.poll() is not None:
            self.status_var.set("🟡 In attesa")
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            return
        try:
            self._stop_reader.set()
            self.proc.terminate()
            for _ in range(40):
                if self.proc.poll() is not None:
                    break
                time.sleep(0.1)
            if self.proc.poll() is None:
                self.proc.kill()
        except Exception:
            pass
        finally:
            self.status_var.set("🟡 In attesa")
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            try:
                if self._reader_thread and self._reader_thread.is_alive():
                    self._reader_thread.join(timeout=0.5)
            except Exception:
                pass
            self._reader_thread = None

    # ----------------------- Realtime WS server ------------------------- #
    def _read_ws_server_stdout(self) -> None:
        if not self.ws_server_proc or not self.ws_server_proc.stdout:
            return
        f = self.ws_server_proc.stdout
        while not self._ws_stop_reader.is_set():
            line = f.readline()
            if not line:
                if self.ws_server_proc and self.ws_server_proc.poll() is not None:
                    break
                time.sleep(0.05)
                continue
            text = line.rstrip("\n")
            self.after(0, lambda t=text: self._append_log(f"[WS-SRV] {t}\n", "WS"))
        rest = f.read()
        if rest:
            self.after(0, lambda rest=rest: self._append_log(f"[WS-SRV] {rest}", "WS"))

    def start_ws_server(self) -> None:
        if self.ws_server_proc and self.ws_server_proc.poll() is None:
            messagebox.showinfo("Server WS", "È già in esecuzione.")
            return
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            args = [sys.executable, "-u", "scripts/realtime_server.py"]
            self.ws_server_proc = subprocess.Popen(
                args,
                cwd=self.root_dir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0,
                env=env,
            )
            self.status_var.set("🟢 Server WS in esecuzione")
            if hasattr(self, "server_menu"):
                self.server_menu.entryconfig(self._srv_start_idx, state="disabled")
                self.server_menu.entryconfig(self._srv_stop_idx, state="normal")
            self._ws_stop_reader.clear()
            self._ws_server_thread = threading.Thread(target=self._read_ws_server_stdout, daemon=True)
            self._ws_server_thread.start()
        except Exception as e:
            messagebox.showerror("Server WS", f"Impossibile avviare il server: {e}")
            self.ws_server_proc = None

    def stop_ws_server(self) -> None:
        if not self.ws_server_proc or self.ws_server_proc.poll() is not None:
            if hasattr(self, "server_menu"):
                self.server_menu.entryconfig(self._srv_start_idx, state="normal")
                self.server_menu.entryconfig(self._srv_stop_idx, state="disabled")
            if not (self.proc and self.proc.poll() is None) and self.ws_client is None:
                self.status_var.set("🟡 In attesa")
            return
        try:
            self._ws_stop_reader.set()
            self.ws_server_proc.terminate()
            for _ in range(40):
                if self.ws_server_proc.poll() is not None:
                    break
                time.sleep(0.1)
            if self.ws_server_proc.poll() is None:
                self.ws_server_proc.kill()
        except Exception:
            pass
        finally:
            self.ws_server_proc = None
            if hasattr(self, "server_menu"):
                self.server_menu.entryconfig(self._srv_start_idx, state="normal")
                self.server_menu.entryconfig(self._srv_stop_idx, state="disabled")
            try:
                if self._ws_server_thread and self._ws_server_thread.is_alive():
                    self._ws_server_thread.join(timeout=0.5)
            except Exception:
                pass
            self._ws_server_thread = None
            if not (self.proc and self.proc.poll() is None) and self.ws_client is None:
                self.status_var.set("🟡 In attesa")

    # ----------------------- Realtime WS control ------------------------- #
    def _on_ws_event(self, ev: str) -> None:
        self._append_log(f"[WS] {ev}\n", "WS")
        if ev == "connected":
            self.ping_var.set("Ping: handshake")
        elif ev == "disconnected":
            self.ping_var.set("Ping: -")
            self.in_level.set(0.0)
            self.out_level.set(0.0)
        elif ev.startswith("error") or ev.startswith("handshake_error"):
            self.ping_var.set("Ping: errore")

    def start_realtime(self) -> None:
        if self.ws_client is not None:
            messagebox.showinfo("Realtime", "Sessione già attiva.")
            return

        # applica dispositivi audio da settings come default
        audio = self.settings.get("audio", {})
        try:
            validate_device_config(audio)
        except ValueError as e:
            messagebox.showerror("Audio", str(e))
            logging.error("Invalid device configuration: %s", e)
            return
        in_dev = audio.get("input_device", None)
        out_dev = audio.get("output_device", None)
        sd.default.device = (in_dev, out_dev)

        rt_conf = self.settings.get("realtime", {})
        url = rt_conf.get("ws_url", WS_URL)
        sr = int(rt_conf.get("sample_rate", SR))
        barge = float(rt_conf.get("barge_rms_threshold", 500.0))
        ping_interval = int(rt_conf.get("ping_interval", 20))
        ping_timeout = int(rt_conf.get("ping_timeout", 20))

        self._append_log(
            f"🔌 Realtime WS → {url}  (sr={sr}, in={sd.default.device[0]}, out={sd.default.device[1]})\n",
            "WS",
        )

        def _handle_partial(text: str, final: bool) -> None:
            self._append_log(f"… {text}\n", "STT")
            if final:
                self._append_chat("user", text)
                self.conv.push_user(text)

        def _handle_answer(text: str) -> None:
            self._append_log(f"🔮 {text}\n", "LLM")
            self._append_chat("assistant", text)
            self.conv.push_assistant(text)

        self.ws_client = RealtimeWSClient(
            url,
            sr,
            on_partial=lambda text, final: self.after(
                0, lambda t=text, f=final: _handle_partial(t, f)
            ),
            on_answer=lambda text: self.after(0, lambda t=text: _handle_answer(t)),
            barge_threshold=barge,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            auto_reconnect=bool(self.ws_reconnect_var.get()),
            on_input_level=lambda lvl: self.after(0, self.in_level.set, lvl),
            on_output_level=lambda lvl: self.after(0, self.out_level.set, lvl),
            on_ping=lambda ms: self.after(0, self.ping_var.set, f"Ping: {ms:.0f} ms"),
            on_event=lambda ev: self.after(0, self._on_ws_event, ev),
            profile_name=self.profile_var.get(),
        )
        try:
            self.ws_client.start()
            self.ws_start_btn.configure(state="disabled")
            self.ws_stop_btn.configure(state="normal")
            self.status_var.set("🟢 In esecuzione (realtime)")
        except Exception as e:
            messagebox.showerror("Realtime", f"Impossibile avviare il WS: {e}")
            self.ws_client = None

    def stop_realtime(self) -> None:
        if self.ws_client is None:
            return
        try:
            self.ws_client.stop()
        finally:
            self.ws_client = None
            self.ws_start_btn.configure(state="normal")
            self.ws_stop_btn.configure(state="disabled")
            self.in_level.set(0.0)
            self.out_level.set(0.0)
            self.ping_var.set("Ping: -")
            if not (self.proc and self.proc.poll() is None):
                self.status_var.set("🟡 In attesa")

    def _poll_process(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
        else:
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")

        if self.ws_server_proc is not None and self.ws_server_proc.poll() is None:
            if hasattr(self, "server_menu"):
                self.server_menu.entryconfig(self._srv_start_idx, state="disabled")
                self.server_menu.entryconfig(self._srv_stop_idx, state="normal")
        else:
            self.ws_server_proc = None
            if hasattr(self, "server_menu"):
                self.server_menu.entryconfig(self._srv_start_idx, state="normal")
                self.server_menu.entryconfig(self._srv_stop_idx, state="disabled")

        if self.proc is not None and self.proc.poll() is None:
            self.status_var.set("🟢 In esecuzione")
        elif self.ws_client is not None:
            self.status_var.set("🟢 In esecuzione (realtime)")
        elif self.ws_server_proc is not None and self.ws_server_proc.poll() is None:
            self.status_var.set("🟢 Server WS in esecuzione")
        else:
            self.status_var.set("🟡 In attesa")

        self.after(500, self._poll_process)

    # ------------------------------ Exit ----------------------------------- #
    def _on_close(self) -> None:
        try:
            self.stop_oracolo()
            self.stop_realtime()
            self.stop_ws_server()
        finally:
            try:
                self._async_loop.call_soon_threadsafe(self._async_loop.stop)
            except Exception:
                pass
            self.destroy()


# Alias per accesso esterno al gestore di log della UI
UILogHandler = OracoloUI.UILogHandler

# ----------------------------- entry point -------------------------------- #
def main() -> None:
    state = UIState()
    controller = UIController(state)
    app = OracoloUI(state=state, controller=controller)
    app.mainloop()


if __name__ == "__main__":
    main()
