# src/ui.py
"""
UI Tkinter per Occhio Onniveggente.

- Carica settings.yaml + settings.local.yaml (overlay).
- Salvataggio "split": debug e audio.(input/output)_device -> settings.local.yaml,
  tutto il resto -> settings.yaml.
- Menu Documenti: Aggiungi / Rimuovi / Aggiorna indice.
- Avvio/Stop di src.main in modalità --autostart con log in tempo reale.
- Controllo client Realtime WebSocket (start/stop) con streaming audio.
"""

from __future__ import annotations

import copy
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk
from typing import Any

from src.retrieval import retrieve

import asyncio
import numpy as np
import sounddevice as sd
import websockets
import yaml

try:  # opzionale: drag&drop documenti
    import tkinterdnd2 as tkdnd  # type: ignore
except Exception:  # pragma: no cover - fallback se non disponibile
    tkdnd = None

WS_URL = os.getenv("ORACOLO_WS_URL", "ws://localhost:8765")
SR = 24_000

# opzionale: compatibilità icona PNG via Pillow
try:
    from PIL import Image, ImageTk  # pip install pillow
except Exception:
    Image = None
    ImageTk = None


# ------------------------------ helpers ---------------------------------- #
def deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def deep_copy(d: dict) -> dict:
    return copy.deepcopy(d or {})


def load_settings_pair(root: Path) -> tuple[dict, dict, dict]:
    """Ritorna (base, local, merged)."""
    base_p = root / "settings.yaml"
    local_p = root / "settings.local.yaml"

    base = {}
    local = {}
    if base_p.exists():
        base = yaml.safe_load(base_p.read_text(encoding="utf-8")) or {}
    if local_p.exists():
        local = yaml.safe_load(local_p.read_text(encoding="utf-8")) or {}

    merged = deep_copy(base)
    merged = deep_update(merged, deep_copy(local))
    return base, local, merged


def routed_save(base_now: dict, local_now: dict, merged_new: dict, root: Path) -> None:
    """
    Salva split:
      - In settings.local.yaml: debug, audio.input_device, audio.output_device
      - In settings.yaml: tutto il resto
    Mantiene eventuali altre chiavi locali preesistenti.
    """
    local_out = deep_copy(local_now)
    local_out.setdefault("audio", {})
    local_out["debug"] = bool(merged_new.get("debug", local_out.get("debug", False)))

    audio_new = deep_copy(merged_new.get("audio", {}))
    if "input_device" in audio_new:
        local_out["audio"]["input_device"] = audio_new.get(
            "input_device", local_out["audio"].get("input_device")
        )
    if "output_device" in audio_new:
        local_out["audio"]["output_device"] = audio_new.get(
            "output_device", local_out["audio"].get("output_device")
        )

    base_out = deep_copy(merged_new)
    base_out.pop("debug", None)
    if "audio" in base_out:
        base_out["audio"].pop("input_device", None)
        base_out["audio"].pop("output_device", None)

    (root / "settings.yaml").write_text(
        yaml.safe_dump(base_out, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (root / "settings.local.yaml").write_text(
        yaml.safe_dump(local_out, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


# ------------------------- Realtime WS client ---------------------------- #
class RealtimeWSClient:
    """Gestisce la connessione WebSocket realtime con audio bidirezionale."""

    def __init__(self, url: str, sr: int, on_partial, on_answer) -> None:
        self.url = url
        self.sr = sr
        self.frame_samples = int(self.sr * 0.02)
        self.on_partial = on_partial
        self.on_answer = on_answer
        self.send_q: "queue.Queue[bytes]" = queue.Queue()
        self.audio_q: "queue.Queue[bytes]" = queue.Queue()
        self.state: dict[str, Any] = {"tts_playing": False, "barge_sent": False}
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.ws = None
        self.stop_event: asyncio.Event | None = None

    async def _mic_worker(self) -> None:
        assert self.ws is not None and self.stop_event is not None
        loop = asyncio.get_running_loop()

        def callback(indata, frames, time_info, status) -> None:  # type: ignore[override]
            data = bytes(indata)
            self.send_q.put_nowait(data)
            if self.state.get("tts_playing") and not self.state.get("barge_sent"):
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                level = float(np.sqrt(np.mean(samples ** 2)))
                if level > 500:
                    self.state["barge_sent"] = True
                    asyncio.run_coroutine_threadsafe(
                        self.ws.send(json.dumps({"type": "barge_in"})), loop
                    )

        with sd.RawInputStream(
            samplerate=self.sr,
            blocksize=self.frame_samples,
            channels=1,
            dtype="int16",
            callback=callback,
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
            if kind == "partial":
                self.on_partial(data.get("text", ""))
            elif kind == "answer":
                self.on_answer(data.get("text", ""))
            if self.stop_event.is_set():
                break

    async def _player(self) -> None:
        assert self.stop_event is not None

        def callback(outdata, frames, time_info, status) -> None:  # type: ignore[override]
            try:
                chunk = self.audio_q.get_nowait()
                n = len(outdata)
                if len(chunk) >= n:
                    outdata[:] = chunk[:n]
                else:
                    outdata[:len(chunk)] = chunk
                    outdata[len(chunk):] = b"\x00" * (n - len(chunk))
                self.state["tts_playing"] = True
            except queue.Empty:
                outdata[:] = b"\x00" * len(outdata)
                self.state["tts_playing"] = False
                self.state["barge_sent"] = False

        with sd.RawOutputStream(
            samplerate=self.sr,
            blocksize=self.frame_samples,
            channels=1,
            dtype="int16",
            callback=callback,
        ):
            while not self.stop_event.is_set():
                await asyncio.sleep(0.1)

    async def _run(self) -> None:
        self.stop_event = asyncio.Event()
        async with websockets.connect(self.url, ping_interval=20, ping_timeout=20) as ws:
            self.ws = ws
            print(
                f"🔌 Realtime WS → {self.url}  (sr={self.sr}, in={sd.default.device[0]}, out={sd.default.device[1]})",
                flush=True,
            )
            # Handshake
            await ws.send(json.dumps({"type": "hello", "sr": self.sr, "format": "pcm_s16le", "channels": 1}))
            try:
                ready_raw = await asyncio.wait_for(ws.recv(), timeout=10)
                data = json.loads(ready_raw) if isinstance(ready_raw, str) else {}
                if data.get("type") != "ready":
                    print("Handshake non valido", flush=True)
                    return
            except Exception:
                print("Handshake non valido", flush=True)
                return

            print("✅ pronto a ricevere audio", flush=True)

            tasks = [
                asyncio.create_task(self._mic_worker()),
                asyncio.create_task(self._sender()),
                asyncio.create_task(self._receiver()),
                asyncio.create_task(self._player()),
            ]
            await self.stop_event.wait()
            for t in tasks:
                t.cancel()

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

    def __init__(self) -> None:
        super().__init__()
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

        # settings
        self.base_settings, self.local_settings, self.settings = load_settings_pair(self.root_dir)

        # quick toggles state
        self.lang_map = {"Auto": "auto", "IT": "it", "EN": "en"}
        self.mode_map = {"Dettagliata": "detailed", "Concisa": "concise"}
        self.lang_choice = tk.StringVar(value="Auto")
        default_mode = (
            "Dettagliata" if self.settings.get("answer_mode", "detailed") == "detailed" else "Concisa"
        )
        self.mode_choice = tk.StringVar(value=default_mode)
        self.style_var = tk.BooleanVar(value=True)

        # process & logs
        self.proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop_reader = threading.Event()

        # realtime client
        self.ws_client: RealtimeWSClient | None = None

        # UI
        self._build_menubar()
        self._build_layout()

        # poll
        self.after(500, self._poll_process)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # --------------------------- Menubar & dialogs ------------------------ #
    def _build_menubar(self) -> None:
        menubar = tk.Menu(self)

        # Documenti
        docs_menu = tk.Menu(menubar, tearoff=0)
        docs_menu.add_command(label="Aggiungi…", command=self._add_documents)
        docs_menu.add_command(label="Rimuovi…", command=self._remove_documents)
        docs_menu.add_separator()
        docs_menu.add_command(label="Aggiorna indice…", command=self._reindex_documents)
        menubar.add_cascade(label="Documenti", menu=docs_menu)

        # Impostazioni
        settings_menu = tk.Menu(menubar, tearoff=0)
        self.debug_var = tk.BooleanVar(value=bool(self.settings.get("debug", False)))
        settings_menu.add_checkbutton(label="Debug", variable=self.debug_var, command=self._update_debug)
        settings_menu.add_command(label="Audio…", command=self._open_audio_dialog)
        settings_menu.add_command(label="Recording…", command=self._open_recording_dialog)
        settings_menu.add_command(label="Luci…", command=self._open_lighting_dialog)
        settings_menu.add_command(label="OpenAI…", command=self._open_openai_dialog)
        settings_menu.add_command(label="Dominio…", command=self._open_domain_dialog)
        settings_menu.add_command(label="Conoscenza…", command=self._open_knowledge_dialog)
        settings_menu.add_separator()
        settings_menu.add_command(label="Salva", command=self.save_settings)
        menubar.add_cascade(label="Impostazioni", menu=settings_menu)

        self.config(menu=menubar)

    # --------------------------- Layout / Widgets ------------------------- #
    def _build_layout(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill="x", padx=16, pady=(12, 8))
        ttk.Label(header, text="Occhio Onniveggente", font=("Helvetica", 18, "bold")).pack(side="left")

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

        self.status_var = tk.StringVar(value="🟡 In attesa")
        ttk.Label(bar, textvariable=self.status_var).pack(side="right")

        opts = ttk.Frame(self)
        opts.pack(fill="x", padx=16, pady=(0, 8))
        ttk.Checkbutton(opts, text="Stile poetico", variable=self.style_var).pack(side="left")
        ttk.Label(opts, text="Lingua:").pack(side="left", padx=(8, 4))
        self.lang_cb = ttk.Combobox(opts, textvariable=self.lang_choice, values=list(self.lang_map.keys()), state="readonly", width=7)
        self.lang_cb.pack(side="left")
        ttk.Label(opts, text="Risposta:").pack(side="left", padx=(8, 4))
        self.mode_cb = ttk.Combobox(opts, textvariable=self.mode_choice, values=list(self.mode_map.keys()), state="readonly", width=11)
        self.mode_cb.pack(side="left")

        log_frame = ttk.Frame(self)
        log_frame.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        self.log = scrolledtext.ScrolledText(
            log_frame,
            wrap="word",
            state="disabled",
            height=22,
            bg=self._mid,
            fg="#d7fff9",
            insertbackground=self._fg,
            relief="flat",
            font=("Consolas", 10),
        )
        self.log.pack(fill="both", expand=True)

        footer = ttk.Frame(self)
        footer.pack(fill="x", padx=16, pady=(0, 12))
        ttk.Button(footer, text="Esci", command=self._on_close).pack(side="right")

    # --------------------------- Log helpers ------------------------------ #
    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    # --------------------------- Document actions ------------------------- #
    def _find_ingest_script(self) -> Path | None:
        p = self.root_dir / "scripts" / "ingest_docs.py"
        return p if p.exists() else None

    def _add_documents(self) -> None:
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
        try:
            subprocess.run(
                [sys.executable, "-m", "scripts.ingest_docs", "--add", *paths],
                check=True,
                cwd=self.root_dir,
            )
            messagebox.showinfo("Successo", "Documenti aggiunti.")
        except subprocess.CalledProcessError as exc:
            messagebox.showerror("Errore", f"Ingest fallito: {exc}")

    def _remove_documents(self) -> None:
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
        try:
            subprocess.run(
                [sys.executable, "-m", "scripts.ingest_docs", "--remove", *paths],
                check=True,
                cwd=self.root_dir,
            )
            messagebox.showinfo("Successo", "Documenti rimossi.")
        except subprocess.CalledProcessError as exc:
            messagebox.showerror("Errore", f"Rimozione fallita: {exc}")

    def _reindex_documents(self) -> None:
        script = self._find_ingest_script()
        if not script:
            messagebox.showwarning("Indice", "Script di ingest non trovato (scripts/ingest_docs.py).")
            return
        tried = (["--reindex"], ["--rebuild"], ["--refresh"])
        for args in tried:
            try:
                subprocess.run(
                    [sys.executable, "-m", "scripts.ingest_docs", *args],
                    check=True,
                    cwd=self.root_dir,
                )
                messagebox.showinfo("Indice", f"Indice aggiornato ({' '.join(args)}).")
                return
            except subprocess.CalledProcessError:
                continue
        messagebox.showerror("Indice", "Impossibile aggiornare l'indice (nessuna delle opzioni supportata).")

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
            # applica immediatamente
            sd.default.device = (audio["input_device"], audio["output_device"])
            win.destroy()

        ttk.Button(win, text="OK", command=on_ok).grid(row=2, column=0, columnspan=2, pady=10)

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
            win.destroy()

        ttk.Button(win, text="OK", command=on_ok).grid(row=len(rows), column=0, columnspan=2, pady=10)

    def _open_domain_dialog(self) -> None:
        win = tk.Toplevel(self)
        win.title("Dominio")
        win.configure(bg=self._bg)

        dom = self.settings.setdefault("domain", {})
        presets = self.settings.setdefault("domain_presets", {})
        preset_var = tk.StringVar()

        tk.Label(win, text="Preset", fg=self._fg, bg=self._bg).grid(row=0, column=0, padx=6, pady=6, sticky="e")
        preset_cb = ttk.Combobox(win, textvariable=preset_var, values=list(presets.keys()), state="readonly", width=20)
        preset_cb.grid(row=0, column=1, padx=6, pady=6, sticky="w")

        kw_var = tk.StringVar(value=", ".join(dom.get("keywords", [])))
        rigid_var = tk.StringVar(value=str(dom.get("accept_threshold", 0.5)))

        tk.Label(win, text="Keyword (separate da virgola)", fg=self._fg, bg=self._bg).grid(
            row=1, column=0, padx=6, pady=6, sticky="e"
        )
        tk.Entry(win, textvariable=kw_var, width=40).grid(row=1, column=1, padx=6, pady=6, sticky="w")

        tk.Label(win, text="Prompt", fg=self._fg, bg=self._bg).grid(
            row=2, column=0, padx=6, pady=6, sticky="ne"
        )
        prompt_box = scrolledtext.ScrolledText(win, width=40, height=6)
        prompt_box.insert("1.0", self.settings.get("oracle_system", ""))
        prompt_box.grid(row=2, column=1, padx=6, pady=6, sticky="w")

        tk.Label(win, text="Rigidità", fg=self._fg, bg=self._bg).grid(
            row=3, column=0, padx=6, pady=6, sticky="e"
        )
        tk.Entry(win, textvariable=rigid_var, width=8).grid(row=3, column=1, padx=6, pady=6, sticky="w")

        def load_preset(*_):
            name = preset_var.get()
            data = presets.get(name)
            if not data:
                return
            kw_var.set(", ".join(data.get("keywords", [])))
            rigid_var.set(str(data.get("accept_threshold", 0.5)))
            prompt_box.delete("1.0", "end")
            prompt_box.insert("1.0", data.get("prompt", ""))

        def save_preset() -> None:
            name = simpledialog.askstring("Salva preset", "Nome preset:", parent=win)
            if not name:
                return
            presets[name] = {
                "keywords": [k.strip() for k in kw_var.get().split(",") if k.strip()],
                "accept_threshold": float(rigid_var.get() or 0.5),
                "prompt": prompt_box.get("1.0", "end").strip(),
            }
            preset_cb.configure(values=list(presets.keys()))
            preset_var.set(name)

        preset_cb.bind("<<ComboboxSelected>>", load_preset)
        ttk.Button(win, text="Salva preset", command=save_preset).grid(row=0, column=2, padx=6, pady=6)

        def on_ok() -> None:
            dom["keywords"] = [k.strip() for k in kw_var.get().split(",") if k.strip()]
            try:
                dom["accept_threshold"] = float(rigid_var.get())
            except Exception:
                pass
            self.settings["oracle_system"] = prompt_box.get("1.0", "end").strip()
            win.destroy()

        ttk.Button(win, text="OK", command=on_ok).grid(row=4, column=0, columnspan=3, pady=10)

    def _open_knowledge_dialog(self) -> None:
        if tkdnd is not None:
            win = tkdnd.TkinterDnD.Toplevel(self)  # type: ignore[attr-defined]
        else:
            win = tk.Toplevel(self)
        win.title("Conoscenza")
        win.configure(bg=self._bg)

        doc_var = tk.StringVar(value=self.settings.get("docstore_path", ""))
        topk_var = tk.StringVar(value=str(self.settings.get("retrieval_top_k", 3)))
        query_var = tk.StringVar()

        tk.Label(win, text="Indice", fg=self._fg, bg=self._bg).grid(row=0, column=0, padx=6, pady=6, sticky="e")
        tk.Entry(win, textvariable=doc_var, width=40).grid(row=0, column=1, padx=6, pady=6, sticky="w")

        def browse() -> None:
            p = filedialog.askopenfilename(
                title="Indice", filetypes=[("JSON", "*.json"), ("Tutti", "*.*")]
            )
            if p:
                doc_var.set(p)

        ttk.Button(win, text="Sfoglia", command=browse).grid(row=0, column=2, padx=6, pady=6)

        tk.Label(win, text="Top-K", fg=self._fg, bg=self._bg).grid(row=1, column=0, padx=6, pady=6, sticky="e")
        tk.Entry(win, textvariable=topk_var, width=6).grid(row=1, column=1, padx=6, pady=6, sticky="w")

        tk.Label(win, text="Test query", fg=self._fg, bg=self._bg).grid(row=2, column=0, padx=6, pady=6, sticky="e")
        tk.Entry(win, textvariable=query_var, width=40).grid(row=2, column=1, padx=6, pady=6, sticky="w")

        result_box = scrolledtext.ScrolledText(win, width=50, height=8)
        result_box.grid(row=3, column=0, columnspan=3, padx=6, pady=6)

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

        ttk.Button(win, text="Prova", command=run_test).grid(row=2, column=2, padx=6, pady=6)

        drop_lab = ttk.Label(
            win,
            text="Trascina qui file o cartelle per importarli",
            relief="ridge",
            padding=6,
        )
        drop_lab.grid(row=4, column=0, columnspan=3, padx=6, pady=6, sticky="we")

        def _ingest_paths(paths: list[str]) -> None:
            if not paths:
                return
            script = self._find_ingest_script()
            if not script:
                messagebox.showwarning("Documento", "Script di ingest non trovato (scripts/ingest_docs.py).")
                return
            try:
                subprocess.run(
                    [sys.executable, "-m", "scripts.ingest_docs", "--add", *paths],
                    check=True,
                    cwd=self.root_dir,
                )
                messagebox.showinfo("Successo", "Documenti aggiunti.")
            except subprocess.CalledProcessError as exc:
                messagebox.showerror("Errore", f"Ingest fallito: {exc}")

        def _handle_drop(event):  # pragma: no cover - dipende da GUI
            try:
                items = drop_lab.tk.splitlist(event.data)  # type: ignore[attr-defined]
            except Exception:
                items = []
            _ingest_paths(list(items))

        if tkdnd is not None:  # pragma: no cover - non testabile headless
            drop_lab.drop_target_register(tkdnd.DND_FILES)  # type: ignore[attr-defined]
            drop_lab.dnd_bind("<<Drop>>", _handle_drop)  # type: ignore[attr-defined]

        def on_ok() -> None:
            self.settings["docstore_path"] = doc_var.get().strip()
            try:
                self.settings["retrieval_top_k"] = int(topk_var.get())
            except Exception:
                pass
            win.destroy()

        ttk.Button(win, text="OK", command=on_ok).grid(row=5, column=0, columnspan=3, pady=10)

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
            win.destroy()

        ttk.Button(win, text="OK", command=on_ok).grid(row=3, column=0, columnspan=2, pady=10)

    # ------------------------------ Save ----------------------------------- #
    def save_settings(self) -> None:
        self.settings["debug"] = bool(self.debug_var.get())
        try:
            routed_save(self.base_settings, self.local_settings, self.settings, self.root_dir)
            self.base_settings, self.local_settings, self.settings = load_settings_pair(self.root_dir)
            messagebox.showinfo("Impostazioni", "Salvate correttamente.")
        except Exception as e:
            messagebox.showerror("Impostazioni", f"Errore nel salvataggio: {e}")

    # --------------------------- Start / Stop + logs ----------------------- #
    def start_oracolo(self) -> None:
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Oracolo", "È già in esecuzione.")
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
            self.after(0, self._append_log, line)
        rest = f.read()
        if rest:
            self.after(0, self._append_log, rest)

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

    # ----------------------- Realtime WS control ------------------------- #
    def start_realtime(self) -> None:
        if self.ws_client is not None:
            messagebox.showinfo("Realtime", "Sessione già attiva.")
            return

        # applica dispositivi audio da settings come default
        audio = self.settings.get("audio", {})
        in_dev = audio.get("input_device", None)
        out_dev = audio.get("output_device", None)
        sd.default.device = (in_dev, out_dev)

        self._append_log(
            f"🔌 Realtime WS → {WS_URL}  (sr={SR}, in={sd.default.device[0]}, out={sd.default.device[1]})\n"
        )

        self.ws_client = RealtimeWSClient(
            WS_URL,
            SR,
            on_partial=lambda text: self.after(0, self._append_log, f"… {text}\n"),
            on_answer=lambda text: self.after(0, self._append_log, f"🔮 {text}\n"),
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
            if not (self.proc and self.proc.poll() is None):
                self.status_var.set("🟡 In attesa")

    def _poll_process(self) -> None:
        if self.proc is not None:
            if self.proc.poll() is None:
                if self.status_var.get() != "🟢 In esecuzione":
                    self.status_var.set("🟢 In esecuzione")
                self.start_btn.configure(state="disabled")
                self.stop_btn.configure(state="normal")
            else:
                self.status_var.set("🟡 In attesa")
                self.start_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
        self.after(500, self._poll_process)

    # ------------------------------ Exit ----------------------------------- #
    def _on_close(self) -> None:
        try:
            self.stop_oracolo()
            self.stop_realtime()
        finally:
            self.destroy()


# ----------------------------- entry point -------------------------------- #
def main() -> None:
    app = OracoloUI()
    app.mainloop()


if __name__ == "__main__":
    main()
