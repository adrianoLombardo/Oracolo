"""Simple Tkinter-based UI for Occhio Onniveggente.

This module provides a minimal graphical interface with a dark futuristic
style. The UI exposes a button to launch the existing CLI ``main`` module in a
separate process, allowing the assistant to run while the window remains
responsive. It now includes a settings menu for adjusting audio and lighting
parameters.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import TextIO

import sounddevice as sd
import yaml


class OracoloUI(tk.Tk):
    """Graphical frontend for the Oracolo assistant."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Occhio Onniveggente")
        self.option_add("*Font", ("Consolas", 12))

        # Process management
        self.proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Futuristic dark theme
        bg = "#000000"
        fg = "#00ffcc"
        self.configure(bg=bg)
        self._bg = bg
        self._fg = fg

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Cmd.TButton",
            background=bg,
            foreground=fg,
            borderwidth=1,
            focusthickness=3,
            focuscolor="none",
        )
        style.map(
            "Cmd.TButton",
            background=[("active", fg)],
            foreground=[("active", bg)],
        )
        style.configure("Cmd.TLabel", background=bg, foreground=fg)
        style.configure("Cmd.TFrame", background=bg)
        style.configure("Cmd.TEntry", fieldbackground=bg, foreground=fg, insertcolor=fg)
        style.configure("Cmd.TMenubutton", background=bg, foreground=fg)
        style.map(
            "Cmd.TMenubutton",
            background=[("active", fg)],
            foreground=[("active", bg)],
        )

        # Load settings
        self.settings_path = Path(__file__).resolve().parents[1] / "settings.yaml"
        try:
            with self.settings_path.open("r", encoding="utf-8") as fh:
                self.settings = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            self.settings = {}

        # Menu bar
        self.debug_var = tk.BooleanVar(value=bool(self.settings.get("debug", False)))
        menubar = tk.Menu(self)

        docs_menu = tk.Menu(menubar, tearoff=0)
        docs_menu.add_command(label="Aggiungi…", command=self._add_documents)
        docs_menu.add_command(label="Rimuovi…", command=self._remove_documents)
        menubar.add_cascade(label="Documenti", menu=docs_menu)

        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_checkbutton(
            label="Debug", variable=self.debug_var, command=self._update_debug
        )
        settings_menu.add_command(
            label="Audio Device…", command=self._open_audio_dialog
        )
        settings_menu.add_command(
            label="Lighting…", command=self._open_lighting_dialog
        )
        settings_menu.add_separator()
        settings_menu.add_command(label="Save", command=self.save_settings)
        menubar.add_cascade(label="Impostazioni", menu=settings_menu)
        self.config(menu=menubar)

        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.canvas.create_oval(10, 10, 390, 390, outline=fg, width=2)
        self.canvas.create_oval(60, 60, 340, 340, outline="#00ffff", width=1)
        self.canvas.lower()

        label = ttk.Label(
            self,
            text="Occhio Onniveggente",
            style="Cmd.TLabel",
            font=("Consolas", 18, "bold"),
        )
        label.pack(pady=20)

        start_btn = ttk.Button(
            self, text="Avvia", command=self.start_oracolo, style="Cmd.TButton"
        )
        start_btn.pack(pady=10)

        quit_btn = ttk.Button(
            self, text="Esci", command=self.close, style="Cmd.TButton"
        )
        quit_btn.pack(pady=10)

        self.viewport = scrolledtext.ScrolledText(
            self,
            height=15,
            bg=bg,
            fg=fg,
            insertbackground=fg,
            state="disabled",
        )
        self.viewport.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.viewport_menu = tk.Menu(self, tearoff=0)
        self.viewport_menu.add_command(label="Clear", command=self.clear_viewport)
        self.viewport.bind("<Button-3>", self._show_viewport_menu)
        self.bind_all("<Control-l>", lambda _e: self.clear_viewport())

        # Ensure cleanup on window close
        self.protocol("WM_DELETE_WINDOW", self.close)

    def start_oracolo(self) -> None:
        """Launch the existing CLI main module in a new process."""
        if self.proc and self.proc.poll() is None:
            return

        self._stop_event.clear()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "src.main"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if self.proc.stdout:
            self._reader = threading.Thread(
                target=self._reader_thread, args=(self.proc.stdout,), daemon=True
            )
            self._reader.start()

    def _reader_thread(self, stream: TextIO) -> None:
        for line in iter(stream.readline, ""):
            if self._stop_event.is_set():
                break
            self.log(line)
        stream.close()
        self._reader = None
        self.proc = None

    def close(self) -> None:
        """Terminate child process and close the UI."""

        self._stop_event.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        if self._reader and self._reader.is_alive():
            self._reader.join(timeout=1)
        self.destroy()

    def log(self, message: str) -> None:
        self.viewport.configure(state="normal")
        self.viewport.insert(tk.END, message)
        self.viewport.see(tk.END)
        self.viewport.configure(state="disabled")

    def clear_viewport(self) -> None:
        self.viewport.configure(state="normal")
        self.viewport.delete("1.0", tk.END)
        self.viewport.configure(state="disabled")

    def _show_viewport_menu(self, event: tk.Event) -> None:
        self.viewport_menu.tk_popup(event.x_root, event.y_root)

    def _add_documents(self) -> None:
        """Add documents to the knowledge base via ingest script."""

        paths = list(filedialog.askopenfilenames(parent=self))
        if not paths:
            directory = filedialog.askdirectory(parent=self)
            if directory:
                paths = [directory]
        if not paths:
            return

        script = Path(__file__).resolve().parents[1] / "scripts" / "ingest_docs.py"
        try:
            subprocess.run([sys.executable, str(script), "--add", *paths], check=True)
        except subprocess.CalledProcessError as exc:
            messagebox.showerror("Errore", f"Impossibile aggiungere i documenti: {exc}")
        else:
            messagebox.showinfo("Successo", "Documenti aggiunti correttamente")

    def _remove_documents(self) -> None:
        """Remove documents from the knowledge base via ingest script."""

        paths = list(filedialog.askopenfilenames(parent=self))
        if not paths:
            directory = filedialog.askdirectory(parent=self)
            if directory:
                paths = [directory]
        if not paths:
            return

        script = Path(__file__).resolve().parents[1] / "scripts" / "ingest_docs.py"
        try:
            subprocess.run([sys.executable, str(script), "--remove", *paths], check=True)
        except subprocess.CalledProcessError as exc:
            messagebox.showerror("Errore", f"Impossibile rimuovere i documenti: {exc}")
        else:
            messagebox.showinfo("Successo", "Documenti rimossi correttamente")

    # ----- settings handling -------------------------------------------------

    def _update_debug(self) -> None:
        self.settings["debug"] = bool(self.debug_var.get())

    def _open_audio_dialog(self) -> None:
        """Dialog to select audio devices."""

        win = tk.Toplevel(self)
        win.title("Dispositivi audio")
        win.configure(bg=self._bg)
        win.option_add("*Font", ("Consolas", 12))

        devices = sd.query_devices()
        names = [d["name"] for d in devices]

        audio = self.settings.setdefault("audio", {})
        in_var = tk.StringVar(value=audio.get("input_device") or "")
        out_var = tk.StringVar(value=audio.get("output_device") or "")

        ttk.Label(win, text="Input", style="Cmd.TLabel").grid(
            row=0, column=0, padx=5, pady=5, sticky="e"
        )
        ttk.OptionMenu(win, in_var, in_var.get(), "", *names, style="Cmd.TMenubutton").grid(
            row=0, column=1, padx=5, pady=5
        )

        ttk.Label(win, text="Output", style="Cmd.TLabel").grid(
            row=1, column=0, padx=5, pady=5, sticky="e"
        )
        ttk.OptionMenu(win, out_var, out_var.get(), "", *names, style="Cmd.TMenubutton").grid(
            row=1, column=1, padx=5, pady=5
        )

        def on_ok() -> None:
            audio["input_device"] = in_var.get() or None
            audio["output_device"] = out_var.get() or None
            win.destroy()

        ttk.Button(win, text="OK", command=on_ok, style="Cmd.TButton").grid(
            row=2, column=0, columnspan=2, pady=10
        )

    def _open_lighting_dialog(self) -> None:
        """Dialog to configure lighting/DMX settings."""

        win = tk.Toplevel(self)
        win.title("Luci")
        win.configure(bg=self._bg)
        win.option_add("*Font", ("Consolas", 12))

        lighting = self.settings.setdefault("lighting", {})
        mode_var = tk.StringVar(value=lighting.get("mode", "sacn"))

        ttk.Label(win, text="Modalità", style="Cmd.TLabel").grid(
            row=0, column=0, padx=5, pady=5, sticky="e"
        )
        ttk.OptionMenu(win, mode_var, mode_var.get(), "sacn", "wled", style="Cmd.TMenubutton").grid(
            row=0, column=1, padx=5, pady=5
        )

        sacn_frame = ttk.Frame(win, style="Cmd.TFrame")
        sacn_conf = lighting.setdefault("sacn", {})
        sacn_ip = tk.StringVar(value=sacn_conf.get("destination_ip", ""))
        sacn_uni = tk.StringVar(value=str(sacn_conf.get("universe", 1)))
        ttk.Label(sacn_frame, text="IP", style="Cmd.TLabel").grid(
            row=0, column=0, padx=5, pady=5, sticky="e"
        )
        ttk.Entry(sacn_frame, textvariable=sacn_ip, style="Cmd.TEntry").grid(
            row=0, column=1, padx=5, pady=5
        )
        ttk.Label(sacn_frame, text="Universe", style="Cmd.TLabel").grid(
            row=1, column=0, padx=5, pady=5, sticky="e"
        )
        ttk.Entry(sacn_frame, textvariable=sacn_uni, style="Cmd.TEntry").grid(
            row=1, column=1, padx=5, pady=5
        )

        wled_frame = ttk.Frame(win, style="Cmd.TFrame")
        wled_conf = lighting.setdefault("wled", {})
        wled_host = tk.StringVar(value=wled_conf.get("host", ""))
        ttk.Label(wled_frame, text="Host", style="Cmd.TLabel").grid(
            row=0, column=0, padx=5, pady=5, sticky="e"
        )
        ttk.Entry(wled_frame, textvariable=wled_host, style="Cmd.TEntry").grid(
            row=0, column=1, padx=5, pady=5
        )

        def update_mode(*_args: object) -> None:
            sacn_frame.grid_remove()
            wled_frame.grid_remove()
            if mode_var.get() == "sacn":
                sacn_frame.grid(row=1, column=0, columnspan=2, pady=5)
            else:
                wled_frame.grid(row=1, column=0, columnspan=2, pady=5)

        mode_var.trace_add("write", update_mode)
        update_mode()

        def on_ok() -> None:
            lighting["mode"] = mode_var.get()
            sacn_conf["destination_ip"] = sacn_ip.get()
            try:
                sacn_conf["universe"] = int(sacn_uni.get())
            except ValueError:
                sacn_conf["universe"] = 1
            wled_conf["host"] = wled_host.get()
            win.destroy()

        ttk.Button(win, text="OK", command=on_ok, style="Cmd.TButton").grid(
            row=2, column=0, columnspan=2, pady=10
        )

    def save_settings(self) -> None:
        """Persist settings back to ``settings.yaml``."""

        self._update_debug()
        with self.settings_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.settings, fh, allow_unicode=True)


def main() -> None:
    """Entry point to launch the UI."""

    app = OracoloUI()
    app.mainloop()


if __name__ == "__main__":
    main()

