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
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import sounddevice as sd
import yaml
from src.config import Settings


try:
    SET = Settings.model_validate_yaml(Path(__file__).resolve().parents[1] / "settings.yaml")
except Exception:  # pragma: no cover - fallback to defaults
    SET = Settings()


class OracoloUI(tk.Tk):
    """Graphical frontend for the Oracolo assistant."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Occhio Onniveggente")

        # Futuristic dark theme
        bg = "#0f0f0f"
        fg = "#00ffe1"
        self.configure(bg=bg)
        self._bg = bg
        self._fg = fg

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "TButton",
            background="#1e1e1e",
            foreground=fg,
            borderwidth=1,
            focusthickness=3,
            focuscolor="none",
        )
        style.map(
            "TButton",
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

        label = tk.Label(
            self,
            text="Occhio Onniveggente",
            fg=fg,
            bg=bg,
            font=("Helvetica", 18, "bold"),
        )
        label.pack(pady=20)

        start_btn = ttk.Button(self, text="Avvia", command=self.start_oracolo)
        start_btn.pack(pady=10)

        reindex_btn = ttk.Button(
            self, text="Aggiorna indice", command=self._reindex_docstore
        )
        reindex_btn.pack(pady=10)

        quit_btn = ttk.Button(self, text="Esci", command=self.destroy)
        quit_btn.pack(pady=10)

    def start_oracolo(self) -> None:
        """Launch the existing CLI main module in a new process."""

        subprocess.Popen([sys.executable, "-m", "src.main"])

    def _reindex_docstore(self) -> None:
        """Rebuild the document store index and report the result."""

        proc = subprocess.Popen(
            [
                sys.executable,
                "scripts/ingest_docs.py",
                "--add",
                SET.docstore_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = proc.communicate()
        if proc.returncode == 0:
            messagebox.showinfo("Indicizzazione", stdout.decode() or "Completata")
        else:
            messagebox.showerror("Indicizzazione", stderr.decode() or "Errore")

    # ----- settings handling -------------------------------------------------

    def _update_debug(self) -> None:
        self.settings["debug"] = bool(self.debug_var.get())

    def _open_audio_dialog(self) -> None:
        """Dialog to select audio devices."""

        win = tk.Toplevel(self)
        win.title("Dispositivi audio")
        win.configure(bg=self._bg)

        devices = sd.query_devices()
        names = [d["name"] for d in devices]

        audio = self.settings.setdefault("audio", {})
        in_var = tk.StringVar(value=audio.get("input_device") or "")
        out_var = tk.StringVar(value=audio.get("output_device") or "")

        tk.Label(win, text="Input", fg=self._fg, bg=self._bg).grid(
            row=0, column=0, padx=5, pady=5, sticky="e"
        )
        tk.OptionMenu(win, in_var, "", *names).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(win, text="Output", fg=self._fg, bg=self._bg).grid(
            row=1, column=0, padx=5, pady=5, sticky="e"
        )
        tk.OptionMenu(win, out_var, "", *names).grid(row=1, column=1, padx=5, pady=5)

        def on_ok() -> None:
            audio["input_device"] = in_var.get() or None
            audio["output_device"] = out_var.get() or None
            win.destroy()

        ttk.Button(win, text="OK", command=on_ok).grid(
            row=2, column=0, columnspan=2, pady=10
        )

    def _open_lighting_dialog(self) -> None:
        """Dialog to configure lighting/DMX settings."""

        win = tk.Toplevel(self)
        win.title("Luci")
        win.configure(bg=self._bg)

        lighting = self.settings.setdefault("lighting", {})
        mode_var = tk.StringVar(value=lighting.get("mode", "sacn"))

        tk.Label(win, text="Modalità", fg=self._fg, bg=self._bg).grid(
            row=0, column=0, padx=5, pady=5, sticky="e"
        )
        tk.OptionMenu(win, mode_var, "sacn", "wled").grid(
            row=0, column=1, padx=5, pady=5
        )

        sacn_frame = tk.Frame(win, bg=self._bg)
        sacn_conf = lighting.setdefault("sacn", {})
        sacn_ip = tk.StringVar(value=sacn_conf.get("destination_ip", ""))
        sacn_uni = tk.StringVar(value=str(sacn_conf.get("universe", 1)))
        tk.Label(sacn_frame, text="IP", fg=self._fg, bg=self._bg).grid(
            row=0, column=0, padx=5, pady=5, sticky="e"
        )
        tk.Entry(sacn_frame, textvariable=sacn_ip).grid(
            row=0, column=1, padx=5, pady=5
        )
        tk.Label(sacn_frame, text="Universe", fg=self._fg, bg=self._bg).grid(
            row=1, column=0, padx=5, pady=5, sticky="e"
        )
        tk.Entry(sacn_frame, textvariable=sacn_uni).grid(
            row=1, column=1, padx=5, pady=5
        )

        wled_frame = tk.Frame(win, bg=self._bg)
        wled_conf = lighting.setdefault("wled", {})
        wled_host = tk.StringVar(value=wled_conf.get("host", ""))
        tk.Label(wled_frame, text="Host", fg=self._fg, bg=self._bg).grid(
            row=0, column=0, padx=5, pady=5, sticky="e"
        )
        tk.Entry(wled_frame, textvariable=wled_host).grid(
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

        ttk.Button(win, text="OK", command=on_ok).grid(
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

