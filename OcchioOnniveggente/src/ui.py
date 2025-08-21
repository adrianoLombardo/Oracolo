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
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import sounddevice as sd
import yaml


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

        # Style for settings panels hosted in a notebook
        style.configure("Settings.TFrame", background=bg)
        style.configure("Settings.TNotebook", background=bg)

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

        quit_btn = ttk.Button(self, text="Esci", command=self.destroy)
        quit_btn.pack(pady=10)

        # Settings notebook (hidden by default)
        self._build_settings_pane()

    def start_oracolo(self) -> None:
        """Launch the existing CLI main module in a new process."""

        subprocess.Popen([sys.executable, "-m", "src.main"])

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

    def _build_settings_pane(self) -> None:
        """Create the notebook hosting settings panels."""

        self.settings_pane = ttk.Frame(self, style="Settings.TFrame")
        self.settings_notebook = ttk.Notebook(
            self.settings_pane, style="Settings.TNotebook"
        )
        self.settings_notebook.pack(fill="both", expand=True)

        self._build_audio_frame()
        self._build_lighting_frame()

        # Hide initially
        self.settings_pane.pack_forget()

    def _build_audio_frame(self) -> None:
        """Create the audio settings panel."""

        self.audio_frame = ttk.Frame(
            self.settings_notebook, style="Settings.TFrame"
        )
        self.settings_notebook.add(self.audio_frame, text="Audio")

        devices = sd.query_devices()
        names = [d["name"] for d in devices]

        audio = self.settings.setdefault("audio", {})
        self.audio_in_var = tk.StringVar(value=audio.get("input_device") or "")
        self.audio_out_var = tk.StringVar(value=audio.get("output_device") or "")

        tk.Label(self.audio_frame, text="Input", fg=self._fg, bg=self._bg).grid(
            row=0, column=0, padx=5, pady=5, sticky="e"
        )
        tk.OptionMenu(
            self.audio_frame, self.audio_in_var, self.audio_in_var.get(), "", *names
        ).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(self.audio_frame, text="Output", fg=self._fg, bg=self._bg).grid(
            row=1, column=0, padx=5, pady=5, sticky="e"
        )
        tk.OptionMenu(
            self.audio_frame, self.audio_out_var, self.audio_out_var.get(), "", *names
        ).grid(row=1, column=1, padx=5, pady=5)

        self.audio_in_var.trace_add("write", lambda *_: self._apply_audio_settings())
        self.audio_out_var.trace_add("write", lambda *_: self._apply_audio_settings())

    def _build_lighting_frame(self) -> None:
        """Create the lighting settings panel."""

        self.lighting_frame = ttk.Frame(
            self.settings_notebook, style="Settings.TFrame"
        )
        self.settings_notebook.add(self.lighting_frame, text="Luci")

        lighting = self.settings.setdefault("lighting", {})
        self.lighting_mode_var = tk.StringVar(value=lighting.get("mode", "sacn"))

        tk.Label(self.lighting_frame, text="Modalità", fg=self._fg, bg=self._bg).grid(
            row=0, column=0, padx=5, pady=5, sticky="e"
        )
        tk.OptionMenu(
            self.lighting_frame, self.lighting_mode_var, self.lighting_mode_var.get(),
            "sacn", "wled"
        ).grid(row=0, column=1, padx=5, pady=5)

        sacn_conf = lighting.setdefault("sacn", {})
        self.sacn_frame = tk.Frame(self.lighting_frame, bg=self._bg)
        self.sacn_ip_var = tk.StringVar(value=sacn_conf.get("destination_ip", ""))
        self.sacn_uni_var = tk.StringVar(value=str(sacn_conf.get("universe", 1)))
        tk.Label(self.sacn_frame, text="IP", fg=self._fg, bg=self._bg).grid(
            row=0, column=0, padx=5, pady=5, sticky="e"
        )
        tk.Entry(self.sacn_frame, textvariable=self.sacn_ip_var).grid(
            row=0, column=1, padx=5, pady=5
        )
        tk.Label(self.sacn_frame, text="Universe", fg=self._fg, bg=self._bg).grid(
            row=1, column=0, padx=5, pady=5, sticky="e"
        )
        tk.Entry(self.sacn_frame, textvariable=self.sacn_uni_var).grid(
            row=1, column=1, padx=5, pady=5
        )

        wled_conf = lighting.setdefault("wled", {})
        self.wled_frame = tk.Frame(self.lighting_frame, bg=self._bg)
        self.wled_host_var = tk.StringVar(value=wled_conf.get("host", ""))
        tk.Label(self.wled_frame, text="Host", fg=self._fg, bg=self._bg).grid(
            row=0, column=0, padx=5, pady=5, sticky="e"
        )
        tk.Entry(self.wled_frame, textvariable=self.wled_host_var).grid(
            row=0, column=1, padx=5, pady=5
        )

        def update_mode(*_args: object) -> None:
            self.sacn_frame.grid_remove()
            self.wled_frame.grid_remove()
            if self.lighting_mode_var.get() == "sacn":
                self.sacn_frame.grid(row=1, column=0, columnspan=2, pady=5)
            else:
                self.wled_frame.grid(row=1, column=0, columnspan=2, pady=5)

        self.lighting_mode_var.trace_add("write", update_mode)
        update_mode()

        for var in (
            self.lighting_mode_var,
            self.sacn_ip_var,
            self.sacn_uni_var,
            self.wled_host_var,
        ):
            var.trace_add("write", lambda *_: self._apply_lighting_settings())

    def _toggle_panel(self, frame: ttk.Frame) -> None:
        """Show the settings frame or hide the pane if already selected."""

        if not self.settings_pane.winfo_ismapped():
            self.settings_pane.pack(fill="x", padx=10, pady=10)
            self.settings_notebook.select(frame)
        elif self.settings_notebook.select() == str(frame):
            self.settings_pane.pack_forget()
        else:
            self.settings_notebook.select(frame)

    def _open_audio_dialog(self) -> None:
        """Toggle visibility of the audio settings panel."""

        self._toggle_panel(self.audio_frame)

    def _open_lighting_dialog(self) -> None:
        """Toggle visibility of the lighting settings panel."""

        self._toggle_panel(self.lighting_frame)

    def _apply_audio_settings(self) -> None:
        audio = self.settings.setdefault("audio", {})
        audio["input_device"] = self.audio_in_var.get() or None
        audio["output_device"] = self.audio_out_var.get() or None

    def _apply_lighting_settings(self) -> None:
        lighting = self.settings.setdefault("lighting", {})
        lighting["mode"] = self.lighting_mode_var.get()
        sacn_conf = lighting.setdefault("sacn", {})
        sacn_conf["destination_ip"] = self.sacn_ip_var.get()
        try:
            sacn_conf["universe"] = int(self.sacn_uni_var.get())
        except ValueError:
            sacn_conf["universe"] = 1
        wled_conf = lighting.setdefault("wled", {})
        wled_conf["host"] = self.wled_host_var.get()

    def save_settings(self) -> None:
        """Persist settings back to ``settings.yaml``."""

        self._update_debug()
        self._apply_audio_settings()
        self._apply_lighting_settings()
        with self.settings_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.settings, fh, allow_unicode=True)


def main() -> None:
    """Entry point to launch the UI."""

    app = OracoloUI()
    app.mainloop()


if __name__ == "__main__":
    main()

