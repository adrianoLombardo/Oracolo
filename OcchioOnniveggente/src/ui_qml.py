"""Qt/QML based UI for Occhio Onniveggente."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6 import QtCore, QtQml
from PySide6.QtWidgets import QApplication

from src.ui_controller import UiController


class Backend(QtCore.QObject):
    """Expose a minimal API to QML."""

    def __init__(self, controller: UiController) -> None:
        super().__init__()
        self.controller = controller

    @QtCore.Slot()
    def start(self) -> None:
        """Stub start command; hook real logic here."""
        self.controller.reload_settings()

    @QtCore.Slot()
    def reload(self) -> None:
        self.controller.reload_settings()

    # ------------------------------------------------------------------
    # Settings & documents helpers exposed to QML

    @QtCore.Slot(result="QVariantList")
    def get_documents(self) -> list:
        """Return the documents loaded from the configured docstore."""
        path = self.controller.settings.get("docstore_path", "DataBase/index.json")
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return []

        if isinstance(data, dict) and "documents" in data:
            docs = data["documents"]
        elif isinstance(data, list):
            docs = data
        else:
            docs = []
        return docs

    @QtCore.Slot("QVariant", result="QVariant")
    def update_rules(self, rules) -> list:
        """Update domain rules/keywords in settings.

        Parameters
        ----------
        rules: list | Any
            Sequence of rule strings coming from QML.  If ``rules`` is a
            comma separated string it will be split automatically.
        """

        dom = self.controller.settings.setdefault("domain", {})
        if isinstance(rules, str):
            items = [r.strip() for r in rules.split(",") if r.strip()]
        else:
            try:
                items = [str(r).strip() for r in rules if str(r).strip()]
            except Exception:
                items = []

        dom["keywords"] = items
        return dom["keywords"]

    @QtCore.Slot()
    def save_config(self) -> None:
        """Persist the current settings to disk."""
        self.controller.save_settings()


def main() -> int:
    app = QApplication(sys.argv)
    controller = UiController(Path(__file__).resolve().parent.parent)
    backend = Backend(controller)
    engine = QtQml.QQmlApplicationEngine()
    engine.rootContext().setContextProperty("backend", backend)
    qml_path = Path(__file__).resolve().parent / "qml" / "MainSciFi.qml"
    engine.load(QtCore.QUrl.fromLocalFile(qml_path))
    if not engine.rootObjects():
        return -1
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
