"""Qt/QML based UI for Occhio Onniveggente."""
from __future__ import annotations

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
