# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path
from PySide6.QtCore import QObject, Slot, Signal, Property, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

ROOT = Path(__file__).resolve().parent

class Bridge(QObject):
    wakeWordRecognizedChanged = Signal()

    def __init__(self):
        super().__init__()
        self._wake = False

    @Slot()
    def onMicTapped(self):
        print("🎙️ Mic tapped (TODO: start/stop capture)")

    @Slot()
    def onSendPressed(self):
        print("💬 Send pressed (TODO: send text to backend)")

    @Property(bool, notify=wakeWordRecognizedChanged)
    def wakeWordRecognized(self):
        return self._wake

    @Slot(bool)
    def setWakeWordRecognized(self, v: bool):
        if self._wake != v:
            self._wake = v
            self.wakeWordRecognizedChanged.emit()

def main():
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()
    bridge = Bridge()
    engine.rootContext().setContextProperty("Bridge", bridge)
    engine.addImportPath(str(ROOT / "qml"))
    engine.load(QUrl.fromLocalFile(str(ROOT / "qml" / "MainSciFi.qml")))
    if not engine.rootObjects():
        sys.exit(1)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
