# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path
from PySide6.QtCore import (
    QObject,
    Slot,
    Signal,
    Property,
    QUrl,
    QAbstractListModel,
    QModelIndex,
    Qt,
)
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from src.ui_controller import UIController
from src.chat import ChatState
import logging

ROOT = Path(__file__).resolve().parent


class HistoryModel(QAbstractListModel):
    ROLE_ROLE = Qt.UserRole + 1
    CONTENT_ROLE = Qt.UserRole + 2

    def __init__(self, chat: ChatState):
        super().__init__()
        self.chat = chat

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # type: ignore[override]
        return len(self.chat.history)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # type: ignore[override]
        if not index.isValid():
            return None
        msg = self.chat.history[index.row()]
        if role == self.ROLE_ROLE:
            return msg.get("role", "")
        if role == self.CONTENT_ROLE:
            return msg.get("content", "")
        return None

    def roleNames(self):  # type: ignore[override]
        return {
            self.ROLE_ROLE: b"role",
            self.CONTENT_ROLE: b"content",
        }

    def refresh(self) -> None:
        self.beginResetModel()
        self.endResetModel()


class Bridge(QObject):
    wakeWordRecognizedChanged = Signal()
    logTextChanged = Signal(str)

    def __init__(self, controller: UIController):
        super().__init__()
        self._wake = False
        self.controller = controller
        self._log_text = ""
        self._history_model = HistoryModel(self.controller.chat_state)

    @Property(QObject, constant=True)
    def historyModel(self) -> QObject:
        return self._history_model

    @Slot()
    def onMicTapped(self):
        # Voice input is treated the same as text once transcribed
        self.controller.submit_user_input("[voice]")
        self._history_model.refresh()

    @Slot(str)
    def sendText(self, text: str) -> None:
        self.controller.submit_user_input(text)
        self._history_model.refresh()

    def appendLog(self, text: str) -> None:
        self._log_text += text + "\n"
        self.logTextChanged.emit(self._log_text)

    @Property(str, notify=logTextChanged)
    def logText(self) -> str:
        return self._log_text

    @Property(bool, notify=wakeWordRecognizedChanged)
    def wakeWordRecognized(self):
        return self._wake

    @Slot(bool)
    def setWakeWordRecognized(self, v: bool):
        if self._wake != v:
            self._wake = v
            self.wakeWordRecognizedChanged.emit()


class _QtLogHandler(logging.Handler):
    def __init__(self, bridge: Bridge):
        super().__init__()
        self.bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.bridge.appendLog(msg)


def main():
    app = QGuiApplication(sys.argv)
    controller = UIController()
    bridge = Bridge(controller)

    logger = logging.getLogger("backend")
    logger.setLevel(logging.INFO)
    logger.addHandler(_QtLogHandler(bridge))

    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("Bridge", bridge)
    engine.addImportPath(str(ROOT / "qml"))
    engine.load(QUrl.fromLocalFile(str(ROOT / "qml" / "MainSciFi.qml")))
    if not engine.rootObjects():
        sys.exit(1)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
