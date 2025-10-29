import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel
from PyQt6.QtCore import QTimer, pyqtSignal, QObject
from threading import Thread
import transcripcion
import re

class TranscriptionWorker(QObject):
    text_update = pyqtSignal(str, bool, bool)
    timer_update = pyqtSignal(float)
    finished = pyqtSignal()  # Señal para cuando termine la transcripción

    def run(self):
        transcripcion.transcribe_streaming(self.update_text, self.update_timer)
        self.finished.emit()  # Emitir señal cuando termine la transcripción

    def update_text(self, text, partial=False, final=False):
        self.text_update.emit(text, partial, final)

    def update_timer(self, elapsed_time):
        self.timer_update.emit(elapsed_time)
