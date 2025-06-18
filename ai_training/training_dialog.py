"""
AI edzési dialógus ablak
"""

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)

from ai_training.training_manager import TrainingManager, TrainingWorker


class TrainingDialog(QDialog):
    def __init__(self, training_manager: TrainingManager, parent=None):
        super().__init__(parent)
        self.training_manager = training_manager
        self.training_worker = None

        self.setWindowTitle("AI Edzés")
        self.setModal(True)
        self.resize(500, 400)

        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """UI felépítése"""
        layout = QVBoxLayout()

        # Edzési beállítások
        settings_group = QGroupBox("Edzési beállítások")
        settings_layout = QVBoxLayout()

        # Edzés típusa
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Edzés típusa:"))
        self.training_type = QComboBox()
        self.training_type.addItems(["Random ellenfél", "Self-play", "Curriculum"])
        type_layout.addWidget(self.training_type)
        settings_layout.addLayout(type_layout)

        # Epizódok száma
        episodes_layout = QHBoxLayout()
        episodes_layout.addWidget(QLabel("Epizódok száma:"))
        self.episodes_spinbox = QSpinBox()
        self.episodes_spinbox.setRange(1000, 100000)
        self.episodes_spinbox.setValue(10000)
        self.episodes_spinbox.setSingleStep(1000)
        episodes_layout.addWidget(self.episodes_spinbox)
        settings_layout.addLayout(episodes_layout)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Progress
        progress_group = QGroupBox("Előrehaladás")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Kész az indításra")

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Log
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        layout.addWidget(QLabel("Edzési napló:"))
        layout.addWidget(self.log_text)

        # Gombok
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Edzés indítása")
        self.stop_button = QPushButton("Leállítás")
        self.stop_button.setEnabled(False)
        self.close_button = QPushButton("Bezárás")

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def connect_signals(self):
        """Jelek összekapcsolása"""
        self.start_button.clicked.connect(self.start_training)
        self.stop_button.clicked.connect(self.stop_training)
        self.close_button.clicked.connect(self.close)

    def start_training(self):
        """Edzés indítása"""
        # Konfiguráció összeállítása
        training_config = {"training_type": self.get_training_type(), "num_episodes": self.episodes_spinbox.value()}

        # Worker létrehozása
        self.training_worker = self.training_manager.create_training_worker(training_config)

        # Jelek összekapcsolása
        self.training_worker.progress_updated.connect(self.update_progress)
        self.training_worker.log_message.connect(self.add_log_message)
        self.training_worker.training_finished.connect(self.training_finished)

        # UI frissítése
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()

        # Edzés indítása
        self.training_worker.start()
        self.add_log_message("Edzés elindítva...")

    def get_training_type(self):
        """Edzés típus lekérése"""
        type_map = {"Random ellenfél": "random", "Self-play": "self_play", "Curriculum": "curriculum"}
        return type_map.get(self.training_type.currentText(), "random")

    @pyqtSlot(int, dict)
    def update_progress(self, episode, stats):
        """Progress frissítése"""
        max_episodes = self.episodes_spinbox.value()
        progress = int((episode / max_episodes) * 100)
        self.progress_bar.setValue(progress)

        win_rate = stats.get("win_rate", 0)
        self.progress_label.setText(f"Epizód: {episode}/{max_episodes}, Győzelmi arány: {win_rate:.1%}")

    @pyqtSlot(str)
    def add_log_message(self, message):
        """Log üzenet hozzáadása"""
        self.log_text.append(message)

    @pyqtSlot(dict)
    def training_finished(self, results):
        """Edzés befejezése"""
        self.progress_bar.setValue(100)
        self.progress_label.setText("Edzés befejezve!")

        final_win_rate = results.get("results", {}).get("final_evaluation", {}).get("win_rate", 0)
        self.add_log_message(f"Végső győzelmi arány: {final_win_rate:.1%}")

        # UI visszaállítása
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def stop_training(self):
        """Edzés leállítása"""
        if self.training_worker:
            self.training_worker.stop_training()
            self.add_log_message("Edzés leállítása...")
