"""
Edzési dialog háttérszálakkal
"""

import sys

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)
from training_manager import TrainingManager, TrainingWorker


class TrainingThread(QThread):
    """Edzési háttérszál"""

    # Jelek
    progress_updated = pyqtSignal(int, dict)  # progress %, stats
    training_finished = pyqtSignal(dict)  # results
    log_message = pyqtSignal(str)  # log üzenet
    stage_started = pyqtSignal(str)  # új szakasz kezdése

    def __init__(self, training_config):
        super().__init__()
        self.training_config = training_config
        self.should_stop = False
        self.training_manager = TrainingManager()

    def stop_training(self):
        """Edzés leállítása"""
        self.should_stop = True

    def run(self):
        """Edzés futtatása háttérszálon"""
        try:
            self.log_message.emit("Edzés indítása...")

            # Progress callback
            def progress_callback(episode: int, stats: dict):
                if self.should_stop:
                    return False

                total_episodes = self.training_config.get("num_episodes", 10000)
                progress_percent = int((episode / total_episodes) * 100)
                self.progress_updated.emit(progress_percent, stats)

                # Folytatás jelzése
                return not self.should_stop

            # Log callback
            def log_callback(message: str):
                self.log_message.emit(message)

            # Stage callback curriculum-hoz
            def stage_callback(stage_name: str):
                self.stage_started.emit(stage_name)

            # Edzés típus alapján futtatás
            training_type = self.training_config.get("training_type", "random")

            if training_type == "curriculum":
                results = self.run_curriculum_training(progress_callback, log_callback, stage_callback)
            else:
                results = self.training_manager.run_training(
                    **self.training_config, progress_callback=progress_callback, log_callback=log_callback
                )

            if not self.should_stop:
                self.training_finished.emit(results)

        except Exception as e:
            self.log_message.emit(f"Hiba az edzés során: {str(e)}")
            print(f"Training error: {e}")

    def run_curriculum_training(self, progress_callback, log_callback, stage_callback):
        """Curriculum edzés speciális kezeléssel"""
        curriculum_stages = [
            {
                "name": "Alapok - Könnyű stratégiai ellenfél",
                "episodes": 15000,
                "opponent": "strategic_easy",
                "description": "Alapvető stratégiák tanulása",
            },
            {
                "name": "Fejlesztés - Közepes ellenfél",
                "episodes": 20000,
                "opponent": "strategic_medium",
                "description": "Közepes nehézségű ellenfél",
            },
            {
                "name": "Self-play szakasz",
                "episodes": 25000,
                "opponent": "self",
                "description": "Self-play a komplex stratégiákért",
            },
            {
                "name": "Finomhangolás - Nehéz ellenfél",
                "episodes": 15000,
                "opponent": "strategic_hard",
                "description": "Nehéz stratégiai ellenfél",
            },
        ]

        total_episodes = sum(stage["episodes"] for stage in curriculum_stages)
        completed_episodes = 0

        # Módosított progress callback curriculum-hoz
        def curriculum_progress_callback(episode: int, stats: dict):
            if self.should_stop:
                return False

            current_total = completed_episodes + episode
            progress_percent = int((current_total / total_episodes) * 100)
            self.progress_updated.emit(progress_percent, stats)

            return not self.should_stop

        results = self.training_manager.curriculum_training(
            curriculum_stages, progress_callback=curriculum_progress_callback, log_callback=log_callback
        )

        return results


class TrainingDialog(QDialog):
    """Edzési dialog ablak"""

    def __init__(self, training_manager, parent=None):
        super().__init__(parent)
        self.training_manager = training_manager
        self.training_thread = None
        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """UI felépítése"""
        self.setWindowTitle("AI Edzés")
        self.setFixedSize(600, 500)

        layout = QVBoxLayout()

        # Edzés típus választás
        type_group = QGroupBox("Edzés típusa")
        type_layout = QVBoxLayout()

        self.training_type = QComboBox()
        self.training_type.addItems(
            [
                "Gyors edzés (20k epizód)",
                "Teljes curriculum (75k epizód)",
                "Stratégiai könnyű (15k epizód)",
                "Stratégiai közepes (25k epizód)",
                "Self-play (30k epizód)",
            ]
        )
        type_layout.addWidget(QLabel("Válassz edzés típust:"))
        type_layout.addWidget(self.training_type)
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)

        # Haladás megjelenítés
        progress_group = QGroupBox("Haladás")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_label = QLabel("Kész az indításra")

        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Log terület
        log_group = QGroupBox("Edzési napló")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setFont(QFont("Consolas", 9))

        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # Gombok
        button_layout = QHBoxLayout()

        self.start_button = QPushButton("Edzés indítása")
        self.stop_button = QPushButton("Leállítás")
        self.close_button = QPushButton("Bezárás")

        self.stop_button.setEnabled(False)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def connect_signals(self):
        """Jelek összekapcsolása"""
        self.start_button.clicked.connect(self.start_training)
        self.stop_button.clicked.connect(self.stop_training)
        self.close_button.clicked.connect(self.close_dialog)

    def start_training(self):
        """Edzés indítása"""
        if self.training_thread and self.training_thread.isRunning():
            return

        # UI állapot
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()

        # Edzési konfiguráció
        training_config = self.get_training_config()

        # Háttérszál létrehozása és indítása
        self.training_thread = TrainingThread(training_config)

        # Jelek összekapcsolása
        self.training_thread.progress_updated.connect(self.update_progress)
        self.training_thread.training_finished.connect(self.training_finished)
        self.training_thread.log_message.connect(self.add_log_message)
        self.training_thread.stage_started.connect(self.stage_started)

        # Edzés indítása
        self.training_thread.start()

        self.add_log_message("Edzés elindítva háttérszálon...")

    def get_training_config(self):
        """Edzési konfiguráció összeállítása"""
        selected = self.training_type.currentText()

        if "Gyors edzés" in selected:
            return {
                "training_type": "strategic_medium",
                "num_episodes": 20000,
                "agent_params": {
                    "learning_rate": 0.2,
                    "discount_factor": 0.99,
                    "epsilon_start": 0.95,
                    "epsilon_end": 0.01,
                    "epsilon_decay": 0.9998,
                },
            }
        elif "Teljes curriculum" in selected:
            return {"training_type": "curriculum", "num_episodes": 75000}  # Összes epizód
        elif "Stratégiai könnyű" in selected:
            return {"training_type": "strategic_easy", "num_episodes": 15000}
        elif "Stratégiai közepes" in selected:
            return {"training_type": "strategic_medium", "num_episodes": 25000}
        elif "Self-play" in selected:
            return {"training_type": "self_play", "num_episodes": 30000}
        else:
            return {"training_type": "random", "num_episodes": 10000}

    def update_progress(self, progress_percent, stats):
        """Haladás frissítése"""
        self.progress_bar.setValue(progress_percent)

        # Statisztikák megjelenítése
        win_rate = stats.get("win_rate", 0)
        games_played = stats.get("games_played", 0)

        self.progress_label.setText(
            f"Haladás: {progress_percent}% | " f"Játékok: {games_played} | " f"Győzelmi arány: {win_rate:.1%}"
        )

    def add_log_message(self, message):
        """Log üzenet hozzáadása"""
        self.log_text.append(message)
        # Automatikus görgetés
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def stage_started(self, stage_name):
        """Új szakasz kezdése"""
        self.add_log_message(f"=== {stage_name} ===")

    def training_finished(self, results):
        """Edzés befejezése"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)

        # Eredmények megjelenítése
        if "results" in results and "final_evaluation" in results["results"]:
            win_rate = results["results"]["final_evaluation"]["win_rate"]
            self.add_log_message(f"Edzés befejezve! Végső győzelmi arány: {win_rate:.1%}")
        else:
            self.add_log_message("Edzés befejezve!")

        # Eredmény dialog
        QMessageBox.information(
            self, "Edzés befejezve", f"Az AI edzés sikeresen befejezve!\n" f"Részletek a naplóban."
        )

    def stop_training(self):
        """Edzés leállítása"""
        if self.training_thread and self.training_thread.isRunning():
            self.add_log_message("Edzés leállítása...")
            self.training_thread.stop_training()

            # Várunk a szál befejezésére (max 5 másodperc)
            if self.training_thread.wait(5000):
                self.add_log_message("Edzés leállítva.")
            else:
                self.add_log_message("Edzés erőszakos leállítása...")
                self.training_thread.terminate()
                self.training_thread.wait()

            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def close_dialog(self):
        """Dialog bezárása"""
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Edzés folyamatban",
                "Az edzés még folyamatban van. Biztosan be akarod zárni?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.stop_training()
                self.accept()
        else:
            self.accept()

    def closeEvent(self, event):
        """Ablak bezárása esemény"""
        if self.training_thread and self.training_thread.isRunning():
            self.stop_training()
        event.accept()
