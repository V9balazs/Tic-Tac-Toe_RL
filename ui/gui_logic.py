import os
import sys

import numpy as np
from PyQt6 import uic
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QComboBox, QLabel, QMainWindow, QMessageBox, QPushButton

# RL komponensek importálása
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from ai_training.q_learning_agent import QLearningAgent
    from ai_training.tictactoe_environment import TicTacToeEnvironment
    from ai_training.training_manager import TrainingManager

    # Konstansok importálása
    try:
        from ai_training.constants import PLAYER_O, PLAYER_X
    except ImportError:
        # Fallback konstansok ha nincs config fájl
        PLAYER_X = 1
        PLAYER_O = -1
        print("Figyelem: config.constants nem található, alapértelmezett értékek használata")
except ImportError as e:
    print(f"Hiba az AI komponensek importálásakor: {e}")
    # Fallback értékek
    PLAYER_X = 1
    PLAYER_O = -1


class TicTacToeGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # UI fájl betöltése
        self.load_ui()

        # Játék állapot inicializálása
        self.init_game_state()

        # AI komponensek inicializálása
        self.init_ai_components()

        # UI beállítások
        self.setup_ui()

        # Jelek összekapcsolása
        self.connect_signals()

    def load_ui(self):
        """UI fájl betöltése"""
        try:
            ui_path = os.path.join(os.path.dirname(__file__), "main_window.ui")
            print(f"UI fájl elérési útja: {ui_path}")

            if not os.path.exists(ui_path):
                raise FileNotFoundError(f"UI fájl nem található: {ui_path}")

            uic.loadUi(ui_path, self)
            print("UI fájl sikeresen betöltve")

        except Exception as e:
            print(f"Hiba a UI betöltésekor: {e}")

    def init_game_state(self):
        """Játék állapot inicializálása"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 = X, -1 = O
        self.game_over = False
        self.winner = None
        self.winning_positions = []

    def setup_ui(self):
        """UI elemek beállítása"""
        try:
            # Gombok listája
            self.board_buttons = [
                self.pushButton_1,
                self.pushButton_2,
                self.pushButton_3,
                self.pushButton_4,
                self.pushButton_5,
                self.pushButton_6,
                self.pushButton_7,
                self.pushButton_8,
                self.pushButton_9,
            ]

            # Panel címek beállítása
            if hasattr(self, "game_panel"):
                self.game_panel.setTitle("Játéktábla")
            if hasattr(self, "control_panel"):
                self.control_panel.setTitle("Vezérlőpult")

            if hasattr(self, "result_text"):
                self.result_text.setText("")

            # Alapértelmezett gomb stílus
            self.default_button_style = """
                QPushButton {
                    border: 3px solid #333;
                    border-radius: 10px;
                    background-color: white;
                    font-size: 28px;
                    font-weight: bold;
                    min-height: 80px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #f0f0f0;
                }
                QPushButton:pressed {
                    background-color: #e0e0e0;
                }
            """

            # X játékos stílus (kék)
            self.x_button_style = """
                QPushButton {
                    border: 3px solid #333;
                    border-radius: 10px;
                    background-color: white;
                    font-size: 28px;
                    font-weight: bold;
                    min-height: 80px;
                    min-width: 80px;
                    color: #2196F3;
                }
            """

            # O játékos stílus (piros)
            self.o_button_style = """
                QPushButton {
                    border: 3px solid #333;
                    border-radius: 10px;
                    background-color: white;
                    font-size: 28px;
                    font-weight: bold;
                    min-height: 80px;
                    min-width: 80px;
                    color: #F44336;
                }
            """

            # Nyerő gomb stílus (zöld háttér)
            self.winning_button_style = """
                QPushButton {
                    border: 3px solid #333;
                    border-radius: 10px;
                    background-color: #4CAF50;
                    font-size: 28px;
                    font-weight: bold;
                    min-height: 80px;
                    min-width: 80px;
                    color: white;
                }
            """

            # Gombok inicializálása
            for button in self.board_buttons:
                button.setText("")
                button.setFont(QFont("Arial", 28, QFont.Weight.Bold))
                button.setStyleSheet(self.default_button_style)

            print("UI beállítások alkalmazva")

        except Exception as e:
            print(f"Hiba az UI beállításakor: {e}")

    def connect_signals(self):
        """Jelek és slotok összekapcsolása"""
        try:
            # Játéktábla gombok
            for i, button in enumerate(self.board_buttons):
                row, col = i // 3, i % 3
                button.clicked.connect(lambda checked, r=row, c=col: self.make_move(r, c))

            # Vezérlő gombok
            if hasattr(self, "new_game"):
                self.new_game.clicked.connect(self.start_new_game)
            if hasattr(self, "ai_training"):
                self.ai_training.clicked.connect(self.start_ai_training)
            if hasattr(self, "game_mode"):
                self.game_mode.currentTextChanged.connect(self.on_game_mode_changed)

            print("Jelek sikeresen összekapcsolva")

        except Exception as e:
            print(f"Hiba a jelek összekapcsolásakor: {e}")

    # Az init_ai_components metódus módosítása:

    def init_ai_components(self):
        """AI komponensek inicializálása"""
        try:
            self.training_manager = TrainingManager()
            self.ai_agent = None

            # Próbáljuk betölteni a legjobb modellt
            self.ai_agent = self.training_manager.load_best_agent()
            if self.ai_agent:
                print("Legjobb AI modell betöltve")
                # Státusz frissítése a GUI-ban
                if hasattr(self, "statusbar"):
                    self.statusbar.showMessage("AI modell betöltve", 3000)
            else:
                print("Nincs betölthető AI modell - új edzés szükséges")
                if hasattr(self, "statusbar"):
                    self.statusbar.showMessage("Nincs AI modell - edzés szükséges", 5000)

        except Exception as e:
            print(f"Hiba az AI inicializálásakor: {e}")
            QMessageBox.warning(
                self,
                "AI inicializálási hiba",
                f"Nem sikerült inicializálni az AI komponenseket:\n{str(e)}\n\n"
                f"Az edzési funkciók nem lesznek elérhetők.",
            )

    def make_move(self, row, col):
        """Játékos lépése"""
        if self.game_over or self.board[row][col] != 0:
            return

        # Emberi játékos lépése
        self.board[row][col] = self.current_player
        self.update_board_display()

        if self.check_winner() or self.is_board_full():
            self.end_game()
            return

        self.current_player *= -1

        # AI lépés ha szükséges
        if self.game_mode.currentText() == "Ember vs AI" and self.current_player == PLAYER_O:
            QTimer.singleShot(500, self.ai_move)

    def on_game_mode_changed(self):
        """Játékmód változás kezelése"""
        current_mode = self.game_mode.currentText()

        if current_mode == "Ember vs AI" and not self.ai_agent:
            # Figyelmeztetés ha nincs AI modell
            QMessageBox.information(
                self,
                "AI modell hiányzik",
                "Nincs betöltött AI modell!\n\n"
                "Kérlek indíts egy edzést az 'AI Edzés' gombbal, "
                "vagy a program egyszerű stratégiai logikát fog használni.",
            )

    def ai_move(self):
        """AI lépése - jobb hibakezelés"""
        if self.game_over:
            return

        if not self.ai_agent:
            # Fallback: egyszerű stratégiai lépés
            self.strategic_fallback_move()
            return

        try:
            # Environment létrehozása az aktuális állapotból
            env = TicTacToeEnvironment()
            env.board = self.board.copy()
            env.current_player = self.current_player

            # Érvényes lépések
            valid_actions = env.get_valid_actions()

            if valid_actions:
                # AI döntés
                action = self.ai_agent.get_best_action(env.get_state(), valid_actions, env)

                # Lépés végrehajtása
                row, col = action // 3, action % 3
                self.board[row][col] = self.current_player
                self.update_board_display()

                # Játék vége ellenőrzése
                if self.check_winner() or self.is_board_full():
                    self.end_game()
                    return

                # Játékos váltás
                self.current_player *= -1

        except Exception as e:
            print(f"Hiba az AI lépésekor: {e}")
            # Fallback: stratégiai lépés
            self.strategic_fallback_move()

    def strategic_fallback_move(self):
        """Stratégiai fallback lépés ha az AI nem működik"""
        if self.game_over:
            return

        # Elérhető pozíciók
        available_moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    available_moves.append((i, j))

        if not available_moves:
            return

        # 1. Nyerő lépés keresése
        for row, col in available_moves:
            test_board = self.board.copy()
            test_board[row][col] = self.current_player
            if self.check_winner_on_board(test_board) == self.current_player:
                self.execute_move(row, col)
                return

        # 2. Ellenfél blokkolása
        opponent = -self.current_player
        for row, col in available_moves:
            test_board = self.board.copy()
            test_board[row][col] = opponent
            if self.check_winner_on_board(test_board) == opponent:
                self.execute_move(row, col)
                return

        # 3. Stratégiai pozíciók (központ, sarkok, oldalak)
        strategic_positions = [
            (1, 1),  # központ
            (0, 0),
            (0, 2),
            (2, 0),
            (2, 2),  # sarkok
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),  # oldalak
        ]

        for row, col in strategic_positions:
            if (row, col) in available_moves:
                self.execute_move(row, col)
                return

        # 4. Random fallback
        import random

        row, col = random.choice(available_moves)
        self.execute_move(row, col)

    def execute_move(self, row, col):
        """Lépés végrehajtása és játék állapot frissítése"""
        self.board[row][col] = self.current_player
        self.update_board_display()

        if self.check_winner() or self.is_board_full():
            self.end_game()
            return

        self.current_player *= -1

    # def random_ai_move(self):
    #     """Fallback random AI lépés"""
    #     if self.game_over:
    #         return

    #     # Elérhető pozíciók keresése
    #     available_moves = []
    #     for i in range(3):
    #         for j in range(3):
    #             if self.board[i][j] == 0:
    #                 available_moves.append((i, j))

    #     if available_moves:
    #         import random

    #         row, col = random.choice(available_moves)

    #         # AI lépés végrehajtása
    #         self.board[row][col] = self.current_player
    #         self.update_board_display()

    #         # Játék vége ellenőrzése
    #         if self.check_winner() or self.is_board_full():
    #             self.end_game()
    #             return

    #         # Játékos váltás
    #         self.current_player *= -1

    def update_board_display(self):
        """Játéktábla megjelenítésének frissítése"""
        for i in range(3):
            for j in range(3):
                button_index = i * 3 + j
                button = self.board_buttons[button_index]
                value = self.board[i][j]

                if value == 1:  # X játékos
                    button.setText("X")
                    button.setStyleSheet(self.x_button_style)
                elif value == -1:  # O játékos
                    button.setText("O")
                    button.setStyleSheet(self.o_button_style)
                else:  # Üres mező
                    button.setText("")
                    button.setStyleSheet(self.default_button_style)

        # UI frissítés kényszerítése
        self.update()

    def check_winner(self):
        """Győztes ellenőrzése"""

        self.winning_positions = []

        # Sorok ellenőrzése
        for row in range(3):
            if abs(sum(self.board[row])) == 3 and 0 not in self.board[row]:
                self.winner = 1 if sum(self.board[row]) == 3 else -1
                self.winning_positions = [(row, col) for col in range(3)]
                return True

        # Oszlopok ellenőrzése
        for col in range(3):
            col_sum = sum(self.board[row][col] for row in range(3))
            if abs(col_sum) == 3:
                self.winner = 1 if col_sum == 3 else -1
                self.winning_positions = [(row, col) for row in range(3)]
                return True

        # Átlók ellenőrzése
        diag1 = sum(self.board[i][i] for i in range(3))
        diag2 = sum(self.board[i][2 - i] for i in range(3))

        diag1 = sum(self.board[i][i] for i in range(3))
        if abs(diag1) == 3:
            self.winner = 1 if diag1 == 3 else -1
            self.winning_positions = [(i, i) for i in range(3)]
            return True

        diag2 = sum(self.board[i][2 - i] for i in range(3))
        if abs(diag2) == 3:
            self.winner = 1 if diag2 == 3 else -1
            self.winning_positions = [(i, 2 - i) for i in range(3)]
            return True

        return False

    def highlight_winning_positions(self):
        """Nyerő pozíciók kiemelése zöld háttérrel"""

        for row, col in self.winning_positions:
            button_index = row * 3 + col
            button = self.board_buttons[button_index]

            # Nyerő gomb stílusa a játékos alapján
            if self.winner == 1:  # X nyert
                winning_style = """
                    QPushButton {
                        border: 3px solid #333;
                        border-radius: 10px;
                        background-color: #4CAF50;
                        font-size: 28px;
                        font-weight: bold;
                        min-height: 80px;
                        min-width: 80px;
                        color: #2196F3;
                    }
                """
            else:  # O nyert
                winning_style = """
                    QPushButton {
                        border: 3px solid #333;
                        border-radius: 10px;
                        background-color: #4CAF50;
                        font-size: 28px;
                        font-weight: bold;
                        min-height: 80px;
                        min-width: 80px;
                        color: #F44336;
                    }
                """

            button.setStyleSheet(winning_style)

    def start_ai_training(self):
        """AI edzés indítása - háttérszálakkal"""
        try:
            # Training manager inicializálása ha szükséges
            if not hasattr(self, "training_manager"):
                self.init_ai_components()

            # Training dialog megnyitása
            from PyQt6.QtWidgets import QDialog, QMessageBox

            from ai_training.training_dialog import TrainingDialog

            dialog = TrainingDialog(self.training_manager, self)
            result = dialog.exec()

            # Ha az edzés sikeresen befejeződött, frissítjük az AI-t
            if result == QDialog.DialogCode.Accepted:
                try:
                    # Legjobb modell újratöltése
                    self.ai_agent = self.training_manager.load_best_agent()
                    if self.ai_agent:
                        QMessageBox.information(
                            self,
                            "AI frissítve",
                            "Az új AI modell sikeresen betöltve!\n" "Most már az újonnan edzett AI ellen játszhatsz.",
                        )
                    else:
                        QMessageBox.warning(
                            self, "Figyelmeztetés", "Az edzés befejeződött, de nem sikerült betölteni az új modellt."
                        )
                except Exception as e:
                    QMessageBox.warning(self, "Modell betöltési hiba", f"Hiba az új modell betöltésekor: {str(e)}")

        except ImportError:
            QMessageBox.critical(
                self,
                "Hiba",
                "A training dialog nem található!\n" "Ellenőrizd, hogy a training_dialog.py fájl létezik.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Edzési hiba", f"Hiba az edzés indításakor: {str(e)}")

    def run_quick_training(self):
        """Gyors edzés javított paraméterekkel"""
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QProgressDialog

        progress_dialog = QProgressDialog("AI edzés folyamatban...", "Megszakítás", 0, 100, self)
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.show()

        def progress_callback(episode, stats):
            progress = int((episode / 20000) * 100)
            progress_dialog.setValue(progress)
            progress_dialog.setLabelText(
                f"Epizód: {episode}/20000\n" f"Győzelmi arány: {stats.get('win_rate', 0):.1%}"
            )
            QApplication.processEvents()
            return not progress_dialog.wasCanceled()

        # Javított paraméterekkel
        training_config = {
            "training_type": "strategic_mixed",
            "num_episodes": 20000,
            "agent_params": {
                "learning_rate": 0.2,
                "discount_factor": 0.99,
                "epsilon_start": 0.95,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.9998,
                "use_symmetries": True,
            },
        }

        results = self.training_manager.run_training(**training_config, progress_callback=progress_callback)

        progress_dialog.close()

        # Eredmény megjelenítése
        win_rate = results["results"]["final_evaluation"]["win_rate"]
        QMessageBox.information(
            self,
            "Edzés befejezve",
            f"Gyors edzés befejezve!\n"
            f"Végső győzelmi arány: {win_rate:.1%}\n"
            f"Edzési idő: {results['training_duration']:.1f} másodperc",
        )

        # AI agent frissítése
        self.ai_agent = self.training_manager.load_best_agent()

    def run_full_curriculum(self):
        """Teljes curriculum edzés"""
        from PyQt6.QtWidgets import QProgressDialog

        progress_dialog = QProgressDialog("Curriculum edzés folyamatban...", "Megszakítás", 0, 100, self)
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.show()

        total_episodes = 75000
        current_episode = 0

        def progress_callback(episode, stats):
            nonlocal current_episode
            current_episode = episode
            progress = int((current_episode / total_episodes) * 100)
            progress_dialog.setValue(progress)
            progress_dialog.setLabelText(
                f"Epizód: {current_episode}/{total_episodes}\n" f"Győzelmi arány: {stats.get('win_rate', 0):.1%}"
            )
            QApplication.processEvents()
            return not progress_dialog.wasCanceled()

        # Javított curriculum
        curriculum_stages = [
            {
                "name": "Alapok - Stratégiai könnyű",
                "episodes": 15000,
                "opponent": "strategic_easy",
                "description": "Alapvető stratégiák tanulása",
            },
            {
                "name": "Fejlesztés - Stratégiai közepes",
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

        results = self.training_manager.curriculum_training(curriculum_stages, progress_callback=progress_callback)

        progress_dialog.close()

        # Eredmény megjelenítése
        final_stage = results["stages"][-1]
        win_rate = final_stage["result"]["final_evaluation"]["win_rate"]

        QMessageBox.information(
            self,
            "Curriculum befejezve",
            f"Teljes curriculum edzés befejezve!\n"
            f"Végső győzelmi arány: {win_rate:.1%}\n"
            f"Összes epizód: {results['total_episodes']}\n"
            f"Edzési idő: {results.get('training_duration', 0):.1f} másodperc",
        )

        # AI agent frissítése
        self.ai_agent = self.training_manager.load_best_agent()

    def is_board_full(self):
        """Ellenőrzi, hogy a tábla tele van-e"""
        return not any(0 in row for row in self.board)

    def end_game(self):
        """JAVÍTOTT Játék befejezése - jobb eredmény megjelenítés"""
        self.game_over = True

        # Eredmény meghatározása
        if self.winner == 1:
            result_text = "X nyert! 🎉"
            if self.game_mode.currentText() == "Ember vs AI":
                result_text = "Te nyertél! 🎉"
        elif self.winner == -1:
            result_text = "O nyert! 🎉"
            if self.game_mode.currentText() == "Ember vs AI":
                result_text = "Az AI nyert! 🤖"
        else:
            result_text = "Döntetlen! 🤝"

        # Eredmény megjelenítése ha van result_text widget
        if hasattr(self, "result_text"):
            self.result_text.setText(result_text)

        # Státusz bar frissítése
        if hasattr(self, "statusbar"):
            self.statusbar.showMessage(result_text, 10000)

        # Gombok letiltása
        for button in self.board_buttons:
            button.setEnabled(False)

        # Győztes pozíciók kiemelése (opcionális)
        self.highlight_winning_positions()

    def start_new_game(self):
        """Új játék indítása"""
        print("Új játék indítása...")

        # Játék állapot visszaállítása
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.winning_positions = []

        # UI frissítése
        for i, button in enumerate(self.board_buttons):
            button.setText("")
            button.setEnabled(True)
            # Alapértelmezett stílus visszaállítása
            button.setStyleSheet(self.default_button_style)

        # Eredmény törlése
        if hasattr(self, "result_text"):
            self.result_text.setText("")
            self.result_text.setStyleSheet("")

        # Tábla megjelenítés frissítése
        self.update_board_display()

        print("Új játék elindítva!")
