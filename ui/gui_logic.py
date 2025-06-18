import os
import sys

import numpy as np
from PyQt6 import uic
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QComboBox, QLabel, QMainWindow, QMessageBox, QPushButton

# RL komponensek importálása
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ai_training.constants import PLAYER_O, PLAYER_X
from ai_training.q_learning_agent import QLearningAgent
from ai_training.tictactoe_environment import TicTacToeEnvironment
from ai_training.training_manager import TrainingManager


class TicTacToeGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # UI fájl betöltése
        self.load_ui()

        # Játék állapot inicializálása
        self.init_game_state()

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

            self.result_text.setText("")

            # Gombok stílusának beállítása
            button_style = """
                QPushButton {
                    border: 2px solid #333;
                    border-radius: 10px;
                    background-color: white;
                    font-size: 24px;
                    font-weight: bold;
                    min-height: 80px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
                QPushButton:pressed {
                    background-color: #d0d0d0;
                }
            """

            for button in self.board_buttons:
                button.setText("")
                button.setFont(QFont("Arial", 24, QFont.Weight.Bold))
                button.setStyleSheet(button_style)

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

    def init_ai_components(self):
        """AI komponensek inicializálása"""
        self.training_manager = TrainingManager()
        self.ai_agent = None

        # Próbáljuk betölteni a legjobb modellt
        try:
            self.ai_agent = self.training_manager.load_best_agent()
            if self.ai_agent:
                print("Legjobb AI modell betöltve")
            else:
                print("Nincs betölthető AI modell")
        except Exception as e:
            print(f"Hiba az AI betöltésekor: {e}")

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

    def on_game_mode_changed(self, mode):
        """Játék mód változása"""
        print(f"Játék mód változott: {mode}")
        # Új játék indítása mód váltáskor
        self.start_new_game()

    def ai_move(self):
        """AI lépése"""
        if self.game_over or not self.ai_agent:
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
            # Fallback: random lépés
            self.random_ai_move()

    def update_board_display(self):
        """Játéktábla megjelenítésének frissítése"""
        for i in range(3):
            for j in range(3):
                button_index = i * 3 + j
                button = self.board_buttons[button_index]
                value = self.board[i][j]

                if value == 1:
                    button.setText("X")
                    button.setStyleSheet(button.styleSheet() + "color: blue;")
                elif value == -1:
                    button.setText("O")
                    button.setStyleSheet(button.styleSheet() + "color: red;")
                else:
                    button.setText("")

    def check_winner(self):
        """Győztes ellenőrzése"""
        # Sorok ellenőrzése
        for row in self.board:
            if abs(sum(row)) == 3:
                self.winner = 1 if sum(row) == 3 else -1
                return True

        # Oszlopok ellenőrzése
        for col in range(3):
            col_sum = sum(self.board[row][col] for row in range(3))
            if abs(col_sum) == 3:
                self.winner = 1 if col_sum == 3 else -1
                return True

        # Átlók ellenőrzése
        diag1 = sum(self.board[i][i] for i in range(3))
        diag2 = sum(self.board[i][2 - i] for i in range(3))

        for diag in [diag1, diag2]:
            if abs(diag) == 3:
                self.winner = 1 if diag == 3 else -1
                return True

        return False

    def start_ai_training(self):
        """AI edzés indítása"""
        from ai_training.training_dialog import TrainingDialog

        dialog = TrainingDialog(self.training_manager, self)
        if dialog.exec() == QMessageBox.DialogCode.Accepted:
            # Új modell betöltése az edzés után
            self.ai_agent = self.training_manager.load_best_agent()
            QMessageBox.information(self, "Edzés", "AI edzés befejezve!")

    def is_board_full(self):
        """Ellenőrzi, hogy a tábla tele van-e"""
        return not any(0 in row for row in self.board)

    def end_game(self):
        """Játék befejezése"""
        self.game_over = True

        # Eredmény megjelenítése
        if hasattr(self, "result_label"):  # Ha van result_label widget
            if self.winner == 1:
                self.result_label.setText("X nyert!")
            elif self.winner == -1:
                self.result_label.setText("O nyert!")
            else:
                self.result_label.setText("Döntetlen!")

        # Gombok letiltása
        for button in self.board_buttons:
            button.setEnabled(False)

    def start_new_game(self):
        """Új játék indítása"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None

        # UI frissítése
        for button in self.board_buttons:
            button.setText("")
            button.setEnabled(True)

        self.result_text.setText("")

        self.update_board_display()
