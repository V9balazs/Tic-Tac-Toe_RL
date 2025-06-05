import os
import sys

import numpy as np
from PyQt6 import uic
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QComboBox, QLabel, QMainWindow, QPushButton


class TicTacToeGUI(QMainWindow):
    def _init(self):
        super().__init__()

        # Load the UI file
        ui_path = os.path.join(os.path.dirname(__file__), "main_window.ui")
        uic.loadUi(ui_path, self)

        # Játékmező állapot tárolása
        self.board = np.zeros((3, 3), dtype=int)

        self.current_player = 1
        self.game_over = False
        self.winnner = None

        # UI elemek
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

        self.init_ui()
        self.connect_sinagls()

    def init_ui(self):
        self.game_panel.setTitle("Játéktábla")
        self.control_panel.setTitle("Vezérlőpult")

        # Játéktábla gombjainak beállítása
        for i in range(3):
            for j in range(3):
                button = self.board_buttons[i][j]
                button.setText("")
                button.setFont(QFont("Arial", 24, QFont.Weight.Bold))
                button.setStyleSheet(
                    """
                    QPushButton {
                        border: 2px solid #333;
                        border-radius: 10px;
                        background-color: white;
                        font-size: 24px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #e0e0e0;
                    }
                    QPushButton:pressed {
                        background-color: #d0d0d0;
                    }
                    """
                )

    def connect_sinals(self):
        for i in range(3):
            for j in range(3):
                button = self.board_buttons[i][j]
                button.clicked.connect(lambda checked, row=i, col=j: self.make_move(row, col))

        self.new_game.clicked.connect(self.start_new_game)
        self.ai_training.clicked.connect(self.start_ai_training)
        self.game_mode.currentTextChanged.connect(self.on_game_mode_changed)

    def make_move(self, row, col):
        if self.game_over or self.board[row][col] != 0:
            return

        self.board[row][col] = self.current_player
        self.update_board_display()

        if self.check_winner() or self.is_board_full():
            self.end_game()
            return

        self.current_player *= -1

        if self.game_over.currentText() == "Ember vs AI" and self.current_player == -1:
            self.ai_move()

    def ai_move(self):
        if self.game_over:
            return

        # Elérhető pozíciók keresése
        available_moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    available_moves.append((i, j))

        if available_moves:
            # Egyszerű AI: random lépés
            import random

            row, col = random.choice(available_moves)

            # AI lépés végrehajtása
            self.board[row][col] = self.current_player
            self.update_board_display()

            # Játék vége ellenőrzése
            if self.check_winner() or self.is_board_full():
                self.end_game()
                return

            # Játékos váltás vissza az emberre
            self.current_player *= -1

    def update_board_display(self):
        for i in range(3):
            for j in range(3):
                button = self.board_buttons[i][j]
                value = self.board[i][j]

                if value == 1:
                    button.setText("X")
                elif value == -1:
                    button.setText("O")
                else:
                    button.setText("")

    def check_winner(self):
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

    def is_board_full(self):
        """Tábla tele van-e"""
        return not any(0 in row for row in self.board)

    def end_game(self):
        """Játék befejezése"""
        self.game_over = True

        # Gombok letiltása
        for i in range(3):
            for j in range(3):
                self.board_buttons[i][j].setEnabled(False)

    def start_new_game(self):
        """Új játék indítása"""
        # Játék állapot visszaállítása
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None

        # UI frissítése
        for i in range(3):
            for j in range(3):
                button = self.board_buttons[i][j]
                button.setText("")
                button.setEnabled(True)
                button.setStyleSheet(
                    """
                    QPushButton {
                        border: 2px solid #333;
                        border-radius: 10px;
                        background-color: white;
                        font-size: 24px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #e0e0e0;
                    }
                    QPushButton:pressed {
                        background-color: #d0d0d0;
                    }
                """
                )

    def start_ai_training(self):
        """AI edzés indítása"""
        self.update_status("AI edzés folyamatban...")
        # Itt később implementáljuk a RL edzést

        # Egyelőre csak egy üzenet
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.information(
            self,
            "AI Edzés",
            "AI edzés funkció hamarosan elérhető!\n" "Itt fog történni a Reinforcement Learning edzés.",
        )


def main():
    """Fő alkalmazás indítása"""
    app = QApplication(sys.argv)

    # Alkalmazás beállítások
    app.setApplicationName("Tic-Tac-Toe RL")
    app.setApplicationVersion("1.0")

    # Fő ablak létrehozása
    window = TicTacToeGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
