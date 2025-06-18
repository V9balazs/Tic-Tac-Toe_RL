#!/usr/bin/env python3
"""
Tic-Tac-Toe Reinforcement Learning
Fő alkalmazás belépési pont
"""

import os
import sys

# Path beállítása
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from ui.gui_logic import TicTacToeGUI


def main():
    """Fő alkalmazás indítása"""
    # QApplication létrehozása
    app = QApplication(sys.argv)

    try:
        # Fő ablak létrehozása és megjelenítése
        window = TicTacToeGUI()
        window.show()

        print("Alkalmazás elindítva!")

        # Alkalmazás futtatása
        sys.exit(app.exec())

    except Exception as e:
        print(f"Hiba az alkalmazás indításakor: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
