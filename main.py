import sys

from PyQt6.QtWidgets import QApplication

from ui.gui_logic import TicTacToeGUI


def main():
    """Fő alkalmazás indítása"""
    app = QApplication(sys.argv)

    # Fő ablak létrehozása
    window = TicTacToeGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
