import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Konstansok importálása
from .constants import (
    BOARD_SIZE,
    EMPTY_CELL,
    PLAYER_O,
    PLAYER_X,
    RESULT_DRAW,
    RESULT_LOSE,
    RESULT_ONGOING,
    RESULT_WIN,
    TOTAL_CELLS,
)


class GameState(Enum):
    """Játék állapotok"""

    ONGOING = "ongoing"
    X_WINS = "x_wins"
    O_WINS = "o_wins"
    DRAW = "draw"


class TicTacToeEnvironment:
    def __init__(self):
        """Environment inicializálása"""
        self.board = None
        self.current_player = None
        self.game_state = None
        self.move_count = 0
        self.game_history = []

        # Statisztikák
        self.stats = {"games_played": 0, "x_wins": 0, "o_wins": 0, "draws": 0, "total_moves": 0}

        self.reset()

    def reset(self) -> np.ndarray:
        """
        Játék újraindítása

        Returns:
            np.ndarray: Kezdő állapot (3x3 tábla)
        """
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = PLAYER_X  # X mindig kezd
        self.game_state = GameState.ONGOING
        self.move_count = 0
        self.game_history = []

        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Lépés végrehajtása

        Args:
            action (int): Akció (0-8, balról jobbra, fentről le)

        Returns:
            Tuple[np.ndarray, float, bool, Dict]:
                - next_state: Következő állapot
                - reward: Jutalom
                - done: Játék véget ért-e
                - info: További információk
        """
        if self.game_state != GameState.ONGOING:
            raise ValueError("A játék már véget ért! Használd a reset() metódust.")

        if not self.is_valid_action(action):
            return (
                self.get_state(),
                -0.1,
                False,
                {"error": "Invalid action", "valid_actions": self.get_valid_actions()},
            )

        # Lépés végrehajtása
        row, col = self.action_to_position(action)
        self.board[row, col] = self.current_player
        self.move_count += 1

        # Lépés mentése a történetbe
        self.game_history.append(
            {"player": self.current_player, "action": action, "position": (row, col), "board_state": self.board.copy()}
        )

        # Játék állapot ellenőrzése
        winner = self.check_winner()
        is_draw = self.is_board_full() and winner is None

        # JAVÍTÁS: Játék állapot frissítése ELŐBB
        done = winner is not None or is_draw
        if done:
            self.update_game_state(winner, is_draw)
            self.update_stats()  # Itt már a helyes game_state van beállítva

        # Jutalom számítása
        reward = self.calculate_reward(winner, is_draw)

        # Info dictionary összeállítása
        info = {
            "winner": winner,
            "is_draw": is_draw,
            "move_count": self.move_count,
            "current_player": self.current_player,
            "valid_actions": self.get_valid_actions() if not done else [],
            "game_state": self.game_state.value,
        }

        # Játékos váltás csak akkor, ha a játék nem ért véget
        if not done:
            self.current_player = -self.current_player

        return self.get_state(), reward, done, info

    def get_state(self) -> np.ndarray:
        """
        Aktuális állapot lekérése

        Returns:
            np.ndarray: 3x3 tábla másolata
        """
        return self.board.copy()

    def get_state_key(self, board: Optional[np.ndarray] = None) -> str:
        """
        Állapot string reprezentációja (Q-táblázat kulcshoz)

        Args:
            board: Opcionális tábla (alapértelmezett: aktuális tábla)

        Returns:
            str: Állapot string reprezentáció
        """
        if board is None:
            board = self.board

        # Tábla -> string konverzió
        state_str = "".join(str(cell) for row in board for cell in row)
        return state_str

    def get_canonical_state_key(self, board: Optional[np.ndarray] = None) -> str:
        """
        Kanonikus állapot kulcs (szimmetriák figyelembevételével)

        Args:
            board: Opcionális tábla

        Returns:
            str: Kanonikus állapot kulcs
        """
        if board is None:
            board = self.board

        # Összes szimmetria generálása
        symmetries = self.get_all_symmetries(board)

        # Lexikografikusan legkisebb választása
        canonical = min(self.get_state_key(sym) for sym in symmetries)
        return canonical

    def get_all_symmetries(self, board: np.ndarray) -> List[np.ndarray]:
        """
        Tábla összes szimmetriájának generálása

        Args:
            board: 3x3 tábla

        Returns:
            List[np.ndarray]: 8 szimmetrikus tábla
        """
        symmetries = []
        current = board.copy()

        # 4 forgatás
        for _ in range(4):
            symmetries.append(current.copy())
            current = np.rot90(current)

        # Tükrözés + 4 forgatás
        current = np.fliplr(board)
        for _ in range(4):
            symmetries.append(current.copy())
            current = np.rot90(current)

        return symmetries

    def is_valid_action(self, action: int) -> bool:
        """
        Akció érvényességének ellenőrzése

        Args:
            action: Akció (0-8)

        Returns:
            bool: Érvényes-e az akció
        """
        if not (0 <= action < TOTAL_CELLS):
            return False

        row, col = self.action_to_position(action)
        return self.board[row, col] == EMPTY_CELL

    def get_valid_actions(self) -> List[int]:
        """
        Érvényes akciók listája

        Returns:
            List[int]: Érvényes akciók (0-8)
        """
        valid_actions = []
        for action in range(TOTAL_CELLS):
            if self.is_valid_action(action):
                valid_actions.append(action)
        return valid_actions

    def action_to_position(self, action: int) -> Tuple[int, int]:
        """
        Akció -> (sor, oszlop) konverzió

        Args:
            action: Akció (0-8)

        Returns:
            Tuple[int, int]: (sor, oszlop)
        """
        row = action // BOARD_SIZE
        col = action % BOARD_SIZE
        return row, col

    def position_to_action(self, row: int, col: int) -> int:
        """
        (sor, oszlop) -> akció konverzió

        Args:
            row: Sor (0-2)
            col: Oszlop (0-2)

        Returns:
            int: Akció (0-8)
        """
        return row * BOARD_SIZE + col

    def check_winner(self) -> Optional[int]:
        """
        Győztes ellenőrzése

        Returns:
            Optional[int]: Győztes játékos (1, -1) vagy None
        """
        # Sorok ellenőrzése
        for row in self.board:
            if abs(sum(row)) == BOARD_SIZE and EMPTY_CELL not in row:
                return PLAYER_X if sum(row) > 0 else PLAYER_O

        # Oszlopok ellenőrzése
        for col in range(BOARD_SIZE):
            col_sum = sum(self.board[row, col] for row in range(BOARD_SIZE))
            if abs(col_sum) == BOARD_SIZE:
                return PLAYER_X if col_sum > 0 else PLAYER_O

        # Főátló ellenőrzése
        main_diag = sum(self.board[i, i] for i in range(BOARD_SIZE))
        if abs(main_diag) == BOARD_SIZE:
            return PLAYER_X if main_diag > 0 else PLAYER_O

        # Mellékátló ellenőrzése
        anti_diag = sum(self.board[i, BOARD_SIZE - 1 - i] for i in range(BOARD_SIZE))
        if abs(anti_diag) == BOARD_SIZE:
            return PLAYER_X if anti_diag > 0 else PLAYER_O

        return None

    def is_board_full(self) -> bool:
        """
        Tábla tele van-e

        Returns:
            bool: Tele van-e a tábla
        """
        return not any(EMPTY_CELL in row for row in self.board)

    def calculate_reward(self, winner: Optional[int], is_draw: bool) -> float:
        """
        Jutalom számítása

        Args:
            winner: Győztes játékos
            is_draw: Döntetlen-e

        Returns:
            float: Jutalom érték
        """

        base_reward = 0.0

        if winner == self.current_player:
            # Győzelem - korai győzelem extra jutalom
            moves_bonus = max(0, (9 - self.move_count) * 0.1)
            base_reward = 1.0 + moves_bonus
        elif winner is not None:
            # Vereség - korai vereség extra büntetés
            moves_penalty = max(0, (9 - self.move_count) * 0.05)
            base_reward = -1.0 - moves_penalty
        elif is_draw:
            base_reward = 0.3  # Döntetlen pozitív (jobb mint vereség)
        else:
            # Köztes jutalmak - FONTOS JAVÍTÁS!
            base_reward = self.get_positional_reward()

        return base_reward

    def get_positional_reward(self) -> float:
        """Pozicionális jutalmak"""
        reward = 0.0
        player = self.current_player

        # Központ elfoglalása
        if self.board[1, 1] == player:
            reward += 0.1

        # Sarok pozíciók értékelése
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        corner_count = sum(1 for r, c in corners if self.board[r, c] == player)
        reward += corner_count * 0.05

        # Két egy sorban - nyerő lehetőség
        reward += self.count_potential_wins(player) * 0.3

        # Ellenfél blokkolása
        opponent = -player
        reward += self.count_blocked_threats(opponent) * 0.4

        return reward

    def count_blocked_threats(self, opponent: int) -> int:
        """Blokkolt ellenfél fenyegetések"""
        count = 0
        lines = self.get_all_lines()

        for line in lines:
            if line.count(opponent) == 2 and line.count(self.current_player) >= 1:
                count += 1

        return count

    def count_potential_wins(self, player: int) -> int:
        """Potenciális nyerő helyzetek számolása"""
        count = 0
        lines = self.get_all_lines()

        for line in lines:
            if line.count(player) == 2 and line.count(0) == 1:
                count += 1

        return count

    def get_all_lines(self) -> List[List[int]]:
        """Összes sor/oszlop/átló"""
        lines = []

        # Sorok
        for i in range(3):
            lines.append([self.board[i, j] for j in range(3)])

        # Oszlopok
        for j in range(3):
            lines.append([self.board[i, j] for i in range(3)])

        # Átlók
        lines.append([self.board[i, i] for i in range(3)])
        lines.append([self.board[i, 2 - i] for i in range(3)])

        return lines

    def update_game_state(self, winner: Optional[int], is_draw: bool):
        """
        Játék állapot frissítése

        Args:
            winner: Győztes játékos
            is_draw: Döntetlen-e
        """
        if winner == PLAYER_X:
            self.game_state = GameState.X_WINS
        elif winner == PLAYER_O:
            self.game_state = GameState.O_WINS
        elif is_draw:
            self.game_state = GameState.DRAW

    def update_stats(self):
        """Statisztikák frissítése"""
        self.stats["games_played"] += 1

        self.stats["total_moves"] += self.move_count

        # A game_state alapján frissítjük a statisztikákat
        if self.game_state == GameState.X_WINS:
            self.stats["x_wins"] += 1
        elif self.game_state == GameState.O_WINS:
            self.stats["o_wins"] += 1
        elif self.game_state == GameState.DRAW:
            self.stats["draws"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """
        Statisztikák lekérése

        Returns:
            Dict[str, Any]: Statisztikák
        """
        stats = self.stats.copy()
        if stats["games_played"] > 0:
            stats["avg_moves_per_game"] = stats["total_moves"] / stats["games_played"]
            stats["x_win_rate"] = stats["x_wins"] / stats["games_played"]
            stats["o_win_rate"] = stats["o_wins"] / stats["games_played"]
            stats["draw_rate"] = stats["draws"] / stats["games_played"]
        else:
            stats["avg_moves_per_game"] = 0
            stats["x_win_rate"] = 0
            stats["o_win_rate"] = 0
            stats["draw_rate"] = 0

        return stats

    def render(self, mode: str = "human") -> Optional[str]:
        """
        Tábla megjelenítése

        Args:
            mode: Megjelenítési mód ('human', 'ansi')

        Returns:
            Optional[str]: String reprezentáció (ha mode != 'human')
        """
        symbols = {EMPTY_CELL: " ", PLAYER_X: "X", PLAYER_O: "O"}

        board_str = "\n"
        board_str += "  0   1   2\n"
        board_str += "┌───┬───┬───┐\n"

        for i in range(BOARD_SIZE):
            row_str = f"{i}│"
            for j in range(BOARD_SIZE):
                symbol = symbols[self.board[i, j]]
                row_str += f" {symbol} │"
            board_str += row_str + "\n"

            if i < BOARD_SIZE - 1:
                board_str += "├───┼───┼───┤\n"

        board_str += "└───┴───┴───┘\n"

        # Játék információk
        if self.game_state == GameState.ONGOING:
            current_symbol = symbols[self.current_player]
            board_str += f"\nKövetkező játékos: {current_symbol} ({self.current_player})\n"
        else:
            board_str += f"\nJáték vége: {self.game_state.value}\n"

        board_str += f"Lépések száma: {self.move_count}\n"

        if mode == "human":
            print(board_str)
        else:
            return board_str

    def clone(self) -> "TicTacToeEnvironment":
        new_env = TicTacToeEnvironment()
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.game_state = self.game_state
        new_env.move_count = self.move_count
        new_env.game_history = self.game_history.copy()
        new_env.stats = self.stats.copy()

        return new_env

    def get_game_history(self) -> List[Dict[str, Any]]:
        """
        Játék történet lekérése

        Returns:
            List[Dict]: Lépések listája részletes információkkal
        """
        return self.game_history.copy()

    def simulate_random_game(self) -> Dict[str, Any]:
        """
        Random játék szimulálása (teszteléshez)

        Returns:
            Dict[str, Any]: Játék eredménye és statisztikái
        """
        self.reset()

        moves = []

        while self.game_state == GameState.ONGOING:
            valid_actions = self.get_valid_actions()
            if not valid_actions:
                break

            action = random.choice(valid_actions)

            # JAVÍTÁS: Aktuális játékos mentése a lépés előtt
            current_player_before_move = self.current_player

            state, reward, done, info = self.step(action)

            moves.append(
                {
                    "action": action,
                    "player": current_player_before_move,  # A lépést végrehajtó játékos
                    "reward": reward,
                    "state": state.copy(),
                }
            )

            if done:
                break

        return {
            "winner": info.get("winner") if "info" in locals() else None,
            "is_draw": info.get("is_draw", False) if "info" in locals() else False,
            "moves": moves,
            "total_moves": len(moves),
            "final_state": self.get_state(),
            "game_state": self.game_state.value,
        }

    def evaluate_position(self, player: int) -> float:
        """
        Pozíció értékelése egy adott játékos szempontjából

        Args:
            player: Játékos (PLAYER_X vagy PLAYER_O)

        Returns:
            float: Pozíció értéke (-1.0 - 1.0)
        """
        winner = self.check_winner()

        if winner == player:
            return 1.0
        elif winner == -player:
            return -1.0
        elif self.is_board_full():
            return 0.0

        # Heurisztikus értékelés (opcionális)
        score = 0.0

        # Központi pozíció értékelése
        if self.board[1, 1] == player:
            score += 0.1
        elif self.board[1, 1] == -player:
            score -= 0.1

        # Sarok pozíciók értékelése
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        for row, col in corners:
            if self.board[row, col] == player:
                score += 0.05
            elif self.board[row, col] == -player:
                score -= 0.05

        return score

    def get_winning_positions(self) -> List[List[Tuple[int, int]]]:
        """
        Összes nyerő pozíció kombinációja

        Returns:
            List[List[Tuple]]: Nyerő pozíciók listája
        """
        winning_positions = []

        # Sorok
        for row in range(BOARD_SIZE):
            positions = [(row, col) for col in range(BOARD_SIZE)]
            winning_positions.append(positions)

        # Oszlopok
        for col in range(BOARD_SIZE):
            positions = [(row, col) for row in range(BOARD_SIZE)]
            winning_positions.append(positions)

        # Főátló
        positions = [(i, i) for i in range(BOARD_SIZE)]
        winning_positions.append(positions)

        # Mellékátló
        positions = [(i, BOARD_SIZE - 1 - i) for i in range(BOARD_SIZE)]
        winning_positions.append(positions)

        return winning_positions

    def find_winning_move(self, player: int) -> Optional[int]:
        """
        Nyerő lépés keresése

        Args:
            player: Játékos

        Returns:
            Optional[int]: Nyerő akció vagy None
        """
        for action in self.get_valid_actions():
            # Lépés szimulálása
            temp_env = self.clone()
            temp_env.step(action)

            if temp_env.check_winner() == player:
                return action

        return None

    def find_blocking_move(self, player: int) -> Optional[int]:
        """
        Blokkoló lépés keresése (ellenfél nyerésének megakadályozása)

        Args:
            player: Saját játékos

        Returns:
            Optional[int]: Blokkoló akció vagy None
        """
        opponent = -player
        return self.find_winning_move(opponent)

    def get_strategic_moves(self, player: int) -> List[Tuple[int, str, float]]:
        """
        Stratégiai lépések prioritás szerint

        Args:
            player: Játékos

        Returns:
            List[Tuple[int, str, float]]: (akció, típus, prioritás) lista
        """
        moves = []

        # 1. Nyerő lépés
        winning_move = self.find_winning_move(player)
        if winning_move is not None:
            moves.append((winning_move, "winning", 1.0))

        # 2. Blokkoló lépés
        blocking_move = self.find_blocking_move(player)
        if blocking_move is not None:
            moves.append((blocking_move, "blocking", 0.9))

        # 3. Központ
        center_action = self.position_to_action(1, 1)
        if self.is_valid_action(center_action):
            moves.append((center_action, "center", 0.7))

        # 4. Sarok pozíciók
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        for row, col in corners:
            action = self.position_to_action(row, col)
            if self.is_valid_action(action):
                moves.append((action, "corner", 0.5))

        # 5. Oldal pozíciók
        sides = [(0, 1), (1, 0), (1, 2), (2, 1)]
        for row, col in sides:
            action = self.position_to_action(row, col)
            if self.is_valid_action(action):
                moves.append((action, "side", 0.3))

        # Prioritás szerint rendezés
        moves.sort(key=lambda x: x[2], reverse=True)
        return moves

    def is_terminal_state(self) -> bool:
        """
        Terminális állapot-e (játék véget ért)

        Returns:
            bool: Terminális állapot-e
        """
        return self.game_state != GameState.ONGOING

    def get_state_features(self) -> np.ndarray:
        """
        Állapot jellemzők kinyerése (neural network-höz)

        Returns:
            np.ndarray: Jellemző vektor
        """
        features = []

        # Tábla állapot (flatten)
        features.extend(self.board.flatten())

        # Aktuális játékos
        features.append(self.current_player)

        # Lépések száma
        features.append(self.move_count / TOTAL_CELLS)  # normalizált

        # Pozíció értékelések
        features.append(self.evaluate_position(PLAYER_X))
        features.append(self.evaluate_position(PLAYER_O))

        # Érvényes lépések száma
        features.append(len(self.get_valid_actions()) / TOTAL_CELLS)

        return np.array(features, dtype=np.float32)

    def reset_stats(self):
        """Statisztikák nullázása"""
        self.stats = {"games_played": 0, "x_wins": 0, "o_wins": 0, "draws": 0, "total_moves": 0}

    def __str__(self) -> str:
        """String reprezentáció"""
        return self.render(mode="ansi")

    def __repr__(self) -> str:
        """Objektum reprezentáció"""
        return f"TicTacToeEnvironment(state={self.game_state.value}, moves={self.move_count})"


# =============================================================================
# SEGÉD FÜGGVÉNYEK
# =============================================================================


def create_environment() -> TicTacToeEnvironment:
    """
    Új environment létrehozása

    Returns:
        TicTacToeEnvironment: Új environment példány
    """
    return TicTacToeEnvironment()


def play_random_games(num_games: int = 1000) -> Dict[str, Any]:
    """
    Random játékok lejátszása statisztikákhoz

    Args:
        num_games: Játékok száma

    Returns:
        Dict[str, Any]: Összesített statisztikák
    """
    env = create_environment()
    results = []

    for _ in range(num_games):
        result = env.simulate_random_game()
        results.append(result)

    # Statisztikák összesítése
    x_wins = sum(1 for r in results if r["winner"] == PLAYER_X)
    o_wins = sum(1 for r in results if r["winner"] == PLAYER_O)
    draws = sum(1 for r in results if r["is_draw"])
    total_moves = sum(r["total_moves"] for r in results)

    return {
        "total_games": num_games,
        "x_wins": x_wins,
        "o_wins": o_wins,
        "draws": draws,
        "x_win_rate": x_wins / num_games,
        "o_win_rate": o_wins / num_games,
        "draw_rate": draws / num_games,
        "avg_moves_per_game": total_moves / num_games,
        "results": results,
    }


def test_environment():
    """Environment tesztelése"""
    print("=== TicTacToe Environment Teszt ===\n")

    env = create_environment()

    # Alapvető funkciók tesztelése
    print("1. Kezdő állapot:")
    env.render()

    print(f"Érvényes akciók: {env.get_valid_actions()}")
    print(f"Állapot kulcs: {env.get_state_key()}")

    # Néhány lépés végrehajtása
    print("\n2. Lépések végrehajtása:")
    actions = [4, 0, 8, 2]  # Központ, bal felső, jobb alsó, jobb felső

    for i, action in enumerate(actions):
        if env.is_valid_action(action):
            state, reward, done, info = env.step(action)
            print(f"\nLépés {i+1}: Akció {action}")
            env.render()
            print(f"Jutalom: {reward}, Vége: {done}")

            if done:
                print(f"Játék vége! Győztes: {info.get('winner', 'Senki')}")
                break

    # Statisztikák
    print(f"\n3. Statisztikák:")
    stats = env.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Random játékok tesztelése
    print(f"\n4. Random játékok tesztelése:")
    random_stats = play_random_games(100)
    print(f"X győzelmi arány: {random_stats['x_win_rate']:.2%}")
    print(f"O győzelmi arány: {random_stats['o_win_rate']:.2%}")
    print(f"Döntetlen arány: {random_stats['draw_rate']:.2%}")
    print(f"Átlagos lépések: {random_stats['avg_moves_per_game']:.1f}")


if __name__ == "__main__":
    test_environment()
