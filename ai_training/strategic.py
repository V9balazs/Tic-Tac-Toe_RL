"""
Stratégiai ellenfél implementáció
"""

import random
from typing import List, Optional

import numpy as np
from tictactoe_environment import TicTacToeEnvironment


class StrategicOpponent:
    """
    Stratégiai ellenfél - nem random lépések
    Különböző nehézségi szintekkel
    """

    def __init__(self, player: int, difficulty: float = 0.8):
        """
        Args:
            player: Játékos azonosító (-1 vagy 1)
            difficulty: Nehézségi szint (0.0 = random, 1.0 = tökéletes)
        """
        self.player = player
        self.difficulty = difficulty
        self.opponent = -player

    def choose_action(
        self, board: np.ndarray, valid_actions: List[int], env: TicTacToeEnvironment, training: bool = True
    ) -> int:
        """
        Stratégiai akció választás

        Args:
            board: Aktuális tábla
            valid_actions: Érvényes akciók
            env: Environment
            training: Tanulási mód (nem használt)

        Returns:
            int: Választott akció
        """
        # Nehézségi szint alapján random esély
        if random.random() > self.difficulty:
            return random.choice(valid_actions)

        # 1. Prioritás: Nyerő lépés
        winning_move = self.find_winning_move(env, valid_actions)
        if winning_move is not None:
            return winning_move

        # 2. Prioritás: Ellenfél blokkolása
        blocking_move = self.find_blocking_move(env, valid_actions)
        if blocking_move is not None:
            return blocking_move

        # 3. Prioritás: Fork létrehozása (két nyerő lehetőség)
        fork_move = self.find_fork_move(env, valid_actions)
        if fork_move is not None:
            return fork_move

        # 4. Prioritás: Ellenfél fork blokkolása
        block_fork_move = self.find_block_fork_move(env, valid_actions)
        if block_fork_move is not None:
            return block_fork_move

        # 5. Prioritás: Stratégiai pozíciók
        strategic_move = self.get_best_strategic_move(env, valid_actions)
        if strategic_move is not None:
            return strategic_move

        # Fallback: random
        return random.choice(valid_actions)

    def find_winning_move(self, env: TicTacToeEnvironment, valid_actions: List[int]) -> Optional[int]:
        """Nyerő lépés keresése"""
        for action in valid_actions:
            temp_env = env.clone()
            temp_env.current_player = self.player
            temp_env.step(action)
            if temp_env.check_winner() == self.player:
                return action
        return None

    def find_blocking_move(self, env: TicTacToeEnvironment, valid_actions: List[int]) -> Optional[int]:
        """Ellenfél nyerő lépésének blokkolása"""
        for action in valid_actions:
            temp_env = env.clone()
            temp_env.current_player = self.opponent
            temp_env.step(action)
            if temp_env.check_winner() == self.opponent:
                return action
        return None

    def find_fork_move(self, env: TicTacToeEnvironment, valid_actions: List[int]) -> Optional[int]:
        """Fork létrehozása - két nyerő lehetőség egyszerre"""
        for action in valid_actions:
            temp_env = env.clone()
            temp_env.current_player = self.player
            temp_env.step(action)

            # Számoljuk meg a nyerő lehetőségeket
            winning_moves = 0
            for next_action in temp_env.get_valid_actions():
                test_env = temp_env.clone()
                test_env.current_player = self.player
                test_env.step(next_action)
                if test_env.check_winner() == self.player:
                    winning_moves += 1

            # Ha 2 vagy több nyerő lehetőség, az fork
            if winning_moves >= 2:
                return action

        return None

    def find_block_fork_move(self, env: TicTacToeEnvironment, valid_actions: List[int]) -> Optional[int]:
        """Ellenfél fork blokkolása"""
        opponent_forks = []

        for action in valid_actions:
            temp_env = env.clone()
            temp_env.current_player = self.opponent
            temp_env.step(action)

            # Számoljuk meg az ellenfél nyerő lehetőségeit
            winning_moves = 0
            for next_action in temp_env.get_valid_actions():
                test_env = temp_env.clone()
                test_env.current_player = self.opponent
                test_env.step(next_action)
                if test_env.check_winner() == self.opponent:
                    winning_moves += 1

            # Ha az ellenfélnek fork-ja lenne
            if winning_moves >= 2:
                opponent_forks.append(action)

        # Ha van ellenfél fork, blokkoljuk
        if opponent_forks:
            # Válasszuk a legjobb blokkoló lépést
            return self.choose_best_blocking_move(env, opponent_forks, valid_actions)

        return None

    def choose_best_blocking_move(
        self, env: TicTacToeEnvironment, fork_actions: List[int], valid_actions: List[int]
    ) -> int:
        """Legjobb blokkoló lépés választása"""
        # Próbáljunk olyan lépést, ami blokkolja a fork-ot ÉS fenyegetést hoz létre
        for action in valid_actions:
            if action in fork_actions:
                continue

            temp_env = env.clone()
            temp_env.current_player = self.player
            temp_env.step(action)

            # Ellenőrizzük, hogy ez fenyegetést hoz-e létre
            next_valid = temp_env.get_valid_actions()
            for next_action in next_valid:
                test_env = temp_env.clone()
                test_env.current_player = self.player
                test_env.step(next_action)
                if test_env.check_winner() == self.player:
                    return action

        # Ha nincs jó blokkoló lépés, válasszunk egyet a fork pozíciók közül
        return random.choice(fork_actions) if fork_actions else random.choice(valid_actions)

    def get_best_strategic_move(self, env: TicTacToeEnvironment, valid_actions: List[int]) -> Optional[int]:
        """Legjobb stratégiai lépés kiválasztása"""
        # Pozíciók prioritási sorrendben
        strategic_priorities = [4, 0, 2, 6, 8, 1, 3, 5, 7]  # Központ (1,1)  # Sarkok  # Oldalak

        for priority_action in strategic_priorities:
            if priority_action in valid_actions:
                # További ellenőrzések a pozíció értékére
                if self.is_good_strategic_position(env, priority_action):
                    return priority_action

        return None

    def is_good_strategic_position(self, env: TicTacToeEnvironment, action: int) -> bool:
        """Ellenőrzi, hogy egy pozíció stratégiailag jó-e"""
        row, col = action // 3, action % 3

        # Központ mindig jó
        if action == 4:
            return True

        # Sarkok jók, ha a központ foglalt
        if action in [0, 2, 6, 8]:
            if env.board[1, 1] != 0:  # Központ foglalt
                return True
            # Vagy ha az ellenfél sarokba lépett
            corners_occupied = sum(1 for corner in [0, 2, 6, 8] if env.board[corner // 3, corner % 3] == self.opponent)
            return corners_occupied > 0

        # Oldalak általában kevésbé jók, de néha szükségesek
        return True

    def evaluate_position(self, env: TicTacToeEnvironment) -> float:
        """Pozíció értékelése a saját szempontunkból"""
        score = 0.0

        # Ellenőrizzük az összes sort/oszlopot/átlót
        lines = self.get_all_lines(env.board)

        for line in lines:
            line_score = self.evaluate_line(line)
            score += line_score

        return score

    def evaluate_line(self, line: List[int]) -> float:
        """Egy sor/oszlop/átló értékelése"""
        my_count = line.count(self.player)
        opp_count = line.count(self.opponent)
        empty_count = line.count(0)

        # Ha mindkét játékos van a sorban, az semleges
        if my_count > 0 and opp_count > 0:
            return 0.0

        # Saját előnyök
        if my_count == 3:
            return 100.0  # Nyerés
        elif my_count == 2 and empty_count == 1:
            return 10.0  # Közel a nyeréshez
        elif my_count == 1 and empty_count == 2:
            return 1.0  # Jó kezdet

        # Ellenfél fenyegetések
        elif opp_count == 3:
            return -100.0  # Vereség
        elif opp_count == 2 and empty_count == 1:
            return -10.0  # Ellenfél közel a nyeréshez
        elif opp_count == 1 and empty_count == 2:
            return -1.0  # Ellenfél jó kezdete

        return 0.0

    def get_all_lines(self, board: np.ndarray) -> List[List[int]]:
        """Összes sor/oszlop/átló lekérése"""
        lines = []

        # Sorok
        for i in range(3):
            lines.append([board[i, j] for j in range(3)])

        # Oszlopok
        for j in range(3):
            lines.append([board[i, j] for i in range(3)])

        # Átlók
        lines.append([board[i, i] for i in range(3)])
        lines.append([board[i, 2 - i] for i in range(3)])

        return lines

    def get_stats(self) -> dict:
        """Statisztikák (kompatibilitás miatt)"""
        return {"player": self.player, "difficulty": self.difficulty, "type": "strategic_opponent"}


# Különböző nehézségi szintű ellenfelek létrehozása
def create_easy_opponent(player: int = -1) -> StrategicOpponent:
    """Könnyű stratégiai ellenfél"""
    return StrategicOpponent(player, difficulty=0.3)


def create_medium_opponent(player: int = -1) -> StrategicOpponent:
    """Közepes stratégiai ellenfél"""
    return StrategicOpponent(player, difficulty=0.6)


def create_hard_opponent(player: int = -1) -> StrategicOpponent:
    """Nehéz stratégiai ellenfél"""
    return StrategicOpponent(player, difficulty=0.9)


def create_perfect_opponent(player: int = -1) -> StrategicOpponent:
    """Tökéletes stratégiai ellenfél"""
    return StrategicOpponent(player, difficulty=1.0)
