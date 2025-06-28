"""
Q-Learning Agent - Tic-Tac-Toe Reinforcement Learning
Epsilon-greedy stratégiával és experience replay-jel
"""

import json
import os
import pickle
import random
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Konstansok importálása
from .constants import (
    DISCOUNT_FACTOR,
    EMPTY_CELL,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    INITIAL_Q_VALUE,
    LEARNING_RATE,
    MODELS_DIR,
    PLAYER_O,
    PLAYER_X,
    Q_TABLE_FILE,
    RESULT_DRAW,
    RESULT_LOSE,
    RESULT_WIN,
    TRAINING_EPISODES,
    TRAINING_STATS_FILE,
)
from .tictactoe_environment import TicTacToeEnvironment


class QLearningAgent:
    """
    Q-Learning Agent Tic-Tac-Toe játékhoz

    Főbb funkciók:
    - Epsilon-greedy akció választás
    - Q-táblázat frissítés Bellman egyenlettel
    - Állapot szimmetriák kezelése
    - Experience replay
    - Modell mentés/betöltés
    """

    def __init__(
        self,
        player: int = PLAYER_X,
        learning_rate: float = LEARNING_RATE,
        discount_factor: float = DISCOUNT_FACTOR,
        epsilon_start: float = EPSILON_START,
        epsilon_end: float = EPSILON_END,
        epsilon_decay: float = EPSILON_DECAY,
        use_symmetries: bool = True,
    ):
        """
        Q-Learning Agent inicializálása

        Args:
            player: Játékos azonosító (PLAYER_X vagy PLAYER_O)
            learning_rate: Tanulási ráta (α)
            discount_factor: Diszkont faktor (γ)
            epsilon_start: Kezdő exploration ráta
            epsilon_end: Végső exploration ráta
            epsilon_decay: Epsilon csökkenési ráta
            use_symmetries: Szimmetriák használata
        """
        self.player = player
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.use_symmetries = use_symmetries

        # Q-táblázat: {state_key: {action: q_value}}
        self.q_table = defaultdict(lambda: defaultdict(lambda: INITIAL_Q_VALUE))

        # Experience replay buffer
        self.experience_buffer = deque(maxlen=10000)
        self.replay_batch_size = 32

        # Statisztikák
        self.stats = {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "total_reward": 0.0,
            "exploration_moves": 0,
            "exploitation_moves": 0,
            "q_updates": 0,
            "training_time": 0.0,
        }

        # Tanulási történet
        self.learning_history = []

        print(f"Q-Learning Agent inicializálva - Játékos: {player}")

    def get_state_key(self, board: np.ndarray, env: Optional[TicTacToeEnvironment] = None) -> str:
        """
        Állapot kulcs generálása

        Args:
            board: Játéktábla
            env: Environment (szimmetriákhoz)

        Returns:
            str: Állapot kulcs
        """
        if self.use_symmetries and env is not None:
            return env.get_canonical_state_key(board)
        else:
            return "".join(str(cell) for row in board for cell in row)

    def get_q_value(self, state_key: str, action: int) -> float:
        """
        Q-érték lekérése

        Args:
            state_key: Állapot kulcs
            action: Akció

        Returns:
            float: Q-érték
        """
        return self.q_table[state_key][action]

    def set_q_value(self, state_key: str, action: int, value: float):
        """
        Q-érték beállítása

        Args:
            state_key: Állapot kulcs
            action: Akció
            value: Új Q-érték
        """
        self.q_table[state_key][action] = value

    def choose_action(
        self,
        board: np.ndarray,
        valid_actions: List[int],
        env: Optional[TicTacToeEnvironment] = None,
        training: bool = True,
    ) -> int:
        """
        Akció választás epsilon-greedy stratégiával

        Args:
            board: Aktuális tábla
            valid_actions: Érvényes akciók listája
            env: Environment
            training: Tanulási mód

        Returns:
            int: Választott akció
        """
        if not valid_actions:
            raise ValueError("Nincsenek érvényes akciók!")

        state_key = self.get_state_key(board, env)

        # Epsilon-greedy döntés
        if training and random.random() < self.epsilon:
            # Exploration: random akció
            action = random.choice(valid_actions)
            self.stats["exploration_moves"] += 1
        else:
            # Exploitation: legjobb Q-értékű akció
            q_values = [(action, self.get_q_value(state_key, action)) for action in valid_actions]

            # Legjobb akciók keresése
            max_q = max(q_values, key=lambda x: x[1])[1]
            best_actions = [action for action, q in q_values if q == max_q]

            # Ha több egyforma legjobb akció van, random választás
            action = random.choice(best_actions)
            self.stats["exploitation_moves"] += 1

        return action

    def update_q_value(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_valid_actions: List[int],
        done: bool,
        env: Optional[TicTacToeEnvironment] = None,
    ):
        """
        Q-érték frissítése Bellman egyenlettel

        Args:
            state: Jelenlegi állapot
            action: Végrehajtott akció
            reward: Kapott jutalom
            next_state: Következő állapot
            next_valid_actions: Következő állapot érvényes akciói
            done: Epizód véget ért-e
            env: Environment
        """
        state_key = self.get_state_key(state, env)
        current_q = self.get_q_value(state_key, action)

        if done:
            # Terminális állapot: nincs jövőbeli jutalom
            target_q = reward
        else:
            # Következő állapot legjobb Q-értéke
            next_state_key = self.get_state_key(next_state, env)

            if next_valid_actions:
                max_next_q = max(self.get_q_value(next_state_key, next_action) for next_action in next_valid_actions)
            else:
                max_next_q = 0.0

            # Bellman egyenlet
            target_q = reward + self.discount_factor * max_next_q

        # Q-érték frissítése
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.set_q_value(state_key, action, new_q)

        self.stats["q_updates"] += 1

    def add_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        valid_actions: List[int],
        next_valid_actions: List[int],
    ):
        """
        Tapasztalat hozzáadása a replay bufferhez

        Args:
            state: Állapot
            action: Akció
            reward: Jutalom
            next_state: Következő állapot
            done: Epizód vége
            valid_actions: Érvényes akciók
            next_valid_actions: Következő érvényes akciók
        """
        experience = {
            "state": state.copy(),
            "action": action,
            "reward": reward,
            "next_state": next_state.copy(),
            "done": done,
            "valid_actions": valid_actions.copy(),
            "next_valid_actions": next_valid_actions.copy(),
        }
        self.experience_buffer.append(experience)

    def replay_experience(self, env: Optional[TicTacToeEnvironment] = None):
        """
        Experience replay - korábbi tapasztalatok újrajátszása

        Args:
            env: Environment
        """
        if len(self.experience_buffer) < self.replay_batch_size:
            return

        # Random batch választása
        batch = random.sample(self.experience_buffer, self.replay_batch_size)

        for experience in batch:
            self.update_q_value(
                experience["state"],
                experience["action"],
                experience["reward"],
                experience["next_state"],
                experience["next_valid_actions"],
                experience["done"],
                env,
            )

    def decay_epsilon(self):
        """Epsilon csökkentése"""
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def reset_epsilon(self):
        """Epsilon visszaállítása kezdő értékre"""
        self.epsilon = self.epsilon_start

    def get_policy(
        self, board: np.ndarray, valid_actions: List[int], env: Optional[TicTacToeEnvironment] = None
    ) -> Dict[int, float]:
        """
        Aktuális policy lekérése (akció valószínűségek)

        Args:
            board: Tábla
            valid_actions: Érvényes akciók
            env: Environment

        Returns:
            Dict[int, float]: {akció: valószínűség}
        """
        if not valid_actions:
            return {}

        state_key = self.get_state_key(board, env)
        q_values = {action: self.get_q_value(state_key, action) for action in valid_actions}

        # Softmax policy
        max_q = max(q_values.values())
        exp_q = {action: np.exp(q - max_q) for action, q in q_values.items()}
        sum_exp = sum(exp_q.values())

        policy = {action: exp_q[action] / sum_exp for action in valid_actions}
        return policy

    def get_best_action(
        self, board: np.ndarray, valid_actions: List[int], env: Optional[TicTacToeEnvironment] = None
    ) -> int:
        """
        Legjobb akció lekérése (greedy, epsilon nélkül)

        Args:
            board: Tábla
            valid_actions: Érvényes akciók
            env: Environment

        Returns:
            int: Legjobb akció
        """
        return self.choose_action(board, valid_actions, env, training=False)

    def evaluate_against_random(self, num_games: int = 1000) -> Dict[str, float]:
        """
        Teljesítmény értékelése random játékos ellen

        Args:
            num_games: Játékok száma

        Returns:
            Dict[str, float]: Eredmény statisztikák
        """
        wins = 0
        losses = 0
        draws = 0

        for _ in range(num_games):
            env = TicTacToeEnvironment()
            done = False

            while not done:
                valid_actions = env.get_valid_actions()

                if env.current_player == self.player:
                    # Agent lép
                    action = self.get_best_action(env.get_state(), valid_actions, env)
                else:
                    # Random ellenfél
                    action = random.choice(valid_actions)

                _, _, done, info = env.step(action)

            # Eredmény kiértékelése
            winner = info.get("winner")
            if winner == self.player:
                wins += 1
            elif winner == -self.player:
                losses += 1
            else:
                draws += 1

        return {
            "win_rate": wins / num_games,
            "loss_rate": losses / num_games,
            "draw_rate": draws / num_games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "total_games": num_games,
        }

    def train_episode(self, env: TicTacToeEnvironment, opponent_agent=None) -> Dict[str, Any]:
        """
        Egy epizód (játék) edzése

        Args:
            env: Environment
            opponent_agent: Ellenfél agent (None = random)

        Returns:
            Dict[str, Any]: Epizód eredménye
        """
        env.reset()
        episode_reward = 0.0
        episode_moves = 0
        episode_experiences = []

        while not env.is_terminal_state():
            current_state = env.get_state()
            valid_actions = env.get_valid_actions()

            if env.current_player == self.player:
                # Saját lépés
                action = self.choose_action(current_state, valid_actions, env, training=True)
            else:
                # Ellenfél lépése
                if opponent_agent is not None:
                    action = opponent_agent.choose_action(current_state, valid_actions, env, training=True)
                else:
                    # Random ellenfél
                    action = random.choice(valid_actions)

            # Lépés végrehajtása
            next_state, reward, done, info = env.step(action)
            next_valid_actions = env.get_valid_actions()

            # Csak saját lépéseket tanuljuk
            if env.current_player == -self.player or done:  # Az előző lépés a miénk volt
                if env.current_player == -self.player:
                    # Játék folytatódik, de most az ellenfél következik
                    actual_reward = 0.0  # Köztes jutalom
                else:
                    # Játék véget ért
                    winner = info.get("winner")
                    if winner == self.player:
                        actual_reward = 1.0  # Győzelem
                    elif winner == -self.player:
                        actual_reward = -1.0  # Vereség
                    else:
                        actual_reward = 0.0  # Döntetlen

                # Q-érték frissítése
                self.update_q_value(current_state, action, actual_reward, next_state, next_valid_actions, done, env)

                # Experience hozzáadása
                self.add_experience(
                    current_state, action, actual_reward, next_state, done, valid_actions, next_valid_actions
                )

                episode_reward += actual_reward

            episode_moves += 1

            if done:
                break

        # Experience replay
        if len(self.experience_buffer) >= self.replay_batch_size:
            self.replay_experience(env)

        # Epsilon csökkentése
        self.decay_epsilon()

        # Statisztikák frissítése
        self.stats["games_played"] += 1
        self.stats["total_reward"] += episode_reward

        winner = info.get("winner") if "info" in locals() else None
        if winner == self.player:
            self.stats["wins"] += 1
        elif winner == -self.player:
            self.stats["losses"] += 1
        else:
            self.stats["draws"] += 1

        return {
            "reward": episode_reward,
            "moves": episode_moves,
            "winner": winner,
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
        }

    def train(
        self,
        num_episodes: int = TRAINING_EPISODES,
        opponent_agent=None,
        evaluation_interval: int = 1000,
        save_interval: int = 5000,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Teljes edzés végrehajtása

        Args:
            num_episodes: Edzési epizódok száma
            opponent_agent: Ellenfél agent
            evaluation_interval: Értékelési gyakoriság
            save_interval: Mentési gyakoriság
            verbose: Részletes kimenet

        Returns:
            Dict[str, Any]: Edzés eredményei
        """
        start_time = time.time()
        env = TicTacToeEnvironment()

        training_history = {"episodes": [], "rewards": [], "win_rates": [], "epsilons": [], "q_table_sizes": []}

        if verbose:
            print(f"Edzés kezdése: {num_episodes} epizód")
            print(f"Játékos: {self.player}, Epsilon: {self.epsilon:.3f}")

        for episode in range(1, num_episodes + 1):
            # Egy epizód edzése
            episode_result = self.train_episode(env, opponent_agent)

            # Történet mentése
            training_history["episodes"].append(episode)
            training_history["rewards"].append(episode_result["reward"])
            training_history["epsilons"].append(episode_result["epsilon"])
            training_history["q_table_sizes"].append(episode_result["q_table_size"])

            # Periodikus értékelés
            if episode % evaluation_interval == 0:
                eval_results = self.evaluate_against_random(100)
                training_history["win_rates"].append(eval_results["win_rate"])

                if verbose:
                    print(
                        f"Epizód {episode:6d}: "
                        f"Győzelmi arány: {eval_results['win_rate']:.1%}, "
                        f"Epsilon: {self.epsilon:.3f}, "
                        f"Q-táblázat: {len(self.q_table)} állapot"
                    )

            # Periodikus mentés
            if episode % save_interval == 0:
                self.save_model(f"checkpoint_episode_{episode}")
                if verbose:
                    print(f"Modell mentve: epizód {episode}")

        # Edzés befejezése
        training_time = time.time() - start_time
        self.stats["training_time"] += training_time

        # Végső értékelés
        final_evaluation = self.evaluate_against_random(1000)

        if verbose:
            print(f"\nEdzés befejezve!")
            print(f"Időtartam: {training_time:.1f} másodperc")
            print(f"Végső győzelmi arány: {final_evaluation['win_rate']:.1%}")
            print(f"Q-táblázat mérete: {len(self.q_table)} állapot")

        # Végső modell mentése
        self.save_model("final_model")

        return {
            "training_time": training_time,
            "final_evaluation": final_evaluation,
            "training_history": training_history,
            "final_stats": self.get_stats(),
        }

    def self_play_training(self, num_episodes: int = TRAINING_EPISODES, verbose: bool = True) -> Dict[str, Any]:
        """
        Self-play edzés (agent saját maga ellen)

        Args:
            num_episodes: Edzési epizódok száma
            verbose: Részletes kimenet

        Returns:
            Dict[str, Any]: Edzés eredményei
        """
        # Második agent létrehozása (ellenfél)
        opponent = QLearningAgent(
            player=-self.player,
            learning_rate=self.learning_rate,
            discount_factor=self.discount_factor,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay=self.epsilon_decay,
            use_symmetries=self.use_symmetries,
        )

        if verbose:
            print("Self-play edzés kezdése...")

        # Edzés végrehajtása
        results = self.train(num_episodes, opponent, verbose=verbose)

        # Ellenfél statisztikái hozzáadása (agent objektum nélkül)
        try:
            opponent_stats = opponent.get_stats()
            results["opponent_stats"] = opponent_stats
        except Exception as e:
            print(f"Hiba az ellenfél statisztikák mentésekor: {e}")
            results["opponent_stats"] = {"error": str(e)}

        # Agent objektum eltávolítása az eredményből (ha van)
        if "agent" in results:
            del results["agent"]

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Statisztikák lekérése

        Returns:
            Dict[str, Any]: Teljes statisztikák
        """
        stats = self.stats.copy()

        if stats["games_played"] > 0:
            stats["win_rate"] = stats["wins"] / stats["games_played"]
            stats["loss_rate"] = stats["losses"] / stats["games_played"]
            stats["draw_rate"] = stats["draws"] / stats["games_played"]
            stats["avg_reward"] = stats["total_reward"] / stats["games_played"]
        else:
            stats["win_rate"] = 0.0
            stats["loss_rate"] = 0.0
            stats["draw_rate"] = 0.0
            stats["avg_reward"] = 0.0

        # Q-táblázat statisztikák
        stats["q_table_size"] = len(self.q_table)
        stats["total_q_values"] = sum(len(actions) for actions in self.q_table.values())

        # Exploration/exploitation arány
        total_moves = stats["exploration_moves"] + stats["exploitation_moves"]
        if total_moves > 0:
            stats["exploration_rate"] = stats["exploration_moves"] / total_moves
            stats["exploitation_rate"] = stats["exploitation_moves"] / total_moves
        else:
            stats["exploration_rate"] = 0.0
            stats["exploitation_rate"] = 0.0

        stats["current_epsilon"] = self.epsilon
        stats["experience_buffer_size"] = len(self.experience_buffer)

        return stats

    def save_model(self, filename: str = "q_learning_model"):
        """
        Modell mentése fájlba

        Args:
            filename: Fájlnév (kiterjesztés nélkül)
        """
        # Könyvtár létrehozása ha nem létezik
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Q-táblázat mentése
        q_table_path = os.path.join(MODELS_DIR, f"{filename}_qtable.pkl")
        with open(q_table_path, "wb") as f:
            # Defaultdict -> dict konverzió
            q_table_dict = {state: dict(actions) for state, actions in self.q_table.items()}
            pickle.dump(q_table_dict, f)

        # Modell paraméterek és statisztikák mentése
        model_data = {
            "player": self.player,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "use_symmetries": self.use_symmetries,
            "stats": self.get_stats(),
            "q_table_size": len(self.q_table),
        }

        model_path = os.path.join(MODELS_DIR, f"{filename}_params.json")
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)

        print(f"Modell mentve: {filename}")

    # A load_model metódus javítása:

    def load_model(self, filename: str = "q_learning_model"):
        """
        Modell betöltése fájlból (javított verzió)

        Args:
            filename: Fájlnév (kiterjesztés nélkül)
        """
        try:
            # Q-táblázat fájl elérési útja
            q_table_path = os.path.join(MODELS_DIR, f"{filename}_qtable.pkl")
            params_path = os.path.join(MODELS_DIR, f"{filename}_params.json")

            # Ellenőrizzük, hogy léteznek-e a fájlok
            if not os.path.exists(q_table_path):
                raise FileNotFoundError(f"Q-táblázat fájl nem található: {q_table_path}")

            if not os.path.exists(params_path):
                raise FileNotFoundError(f"Paraméter fájl nem található: {params_path}")

            # Q-táblázat betöltése
            print(f"Q-táblázat betöltése: {q_table_path}")
            with open(q_table_path, "rb") as f:
                q_table_dict = pickle.load(f)

            # dict -> defaultdict konverzió
            self.q_table = defaultdict(lambda: defaultdict(lambda: INITIAL_Q_VALUE))
            for state, actions in q_table_dict.items():
                for action, q_value in actions.items():
                    self.q_table[state][action] = q_value

            # Modell paraméterek betöltése
            print(f"Paraméterek betöltése: {params_path}")
            with open(params_path, "r", encoding="utf-8") as f:
                model_data = json.load(f)

            # Paraméterek beállítása
            self.player = model_data.get("player", self.player)
            self.learning_rate = model_data.get("learning_rate", self.learning_rate)
            self.discount_factor = model_data.get("discount_factor", self.discount_factor)
            self.epsilon = model_data.get("epsilon", self.epsilon)
            self.epsilon_start = model_data.get("epsilon_start", self.epsilon_start)
            self.epsilon_end = model_data.get("epsilon_end", self.epsilon_end)
            self.epsilon_decay = model_data.get("epsilon_decay", self.epsilon_decay)
            self.use_symmetries = model_data.get("use_symmetries", self.use_symmetries)

            # Statisztikák visszaállítása (opcionális)
            if "stats" in model_data:
                self.stats.update(model_data["stats"])

            print(f"Modell sikeresen betöltve: {filename}")
            print(f"Q-táblázat mérete: {len(self.q_table)} állapot")
            print(f"Játékos: {self.player}")

            # Modell információk kiírása
            if "stats" in model_data:
                stats = model_data["stats"]
                if "win_rate" in stats:
                    print(f"Modell győzelmi aránya: {stats['win_rate']:.1%}")

        except FileNotFoundError as e:
            print(f"Modell fájl nem található: {e}")
            raise
        except Exception as e:
            print(f"Hiba a modell betöltésekor: {e}")
            raise

    def analyze_q_table(self) -> Dict[str, Any]:
        """
        Q-táblázat elemzése

        Returns:
            Dict[str, Any]: Elemzési eredmények
        """
        if not self.q_table:
            return {"error": "Q-táblázat üres"}

        all_q_values = []
        state_action_counts = []

        for state, actions in self.q_table.items():
            state_action_counts.append(len(actions))
            for action, q_value in actions.items():
                all_q_values.append(q_value)

        analysis = {
            "total_states": len(self.q_table),
            "total_state_action_pairs": len(all_q_values),
            "avg_actions_per_state": np.mean(state_action_counts),
            "q_value_stats": {
                "min": np.min(all_q_values),
                "max": np.max(all_q_values),
                "mean": np.mean(all_q_values),
                "std": np.std(all_q_values),
            },
        }

        return analysis

    def get_move_explanation(
        self, board: np.ndarray, valid_actions: List[int], env: Optional[TicTacToeEnvironment] = None
    ) -> Dict[str, Any]:
        """
        Lépés magyarázata (debug célokra)

        Args:
            board: Tábla
            valid_actions: Érvényes akciók
            env: Environment

        Returns:
            Dict[str, Any]: Lépés magyarázat
        """
        state_key = self.get_state_key(board, env)

        action_analysis = []
        for action in valid_actions:
            q_value = self.get_q_value(state_key, action)
            row, col = action // 3, action % 3

            action_analysis.append({"action": action, "position": f"({row}, {col})", "q_value": q_value})

        # Legjobb akció
        best_action = max(action_analysis, key=lambda x: x["q_value"])

        return {
            "state_key": state_key,
            "epsilon": self.epsilon,
            "will_explore": random.random() < self.epsilon,
            "action_analysis": sorted(action_analysis, key=lambda x: x["q_value"], reverse=True),
            "recommended_action": best_action,
            "q_table_has_state": state_key in self.q_table,
        }

    def reset_stats(self):
        """Statisztikák nullázása"""
        self.stats = {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "total_reward": 0.0,
            "exploration_moves": 0,
            "exploitation_moves": 0,
            "q_updates": 0,
            "training_time": 0.0,
        }

    def __str__(self) -> str:
        """String reprezentáció"""
        stats = self.get_stats()
        return (
            f"QLearningAgent(player={self.player}, "
            f"games={stats['games_played']}, "
            f"win_rate={stats['win_rate']:.1%}, "
            f"epsilon={self.epsilon:.3f}, "
            f"q_table_size={len(self.q_table)})"
        )

    def __repr__(self) -> str:
        """Objektum reprezentáció"""
        return self.__str__()
