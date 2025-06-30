"""
Training Manager - Tic-Tac-Toe RL edzés koordinálása
Különböző edzési módok és kiértékelések kezelése
"""

import json
import os
import random
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from .constants import (
    DISCOUNT_FACTOR,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    LEARNING_RATE,
    LOGS_DIR,
    MODELS_DIR,
    PLAYER_O,
    PLAYER_X,
    TRAINING_EPISODES,
)
from .q_learning_agent import QLearningAgent
from .tictactoe_environment import TicTacToeEnvironment, create_environment


class TrainingWorker(QThread):
    """
    Edzés háttérszálon futtatásához
    """

    progress_updated = pyqtSignal(int, dict)  # progress, stats
    training_finished = pyqtSignal(dict)  # results
    log_message = pyqtSignal(str)  # log message

    def __init__(self, training_config: Dict[str, Any]):
        super().__init__()
        self.training_config = training_config
        self.should_stop = False

    def stop_training(self):
        """Edzés leállítása"""
        self.should_stop = True

    def run(self):
        """Edzés futtatása háttérszálon"""
        try:
            manager = TrainingManager()

            # Progress callback beállítása
            def progress_callback(episode: int, stats: Dict[str, Any]):
                if not self.should_stop:
                    self.progress_updated.emit(episode, stats)
                return not self.should_stop  # Continue if not stopped

            # Log callback beállítása
            def log_callback(message: str):
                self.log_message.emit(message)

            # Edzés végrehajtása
            results = manager.run_training(
                **self.training_config, progress_callback=progress_callback, log_callback=log_callback
            )

            if not self.should_stop:
                self.training_finished.emit(results)

        except Exception as e:
            self.log_message.emit(f"Hiba az edzés során: {str(e)}")


class TrainingManager:
    """
    Edzés menedzser - különböző edzési módok koordinálása

    Funkciók:
    - Self-play edzés
    - Random ellenfél elleni edzés
    - Curriculum learning
    - Teljesítmény kiértékelés
    - Edzési eredmények mentése és vizualizálása
    """

    def __init__(self):
        """Training Manager inicializálása"""
        self.create_directories()

        # Edzési történet
        self.training_history = {"sessions": [], "best_models": [], "evaluations": []}

        # Aktuális edzési session
        self.current_session = None

        print("Training Manager inicializálva")

    def create_directories(self):
        """Szükséges könyvtárak létrehozása"""
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)
        os.makedirs(os.path.join(LOGS_DIR, "plots"), exist_ok=True)

    def create_agent(self, player: int = PLAYER_X, **kwargs) -> QLearningAgent:
        """
        Új agent létrehozása

        Args:
            player: Játékos azonosító
            **kwargs: Agent paraméterek

        Returns:
            QLearningAgent: Új agent
        """
        agent_params = {
            "learning_rate": LEARNING_RATE,
            "discount_factor": DISCOUNT_FACTOR,
            "epsilon_start": EPSILON_START,
            "epsilon_end": EPSILON_END,
            "epsilon_decay": EPSILON_DECAY,
            "use_symmetries": True,
        }
        agent_params.update(kwargs)

        return QLearningAgent(player=player, **agent_params)

    def train_against_random(
        self,
        num_episodes: int = TRAINING_EPISODES,
        agent_params: Optional[Dict] = None,
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Edzés random ellenfél ellen

        Args:
            num_episodes: Edzési epizódok száma
            agent_params: Agent paraméterek
            progress_callback: Progress callback függvény
            log_callback: Log callback függvény

        Returns:
            Dict[str, Any]: Edzés eredményei
        """
        if log_callback:
            log_callback("Random ellenfél elleni edzés kezdése...")

        # Agent létrehozása
        agent = self.create_agent(PLAYER_X, **(agent_params or {}))

        # Edzés végrehajtása
        results = agent.train(
            num_episodes=num_episodes,
            opponent_agent=None,  # Random ellenfél
            evaluation_interval=max(100, num_episodes // 20),
            save_interval=max(1000, num_episodes // 10),
            verbose=False,
        )

        # Progress callback hívása
        if progress_callback:
            progress_callback(num_episodes, agent.get_stats())

        if log_callback:
            log_callback(f"Edzés befejezve. Végső győzelmi arány: {results['final_evaluation']['win_rate']:.1%}")

        return {"type": "random_opponent", "agent": agent, "results": results, "timestamp": datetime.now().isoformat()}

    def train_self_play(
        self,
        num_episodes: int = TRAINING_EPISODES,
        agent_params: Optional[Dict] = None,
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Self-play edzés

        Args:
            num_episodes: Edzési epizódok száma
            agent_params: Agent paraméterek
            progress_callback: Progress callback függvény
            log_callback: Log callback függvény

        Returns:
            Dict[str, Any]: Edzés eredményei
        """
        if log_callback:
            log_callback("Self-play edzés kezdése...")

        # Agent létrehozása
        agent = self.create_agent(PLAYER_X, **(agent_params or {}))

        # Self-play edzés
        results = agent.self_play_training(num_episodes=num_episodes, verbose=False)

        # Progress callback hívása
        if progress_callback:
            progress_callback(num_episodes, agent.get_stats())

        if log_callback:
            log_callback(
                f"Self-play edzés befejezve. Végső győzelmi arány: {results['final_evaluation']['win_rate']:.1%}"
            )

        return {"type": "self_play", "agent": agent, "results": results, "timestamp": datetime.now().isoformat()}

    def curriculum_training(
        self,
        stages: List[Dict[str, Any]],
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Curriculum learning - fokozatos nehezítés

        Args:
            stages: Edzési szakaszok listája
            progress_callback: Progress callback függvény
            log_callback: Log callback függvény

        Returns:
            Dict[str, Any]: Edzés eredményei
        """
        if log_callback:
            log_callback("Curriculum learning kezdése...")

        # Agent létrehozása
        agent = self.create_agent(PLAYER_X)

        total_episodes = 0
        stage_results = []

        for i, stage in enumerate(stages):
            stage_name = stage.get("name", f"Szakasz {i+1}")
            episodes = stage.get("episodes", 1000)
            opponent_type = stage.get("opponent", "random")

            if log_callback:
                log_callback(f"{stage_name} kezdése ({episodes} epizód)...")

            try:
                if opponent_type == "random":
                    # Random ellenfél
                    stage_result = agent.train(num_episodes=episodes, opponent_agent=None, verbose=False)
                elif opponent_type == "self":
                    # Self-play
                    stage_result = agent.self_play_training(num_episodes=episodes, verbose=False)
                else:
                    # Egyéb ellenfél típusok...
                    stage_result = agent.train(num_episodes=episodes, opponent_agent=None, verbose=False)

                total_episodes += episodes

                # Stage eredmény tisztítása (agent objektum eltávolítása)
                clean_stage_result = {}
                for key, value in stage_result.items():
                    if key != "agent" and not isinstance(value, QLearningAgent):
                        clean_stage_result[key] = value

                stage_info = {
                    "stage": stage_name,
                    "episodes": episodes,
                    "opponent": opponent_type,
                    "result": clean_stage_result,
                }

                stage_results.append(stage_info)

                # Progress callback
                if progress_callback:
                    progress_callback(total_episodes, agent.get_stats())

                if log_callback:
                    final_eval = stage_result.get("final_evaluation", {})
                    win_rate = final_eval.get("win_rate", 0)
                    log_callback(f"{stage_name} befejezve. Győzelmi arány: {win_rate:.1%}")

            except Exception as e:
                error_msg = f"Hiba a {stage_name} során: {str(e)}"
                if log_callback:
                    log_callback(error_msg)
                print(error_msg)

                # Hiba esetén is adjunk vissza valamilyen eredményt
                stage_results.append(
                    {"stage": stage_name, "episodes": episodes, "opponent": opponent_type, "result": {"error": str(e)}}
                )

        # Végső eredmény összeállítása
        final_result = {
            "type": "curriculum",
            "agent": agent,  # Ez lesz eltávolítva a mentéskor
            "stages": stage_results,
            "total_episodes": total_episodes,
            "timestamp": datetime.now().isoformat(),
        }

        return final_result

    def evaluate_agents(self, agents: List[QLearningAgent], num_games: int = 1000) -> Dict[str, Any]:
        """
        Több agent teljesítményének összehasonlítása

        Args:
            agents: Agent lista
            num_games: Értékelési játékok száma

        Returns:
            Dict[str, Any]: Értékelési eredmények
        """
        results = {}

        for i, agent in enumerate(agents):
            agent_name = f"Agent_{i+1}"

            # Random ellenfél ellen
            random_results = agent.evaluate_against_random(num_games)

            # Q-táblázat elemzése
            q_analysis = agent.analyze_q_table()

            results[agent_name] = {
                "random_opponent": random_results,
                "q_table_analysis": q_analysis,
                "stats": agent.get_stats(),
            }

        # Agent vs Agent értékelés
        if len(agents) >= 2:
            results["agent_vs_agent"] = self.evaluate_agent_vs_agent(agents[0], agents[1], num_games)

        return results

    def evaluate_agent_vs_agent(
        self, agent1: QLearningAgent, agent2: QLearningAgent, num_games: int = 1000
    ) -> Dict[str, Any]:
        """
        Két agent egymás elleni értékelése

        Args:
            agent1: Első agent
            agent2: Második agent
            num_games: Játékok száma

        Returns:
            Dict[str, Any]: Értékelési eredmények
        """
        agent1_wins = 0
        agent2_wins = 0
        draws = 0

        for game in range(num_games):
            env = create_environment()

            # Játékosok felváltva kezdenek
            if game % 2 == 0:
                current_agents = {PLAYER_X: agent1, PLAYER_O: agent2}
            else:
                current_agents = {PLAYER_X: agent2, PLAYER_O: agent1}

            while not env.is_terminal_state():
                current_player = env.current_player
                agent = current_agents[current_player]

                valid_actions = env.get_valid_actions()
                action = agent.get_best_action(env.get_state(), valid_actions, env)

                _, _, done, info = env.step(action)

                if done:
                    winner = info.get("winner")
                    if winner is None:
                        draws += 1
                    elif (game % 2 == 0 and winner == PLAYER_X) or (game % 2 == 1 and winner == PLAYER_O):
                        agent1_wins += 1
                    else:
                        agent2_wins += 1
                    break

        return {
            "agent1_wins": agent1_wins,
            "agent2_wins": agent2_wins,
            "draws": draws,
            "agent1_win_rate": agent1_wins / num_games,
            "agent2_win_rate": agent2_wins / num_games,
            "draw_rate": draws / num_games,
            "total_games": num_games,
        }

    def run_training(
        self,
        training_type: str = "random",
        num_episodes: int = TRAINING_EPISODES,
        agent_params: Optional[Dict] = None,
        curriculum_stages: Optional[List[Dict]] = None,
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Edzés futtatása a megadott paraméterekkel

        Args:
            training_type: Edzés típusa ('random', 'self_play', 'curriculum')
            num_episodes: Epizódok száma
            agent_params: Agent paraméterek
            curriculum_stages: Curriculum szakaszok
            progress_callback: Progress callback
            log_callback: Log callback

        Returns:
            Dict[str, Any]: Edzés eredményei
        """
        start_time = time.time()

        try:
            if training_type == "random":
                results = self.train_against_random(num_episodes, agent_params, progress_callback, log_callback)
            elif training_type == "self_play":
                results = self.train_self_play(num_episodes, agent_params, progress_callback, log_callback)
            elif training_type == "curriculum":
                if not curriculum_stages:
                    curriculum_stages = self.get_default_curriculum()
                results = self.curriculum_training(curriculum_stages, progress_callback, log_callback)
            else:
                raise ValueError(f"Ismeretlen edzés típus: {training_type}")

            # Edzési idő hozzáadása
            results["training_duration"] = time.time() - start_time

            # Session mentése
            session_id = self.save_training_session(results)
            if session_id:
                results["session_id"] = session_id

            return results

        except Exception as e:
            error_msg = f"Hiba az edzés során: {str(e)}"
            if log_callback:
                log_callback(error_msg)
            print(error_msg)

            # Hiba esetén is adjunk vissza valamilyen eredményt
            return {
                "type": training_type,
                "error": str(e),
                "training_duration": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
            }

    def get_default_curriculum(self) -> List[Dict[str, Any]]:
        """
        Alapértelmezett curriculum szakaszok

        Returns:
            List[Dict[str, Any]]: Curriculum szakaszok
        """
        return [
            {
                "name": "Alapok - Random ellenfél",
                "episodes": 10000,
                "opponent": "random",
                "description": "Alapvető stratégiák tanulása random ellenfél ellen",
            },
            {
                "name": "Fejlesztés - Self-play",
                "episodes": 30000,
                "opponent": "self",
                "description": "Haladó stratégiák fejlesztése self-play módban",
            },
            {
                "name": "Finomhangolás - Mixed",
                "episodes": 10000,
                "opponent": "random",
                "description": "Végső finomhangolás vegyes ellenfelekkel",
            },
        ]

    # A save_training_session metódus javítása:
    def save_training_session(self, session_data: Dict[str, Any]):
        """
        Edzési session mentése

        Args:
            session_data: Session adatok
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"training_session_{timestamp}"

            # Session adatok kiegészítése
            session_data_copy = session_data.copy()
            session_data_copy["session_id"] = session_id
            session_data_copy["timestamp"] = timestamp

            # Agent objektum kezelése
            agent = None
            if "agent" in session_data_copy:
                agent = session_data_copy["agent"]

                # Agent statisztikák és elemzések mentése
                try:
                    session_data_copy["agent_stats"] = agent.get_stats()
                    session_data_copy["agent_q_table_analysis"] = agent.analyze_q_table()
                    session_data_copy["agent_player"] = agent.player
                    session_data_copy["agent_epsilon"] = agent.epsilon
                    session_data_copy["agent_q_table_size"] = len(agent.q_table)
                except Exception as e:
                    print(f"Hiba az agent adatok mentésekor: {e}")
                    session_data_copy["agent_stats"] = {}
                    session_data_copy["agent_q_table_analysis"] = {"error": str(e)}

                # Agent objektum eltávolítása a JSON-ból
                del session_data_copy["agent"]

            # Curriculum stages kezelése (ha van)
            if "stages" in session_data_copy:
                for stage in session_data_copy["stages"]:
                    if "result" in stage and "agent" in stage["result"]:
                        # Stage-ben lévő agent objektum eltávolítása
                        del stage["result"]["agent"]

            # JSON mentés
            session_file = os.path.join(LOGS_DIR, f"{session_id}.json")

            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session_data_copy, f, indent=2, ensure_ascii=False, default=str)

            # Agent modell mentése (ha van)
            if agent is not None:
                try:
                    model_name = f"model_{session_id}"
                    agent.save_model(model_name)
                    session_data_copy["model_saved"] = model_name
                    print(f"Agent modell mentve: {model_name}")
                except Exception as e:
                    print(f"Hiba az agent modell mentésekor: {e}")
                    session_data_copy["model_saved"] = None

            # Training history frissítése
            self.training_history["sessions"].append(session_data_copy)

            print(f"Edzési session mentve: {session_id}")

            return session_id

        except Exception as e:
            print(f"Hiba a session mentésekor: {e}")
            return None

    def load_training_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Edzési session betöltése

        Args:
            session_id: Session azonosító

        Returns:
            Optional[Dict[str, Any]]: Session adatok vagy None
        """
        try:
            session_file = os.path.join(LOGS_DIR, f"{session_id}.json")

            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            return session_data

        except FileNotFoundError:
            print(f"Session nem található: {session_id}")
            return None
        except Exception as e:
            print(f"Hiba a session betöltésekor: {e}")
            return None

    def list_training_sessions(self) -> List[Dict[str, Any]]:
        """
        Összes edzési session listázása

        Returns:
            List[Dict[str, Any]]: Session lista
        """
        sessions = []

        try:
            for filename in os.listdir(LOGS_DIR):
                if filename.startswith("training_session_") and filename.endswith(".json"):
                    session_id = filename[:-5]  # .json eltávolítása

                    session_data = self.load_training_session(session_id)
                    if session_data:
                        # Csak alapvető információk
                        session_info = {
                            "session_id": session_id,
                            "timestamp": session_data.get("timestamp"),
                            "type": session_data.get("type"),
                            "training_duration": session_data.get("training_duration"),
                        }

                        # Eredmények hozzáadása ha vannak
                        if "results" in session_data:
                            results = session_data["results"]
                            if "final_evaluation" in results:
                                session_info["final_win_rate"] = results["final_evaluation"].get("win_rate")

                        sessions.append(session_info)

        except Exception as e:
            print(f"Hiba a sessionök listázásakor: {e}")

        # Időrend szerint rendezés (legújabb először)
        sessions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return sessions

    def create_training_plots(self, session_data: Dict[str, Any], save_plots: bool = True) -> Dict[str, Any]:
        """
        Edzési grafikonok létrehozása

        Args:
            session_data: Session adatok
            save_plots: Grafikonok mentése

        Returns:
            Dict[str, Any]: Grafikon adatok
        """
        plots_data = {}

        try:
            # Edzési történet kinyerése
            if "results" in session_data and "training_history" in session_data["results"]:
                history = session_data["results"]["training_history"]

                # 1. Győzelmi arány változása
                if "win_rates" in history and "episodes" in history:
                    episodes = (
                        history["episodes"][:: len(history["episodes"]) // len(history["win_rates"])]
                        if history["win_rates"]
                        else []
                    )
                    win_rates = history["win_rates"]

                    if episodes and win_rates:
                        plt.figure(figsize=(10, 6))
                        plt.plot(episodes[: len(win_rates)], win_rates, "b-", linewidth=2)
                        plt.title("Győzelmi arány változása az edzés során")
                        plt.xlabel("Epizódok")
                        plt.ylabel("Győzelmi arány")
                        plt.grid(True, alpha=0.3)
                        plt.ylim(0, 1)

                        if save_plots:
                            plot_file = os.path.join(LOGS_DIR, "plots", f"{session_data['session_id']}_win_rate.png")
                            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                            plots_data["win_rate_plot"] = plot_file

                        plt.close()

                # 2. Epsilon változása
                if "epsilons" in history and "episodes" in history:
                    episodes = history["episodes"]
                    epsilons = history["epsilons"]

                    if episodes and epsilons and len(episodes) == len(epsilons):
                        plt.figure(figsize=(10, 6))
                        plt.plot(episodes, epsilons, "r-", linewidth=2)
                        plt.title("Epsilon (exploration rate) változása")
                        plt.xlabel("Epizódok")
                        plt.ylabel("Epsilon")
                        plt.grid(True, alpha=0.3)
                        plt.ylim(0, 1)

                        if save_plots:
                            plot_file = os.path.join(LOGS_DIR, "plots", f"{session_data['session_id']}_epsilon.png")
                            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                            plots_data["epsilon_plot"] = plot_file

                        plt.close()

                # 3. Q-táblázat méret változása
                if "q_table_sizes" in history and "episodes" in history:
                    episodes = history["episodes"]
                    q_sizes = history["q_table_sizes"]

                    if episodes and q_sizes and len(episodes) == len(q_sizes):
                        plt.figure(figsize=(10, 6))
                        plt.plot(episodes, q_sizes, "g-", linewidth=2)
                        plt.title("Q-táblázat méretének változása")
                        plt.xlabel("Epizódok")
                        plt.ylabel("Állapotok száma")
                        plt.grid(True, alpha=0.3)

                        if save_plots:
                            plot_file = os.path.join(
                                LOGS_DIR, "plots", f"{session_data['session_id']}_qtable_size.png"
                            )
                            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                            plots_data["qtable_size_plot"] = plot_file

                        plt.close()

                # 4. Jutalmak eloszlása
                if "rewards" in history:
                    rewards = history["rewards"]

                    if rewards:
                        plt.figure(figsize=(10, 6))
                        plt.hist(rewards, bins=50, alpha=0.7, color="purple", edgecolor="black")
                        plt.title("Jutalmak eloszlása")
                        plt.xlabel("Jutalom")
                        plt.ylabel("Gyakoriság")
                        plt.grid(True, alpha=0.3)

                        if save_plots:
                            plot_file = os.path.join(
                                LOGS_DIR, "plots", f"{session_data['session_id']}_rewards_dist.png"
                            )
                            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                            plots_data["rewards_dist_plot"] = plot_file

                        plt.close()

        except Exception as e:
            print(f"Hiba a grafikonok létrehozásakor: {e}")

        return plots_data

    def generate_training_report(self, session_id: str) -> str:
        """
        Edzési jelentés generálása

        Args:
            session_id: Session azonosító

        Returns:
            str: Jelentés szövege
        """
        session_data = self.load_training_session(session_id)
        if not session_data:
            return "Session nem található!"

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append(f"EDZÉSI JELENTÉS - {session_id}")
        report_lines.append("=" * 60)
        report_lines.append("")

        # Alapvető információk
        report_lines.append("ALAPVETŐ INFORMÁCIÓK:")
        report_lines.append(f"  Edzés típusa: {session_data.get('type', 'N/A')}")
        report_lines.append(f"  Időbélyeg: {session_data.get('timestamp', 'N/A')}")
        report_lines.append(f"  Edzési idő: {session_data.get('training_duration', 0):.1f} másodperc")
        report_lines.append("")

        # Eredmények
        if "results" in session_data:
            results = session_data["results"]

            if "final_evaluation" in results:
                eval_data = results["final_evaluation"]
                report_lines.append("VÉGSŐ TELJESÍTMÉNY (Random ellenfél ellen):")
                report_lines.append(f"  Győzelmi arány: {eval_data.get('win_rate', 0):.1%}")
                report_lines.append(f"  Vereség arány: {eval_data.get('loss_rate', 0):.1%}")
                report_lines.append(f"  Döntetlen arány: {eval_data.get('draw_rate', 0):.1%}")
                report_lines.append(f"  Összes teszt játék: {eval_data.get('total_games', 0)}")
                report_lines.append("")

        # Agent statisztikák
        if "agent_stats" in session_data:
            stats = session_data["agent_stats"]
            report_lines.append("AGENT STATISZTIKÁK:")
            report_lines.append(f"  Edzési játékok: {stats.get('games_played', 0)}")
            report_lines.append(f"  Győzelmek: {stats.get('wins', 0)}")
            report_lines.append(f"  Vereségek: {stats.get('losses', 0)}")
            report_lines.append(f"  Döntetlenek: {stats.get('draws', 0)}")
            report_lines.append(f"  Átlagos jutalom: {stats.get('avg_reward', 0):.3f}")
            report_lines.append(f"  Q-táblázat mérete: {stats.get('q_table_size', 0)} állapot")
            report_lines.append(f"  Exploration arány: {stats.get('exploration_rate', 0):.1%}")
            report_lines.append("")

        # Q-táblázat elemzés
        if "agent_q_table_analysis" in session_data:
            q_analysis = session_data["agent_q_table_analysis"]
            if "error" not in q_analysis:
                report_lines.append("Q-TÁBLÁZAT ELEMZÉS:")
                report_lines.append(f"  Összes állapot: {q_analysis.get('total_states', 0)}")
                report_lines.append(f"  Állapot-akció párok: {q_analysis.get('total_state_action_pairs', 0)}")
                report_lines.append(f"  Átlag akciók/állapot: {q_analysis.get('avg_actions_per_state', 0):.1f}")

                if "q_value_stats" in q_analysis:
                    q_stats = q_analysis["q_value_stats"]
                    report_lines.append(f"  Q-értékek - Min: {q_stats.get('min', 0):.3f}")
                    report_lines.append(f"  Q-értékek - Max: {q_stats.get('max', 0):.3f}")
                    report_lines.append(f"  Q-értékek - Átlag: {q_stats.get('mean', 0):.3f}")
                    report_lines.append(f"  Q-értékek - Szórás: {q_stats.get('std', 0):.3f}")
                report_lines.append("")

        # Curriculum szakaszok (ha van)
        if session_data.get("type") == "curriculum" and "stages" in session_data:
            report_lines.append("CURRICULUM SZAKASZOK:")
            for i, stage in enumerate(session_data["stages"]):
                report_lines.append(f"  {i+1}. {stage.get('stage', 'Névtelen szakasz')}")
                report_lines.append(f"     Epizódok: {stage.get('episodes', 0)}")
                report_lines.append(f"     Ellenfél: {stage.get('opponent', 'N/A')}")
                if "result" in stage and "final_evaluation" in stage["result"]:
                    win_rate = stage["result"]["final_evaluation"].get("win_rate", 0)
                    report_lines.append(f"     Végső győzelmi arány: {win_rate:.1%}")
                report_lines.append("")

        report_lines.append("=" * 60)

        return "\n".join(report_lines)

    def compare_training_sessions(self, session_ids: List[str]) -> str:
        """
        Több edzési session összehasonlítása

        Args:
            session_ids: Session azonosítók listája

        Returns:
            str: Összehasonlító jelentés
        """
        sessions = []
        for session_id in session_ids:
            session_data = self.load_training_session(session_id)
            if session_data:
                sessions.append(session_data)

        if not sessions:
            return "Nincsenek betölthető sessionök!"

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("EDZÉSI SESSIONÖK ÖSSZEHASONLÍTÁSA")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Táblázat fejléc
        report_lines.append(
            f"{'Session ID':<25} {'Típus':<12} {'Győzelmi arány':<15} {'Edzési idő':<12} {'Q-táblázat':<10}"
        )
        report_lines.append("-" * 80)

        # Session adatok
        for session in sessions:
            session_id = session.get("session_id", "N/A")[:24]
            session_type = session.get("type", "N/A")[:11]

            win_rate = "N/A"
            if "results" in session and "final_evaluation" in session["results"]:
                win_rate = f"{session['results']['final_evaluation'].get('win_rate', 0):.1%}"

            training_time = f"{session.get('training_duration', 0):.1f}s"

            q_table_size = "N/A"
            if "agent_stats" in session:
                q_table_size = str(session["agent_stats"].get("q_table_size", 0))

            report_lines.append(
                f"{session_id:<25} {session_type:<12} {win_rate:<15} {training_time:<12} {q_table_size:<10}"
            )

        report_lines.append("")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def get_best_model(self) -> Optional[str]:
        """
        Legjobb modell megkeresése az eddigi edzések alapján

        Returns:
            Optional[str]: Legjobb modell session ID-ja vagy None
        """
        try:
            # Összes modell fájl keresése
            if not os.path.exists(MODELS_DIR):
                return None

            model_files = []

            # .pkl fájlok keresése (Q-táblázatok)
            for filename in os.listdir(MODELS_DIR):
                if filename.endswith("_qtable.pkl"):
                    # Fájl információk
                    file_path = os.path.join(MODELS_DIR, filename)
                    file_time = os.path.getmtime(file_path)

                    # Modell név kinyerése (qtable.pkl eltávolítása)
                    model_name = filename.replace("_qtable.pkl", "")

                    # Ellenőrizzük, hogy van-e hozzá params fájl is
                    params_file = os.path.join(MODELS_DIR, f"{model_name}_params.json")
                    if os.path.exists(params_file):
                        model_files.append(
                            {
                                "name": model_name,
                                "time": file_time,
                                "qtable_file": file_path,
                                "params_file": params_file,
                            }
                        )

            if not model_files:
                return None

            # Legfrissebb modell keresése
            latest_model = max(model_files, key=lambda x: x["time"])

            # Session alapú értékelés (ha van session info)
            sessions = self.list_training_sessions()
            if sessions:
                # Próbáljuk megtalálni a megfelelő session-t
                for session in sessions:
                    session_id = session.get("session_id", "")
                    if session_id in latest_model["name"]:
                        # Ha van győzelmi arány info, azt is figyelembe vesszük
                        win_rate = session.get("final_win_rate", 0)
                        print(f"Legfrissebb modell: {latest_model['name']}, győzelmi arány: {win_rate:.1%}")
                        break

                return latest_model["name"]

        except Exception as e:
            print(f"Hiba a legjobb modell keresésekor: {e}")
            return None

    def load_best_agent(self) -> Optional[QLearningAgent]:
        """
        Legjobb agent betöltése

        Returns:
            Optional[QLearningAgent]: Legjobb agent vagy None
        """
        try:
            # Legfrissebb modell keresése
            best_model_name = self.get_best_model()

            if not best_model_name:
                print("Nincs betölthető modell")
                return None

            print(f"Modell betöltése: {best_model_name}")

            # Agent létrehozása és modell betöltése
            agent = self.create_agent()
            agent.load_model(best_model_name)

            print(f"Modell sikeresen betöltve: {best_model_name}")
            return agent

        except Exception as e:
            print(f"Hiba a legjobb agent betöltésekor: {e}")
            return None

    def load_any_available_model(self) -> Optional[QLearningAgent]:
        """
        Bármilyen elérhető modell betöltése fallback-ként

        Returns:
            Optional[QLearningAgent]: Agent vagy None
        """
        try:
            # Keressük az összes .pkl fájlt a models mappában
            if not os.path.exists(MODELS_DIR):
                return None

            pkl_files = [f for f in os.listdir(MODELS_DIR) if f.endswith("_qtable.pkl")]

            if not pkl_files:
                print("Nincs elérhető modell fájl")
                return None

            # Próbáljuk betölteni az első elérhető modellt
            for pkl_file in pkl_files:
                try:
                    model_name = pkl_file.replace("_qtable.pkl", "")
                    agent = self.create_agent()
                    agent.load_model(model_name)
                    print(f"Fallback modell betöltve: {model_name}")
                    return agent
                except Exception as e:
                    print(f"Nem sikerült betölteni: {model_name} - {e}")
                    continue

            return None

        except Exception as e:
            print(f"Hiba a fallback modell betöltésekor: {e}")
            return None

    def cleanup_old_sessions(self, keep_count: int = 10):
        """
        Régi edzési sessionök törlése (csak a legújabbakat tartja meg)

        Args:
            keep_count: Megtartandó sessionök száma
        """
        sessions = self.list_training_sessions()

        if len(sessions) <= keep_count:
            return

        # Törlendő sessionök
        sessions_to_delete = sessions[keep_count:]

        for session in sessions_to_delete:
            session_id = session["session_id"]

            try:
                # JSON fájl törlése
                json_file = os.path.join(LOGS_DIR, f"{session_id}.json")
                if os.path.exists(json_file):
                    os.remove(json_file)

                # Modell fájlok törlése
                model_name = f"model_{session_id}"
                qtable_file = os.path.join(MODELS_DIR, f"{model_name}_qtable.pkl")
                params_file = os.path.join(MODELS_DIR, f"{model_name}_params.json")

                if os.path.exists(qtable_file):
                    os.remove(qtable_file)
                if os.path.exists(params_file):
                    os.remove(params_file)

                # Grafikon fájlok törlése
                plot_files = [
                    f"{session_id}_win_rate.png",
                    f"{session_id}_epsilon.png",
                    f"{session_id}_qtable_size.png",
                    f"{session_id}_rewards_dist.png",
                ]

                for plot_file in plot_files:
                    plot_path = os.path.join(LOGS_DIR, "plots", plot_file)
                    if os.path.exists(plot_path):
                        os.remove(plot_path)

                print(f"Session törölve: {session_id}")

            except Exception as e:
                print(f"Hiba a session törlése során ({session_id}): {e}")

    def export_training_data(self, output_file: str):
        """
        Összes edzési adat exportálása

        Args:
            output_file: Kimeneti fájl neve
        """
        sessions = self.list_training_sessions()

        export_data = {"export_timestamp": datetime.now().isoformat(), "total_sessions": len(sessions), "sessions": []}

        for session_info in sessions:
            session_id = session_info["session_id"]
            session_data = self.load_training_session(session_id)

            if session_data:
                export_data["sessions"].append(session_data)

        # Export fájl mentése
        export_path = os.path.join(LOGS_DIR, output_file)
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"Edzési adatok exportálva: {export_path}")

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Elérhető modellek listázása

        Returns:
            List[Dict[str, Any]]: Modellek listája
        """
        models = []

        try:
            if not os.path.exists(MODELS_DIR):
                return models

            # .pkl fájlok keresése
            for filename in os.listdir(MODELS_DIR):
                if filename.endswith("_qtable.pkl"):
                    model_name = filename.replace("_qtable.pkl", "")
                    file_path = os.path.join(MODELS_DIR, filename)
                    params_file = os.path.join(MODELS_DIR, f"{model_name}_params.json")

                    if os.path.exists(params_file):
                        # Fájl információk
                        file_time = os.path.getmtime(file_path)
                        file_size = os.path.getsize(file_path)

                        # Paraméterek betöltése
                        try:
                            with open(params_file, "r", encoding="utf-8") as f:
                                params = json.load(f)

                            models.append(
                                {
                                    "name": model_name,
                                    "file_time": file_time,
                                    "file_size": file_size,
                                    "timestamp": datetime.fromtimestamp(file_time).isoformat(),
                                    "q_table_size": params.get("q_table_size", 0),
                                    "player": params.get("player", 1),
                                    "stats": params.get("stats", {}),
                                }
                            )

                        except Exception as e:
                            print(f"Hiba a modell paraméterek betöltésekor ({model_name}): {e}")

        except Exception as e:
            print(f"Hiba a modellek listázásakor: {e}")

        # Időrend szerint rendezés (legfrissebb először)
        models.sort(key=lambda x: x["file_time"], reverse=True)

        return models

    def create_training_worker(self, training_config: Dict[str, Any]) -> TrainingWorker:
        """
        Training worker létrehozása GUI-hoz

        Args:
            training_config: Edzési konfiguráció

        Returns:
            TrainingWorker: Worker thread
        """
        return TrainingWorker(training_config)

    def get_training_recommendations(self) -> Dict[str, Any]:
        """
        Edzési javaslatok generálása a korábbi eredmények alapján

        Returns:
            Dict[str, Any]: Javaslatok
        """
        sessions = self.list_training_sessions()

        recommendations = {
            "suggested_episodes": TRAINING_EPISODES,
            "suggested_type": "random",
            "reasoning": [],
            "parameter_suggestions": {},
        }

        if not sessions:
            recommendations["reasoning"].append("Első edzés - kezdés random ellenféllel ajánlott")
            return recommendations

        # Legjobb eredmény keresése
        best_win_rate = 0
        best_type = "random"

        for session in sessions:
            if session.get("final_win_rate"):
                if session["final_win_rate"] > best_win_rate:
                    best_win_rate = session["final_win_rate"]
                    best_type = session.get("type", "random")

        # Javaslatok generálása
        if best_win_rate < 0.7:
            recommendations["suggested_type"] = "random"
            recommendations["suggested_episodes"] = min(TRAINING_EPISODES * 2, 50000)
            recommendations["reasoning"].append(
                f"Alacsony győzelmi arány ({best_win_rate:.1%}) - több edzés szükséges"
            )
        elif best_win_rate < 0.9:
            recommendations["suggested_type"] = "self_play"
            recommendations["reasoning"].append(f"Jó alapteljesítmény ({best_win_rate:.1%}) - self-play javasolt")
        else:
            recommendations["suggested_type"] = "curriculum"
            recommendations["reasoning"].append(
                f"Kiváló teljesítmény ({best_win_rate:.1%}) - curriculum learning javasolt"
            )

        # Paraméter javaslatok
        if best_win_rate < 0.5:
            recommendations["parameter_suggestions"] = {
                "learning_rate": 0.2,  # Magasabb tanulási ráta
                "epsilon_start": 0.9,  # Több exploration
                "epsilon_decay": 0.9995,  # Lassabb epsilon csökkenés
            }
            recommendations["reasoning"].append("Alacsony teljesítmény - agresszívebb tanulási paraméterek")

        return recommendations

    def __str__(self) -> str:
        """String reprezentáció"""
        sessions_count = len(self.list_training_sessions())
        return f"TrainingManager(sessions: {sessions_count})"


# Segédfüggvények
def create_training_manager() -> TrainingManager:
    """Training Manager factory függvény"""
    return TrainingManager()


def quick_train(training_type: str = "random", episodes: int = 5000, verbose: bool = True) -> Dict[str, Any]:
    """
    Gyors edzés futtatása

    Args:
        training_type: Edzés típusa
        episodes: Epizódok száma
        verbose: Részletes kimenet

    Returns:
        Dict[str, Any]: Edzés eredményei
    """
    manager = create_training_manager()

    return manager.run_training(
        training_type=training_type, num_episodes=episodes, log_callback=print if verbose else None
    )


if __name__ == "__main__":
    # Példa használat
    print("Training Manager teszt...")

    # Gyors edzés
    results = quick_train("random", 1000, verbose=True)

    print(f"Edzés befejezve. Győzelmi arány: {results['results']['final_evaluation']['win_rate']:.1%}")
