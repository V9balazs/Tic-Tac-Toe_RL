# =============================================================================
# JÁTÉK KONSTANSOK
# =============================================================================

# Játékos értékek
PLAYER_X = 1
PLAYER_O = -1
EMPTY_CELL = 0

# Játéktábla
BOARD_SIZE = 3
TOTAL_CELLS = BOARD_SIZE * BOARD_SIZE

# Játék eredmények
RESULT_WIN = 1
RESULT_LOSE = -1
RESULT_DRAW = 0
RESULT_ONGOING = None

# =============================================================================
# Q-LEARNING PARAMÉTEREK
# =============================================================================

# Tanulási paraméterek
LEARNING_RATE = 0.1  # α - tanulási ráta
DISCOUNT_FACTOR = 0.9  # γ - diszkont faktor
EPSILON_START = 0.9  # kezdő exploration ráta
EPSILON_END = 0.01  # végső exploration ráta
EPSILON_DECAY = 0.995  # epsilon csökkenési ráta

# Edzés paraméterek
TRAINING_EPISODES = 50000  # edzési epizódok száma
EVALUATION_INTERVAL = 1000  # értékelési gyakoriság
SAVE_INTERVAL = 5000  # mentési gyakoriság
BATCH_SIZE = 32  # batch méret (ha szükséges)

# Q-táblázat inicializálás
INITIAL_Q_VALUE = 0.0  # kezdő Q-értékek

# =============================================================================
# FÁJL ÚTVONALAK
# =============================================================================

import os

# Projekt gyökér
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# UI fájlok
UI_DIR = os.path.join(PROJECT_ROOT, "ui")
UI_FILE = os.path.join(UI_DIR, "main_window.ui")

# Modell mentés
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
Q_TABLE_FILE = os.path.join(MODELS_DIR, "q_table.pkl")
TRAINING_STATS_FILE = os.path.join(MODELS_DIR, "training_stats.json")

# Logok
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
TRAINING_LOG_FILE = os.path.join(LOGS_DIR, "training.log")

# =============================================================================
# JÁTÉK MÓDOK
# =============================================================================

GAME_MODE_HUMAN_VS_AI = "Ember vs AI"
GAME_MODE_AI_VS_AI = "AI vs AI"

GAME_MODES = [GAME_MODE_HUMAN_VS_AI, GAME_MODE_AI_VS_AI]

# =============================================================================
# ÜZENETEK ÉS SZÖVEGEK
# =============================================================================

# AI üzenetek
MSG_AI_TRAINING_START = "AI edzés folyamatban..."
MSG_AI_TRAINING_COMPLETE = "AI edzés befejezve!"
MSG_AI_THINKING = "AI gondolkodik..."

# =============================================================================
# DEBUG ÉS LOGGING
# =============================================================================

# Debug szintek
DEBUG_LEVEL_NONE = 0
DEBUG_LEVEL_ERROR = 1
DEBUG_LEVEL_WARNING = 2
DEBUG_LEVEL_INFO = 3
DEBUG_LEVEL_DEBUG = 4

# Aktuális debug szint
DEBUG_LEVEL = DEBUG_LEVEL_INFO

# Logging formátum
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# =============================================================================
# TELJESÍTMÉNY BEÁLLÍTÁSOK
# =============================================================================

# AI válaszidő (milliszekundum)
AI_RESPONSE_DELAY = 500

# Animáció beállítások
ANIMATION_DURATION = 200
FADE_DURATION = 150

# Memória optimalizáció
MAX_Q_TABLE_SIZE = 10000  # Maximum Q-táblázat méret
CLEANUP_THRESHOLD = 0.8  # Cleanup küszöb (80%-nál)
