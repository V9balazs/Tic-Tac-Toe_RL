Tic-Tac-Toe Reinforcement Learning

  Egy modern PyQt6-alapú Tic-Tac-Toe játék megerősítéses tanulással (Reinforcement Learning),
  amely Q-learning algoritmust használ intelligens AI ellenfél edzéséhez.

  Mi ez a projekt?

  Ez egy oktatási célú projekt, amely bemutatja a megerősítéses tanulás alapjait egy klassikus
  játék kontextusában. Az AI agent Q-learning algoritmussal tanul meg optimális stratégiákat a
  Tic-Tac-Toe játékban.

  Főbb funkciók

  - Interaktív GUI: Modern PyQt6 felhasználói felület
  - Q-Learning AI: Megerősítéses tanulással tanuló intelligens ellenfél
  - Többféle edzési mód: Random, Self-play, Curriculum learning
  - Teljesítményelemzés: Részletes statisztikák és grafikonok
  - Modell mentés: Edzett AI modellek mentése és betöltése
  - Stratégiai fallback: Intelligens stratégia AI modell nélkül is

  Technológiai stack

  - Python 3.8+ - Főnyelv
  - PyQt6 - Modern GUI framework
  - NumPy - Numerikus számítások és mátrix műveletek
  - Matplotlib - Edzési grafikonok és vizualizáció
  - Pickle - Modell szerializáció
  - JSON - Konfiguráció és adatok tárolása

  AI Algoritmus

  Q-Learning implementáció kulcs funkciókkal:
  - Epsilon-greedy strategy - Exploration vs Exploitation egyensúly
  - Experience replay - Múltbeli tapasztalatok újrahasznosítása
  - State symmetries - Szimmetrikus állapotok hatékony kezelése
  - Prioritized replay - Fontosabb tapasztalatok előnyben részesítése