# DECISIONS

A running log of major implementation decisions. Each milestone appends to it.

## D0 — Project-level decisions (M0)

* **Dependencies:** only `pygame` (UI) and `pytest` (tests). All AI is hand-written Python
  (no numpy, no SAT/AI libraries), so every algorithm is visibly and genuinely implemented.
* **Hidden vs. known state:** `wumpus/env` holds the ground truth; `wumpus/ai` may only import
  `wumpus/core`. The agent receives a frozen `PublicInfo` once and a frozen `Percept` per
  turn. A test (`test_separation.py`) enforces this at the import level.
* **Determinism:** the world uses `random.Random(config.seed)`; the agent has no randomness
  and breaks ties by sorted coordinates; scripted Wumpus controllers take their own seed.
  The same seed + same human inputs therefore reproduce a game exactly.
* **Rules are written down first** (RULES.md) so the agent's inference is derived from
  public rules, never from the implementation of the environment.
* **The Wumpus moves** (it is human-controlled). This is the single biggest design driver:
  static facts (pits) go to a propositional KB; the moving Wumpus is tracked with a Bayesian
  filter (see D2/D3).
