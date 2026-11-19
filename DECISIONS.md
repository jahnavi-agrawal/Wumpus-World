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

## D1 — Environment (M1)

* **Mechanics are assigned so each one creates a real decision for an AI component:**
  * *Jumping* (explorer, 2 charges) gives A* a second edge type that skips a hazardous cell.
  * *Teleport pads* (public, used by both sides) add non-local edges to A* and to the Wumpus
    motion model; the A* heuristic must account for them to remain admissible.
  * *Collapsing tiles* (Wumpus power) change the map mid-game, so plans become invalid and the
    agent must replan. A collapse also leaks information: the Wumpus stood still within distance 2.
  * *Shuffling* (Wumpus power, once) re-places pits on unvisited cells, invalidating old breeze
    evidence. This forces the agent to handle knowledge that expires, not just knowledge that grows.
* **Collapsed tiles do not cause breeze** and are never pits. This keeps the breeze semantics
  ("breeze ⇔ adjacent pit") static, so the propositional encoding stays simple and correct.
* **Wumpus never enters pits/collapsed tiles**, so its motion model only needs the agent's pit
  beliefs, not an extra "Wumpus died in a pit" event.
* **Solvability:** generation retries until a pit-free walk from start to gold exists. Without
  this, a seed could produce an unwinnable game and the AI would be judged unfairly.
* **Safe start zone** (start + its neighbours): the explorer always gets a first move without a
  forced gamble. This is a public rule and appears in the KB as facts.
* **Cooldown semantics:** after a collapse the Wumpus must wait 3 of its own turns.
