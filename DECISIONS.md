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

## D2 — Propositional knowledge base (M2)

* **Why DPLL + refutation (not forward chaining or resolution):** the breeze rule
  `B ⇔ P1 ∨ P2 ∨ …` is not Horn, so forward/backward chaining cannot derive "pit at (2,0)".
  Full resolution is complete but explodes in clause count. DPLL is complete for propositional
  entailment (`KB ⊨ α` iff `KB ∧ ¬α` is UNSAT) and fast at this size (≤128 symbols).
* **Encoding:** symbols `Pxy` (pit) and `Bxy` (breeze). We only add breeze biconditionals for
  cells the explorer actually sensed, in textbook CNF:
  `(¬Bc ∨ Pn1 ∨ … )` and `(Bc ∨ ¬Pni)` for each neighbour. Facts: start zone, pads,
  visited and collapsed cells are `¬P`.
* **The KB is rebuilt from the evidence list every turn** instead of being incrementally
  modified. That makes *shuffle* trivial and correct: stale breeze observations are simply
  not included any more (non-monotonic knowledge handled by recomputation, not retraction).
* **Only the connected component** of the query symbol is passed to DPLL. This is sound
  because the KB is checked to be satisfiable, and it is also the honest "evidence" shown in
  the UI: the inspector lists exactly the clauses (with their source) that the solver used.
* **The Wumpus is not in the KB.** A moving Wumpus would need time-indexed symbols `W_x_y_t`
  plus a transition model; the knowledge would be almost always "unknown". A probabilistic
  filter expresses this far better (D3).

## D3 — Probabilities and uncertainty (M3)

### Pits: exact Bayesian posterior by frontier enumeration
* **Model:** each still-unknown cell has a pit independently with prior
  `δ = (pit_count − proven_pits) / (#unknown cells)`. Evidence = valid breeze observations.
* **Calculation:** `P(Px | e) = α Σ_{m consistent, m[x]=pit} δ^k (1−δ)^(n−k)` over all
  assignments `m` of the *frontier* (unknown cells that occur in some breeze constraint).
  Cells outside the frontier are independent of the evidence, so their posterior equals δ.
  The KB's proven cells are fixed (0 or 1) before enumeration; constraints already satisfied by
  a proven pit are dropped.
* **Efficiency:** the frontier is split into independent connected components, and backtracking
  drops a partial model when a breezy cell can no longer get a pit. Verified against the AIMA
  textbook numbers (0.86 / 0.31) and against full-joint brute force on random evidence.
* **Simplification:** the exact pit *count* is known but we use the independent-δ prior
  (as AIMA does). Exact counting would couple every cell on the board; the error is small
  because δ is re-estimated from the remaining count every turn.
* **Cap:** a component larger than 18 cells uses a per-constraint approximation
  (`δ / (1 − (1−δ)^|U|)`), clearly labelled `approx` in the inspector. On 8×8 this is rare.

### Wumpus: discrete Bayesian filter (HMM forward algorithm)
* The Wumpus moves, so its position is a hidden Markov state. Each turn the filter
  1. removes positions contradicted by the explorer's own action (it survived entering p;
     arrow missed along a ray),
  2. predicts with a motion model, or applies "stayed still" if a COLLAPSE/SHUFFLE event shows the
     Wumpus spent its turn on a power (COLLAPSE also bounds its distance to the target),
  3. removes the explorer's cell (it was not eaten),
  4. conditions on stench / no stench,
  5. renormalises.
* **Motion-model assumption:** the human is modelled as `15% stay`, otherwise `50%` greedy
  pursuit (moves that reduce Manhattan distance to the explorer) and `50%` uniform wander.
  Moves are weighted by `1 − P(pit)` because the Wumpus cannot enter pits. All likelihoods are 0/1
  because percepts are noise-free in this game.
* **Robustness:** if the human does something the model gives zero probability (e.g. a long
  teleport chain), all mass vanishes; the filter then resets to uniform over cells consistent
  with the current turn and logs that it did so. No silent failure.
* **How the agent uses it:** `entry_risk(c) = P(W at c) + Σ_n P(W at n)·T(n→c)`. This is the
  chance of being eaten if the explorer steps into `c` now, before the Wumpus moves or right
  after. Total cell risk = `1 − (1 − P(pit)) (1 − entry_risk)`.
