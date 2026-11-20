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

## D4 — A* pathfinding (M4)

* **Why A*:** optimal for the risk-weighted cost model, and the heuristic is cheap. Alternatives:
  BFS ignores risk; Dijkstra is optimal too, but expands more nodes (the test checks this).
* **State = (cell, jumps_left)**, not just the cell. Jumps are a consumable resource, so a
  path that spends a jump is a different state from one that does not. Without it A* could
  plan two jumps when only one is left.
* **Cost model:** `move = 1 + risk_weight·risk(landing)`, `jump = 2.5 + risk_weight·risk(landing)`.
  `risk_weight = 25` means a 4% death risk costs as much as one extra step. Jumps cost a bit more
  than two moves, so they are used only when they avoid risk or a blocked cell.
* **Admissible heuristic with teleports:** plain Manhattan distance is NOT admissible once pads
  exist (a pad can cross the board in one step). We use
  `min(|p−g|, |p−A|+|B−g|, |p−B|+|A−g|)`, tested to be ≤ the true cost and to give the same
  cost as Dijkstra on random maps.

## D5 — Minimax with alpha-beta (M5)

* **Why Minimax:** the Wumpus is a human adversary, not a random hazard. When it is near, the
  explorer has to reason "if I do X, what is the Wumpus' best reply?". Alpha-beta prunes the
  tree without changing the result; tests check equality with plain minimax and that cutoffs happen.
* **When it runs:** only when `P(Wumpus within distance 2) ≥ 0.30` and the Wumpus is alive.
  Otherwise the Wumpus is far away, and A* exploration over risk costs is the better tool.
* **Partial observability → determinization.** Minimax needs a known state, and the explorer
  does not know where the Wumpus is. We take the top-4 cells of the Bayesian filter,
  renormalise their probabilities, and run one exact search per hypothesis. Each root action is
  scored by `E[V(a)] = Σ_h P(h)·V_h(a)`. Every root action is searched with a full window, so
  `V_h(a)` is exact, not an alpha-beta bound. That makes the average meaningful and lets the
  inspector show real values.
* **Assumptions Minimax makes (listed in every [MINIMAX] trace entry):**
  1. The Wumpus plays optimally against the explorer (worst case), not like the filter's
     stochastic motion model. This is deliberate: in close combat, assume the worst.
  2. The Wumpus is at one of the hypotheses. Each hypothesis is treated as if the explorer knew it.
     This is the known weakness of determinization ("strategy fusion"/clairvoyance).
  3. Collapse and shuffle powers are ignored inside the search, to keep branching at 5 Wumpus replies.
  4. Pit uncertainty is not a chance node: cells above the risk cap are pruned from the explorer's
     moves, and the accumulated `P(pit)` along a line is charged at death cost (−1000·risk).
     So we get pure minimax rather than expectiminimax.
* **Depth 4 plies** (explorer, Wumpus, explorer, Wumpus): enough to see "shoot now vs. flee and
  get cornered", and under ~0.3 s in pure Python with move ordering (shots first, then goal distance).
* **Leaf evaluation:** −1000 if eaten (or adjacent with the Wumpus to move);
  `−1000·risk − 4·goal_distance + 60·arrows + 6·min(dist_to_Wumpus, 4)`; +300 if the Wumpus is dead.

## D6 — Explorer agent, Brain Activity, dynamic replanning (M6)

* **Per-turn pipeline:** percept → own-action evidence → events → KB rebuild + DPLL classification
  → Bayesian pit posteriors → Wumpus filter → (hunt?) → goal selection → plan validation /
  A* → Minimax if threatened → action. Each stage writes one or more BrainLog entries. Their
  `details` come from the stage's result object (`InferenceResult.explain()`,
  `CellRiskReport.explain()`, `TrackerReport.explain()`, `PathResult.explain()`,
  `MinimaxResult.explain()`), and DECISION entries carry `refs` to the entries they used.
* **Combined risk** `risk(c) = 1 − (1 − P(pit))(1 − P(Wumpus))`. P(Wumpus) is the one-step
  `entry_risk` for cells within 2 of the explorer, and the current belief for farther cells,
  since the Wumpus will have moved by the time we get there. (Simplification.)
* **Dynamic replanning:** a plan is kept (commitment avoids dithering between equal goals) until
  one of these triggers fires, and the reason is logged as [REPLAN]: goal changed, a path cell
  collapsed, a path cell's risk rose by > 5 points, no jumps left, or Minimax deviated from the plan.
  The deterministic test feeds a COLLAPSE on the path home; the agent replans and jumps over the hole.
* **Handling uncertainty when stuck** (found by headless simulation: an idle Wumpus parked in the
  only pit-safe corridor made the agent oscillate until the turn limit):
  1. after 12 turns without visiting a new cell, **hunt**: choose the reachable pit-safe cell and
     direction maximising `P(hit) = Σ belief(ray)`, if P(hit) ≥ 35%;
  2. otherwise relax the risk cap from 30% to 40%;
  3. after 36 turns without progress, retreat and climb out (a draw beats a timeout loss).
* **Tuning found by simulation:** with an arrow worth only 15 points, Minimax fired its only arrow
  at a 6% hypothesis. The arrow is now worth 60 (`w_kill = 300`), so firing pays only above ~20% hit chance.
* **Measured play strength** (30 seeds each): idle Wumpus 24 W / 2 D / 4 L; random 18/2/10;
  omniscient "hunter" (shortest-path pursuit) 12/1/17. The hunter is a perfect chaser that
  always knows where the explorer is, so it is expected to beat the agent more often than not.
* **Separation is tested twice:** an AST import check, and a replay test showing that the agent's
  actions are a pure function of the percept stream.

## D7 — Environment v2 (M7, user specification)

The user's environment spec replaced several v1 rules. v1 decisions D1 (Wumpus collapse/shuffle
*powers*, public teleport pads, absolute moves) are **superseded** by:

* **Facing + FORWARD / TURN_LEFT / TURN_RIGHT**, start facing RIGHT. A* state is now
  `(cell, facing, jump_available)`; turns cost one action like the score does, so A* minimises
  the real action count.
* **Two hidden portal pairs** with a *portal signal* percept (like breeze). The KB gets symbols
  `Oxy`/`Sxy` and the rule `Sc ⇔ ∨ O(neighbours)`. A pair is learned when the explorer teleports.
  A cell proven to be a portal with an unknown destination is avoided unless the agent is stuck,
  because its destination is unpredictable.
* **Collapsing tiles** are hidden and collapse *after the explorer leaves*. The explorer feels an
  unstable-floor warning while standing on one, so it knows exactly which cell becomes a pit and
  when. Breezes sensed next to an already-collapsed tile are explained by it and are left out of
  the pit inference (`valid_breeze`). The P symbols stay "original pits", so the KB stays static.
* **Shuffle tiles** (single use): seeded deterministic swap of one pit with one empty cell, never on
  visited cells, specials, the start zone or the generation route (so the route check stays valid).
  The agent drops all older breeze evidence on the shuffle indicator, as in v1.
* **Jump:** one use, lands 3 ahead, skips everything in between; hitting the wall = bump, jump kept.
  `jump_cost = 3` keeps the A* heuristic admissible (3 cells for 3 cost).
* **Percept type stays in `core/`**, because the AI must not import `environment/`.
  `environment/percepts.py` *builds* it.
* **The Wumpus only moves** (one legal cell or STAY). Its v1 powers were removed because the spec
  makes collapse and shuffle tile mechanics.

### Problems found by simulation after the switch, and their fixes
Measured over 30 seeds per opponent. Before fixes: idle 8 W, random 7 W, hunter 0 W; 8 timeouts.

1. **Trapped and idle.** A tile collapsing behind the explorer cut the proven-safe network from
   the exit. The agent tried to "retreat", found no path and turned in place until the turn limit.
   *Fix:* when cut off from the **exit**, A* may route through unexplored cells, still capped at
   the normal risk limit and risk-costed. If even that fails, explore the cheapest frontier cell
   (above the cap only when cut off from the exit, because staying is a certain loss).
   Exploration targets never use this fallback.
2. **Dithering.** Minimax alternated TURN_LEFT/TURN_RIGHT because turning did not change any leaf
   value. *Fix:* the leaf goal distance is facing-aware (+1 sideways, +2 backwards), and
   *anti-paralysis*: after 4 Minimax decisions in the same cell without being attacked, the Wumpus
   is evidently not the worst-case attacker Minimax assumes, and the plan is followed until we leave
   that cell (logged as WARNING).
3. **Facing makes close combat lethal.** Turning costs a turn the Wumpus can use. *Fixes:* a leaf
   with the Wumpus on our arrow line and our move is scored as a kill (the next ply is SHOOT).
   "Adjacent, our move, not on our line" is scored close to lost (−600), which knows something
   beyond the 4-ply horizon. On ties, prefer actions that do not die immediately.
4. Found along the way: the agent preferred a 20 %-risk cell over a safe cell a few turns away.
   *Fix:* safe-first tier: if any frontier cell has risk ≤ 5 %, only those are candidates.

After the fixes: **idle 18 W / 12 L, random 16 W / 1 D / 13 L, hunter 3 W / 27 L**, worst decision 0.13 s.
**Known trade-offs.** (a) The worst-case Minimax assumption occasionally makes the agent step into
a 30–50 % Wumpus cell rather than stay next to a Wumpus it believes will attack. This costs games
against a passive Wumpus but is right against an aggressive human. (b) The omniscient "hunter"
test opponent always knows where the explorer is, while the explorer only smells it when adjacent.
Under the spec's rules it wins most games. A human player is not omniscient in intent, and the
agent beats random play about half the time.

## D8 — Terminal version (M8)

* **Human sees the god view, the AI's views are separate.** The board shows the true dungeon,
  because the human Wumpus is omniscient. `k` and `m` render only the explorer's beliefs (pit
  probabilities, KB-proven cells, belief over the Wumpus position) as of its last observation.
* **Traceability:** each explorer turn prints the key entries with ids, and `i <id>` prints that
  entry's stored details (the clauses DPLL used, the enumerated pit models, the HMM filter steps,
  Minimax values per hypothesis) and the entries it depends on. Nothing is generated at display time.
* **Testable by construction:** rendering and command parsing are pure functions, and the game loop
  takes injectable `inp`/`out`. Tests drive whole games with scripted keystrokes.
* **Windows console:** stdout is reconfigured to UTF-8 (`errors="replace"`) so logic symbols render;
  the board itself is ASCII only.

## D9 — Knowledge base completed per the KB specification (Prompt 1)

* **Layout:** `wumpus/agents/knowledge_base.py` (was `ai/logic.py`), `wumpus/agents/explorer.py`.
  Renamed with `git mv`; the code was extended, not rewritten. Log tags follow the GUI spec:
  `INFERENCE` (was KB_INFER), `A*_PLANNER` (was ASTAR).
* **Propositions:** `PIT_x_y, BREEZE_x_y, WUMPUS_x_y, STENCH_x_y, SAFE_x_y, PORTAL_x_y, SIGNAL_x_y`.
* **The Wumpus is now in the KB, as time-sensitive facts.** Because it moves every turn, only the
  current turn's evidence is valid: `¬WUMPUS(own cell)` and `STENCH_c ⇔ ∨ WUMPUS(neighbours)`,
  tagged `valid_turn = t`. Rebuilding the KB each turn expires them (logged as "N expired").
  That is the belief revision for a moving Wumpus: old location assumptions never stay valid.
  A scream makes `¬WUMPUS` permanent everywhere. Public rule `¬PIT_c ∨ ¬WUMPUS_c`.
* **SAFE_c ⇔ ¬PIT_c ∧ ¬WUMPUS_c**, queried for frontier cells. Far from the explorer the Wumpus can
  never be ruled out logically, so "known safe" is deliberately small. The Bayesian filter supplies
  the quantitative Wumpus risk.
* **Why the HMM filter is kept next to the logic:** logic can only say "possible / impossible right
  now". Across turns, motion makes everything possible again, and only a probabilistic model can
  carry information forward. The KB gives the certain facts, the filter gives the degrees.
* **No at-most-one-Wumpus axiom** (2016 pairwise clauses): it is never needed to entail a single
  WUMPUS literal from stench evidence, and it would slow every query. (Simplification.)
* **Contradictions:** `KnowledgeBase.consistency()` returns a *minimal* unsatisfiable subset
  (deletion-based). The agent logs it and revises: it drops the oldest observation in the core and
  re-checks. Under correct rules this never fires; the test injects an impossible history.
* **Explanations:** every inference stores the DPLL proof, the clauses used, and an "X because: …"
  sentence built from the clause sources. `ExplorerAgent.knowledge()` returns a frozen
  `KnowledgeSnapshot` (frozensets and mappingproxy objects) with all the lists the spec names.

## D10 — Hazard estimation and planning per the Bayes/A* specification (Prompt 2)

* **`agents/bayesian.py`** (was `probability.py`) now also returns one `HazardEstimate` per cell:
  `p_pit`, `p_wumpus` (here now), `p_wumpus_in` (moves in right after we arrive, for reachable
  cells), `p_death`, `confidence`, evidence. RISK_CALC entries show all of them plus the pit
  enumeration.
* **Combining pit and Wumpus:** `P(death) = P(pit) + (1 − P(pit))·(P(W now) + P(W moves in))`.
  Pit and Wumpus are *mutually exclusive* in a cell (the Wumpus never stands on a pit), and the
  filter's motion model already weights cells by `1 − P(pit)`. So the second term is a probability
  conditional on "no pit", not an independent marginal. (Approximation: the filter uses the pit
  marginals, not the full joint.)
* **Dependencies between neighbouring pits** come from the joint enumeration of frontier models.
  The test shows `P(both) ≠ P(a)·P(b)` for two cells sharing a breeze.
* **Confidence** = 1 for proven facts, otherwise `1 − H(p_death)` (binary entropy in bits).
  It is high when the estimate is near 0 or 1, and 0 at 50/50. It measures how decisive the evidence is.
* **Logic overrides the filter where it is certain:** KB "Wumpus ruled out" gives p_wumpus = 0,
  KB "Wumpus certain" gives 1. Proven pit-free gives p_pit = 0; collapsed or proven pit gives p_death = 1.
* **Planning modes** (`agents/pathfinding.py`): `STRICT` (every intermediate cell proven pit-free;
  only the goal may be unproven) and `TOLERANT` (unproven cells allowed when risk ≤ tolerance).
  The agent plans strict and uses tolerant only when cut off from the exit (D7 fix 1).
* **Route reports:** `plan_routes` returns the best route plus variants (strict, tolerant,
  without jump, without portals), the reason each variant was rejected (no path, with counts per
  rejection cause, or costlier by how much), per-step labels (`proven pit-free` / `UNPROVEN, risk x%` /
  `known portal`, never "safe" for an unproven cell), and `to_dict()` for the Brain View.
* The A* core (`astar`) is unchanged; the planning layer only builds the cost function from a
  `RiskModel` (knowledge only) and runs it in several configurations. Play results are identical
  to before the refactor (30-seed sweep).

## D11 — Minimax per the adversarial-search specification (Prompt 3)

* **Evaluation terms** (`DuelGame.evaluate`, leaf from the explorer's side):
  survival (−1000 if eaten, or adjacent with the Wumpus to move; −600 adjacent and not in our
  line of fire), progress toward the current goal (facing-aware BFS distance, ×4), ability to
  return to the entrance (−1 per step of exit distance, −40 if the exit is unreachable), estimated
  hazard risk (−1000 × accumulated P(pit)), distance from the hypothesised Wumpus (+6 per step,
  capped at 4), remaining abilities (+60 arrow, +20 unused jump), escape routes (+8 per usable
  neighbour not adjacent to the Wumpus, up to 3), and +300 for a dead Wumpus or a ready shot.
  The search sees *gold* progress only through the current goal, because the gold's location is
  unknown until glitter; after the grab, the goal is the exit.
* **Structured output:** `MinimaxResult.to_dict()` holds candidate actions with E[V], the predicted
  Wumpus reply to each candidate under each hypothesis, depth, the selected action, cutoff count,
  the number of pruned sibling moves, and the first 40 pruned branches (path, cut type, α, β, skipped moves).
* **Prediction checking:** after a search, the agent stores the reply predicted for the chosen action
  under the most likely hypothesis, and the cell it implies. On the next turn, if the filter gives
  that cell < 5% after the new evidence, the human did something else: the prediction and the plan
  built on it are discarded (logged as REPLAN) and the agent replans from the updated belief.
* **AI-controlled Wumpus** (`MinimaxWumpusController`, `--wumpus minimax`): same legal actions, same
  `DuelGame` and evaluation, from the MIN side and with perfect information, because the Wumpus is
  omniscient. Used only for AI-vs-AI benchmarking, never mixed into human-vs-AI results.

### Assumptions and limitations under partial observability
1. **Determinization** treats each of the top-4 Wumpus positions as if it were known. The agent
   may then pick an action that is good on average over worlds it can't tell apart, when no single
   real strategy achieves that average ("strategy fusion"). The fix would be belief-state search
   (POMDP / information-set search), far too expensive at this size in pure Python.
2. **Worst-case opponent:** MIN is assumed optimal. Against a passive human this makes the agent
   over-cautious (D7 trade-off a); anti-paralysis limits the damage.
3. **No chance nodes for pits:** pit risk is charged along the line (−1000·risk) instead of
   branching, so it is pure minimax, not expectiminimax.
4. **Horizon of 4 plies:** threats that develop later are only approximated by the leaf terms
   ("adjacent and not facing it ≈ lost").
5. **Belief truncation:** hypotheses outside the top 4 are ignored; their mass is renormalised away.
* **Performance:** KB connected components are now computed once per KB with union-find (cached),
  and clause texts are built only when an explanation is displayed. Three games went from 34 s to 4.3 s.

## D12 — Integrated decision pipeline (Prompt 4)

* **`decide()` = priorities → plan/Minimax → hazard guard → `DecisionReport`.** The report (one per
  decision, `agent.reports`) holds the current observations, facts newly inferred this turn,
  hazard estimates of frontier cells, all 7 candidate actions with an assessment, the A* route
  (with alternatives and rejections) if one was planned, the Minimax output if it ran, the selected
  action, and the explanation. It is JSON-serialisable.
* **Priority order** (as specified): grab on glitter → climb with the gold → hunt if stuck → explore
  (tier 1: KB-proven pit-free cells; tier 2: risk ≤ 5 %; tier 3: risk ≤ cap) → Minimax when the
  Wumpus is near → return home with the gold → anti-loop rules (stuck counter, anti-paralysis,
  retreat). A scenario showed the 5 % tier wrongly treated an *unproven* 4 %-prior cell like a
  proven one, so tier 1 is now strictly "proven". Pit deaths against the idle Wumpus fell from 7 to 4 per 30 games.
* **Hazard guard:** whatever produced the action, if a FORWARD/JUMP would land (after known portals)
  on a proven pit, a collapsed tile or a cell where the KB says the Wumpus certainly is, it is
  replaced by a turn and logged as WARNING.
* **Escape-route preservation:** exploration targets must keep a known way back to the entrance:
  walking over proven pit-free cells (with the unstable tile we stand on counted as collapsed), or
  the unused jump landing in that network. A target reachable only by spending the jump, whose only
  way back is the jump, is rejected. The `jump_opportunity` scenario found this bug: the agent
  jumped *into* a dead end and was trapped with the gold.
* **Fixed-dungeon integration tests** live in `evaluation/scenarios.py` (10 scenarios, shared with
  the evaluation) and `tests/test_integration.py`.
* Sweep after these changes (30 seeds): idle 21 W / 9 L, random 15 W / 2 D / 13 L, hunter 4 W / 26 L.

## D13 — Pygame interface (Prompt 5)

* **Layout and design tokens** come from the specification (`ui/theme.py`): 1280×800, 56 px header,
  two equal cards (REALITY | AGENT'S MIND), 204 px Brain Activity panel, 50 px tiles with 6 px gaps,
  14 px radii, the exact hex colours, and monospace for telemetry and logs. A test checks these values.
* **Two independent data sources.** `ui/session.py` builds a `RealitySnapshot` from a copy of the
  hidden world, and a `MindSnapshot` **only** from the explorer (`agent.knowledge()`, hazards, plan,
  decision report). `draw_mind` receives only the MindSnapshot; a test draws it from an agent that
  never had an engine. The Mind panel shows the agent "as perceived at turn N": the explorer
  appears where it last observed itself, because it learns the result of its newest action only
  from the next percept. The Reality panel already shows the new position.
* **Non-blocking AI.** The explorer's `act()` (KB, Bayes, A*, Minimax) runs in a worker thread.
  The engine is only mutated on the main thread, after the worker finishes. A test makes `act()`
  sleep 0.4 s and checks frames keep flowing (< 0.25 s each).
* **Brain Activity rows are the agent's `TraceEntry` objects** (structured: id, turn, tag, summary,
  details, refs, cell, timestamp). Colours: INFERENCE #10B981, RISK_CALC #FACC15, A*_PLANNER
  #38BDF8, MINIMAX #A78BFA, DECISION #F87171. Clicking a row opens the stored evidence; clicking a
  dependency follows it. "F" toggles key tags vs. everything.
* **Icons are vector-drawn** (`ui/icons.py`): chevron explorer, horned Wumpus, vortex pit, faceted
  octagon gold, hexagonal S/B, violet portal, cracked floor, interlocking arrows. No assets, no emoji.
  Logic symbols (∨, ¬, Σ, …) are mapped to ASCII when rendered, so any monospace font works.
* **Animations** are driven by the engine's structured `GameEvent`s: movement tweens (jumps arc),
  teleport rings, collapse/shuffle/gold/hit flashes, and pulses on newly inferred cells and new A*
  routes. The next step waits until movement tweens finish.
* **Bug found by the UI tests:** cached `pygame.font.Font` objects became invalid after
  `pygame.quit()` and crashed the next session natively; the cache is now reset on every init.

## D14 — Evaluation (Prompt 6)

* **Ablations share one code base.** `AgentConfig(use_bayes, use_minimax)` switches components off
  inside the same agent. logic+A* treats every unproven cell as fatal and uses only KB Wumpus facts;
  logic+bayes+A* skips Minimax. The random baseline is a separate 10-line agent.
* **Held-out maps:** development used seeds 0-29; the benchmark uses seeds 1000-1099 for every
  cell of the table, so all agents face identical dungeons.
* **Opponents are automated (AI vs AI)**: random, greedy omniscient hunter, and the Minimax Wumpus.
  Human games are only demonstrated (GUI/terminal) and never mixed into the statistics.
* **Metric definitions:** survival = explorer alive at the end (win, draw or timeout); timeout =
  turn 150 reached; planning time = wall-clock time of `agent.act()` per decision.
* **Results are reported as measured**, including the unflattering ones: logic-only gets the best
  average score through cautious draws; nothing beats the omniscient Minimax Wumpus more than 6% of
  the time; and the held-out win rate is below the development-seed rate.
