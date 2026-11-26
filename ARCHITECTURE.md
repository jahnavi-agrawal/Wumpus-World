# WUMPUS: THE MIND GAMES — Architecture

Human plays the **Wumpus** (omniscient, sees the whole dungeon).
AI plays the **Explorer** (partial observability, sees only its own percepts).

## 1. Folder structure (current)

```
.
├── main.py                     # GUI (default) | --terminal | --seed N | --wumpus human|idle|random|hunter|minimax
├── requirements.txt            # pygame, pytest
├── README.md, RULES.md, DECISIONS.md, ARCHITECTURE.md
├── wumpus/
│   ├── config.py               # GameConfig (rules, seed), AgentConfig (agent parameters, ablation flags)
│   ├── core/                   # SHARED, no hidden information: geometry, actions, percepts (Percept, PublicInfo)
│   ├── environment/            # HIDDEN GROUND TRUTH - the explorer may never import this
│   │   ├── world.py            # WorldState, seeded generation, route check
│   │   ├── mechanics.py        # transition rules (forward/turn/jump/teleport/collapse/shuffle/shoot/Wumpus)
│   │   ├── percepts.py         # builds the explorer's Percept from the hidden world
│   │   ├── events.py           # structured GameEvent for every action
│   │   └── engine.py           # GameEngine: turn sequence, score, status, outcome
│   ├── agents/                 # the explorer - may only import core/ (and config)
│   │   ├── knowledge_base.py   # propositional KB, DPLL, entailment, contradiction cores, DungeonKB
│   │   ├── bayesian.py         # pit enumeration + HazardEstimate (pit / Wumpus / death / confidence)
│   │   ├── wumpus_tracker.py   # HMM forward filter over the moving Wumpus
│   │   ├── pathfinding.py      # A*, strict / risk-tolerant planning, alternative routes
│   │   ├── minimax.py          # alpha-beta, DuelGame, determinized decision, structured output
│   │   ├── explorer.py         # decision pipeline, KnowledgeSnapshot, DecisionReport
│   │   └── trace.py            # BrainLog / TraceEntry
│   ├── control/                # Wumpus controllers: human, scripted (idle/random/hunter), Minimax AI
│   ├── ui/                     # Pygame: theme, icons, session (threaded AI), anim, board_view, brain_panel, app
│   ├── terminal.py             # terminal version
│   ├── runner.py               # headless game loop
│   ├── metrics.py              # evaluation metrics: agent beliefs vs the hidden world
│   └── human_log.py            # records human-Wumpus games
├── evaluation/                 # scenarios.py (10 fixed dungeons), benchmark.py, oracle.py, human_games.py, results/
├── docs/                       # screenshots/ (from the real interface), REPORT.md, SLIDE_SCRIPT.md
└── tests/                      # 17 smoke tests (one per component), helpers.py
```

## 2. Module responsibilities

| Module | Responsibility | May import |
|---|---|---|
| `core.*` | Value types shared by everyone: `Pos`, `Direction`, actions, `Percept`, `PublicInfo`. No randomness, no hidden state. | stdlib |
| `config` | `GameConfig` (rules, seed) and `AgentConfig` (agent parameters, ablation switches `use_bayes`, `use_minimax`). | stdlib |
| `environment.world` | Hidden dungeon: pits, gold, Wumpus, portal pairs, unstable and shuffle tiles. Seeded generation that retries until a walkable route to the gold exists. | core |
| `environment.mechanics` | Pure functions implementing the RULES.md transitions: move, turn, jump, teleport, collapse, shuffle, shoot, Wumpus move. | core, environment.world |
| `environment.percepts` | Builds the explorer's `Percept` from the hidden world. | core, environment.world |
| `environment.engine` | Owns the `WorldState`, runs the turn sequence, validates actions, keeps the score, decides the outcome. The only thing that touches both sides. | core, environment |
| `agents.knowledge_base` | Generic propositional logic (integer literals, CNF, DPLL with unit propagation, entailment by refutation, minimal unsatisfiable cores) and `DungeonKB`, the Wumpus-world rules rebuilt from the evidence every turn. Wumpus facts are time-stamped and expire after the turn. | core |
| `agents.bayesian` | `P(pit \| breezes)` per cell by exact enumeration of frontier models, split into independent components; combines pit and Wumpus risk into a `HazardEstimate` per cell. | core |
| `agents.wumpus_tracker` | HMM forward filter over the Wumpus' cell: predict with the motion model, update with stench, survival and missed-arrow evidence; `entry_risk` for a cell. | core |
| `agents.pathfinding` | A* over `(cell, facing, jump available)` states with risk-weighted step costs and a portal-aware admissible heuristic; strict / tolerant planning with alternative routes and rejection reasons. | core |
| `agents.minimax` | Depth-limited minimax and alpha-beta, `DuelGame` (explorer vs Wumpus), and the determinized decision over the top-k Wumpus hypotheses. | core |
| `agents.explorer` | The per-turn pipeline, goal selection, replanning, the walk-in check, `KnowledgeSnapshot` and `DecisionReport`; writes every step to the `BrainLog`. | core, config, agents.* |
| `agents.trace` | `TraceEntry(id, turn, tag, summary, details, refs, cell)`; `details` is the real data the algorithm produced. | core |
| `control.wumpus_controllers` | Who decides the Wumpus' action: human (a queue filled by the UI), scripted idle / random / hunter, or the Minimax AI. The Wumpus is omniscient, so controllers may read the world. | core, agents.minimax |
| `ui.*` | Rendering and input. The explorer runs in a worker thread; the engine is changed only on the main thread. No game logic. | everything |
| `runner`, `terminal` | Headless game loop; terminal version. | everything |
| `metrics`, `human_log` | Evaluation metrics measured against the hidden world; recording of human-Wumpus games. | everything |

**Separation guarantee:** a test in `tests/test_smoke.py` parses every file in `wumpus/agents/` with `ast`
and fails if it imports `environment`, `ui`, `control`, `runner` or `pygame`. The engine only
hands the agent `PublicInfo` (once) and `Percept` objects (every turn).

## 3. Class interfaces (abridged)

```python
# core
class Pos(NamedTuple): x: int; y: int
class Direction(Enum): N, E, S, W
class ExplorerAction(Enum): FORWARD, TURN_LEFT, TURN_RIGHT, JUMP, GRAB, SHOOT, CLIMB
@dataclass(frozen=True) class WumpusAction: kind: WumpusActionType; direction: Direction | None = None
@dataclass(frozen=True) class Percept: turn, step, position, facing, breeze, stench, glitter, bump, scream,
                                       portal_signal, unstable, shuffled, teleported, has_gold, arrows,
                                       jump_available, score
@dataclass(frozen=True) class PublicInfo: size, start, start_facing, pit_count, portal_pairs, unstable_tiles,
                                          shuffle_tiles, safe_start_zone, wumpus_min_start_distance, arrows,
                                          jumps, jump_distance, max_turns

# environment (hidden)
class WorldState: size, start, pits, gold, wumpus, explorer, facing, portals, unstable, shuffle_tiles, route,
                  arrows, jump_available, wumpus_alive, explorer_alive, has_gold, collapsed, visited, ...
def generate_world(config, rng) -> WorldState
class GameEngine:
    __init__(config=None, world=None)
    public_info() -> PublicInfo
    percept() -> Percept
    status() -> GameStatus
    legal_explorer_actions() / legal_wumpus_actions() -> list
    apply_explorer_action(a) -> list[GameEvent]
    apply_wumpus_action(a)   -> list[GameEvent]
    phase: Phase                     # EXPLORER | WUMPUS | OVER
    result: GameResult | None        # (outcome, reason)

# agents (the AI)
def dpll(clauses, assignment=None, stats=None) -> dict[int, bool] | None
class KnowledgeBase: tell(literals, source, valid_turn=None); ask(name, positive=True) -> InferenceResult
                     consistency() -> ConsistencyReport
class DungeonKB:     rebuild(visited, collapsed, breeze_obs, ...) -> KnowledgeBase
                     classify(cells) / classify_wumpus(cells) / classify_safe(cells) -> dict[Pos, CellStatus]
def estimate_pits(size, statuses, breeze_obs, prior) -> dict[Pos, CellRiskReport]
class WumpusTracker: predict(explorer, blocked, pit_prob); observe_stench(p, stench); observe_survived_at(p)
                     observe_arrow_missed(cells); normalize(); entry_risk(c, blocked, pit_prob); top(k)
def astar(start, facing, goals, size, cell_cost, jump=False, portals=None, ...) -> PathResult
def plan_routes(start, facing, goals, model, mode, tolerance, jump) -> RoutePlan   # best route + rejected variants
def alphabeta(game, state, depth, alpha, beta, stats) -> float
def determinized_decision(game, root, hypotheses, depth) -> MinimaxResult          # E[V] per root action
class ExplorerAgent: __init__(info: PublicInfo, config: AgentConfig | None = None)
                     act(percept) -> ExplorerAction       # = observe(percept), then decide()
                     knowledge() -> KnowledgeSnapshot; log: BrainLog; reports: list[DecisionReport]
```

## 4. State representation

* **Hidden state (`WorldState`):** exact ground truth, owned by the engine. Sets and dicts of `Pos`.
* **Explorer knowledge (`ExplorerAgent`):** completely separate objects:
  * `visited`, `collapsed` (cell → turn it became a pit), `portals` (pairs learned by teleporting),
    `breeze_obs` / `signal_obs` (`Pos → (bool, turn)`; only observations after the last shuffle are valid)
  * `dkb: DungeonKB`: symbols `PIT_x_y`, `BREEZE_x_y`, `WUMPUS_x_y`, `STENCH_x_y`, `SAFE_x_y`, `PORTAL_x_y`,
    `SIGNAL_x_y`; CNF clauses tagged with their source, rebuilt from the evidence every turn
  * `pit_prob` with a `CellRiskReport` per cell (the evidence for each number), and `hazards`
    (`HazardEstimate`: P(pit), P(Wumpus now), P(Wumpus moves in), P(death), confidence)
  * `tracker: WumpusTracker`: the belief over the Wumpus' cell, and whether it is alive
  * `plan` (actions, goal, the reason it was made), `prediction` (the Wumpus reply Minimax expects)
* **Coordinates:** `(x, y)`, 0-based, origin bottom-left, start at `(0, 0)` facing east.

## 5. Turn sequence

```
loop:
  1. Engine builds the Percept (breeze, stench, glitter, bump, scream, portal signal, unstable floor,
     shuffled, teleported)
  2. Explorer.observe(percept): evidence from its own last action → DungeonKB rebuild + DPLL
     classification → Bayesian pit probabilities → Wumpus filter → hazard estimates
  3. Explorer.decide(): goal selection → keep or replan the A* route → Minimax if the Wumpus is
     probably within 2 cells (then the walk-in check) → hazard guard → action + DecisionReport
  4. Engine applies the explorer action (move, turn, jump, teleport, shoot, grab, climb) and resolves
     hazards; leaving an unstable tile collapses it, stepping on a shuffle tile moves a pit
  5. If the game is not over: the Wumpus controller (human or scripted) moves one cell or stays;
     the engine validates it and resolves "Wumpus enters the explorer's cell"
  6. turn += 1; check the turn limit (150)
```

## 6. Development milestones (each ends green + git commit)

| # | Milestone | Verification |
|---|---|---|
| M0 | Scaffold, venv, docs, core types | pytest on core |
| M1 | Hidden world, mechanics, engine, scripted Wumpus, headless sim | rule tests per mechanic, seeded determinism |
| M2 | Propositional KB + DPLL | entailment tests incl. classic textbook cases |
| M3 | Bayesian pit estimation + Wumpus HMM filter | exact-number tests vs hand calculation |
| M4 | A* with jump/teleport | optimality vs Dijkstra, admissibility |
| M5 | Minimax + alpha-beta | equals plain minimax value, prunes > 0 |
| M6 | ExplorerAgent + BrainLog + replanning | separation test, seeded full-game integration tests |
| M7 | Environment v2 (user spec): facing, 2 portal pairs, collapsing/shuffle tiles, 3-cell jump, 8 percepts, score, events; AI adapted | env tests + 90-game simulation sweep |
| M8 | Terminal version: human Wumpus, `i <id>` inspector, knowledge/belief maps | scripted-keystroke tests |
| M9 | Pygame UI, clickable inspector | headless (SDL dummy) smoke test |
| M10 | Polish, how-to-play screen, final DECISIONS | full suite |
| M11 | Evaluation: 500 held-out maps, confidence intervals, paired tests, oracle, belief metrics, human-game log (D16) | benchmark + oracle smoke test |
| M12 | Minimax walk-in fix (D17), full benchmark re-run | regression test on map 1007 |

## 7. Technical risks

| Risk | Mitigation |
|---|---|
| Frontier enumeration is exponential | Split frontier into independent components; backtracking that prunes inconsistent partial models; cap with a labelled approximation |
| DPLL slow in pure Python | Integer literals, unit propagation, only query frontier cells, connected components cached per KB |
| Moving Wumpus breaks a static propositional KB | Wumpus facts are time-stamped and expire each turn; the HMM filter carries information across turns |
| Human's Wumpus policy unknown → filter model mismatch | Mixture motion model (15% stay, then half pursuit, half wander); belief reset on zero-likelihood evidence, logged; walk-in check after Minimax (D17). Not learned online (a known limitation) |
| Minimax under partial observability | Determinized minimax over the top-4 Wumpus hypotheses, weighted by belief (documented assumptions) |
| Agent leaks hidden info by accident | AST separation test; agent only receives frozen `Percept` / `PublicInfo` dataclasses |
| Fabricated explanations in UI | Brain entries are built from the return objects of the algorithms, never from templates alone |
| UI blocking AI | The explorer runs in a worker thread and the engine is changed only on the main thread; decisions average 10-37 ms |
| Agent oscillation / getting stuck | Plan commitment + explicit replanning triggers; anti-paralysis; hunt, then retreat rules; turn limit |
