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
│   ├── agents/                 # the explorer - may only import core/
│   │   ├── knowledge_base.py   # propositional KB, DPLL, entailment, contradiction cores, DungeonKB
│   │   ├── bayesian.py         # pit enumeration + HazardEstimate (pit / Wumpus / death / confidence)
│   │   ├── wumpus_tracker.py   # HMM forward filter over the moving Wumpus
│   │   ├── pathfinding.py      # A*, strict / risk-tolerant planning, alternative routes
│   │   ├── minimax.py          # alpha-beta, DuelGame, determinized decision, structured output
│   │   ├── explorer.py         # decision pipeline, KnowledgeSnapshot, DecisionReport
│   │   └── trace.py            # BrainLog / TraceEntry
│   ├── control/                # Wumpus controllers: human, scripted, Minimax AI
│   ├── ui/                     # Pygame: theme, icons, session (threaded AI), anim, board_view, brain_panel, app
│   ├── terminal.py             # terminal version
│   └── runner.py               # headless game loop
├── evaluation/                 # scenarios.py (10 fixed dungeons), benchmark.py, results/
├── docs/screenshots/           # captured from the real interface
└── tests/                      # unit, integration, separation and UI tests
```

## 2. Module responsibilities

| Module | Responsibility | May import |
|---|---|---|
| `core.*` | Value types shared by everyone. No randomness, no hidden state. | stdlib |
| `environment.world` | Hidden dungeon: pits, gold, Wumpus, collapsed tiles, pads. Seeded generation with a solvability check. | core |
| `environment.mechanics` | Pure functions implementing RULES.md transitions. | core, environment.world |
| `environment.engine` | Owns the `WorldState`, runs the turn sequence, validates actions, builds `Percept`s, decides the outcome. Only thing that touches both sides. | core, environment |
| `ai.logic` | Generic propositional logic: literals as ints, CNF clauses, DPLL SAT, entailment `KB ⊨ α`. Wumpus-specific KB built on top. | stdlib |
| `ai.probability` | `P(pit | evidence)` per cell by exact enumeration of frontier models, split into independent components. | core, ai.logic results |
| `ai.wumpus_tracker` | Forward filter over Wumpus location (predict with motion model, update with stench/arrow/survival evidence). | core |
| `ai.pathfinding` | Generic A* over `(pos, jumps_left)` states, admissible heuristic that accounts for teleport pads. | core |
| `ai.minimax` | Depth-limited minimax + alpha-beta over an abstract adversarial state built from the explorer's beliefs. | core |
| `ai.explorer` | Orchestrates everything per turn and writes every decision to the `BrainLog`. | core, ai.* |
| `ai.trace` | `TraceEntry(id, turn, tag, summary, details)`; `details` is the real data the algorithm produced. | stdlib |
| `control.*` | Who decides the Wumpus action: human (queue filled by UI) or scripted (tests/demo). | core |
| `ui.*` | Rendering and input. No game logic. | everything |

**Separation guarantee:** a test in `tests/test_smoke.py` parses every file in `wumpus/agents/` with `ast`
and fails if it imports `wumpus.environment` (or `wumpus.ui`). The engine only hands the agent
`PublicInfo` (once) and `Percept` objects (every turn).

## 3. Class interfaces (abridged)

```python
# core
class Pos(NamedTuple): x: int; y: int
class Direction(Enum): N, E, S, W           # .dx, .dy
@dataclass(frozen=True) class ExplorerAction: kind: ExplorerActionType; direction: Direction | None
@dataclass(frozen=True) class WumpusAction:  kind: WumpusActionType;  direction=None; target: Pos | None
@dataclass(frozen=True) class Percept: turn, position, stench, breeze, glitter, scream, events, has_gold, arrows, jumps_left
@dataclass(frozen=True) class PublicInfo: size, start, pit_count, pads, safe_start_zone, wumpus_min_start_distance, collapse_range, ...

# env
class WorldState: size, pits:set[Pos], gold:Pos|None, wumpus:Pos, wumpus_alive, explorer:Pos, collapsed:set[Pos], pads, visited, ...
    @staticmethod generate(config, rng) -> WorldState
class GameEngine:
    __init__(config, wumpus_controller)
    public_info() -> PublicInfo
    current_percept() -> Percept
    legal_explorer_actions() / legal_wumpus_actions() -> list
    apply_explorer_action(a) -> list[str]   # resolution log
    apply_wumpus_action(a)   -> list[str]
    outcome: Outcome | None

# ai
class KnowledgeBase: tell(clause, source), ask(literal) -> InferenceResult(entailed, proof_info)
class PitEstimator: estimate(kb_facts, observations, prior) -> dict[Pos, CellRiskReport]
class WumpusTracker: predict(...), observe(...), belief: dict[Pos,float], report()
def astar(start, goal_test, successors, heuristic) -> PathResult(path, actions, cost, expanded)
def alphabeta(state, depth, game) -> MinimaxResult(best_action, value, root_values, nodes, prunes, pv)
class ExplorerAgent: __init__(PublicInfo, seed); observe(Percept); decide() -> ExplorerAction; log: BrainLog
```

## 4. State representation

* **Hidden state (`WorldState`)** — exact ground truth, owned by the engine. Sets of `Pos`.
* **Explorer knowledge (`ExplorerAgent`)** — completely separate objects:
  * `visited: set[Pos]`, `collapsed: set[Pos]` (public), `breeze_obs: dict[Pos, (bool, turn)]`
    (only observations made after the last shuffle are valid)
  * `kb: KnowledgeBase` — symbols `P_x_y` (pit) and `B_x_y` (breeze), CNF clauses tagged with their source
  * `pit_prob: dict[Pos, float]` + per-cell `CellRiskReport` (the evidence for each number)
  * `wumpus belief: dict[Pos, float]` + `wumpus_alive`
  * `plan: list[ExplorerAction]` + the goal and the belief snapshot it was built on (for replanning)
* **Coordinates** — `(x, y)`, 0-based, origin bottom-left, start at `(0, 0)`.

## 5. Turn sequence

```
loop:
  1. Engine builds Percept for the explorer (stench/breeze/glitter/scream + public events since last turn)
  2. Explorer.observe(percept): KB TELL → KB ASK (DPLL) → Bayesian pits → Wumpus filter
  3. Explorer.decide(): goal selection → plan validation / A* replan → Minimax if Wumpus threat → action
  4. Engine applies explorer action (move/jump/teleport/shoot/grab/climb) and resolves hazards
  5. If game not over: Wumpus controller (human) chooses action; engine validates and applies it
     (move/teleport, collapse, shuffle); resolves "Wumpus enters explorer cell"
  6. turn += 1; check turn limit
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
| M10 | Polish, final DECISIONS | full suite |

## 7. Technical risks

| Risk | Mitigation |
|---|---|
| Frontier enumeration is exponential | Split frontier into independent components; backtracking that prunes inconsistent partial models; cap with documented fallback |
| DPLL slow in pure Python | Integer literals, unit propagation, only query frontier cells, cache when KB unchanged |
| Moving Wumpus breaks a static propositional KB | KB is used only for static facts (pits); the moving Wumpus is tracked with a Bayesian filter |
| Human's Wumpus policy unknown → filter model mismatch | Mixture motion model (random + pursuit); belief reset on zero-likelihood evidence, logged |
| Minimax under partial observability | Determinized minimax over top-k Wumpus hypotheses, weighted by belief (documented assumptions) |
| Agent leaks hidden info by accident | AST separation test; agent only receives frozen `Percept` dataclasses |
| Fabricated explanations in UI | Brain entries are built from the return objects of the algorithms, never from templates alone |
| UI blocking AI | AI step is fast (<0.5 s target); measured in tests |
| Agent oscillation / getting stuck | Plan commitment + explicit replanning triggers; turn limit; retreat rule |
