# Game rules and state-transition rules (v2)

Implemented in `wumpus/environment/` (`world.py`, `mechanics.py`, `percepts.py`, `events.py`,
`engine.py`); every rule has a test in `tests/test_mechanics.py` / `tests/test_engine.py`.
**(public)** = known to the explorer AI via `PublicInfo` / `Percept`. Everything else is hidden.

## Setup (seeded, reproducible)

* 8×8 grid, `(x, y)`, origin bottom-left. Explorer starts at `S = (0,0)` facing **RIGHT (E)**. **(public)**
* **Safe start zone:** `S` and its orthogonal neighbours contain no pit, portal, special tile or
  Wumpus, and the Wumpus starts at Manhattan distance ≥ 3 from `S`. **(public rule)**
* **Special cells** (hidden locations, counts **(public)**), all outside the start zone and disjoint:
  * 2 **portal pairs** (4 cells); the two cells of a pair are ≥ 3 apart.
  * 3 **collapsing (unstable) tiles**.
  * 2 **shuffle tiles**.
* **Pits:** `round(13.5% × eligible)` pits, where *eligible* = cells outside the start zone that
  are not special. With defaults: 52 eligible cells, 7 pits (13.5%). Count **(public)**.
* **Gold** on an eligible non-pit cell. **Wumpus** on an eligible non-pit, non-gold cell.
* **Route check:** a map is accepted only if a walking route `S → gold` exists that avoids pits,
  unstable tiles and portals. Walking it back is then also valid, because no rule can ever change
  a cell on it (see Shuffle). Otherwise the generator retries with the same RNG stream.

## Explorer actions (one per turn; each costs 1 point)

| Action | Transition |
|---|---|
| `FORWARD` | target = pos + facing. Out of bounds → **bump**, nothing else happens. Else **leave** pos, **arrive** at target. |
| `TURN_LEFT` / `TURN_RIGHT` | facing rotates 90°. |
| `JUMP` | Only if the jump is unused and pos + 3·facing is in bounds (else **bump**, jump not used). Jump is used up; the 2 intermediate cells are crossed with no effect (pits, Wumpus, tiles and portals are all skipped); **leave** pos, **arrive** at pos + 3·facing. |
| `GRAB` | If gold is here: `has_gold = True`. Otherwise nothing. |
| `SHOOT` | If arrows > 0: arrows −1, extra −10 points; the arrow flies along facing to the wall (over pits). If the living Wumpus is on that ray it dies → **scream**. |
| `CLIMB` | Only at `S`: game ends. Elsewhere nothing. |

**Leave cell c:** if `c` is an unstable tile, it **collapses into a pit** (permanent).
**Arrive at cell c:**
1. If `c` is a portal, the entity moves to the paired portal (no chaining) → `teleported`.
2. Final cell is a pit → explorer **falls** (dies). Final cell holds the living Wumpus → **eaten**.
3. Otherwise the cell is visited. If it is an unused shuffle tile → **shuffle** (below), then the tile is used up.

**Shuffle (seeded deterministic swap):** let *E* = cells that are not visited, not the explorer or
Wumpus cell, not gold, not special, not in the start zone, not a collapsed pit and not on the
generation route. With `rng = Random(f"{seed}:shuffle:{tile}:{step}")`, pick one original pit in
*E* and one empty cell in *E* (both lists sorted), and swap their contents (the pit moves). If
either list is empty, nothing moves. The explorer perceives only that a shuffle happened.

## Wumpus (human) actions (one per turn)

* `MOVE d` (N/E/S/W): target in bounds and not a pit (collapsed tiles are pits). Walls and pits
  block it. Portals teleport it like the explorer. Entering the explorer's cell **eats** the explorer.
* `STAY`: always legal. Unstable and shuffle tiles have no effect on the Wumpus.
* A dead Wumpus has no turn.

## Percepts (at the start of each explorer turn)

| Percept | True iff |
|---|---|
| breeze | a pit is orthogonally adjacent (collapsed tiles are pits) |
| stench | the living Wumpus is orthogonally adjacent |
| glitter | gold is in the explorer's cell |
| bump | the last FORWARD/JUMP hit the wall |
| scream | the last SHOOT killed the Wumpus |
| portal_signal | a portal is orthogonally adjacent |
| unstable | the explorer stands on an unstable tile (it collapses when it is left) |
| shuffled | the last action triggered a shuffle tile |
| teleported | the last action ended in a teleport |

The percept also carries the explorer's own state: position, facing, step, score, has_gold, arrows,
jump_available. These are things the explorer knows about itself, not hidden world facts.

## Turn sequence and end conditions

`explorer action → (if game not over and Wumpus alive) Wumpus action → turn + 1`.
Explorer climbs out with gold: **win** (+1000). Climbs out without gold: **draw**. Falls or is
eaten: **loss** (−1000). Turn 150 is reached: **loss** (timeout).
Every action produces structured `GameEvent`s (see `environment/events.py`).
