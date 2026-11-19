# Game rules and state-transition rules

All rules below are implemented in `wumpus/env/mechanics.py` and `wumpus/env/engine.py`
and each has a test in `tests/test_mechanics.py` / `tests/test_engine.py`.
Everything marked **(public)** is known to the explorer AI through `PublicInfo`/`Percept`.

## Setup (seeded, `random.Random(config.seed)`)

* 8×8 grid, coordinates `(x, y)`, start `S = (0,0)` bottom-left. **(public)**
* **Safe start zone**: `S` and its orthogonal neighbours contain no pit. **(public)**
* `pit_count` pits (default 7) placed uniformly on the remaining cells. Count is **(public)**, locations are hidden.
* One pair of **teleporter pads** `A, B`; never pits, never in the start zone. Locations **(public)**.
* **Gold** on a non-pit, non-pad cell outside the start zone. Hidden.
* **Wumpus** on a non-pit, non-pad cell with Manhattan distance ≥ 3 from `S` **(rule public)**.
* Generation retries until a pit-free walking path from `S` to the gold exists.

## Percepts (delivered at the start of each explorer turn)

| Percept | True iff |
|---|---|
| `stench` | the living Wumpus is orthogonally adjacent to the explorer |
| `breeze` | at least one orthogonally adjacent cell contains a pit (collapsed tiles do **not** cause breeze) |
| `glitter` | the gold is in the explorer's cell |
| `scream` | the Wumpus was killed by the explorer's arrow this turn |
| `events` | public events since the explorer's last turn: `COLLAPSE(pos)`, `SHUFFLE` |

## Explorer actions (one per turn)

| Action | Precondition | Transition |
|---|---|---|
| `MOVE d` | target in bounds | explorer → target; then **resolve landing** |
| `JUMP d` | `jumps_left > 0`, landing `pos+2d` in bounds | `jumps_left -= 1`; middle cell is skipped completely (no hazard, no percept, not visited); explorer → landing; **resolve landing** |
| `SHOOT d` | `arrows > 0` | `arrows -= 1`; arrow flies along the ray until the wall, over pits; if the living Wumpus is on the ray it dies → `scream` |
| `GRAB` | gold in cell | `has_gold = True`, gold removed from map |
| `CLIMB` | explorer at `S` | game ends: with gold → **Explorer wins**, without → **Draw (retreat)** |
| `WAIT` | — | nothing |

**Resolve landing** (explorer or Wumpus arriving at cell `c` by walking or jumping):
1. If `c` is a pad, the entity is moved to the paired pad (`c := pair(c)`); no chaining.
2. Explorer: `c` pit or collapsed → **dies (fell)**; `c` holds the living Wumpus → **dies (eaten)**.
   Otherwise `c` is marked visited.

## Wumpus (human) actions (one per turn)

| Action | Precondition | Transition |
|---|---|---|
| `MOVE d` | target in bounds, not a pit, not collapsed | Wumpus → target; pad teleports it; if it ends on the explorer → **explorer eaten** |
| `STAY` | — | nothing |
| `COLLAPSE c` | charges > 0, cooldown = 0, `manhattan(wumpus, c) ≤ collapse_range` (2), `c` not `S`, not explorer cell, not Wumpus cell, not pit, not pad, not gold, not collapsed | `c` becomes collapsed (permanent, deadly, impassable for both). Wumpus does not move. Event `COLLAPSE(c)` is public. charges −1, cooldown := 3 |
| `SHUFFLE` | charges > 0 (default 1) | All pits are removed and the same number re-placed uniformly at random on cells that are **not** visited by the explorer, not the explorer/Wumpus cell, not collapsed, not pads, not gold, not the safe start zone. Wumpus does not move. Event `SHUFFLE` is public. |

The cooldown decreases by 1 at the end of every Wumpus turn. The dead Wumpus has no actions.

## Consequences the explorer can rely on (all derivable from the public rules)

* A visited cell is pit-free forever (shuffle never puts pits on visited cells).
* A collapsed cell was not a pit before collapsing.
* A `COLLAPSE(c)` event means the Wumpus stood still this turn within distance 2 of `c`.
* A `SHUFFLE` invalidates every breeze observation made before it.
* Surviving a move into `c` means the Wumpus was not in `c`.
* A missed arrow means the Wumpus was not on that ray.

## End conditions

* Explorer climbs out with gold → **Explorer wins**.
* Explorer climbs out without gold → **Draw**.
* Explorer falls or is eaten, or `max_turns` (150) is reached → **Wumpus wins**.
