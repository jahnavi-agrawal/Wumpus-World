import random

import pytest

from wumpus.ai.pathfinding import astar, pad_aware_distance
from wumpus.core.actions import ExplorerActionType
from wumpus.core.geometry import Pos, all_cells


def free(_):
    return 0.0


def test_straight_path():
    r = astar(Pos(0, 0), {Pos(3, 0)}, 8, free)
    assert r.found and r.cost == 3 and r.cells[-1] == Pos(3, 0) and len(r.actions) == 3


def test_blocked_wall_no_path_without_jump():
    wall = {Pos(2, y) for y in range(8)}
    cost = lambda c: None if c in wall else 0.0
    assert not astar(Pos(0, 0), {Pos(4, 0)}, 8, cost).found
    r = astar(Pos(0, 0), {Pos(4, 0)}, 8, cost, jumps=1)
    assert r.found and sum(a.kind is ExplorerActionType.JUMP for a in r.actions) == 1


def test_jump_avoids_risky_cell():
    cost = lambda c: 50.0 if c == Pos(1, 0) else 0.0
    r = astar(Pos(0, 0), {Pos(2, 0)}, 3, lambda c: None if c.y > 0 and c.x == 1 else cost(c), jumps=1)
    assert r.actions[0].kind is ExplorerActionType.JUMP and r.cost == 2.5


def test_teleport_shortcut():
    pads = (Pos(1, 0), Pos(7, 7))
    r = astar(Pos(0, 0), {Pos(7, 6)}, 8, free, pads=pads)
    assert r.found and r.cost == 2 and r.cells == [Pos(7, 7), Pos(7, 6)]


def test_multiple_goals_picks_nearest():
    r = astar(Pos(0, 0), {Pos(5, 5), Pos(0, 2)}, 8, free)
    assert r.goal == Pos(0, 2)


@pytest.mark.parametrize("seed", range(30))
def test_astar_equals_dijkstra_and_heuristic_admissible(seed):
    rng = random.Random(seed)
    costs = {c: (None if rng.random() < 0.2 else rng.choice([0.0, 0.0, 1.5, 5.0])) for c in all_cells(8)}
    pads = tuple(rng.sample(all_cells(8), 2))
    for p in pads:
        costs[p] = 0.0
    costs[Pos(0, 0)] = 0.0
    goal = rng.choice([c for c in all_cells(8) if costs[c] is not None and c not in pads])
    a = astar(Pos(0, 0), {goal}, 8, costs.get, jumps=1, pads=pads)
    d = astar(Pos(0, 0), {goal}, 8, costs.get, jumps=1, pads=pads, use_heuristic=False)
    assert a.found == d.found
    if a.found:
        assert a.cost == pytest.approx(d.cost)
        assert a.heuristic_start <= a.cost + 1e-9
        assert a.expanded <= d.expanded


def test_pad_distance():
    assert pad_aware_distance(Pos(0, 0), Pos(7, 7), (Pos(0, 1), Pos(7, 6))) == 2
