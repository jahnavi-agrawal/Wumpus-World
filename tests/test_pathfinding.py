import random

import pytest

from wumpus.ai.pathfinding import astar, portal_aware_distance, portal_exit_distances
from wumpus.core.actions import FORWARD, JUMP, TURN_LEFT, TURN_RIGHT
from wumpus.core.geometry import Direction as D, Pos, all_cells


def free(_):
    return 0.0


def test_straight_path_no_turns():
    r = astar(Pos(0, 0), D.E, {Pos(3, 0)}, 8, free)
    assert r.found and r.actions == [FORWARD] * 3 and r.cost == 3


def test_turns_cost_an_action():
    r = astar(Pos(0, 0), D.E, {Pos(0, 2)}, 8, free)
    assert r.actions == [TURN_LEFT, FORWARD, FORWARD] and r.cost == 3
    r = astar(Pos(3, 0), D.E, {Pos(1, 0)}, 8, free)        # behind us: two turns
    assert r.actions[:2] in ([TURN_LEFT, TURN_LEFT], [TURN_RIGHT, TURN_RIGHT]) and r.cost == 4


def test_jump_three_cells_over_a_blocked_strip():
    wall = {Pos(1, y) for y in range(8)} | {Pos(2, y) for y in range(8)}
    cost = lambda c: None if c in wall else 0.0
    assert not astar(Pos(0, 0), D.E, {Pos(3, 0)}, 8, cost).found
    r = astar(Pos(0, 0), D.E, {Pos(3, 0)}, 8, cost, jump=True)
    assert r.found and r.actions == [JUMP] and r.cells == [Pos(3, 0)]


def test_jump_used_at_most_once():
    wall = {Pos(1, y) for y in range(8)} | {Pos(2, y) for y in range(8)} | \
           {Pos(4, y) for y in range(8)} | {Pos(5, y) for y in range(8)}
    cost = lambda c: None if c in wall else 0.0
    assert not astar(Pos(0, 0), D.E, {Pos(6, 0)}, 8, cost, jump=True).found


def test_known_portal_shortcut_and_facing_kept():
    portals = {Pos(1, 0): Pos(6, 6), Pos(6, 6): Pos(1, 0)}
    r = astar(Pos(0, 0), D.E, {Pos(7, 6)}, 8, free, portals=portals)
    assert r.found and r.cells[:2] == [Pos(6, 6), Pos(7, 6)] and r.cost == 2


def test_two_portal_pairs_heuristic_admissible():
    portals = {Pos(0, 1): Pos(4, 4), Pos(4, 4): Pos(0, 1), Pos(4, 5): Pos(7, 7), Pos(7, 7): Pos(4, 5)}
    D_ = portal_exit_distances({Pos(7, 6)}, portals)
    assert portal_aware_distance(Pos(0, 0), {Pos(7, 6)}, portals, D_) == 3      # 1 + 1 + 1 via chain


@pytest.mark.parametrize("seed", range(25))
def test_astar_equals_dijkstra_and_heuristic_admissible(seed):
    rng = random.Random(seed)
    costs = {c: (None if rng.random() < 0.2 else rng.choice([0.0, 0.0, 1.5, 5.0])) for c in all_cells(8)}
    cells = rng.sample(all_cells(8), 4)
    portals = {cells[0]: cells[1], cells[1]: cells[0], cells[2]: cells[3], cells[3]: cells[2]}
    for p in portals:
        costs[p] = 0.0
    costs[Pos(0, 0)] = 0.0
    goal = rng.choice([c for c in all_cells(8) if costs[c] is not None and c not in portals])
    f = rng.choice(list(D))
    a = astar(Pos(0, 0), f, {goal}, 8, costs.get, jump=True, portals=portals)
    d = astar(Pos(0, 0), f, {goal}, 8, costs.get, jump=True, portals=portals, use_heuristic=False)
    assert a.found == d.found
    if a.found:
        assert a.cost == pytest.approx(d.cost)
        assert a.heuristic_start <= a.cost + 1e-9
        assert a.expanded <= d.expanded
