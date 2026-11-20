import random

import pytest

from wumpus.config import GameConfig
from wumpus.core.geometry import Direction, Pos, manhattan, neighbors
from wumpus.environment.world import find_route, generate_world, safe_start_zone


@pytest.mark.parametrize("seed", range(40))
def test_generation_respects_rules(seed):
    cfg = GameConfig(seed=seed)
    w = generate_world(cfg, random.Random(seed))
    zone = safe_start_zone(w.start, w.size)
    specials = set(w.portals) | w.unstable | w.shuffle_tiles
    # start completely safe, facing right
    assert w.explorer == Pos(0, 0) and w.facing is Direction.E
    assert not (zone & (w.pits | specials)) and w.wumpus not in zone
    assert manhattan(w.wumpus, w.start) >= cfg.wumpus_min_start_distance
    # counts
    assert len(w.portals) == 4 and len(w.unstable) == 3 and len(w.shuffle_tiles) == 2
    assert len(specials) == 9
    assert all(w.portals[w.portals[p]] == p and manhattan(p, w.portals[p]) >= 3 for p in w.portals)
    eligible = 64 - len(zone) - len(specials)
    assert 0.12 <= len(w.pits) / eligible <= 0.15
    # contents disjoint
    assert w.gold not in w.pits | specials and w.wumpus not in w.pits | specials and w.wumpus != w.gold
    # witness route to the gold (and back) avoids every hazard and every changeable cell
    assert w.route[0] == w.start and w.route[-1] == w.gold
    assert not set(w.route) & (w.pits | w.unstable | set(w.portals))
    assert all(manhattan(a, b) == 1 for a, b in zip(w.route, w.route[1:]))


def test_generation_is_deterministic():
    a = generate_world(GameConfig(seed=42), random.Random(42))
    b = generate_world(GameConfig(seed=42), random.Random(42))
    assert (a.pits, a.gold, a.wumpus, a.portals, a.unstable, a.shuffle_tiles) == \
           (b.pits, b.gold, b.wumpus, b.portals, b.unstable, b.shuffle_tiles)
    c = generate_world(GameConfig(seed=43), random.Random(43))
    assert (a.pits, a.gold) != (c.pits, c.gold)


def test_pit_density_config():
    cfg = GameConfig()
    assert cfg.eligible_count() == 52 and cfg.pit_count() == 7


def test_find_route():
    assert find_route(8, Pos(0, 0), Pos(2, 0), {Pos(1, 0)}) == [Pos(0, 0), Pos(0, 1), Pos(1, 1), Pos(2, 1), Pos(2, 0)]
    wall = {Pos(1, y) for y in range(8)}
    assert find_route(8, Pos(0, 0), Pos(2, 0), wall) is None
