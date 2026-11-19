import random

import pytest

from wumpus.config import GameConfig
from wumpus.core.geometry import Pos, manhattan
from wumpus.env.world import _reachable, generate_world, safe_start_zone


@pytest.mark.parametrize("seed", range(25))
def test_generation_respects_public_rules(seed):
    cfg = GameConfig(seed=seed)
    w = generate_world(cfg, random.Random(seed))
    zone = safe_start_zone(w.start, w.size)
    assert len(w.pits) == cfg.pit_count
    assert not (w.pits & zone)
    assert w.gold not in w.pits and w.gold not in zone
    assert w.pads and not (set(w.pads) & (w.pits | zone)) and manhattan(*w.pads) >= 4
    assert w.wumpus not in w.pits and manhattan(w.wumpus, w.start) >= cfg.wumpus_min_start_distance
    assert _reachable(w.size, w.start, w.gold, w.pits)


def test_generation_is_deterministic():
    a = generate_world(GameConfig(seed=42), random.Random(42))
    b = generate_world(GameConfig(seed=42), random.Random(42))
    assert (a.pits, a.gold, a.wumpus, a.pads) == (b.pits, b.gold, b.wumpus, b.pads)
    c = generate_world(GameConfig(seed=43), random.Random(43))
    assert (a.pits, a.gold, a.wumpus) != (c.pits, c.gold, c.wumpus)


def test_safe_zone():
    assert safe_start_zone(Pos(0, 0), 8) == {Pos(0, 0), Pos(0, 1), Pos(1, 0)}
