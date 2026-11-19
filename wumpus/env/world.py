"""Hidden ground truth of the dungeon. Only `wumpus.env` and the UI may read this."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field, replace

from ..config import GameConfig
from ..core.geometry import Pos, all_cells, manhattan, neighbors


def safe_start_zone(start: Pos, size: int) -> frozenset[Pos]:
    return frozenset([start, *neighbors(start, size)])


@dataclass
class WorldState:
    size: int
    start: Pos
    pits: set[Pos]
    gold: Pos | None
    wumpus: Pos
    explorer: Pos
    pads: tuple[Pos, Pos] | None
    arrows: int
    jumps_left: int
    collapse_charges: int
    shuffle_charges: int
    collapse_cooldown: int = 0
    wumpus_alive: bool = True
    explorer_alive: bool = True
    has_gold: bool = False
    collapsed: set[Pos] = field(default_factory=set)
    visited: set[Pos] = field(default_factory=set)

    def copy(self) -> "WorldState":
        return replace(self, pits=set(self.pits), collapsed=set(self.collapsed), visited=set(self.visited))

    def pad_pair(self, p: Pos) -> Pos | None:
        if self.pads is None:
            return None
        a, b = self.pads
        return b if p == a else a if p == b else None

    def is_blocked_for_wumpus(self, p: Pos) -> bool:
        return p in self.pits or p in self.collapsed


def _reachable(size: int, start: Pos, goal: Pos, blocked: set[Pos]) -> bool:
    seen, q = {start}, deque([start])
    while q:
        p = q.popleft()
        if p == goal:
            return True
        for n in neighbors(p, size):
            if n not in seen and n not in blocked:
                seen.add(n)
                q.append(n)
    return False


def generate_world(config: GameConfig, rng: random.Random | None = None) -> WorldState:
    """Seeded generation following RULES.md 'Setup'. Retries until gold is reachable on foot."""
    rng = rng or random.Random(config.seed)
    size, start = config.size, Pos(0, 0)
    zone = safe_start_zone(start, size)
    outside = [c for c in all_cells(size) if c not in zone]
    for _ in range(10_000):
        pads = None
        if config.use_pads:
            a, b = rng.sample(outside, 2)
            if manhattan(a, b) < 4:
                continue
            pads = (a, b)
        free = [c for c in outside if not pads or c not in pads]
        pits = set(rng.sample(free, config.pit_count))
        non_pit = [c for c in free if c not in pits]
        gold = rng.choice(non_pit)
        wumpus_cells = [c for c in non_pit if c != gold and manhattan(c, start) >= config.wumpus_min_start_distance]
        if not wumpus_cells:
            continue
        wumpus = rng.choice(wumpus_cells)
        if not _reachable(size, start, gold, pits):
            continue
        return WorldState(
            size=size, start=start, pits=pits, gold=gold, wumpus=wumpus, explorer=start, pads=pads,
            arrows=config.arrows, jumps_left=config.jumps, collapse_charges=config.collapse_charges,
            shuffle_charges=config.shuffle_charges, visited={start},
        )
    raise RuntimeError("could not generate a solvable world")
