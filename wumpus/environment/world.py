"""Hidden ground truth of the dungeon. Only `wumpus.environment` and the UIs may read this."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field, replace

from ..config import GameConfig
from ..core.geometry import Direction, Pos, all_cells, manhattan, neighbors


def safe_start_zone(start: Pos, size: int) -> frozenset[Pos]:
    return frozenset([start, *neighbors(start, size)])


@dataclass
class WorldState:
    size: int
    start: Pos
    pits: set[Pos]                       # includes collapsed tiles
    gold: Pos | None
    wumpus: Pos
    explorer: Pos
    facing: Direction
    portals: dict[Pos, Pos]              # symmetric: a -> b and b -> a
    unstable: set[Pos]                   # not yet collapsed
    shuffle_tiles: set[Pos]              # not yet used
    route: tuple[Pos, ...]               # generation witness route S -> gold (never modified)
    arrows: int
    jump_available: bool
    seed: int = 0
    wumpus_alive: bool = True
    explorer_alive: bool = True
    has_gold: bool = False
    collapsed: set[Pos] = field(default_factory=set)
    used_shuffle: set[Pos] = field(default_factory=set)
    visited: set[Pos] = field(default_factory=set)

    def copy(self) -> "WorldState":
        return replace(self, pits=set(self.pits), portals=dict(self.portals), unstable=set(self.unstable),
                       shuffle_tiles=set(self.shuffle_tiles), collapsed=set(self.collapsed),
                       used_shuffle=set(self.used_shuffle), visited=set(self.visited))

    def portal_pair(self, p: Pos) -> Pos | None:
        return self.portals.get(p)

    def special_cells(self) -> set[Pos]:
        return (set(self.portals) | self.unstable | self.shuffle_tiles | self.collapsed | self.used_shuffle)


def find_route(size: int, start: Pos, goal: Pos, blocked: set[Pos]) -> list[Pos] | None:
    """BFS walking route (orthogonal steps) avoiding `blocked`; deterministic neighbour order."""
    prev: dict[Pos, Pos | None] = {start: None}
    q = deque([start])
    while q:
        p = q.popleft()
        if p == goal:
            path = [p]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            return path[::-1]
        for n in neighbors(p, size):
            if n not in prev and n not in blocked:
                prev[n] = p
                q.append(n)
    return None


def generate_world(config: GameConfig, rng: random.Random | None = None) -> WorldState:
    """Seeded generation following RULES.md 'Setup'."""
    rng = rng or random.Random(config.seed)
    size, start = config.size, Pos(0, 0)
    zone = safe_start_zone(start, size)
    outside = [c for c in all_cells(size) if c not in zone]
    for _ in range(10_000):
        specials = rng.sample(outside, config.special_count())
        k = 2 * config.portal_pairs
        portal_cells = specials[:k]
        pairs = [(portal_cells[i], portal_cells[i + 1]) for i in range(0, k, 2)]
        if any(manhattan(a, b) < 3 for a, b in pairs):
            continue
        unstable = set(specials[k:k + config.unstable_tiles])
        shuffle = set(specials[k + config.unstable_tiles:])
        eligible = [c for c in outside if c not in specials]
        pits = set(rng.sample(eligible, config.pit_count()))
        free = [c for c in eligible if c not in pits]
        gold = rng.choice(free)
        wcells = [c for c in free if c != gold and manhattan(c, start) >= config.wumpus_min_start_distance]
        if not wcells:
            continue
        wumpus = rng.choice(wcells)
        route = find_route(size, start, gold, pits | unstable | set(portal_cells))
        if route is None:
            continue
        portals = {}
        for a, b in pairs:
            portals[a], portals[b] = b, a
        return WorldState(size=size, start=start, pits=pits, gold=gold, wumpus=wumpus, explorer=start,
                          facing=Direction.E, portals=portals, unstable=unstable, shuffle_tiles=shuffle,
                          route=tuple(route), arrows=config.arrows, jump_available=config.jumps > 0,
                          seed=config.seed, visited={start})
    raise RuntimeError("could not generate a valid world")
