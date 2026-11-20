"""Who decides the Wumpus' action. The Wumpus is omniscient, so controllers may read the world."""
from __future__ import annotations

import random
from collections import deque

from ..core.actions import STAY, WumpusAction, WumpusActionType
from ..core.geometry import Pos, neighbors


class HumanWumpusController:
    """Filled by the UI. `choose` returns None while the human has not decided yet."""

    def __init__(self) -> None:
        self._pending: WumpusAction | None = None

    def submit(self, action: WumpusAction) -> None:
        self._pending = action

    def choose(self, world, legal: list[WumpusAction]) -> WumpusAction | None:
        a, self._pending = self._pending, None
        return a if a in legal else None


class ScriptedWumpusController:
    """Deterministic (seeded) opponent used for tests and the headless demo.

    modes: 'idle' (always stays), 'random' (random legal move/stay), 'hunter'
    (walks the shortest pit-free path towards the explorer, random tie-break).
    """

    def __init__(self, mode: str = "hunter", seed: int = 0):
        self.mode = mode
        self.rng = random.Random(seed)

    def choose(self, world, legal: list[WumpusAction]) -> WumpusAction:
        moves = [a for a in legal if a.kind in (WumpusActionType.MOVE, WumpusActionType.STAY)]
        if self.mode == "idle" or not moves:
            return STAY
        if self.mode == "random":
            return self.rng.choice(moves)
        dist = _bfs_dist(world, world.explorer)
        def score(a: WumpusAction) -> int:
            p = world.wumpus if a.kind is WumpusActionType.STAY else world.wumpus.step(a.direction)
            return dist.get(p, 999)
        best = min(score(a) for a in moves)
        return self.rng.choice([a for a in moves if score(a) == best])


def _bfs_dist(world, src: Pos) -> dict[Pos, int]:
    dist, q = {src: 0}, deque([src])
    while q:
        p = q.popleft()
        for n in neighbors(p, world.size):
            if n not in dist and n not in world.pits:
                dist[n] = dist[p] + 1
                q.append(n)
    return dist
