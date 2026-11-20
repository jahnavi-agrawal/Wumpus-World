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


def make_controller(mode: str, seed: int = 0):
    """'idle' | 'random' | 'hunter' | 'minimax' (AI-vs-AI benchmarking)."""
    if mode == "minimax":
        return MinimaxWumpusController(seed=seed)
    return ScriptedWumpusController(mode, seed)


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


class MinimaxWumpusController:
    """AI-controlled Wumpus for testing and benchmarking (never used in human-vs-AI statistics).

    It uses the same legal actions and the same DuelGame model and evaluation as the explorer's
    search, but from the MIN side and with perfect information (the Wumpus is omniscient):
    true pits, true explorer position and facing. It picks the move that minimises the explorer's
    alpha-beta value.
    """

    def __init__(self, depth: int = 4, seed: int = 0):
        self.depth = depth
        self.rng = random.Random(seed)
        self.last_values: dict = {}

    def choose(self, world, legal: list[WumpusAction]) -> WumpusAction:
        from ..agents.minimax import DuelGame, DuelState, WumpusStep, alphabeta
        if not legal:
            return STAY
        goal = world.start if world.has_gold or world.gold is None else world.gold
        dist = _bfs_dist_explorer(world, goal)
        exit_dist = _bfs_dist_explorer(world, world.start)
        game = DuelGame(size=world.size, portals=dict(world.portals), blocked=frozenset(world.pits),
                        pit_risk={}, goal_dist=dist, exit_dist=exit_dist)
        root = DuelState(world.explorer, world.facing, world.wumpus, world.wumpus_alive, world.arrows,
                         world.jump_available, explorer_to_move=False)
        values = {}
        for a in legal:
            step = WumpusStep(a.direction if a.kind is WumpusActionType.MOVE else None)
            values[a] = alphabeta(game, game.result(root, step), self.depth - 1)
        self.last_values = values
        best = min(values.values())
        return self.rng.choice([a for a in legal if values[a] == best])


def _bfs_dist_explorer(world, goal: Pos) -> dict[Pos, float]:
    """Explorer walking distance to `goal` over non-pit cells (the Wumpus knows the true map)."""
    dist, q = {goal: 0.0}, deque([goal])
    while q:
        p = q.popleft()
        for n in neighbors(p, world.size):
            if n not in dist and n not in world.pits:
                dist[n] = dist[p] + 1
                q.append(n)
    return dist
