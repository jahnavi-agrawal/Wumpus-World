"""Winnability oracle: an omniscient explorer that sees the whole dungeon, against a Wumpus that stays put.

A* search over the real game mechanics (pits, portals, collapsing tiles, shuffle tiles, the jump and
the arrow), one explorer action per turn, within the turn limit. It returns the fewest actions needed
to escape with the gold, or None if no action sequence wins.

This is an upper bound on hazard handling only: a map the oracle cannot win is unwinnable for any
explorer; a map it wins may still be lost against a moving Wumpus.

    python -m evaluation.oracle 1000 1010      # shortest win for seeds 1000-1009
"""
from __future__ import annotations

import heapq
import itertools
import sys
from collections import deque

from wumpus.config import GameConfig
from wumpus.core.actions import CLIMB, FORWARD, GRAB, JUMP, SHOOT, TURN_LEFT, TURN_RIGHT
from wumpus.core.geometry import all_cells, neighbors, ray
from wumpus.environment import mechanics as M
from wumpus.environment.engine import GameEngine
from wumpus.environment.events import EventKind as K


def _useful(w):
    """Actions that can change the state. Pruned ones are no-ops or only waste a resource."""
    acts = [FORWARD, TURN_LEFT, TURN_RIGHT]
    if w.jump_available:
        acts.append(JUMP)
    if w.gold is not None and w.gold == w.explorer:
        acts.append(GRAB)
    if w.arrows and w.wumpus_alive and w.wumpus in ray(w.explorer, w.facing, w.size):
        acts.append(SHOOT)
    if w.explorer == w.start and w.has_gold:
        acts.append(CLIMB)
    return acts


def _key(w):
    return (w.explorer, w.facing, w.has_gold, w.jump_available, w.wumpus_alive,
            frozenset(w.unstable), frozenset(w.shuffle_tiles), frozenset(w.pits))


def _relaxed_dist(w, target) -> dict:
    """Fewest FORWARD moves from each cell to `target`, ignoring pits and facing, with portals
    (stepping onto a portal cell lands on its pair). A lower bound on the real number of actions."""
    pred = {}                                  # cell -> cells that reach it in one move
    for c in all_cells(w.size):                # incl. portal cells: a teleport lands ON the paired portal
        for n in neighbors(c, w.size):
            pred.setdefault(w.portals.get(n, n), []).append(c)
    dist, queue = {target: 0}, deque([target])
    while queue:
        c = queue.popleft()
        for p in pred.get(c, ()):
            if p not in dist:
                dist[p] = dist[c] + 1
                queue.append(p)
    return dist


def shortest_win(config: GameConfig) -> int | None:
    world = GameEngine(config).world
    far = config.max_turns + 1
    to_start = _relaxed_dist(world, world.start)
    to_gold = _relaxed_dist(world, world.gold)
    gold_leg = to_start.get(world.gold, far)

    def h(w):                                  # admissible: GRAB/CLIMB cost 1, a jump saves at most 2 moves
        if w.has_gold:
            d = to_start.get(w.explorer, far) + 1
        else:
            d = to_gold.get(w.explorer, far) + 1 + gold_leg + 1
        return max(0, d - (config.jump_distance - 1) * w.jump_available)

    tie = itertools.count()
    best = {_key(world): 0}
    heap = [(h(world), next(tie), 0, world)]
    while heap:
        _, _, steps, w = heapq.heappop(heap)
        if w is None:                          # the escape, popped in cost order -> optimal
            return steps
        if steps > best.get(_key(w), far) or steps >= config.max_turns:
            continue
        for a in _useful(w):
            w2 = w.copy()
            evs, _ = M.apply_explorer_action(w2, a, steps + 1, steps + 1, config)
            if not w2.explorer_alive:
                continue
            if any(e.kind is K.CLIMBED_OUT for e in evs):
                heapq.heappush(heap, (steps + 1, next(tie), steps + 1, None))
                continue
            k = _key(w2)
            if steps + 1 < best.get(k, far):
                best[k] = steps + 1
                heapq.heappush(heap, (steps + 1 + h(w2), next(tie), steps + 1, w2))
    return None


if __name__ == "__main__":
    first, stop = int(sys.argv[1]), int(sys.argv[2])
    for s in range(first, stop):
        print(s, shortest_win(GameConfig(seed=s)))
