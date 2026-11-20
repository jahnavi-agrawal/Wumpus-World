"""A* search for the explorer (v2 rules: facing, turns, 3-cell jump, portal pairs).

State: (cell, facing, jump_available). Edges (every action costs >= 1, like the score):
  TURN_LEFT / TURN_RIGHT : cost 1
  FORWARD                : cost 1 + cell_cost(landing)
  JUMP                   : cost jump_cost + cell_cost(landing), lands jump_distance ahead, once
Landing on a known portal continues at its pair. `cell_cost(c)` returns None for impassable.

Heuristic: portal-aware Manhattan distance  h(x) = min( |x-g|, min_p |x-p| + D[pair(p)] ),
where D[q] is the relaxed distance from portal exit q to the goal (Bellman-Ford over the
portal cells). Admissible: every action covers at most 1 cell per unit cost (a jump covers
jump_distance cells for jump_cost >= jump_distance), and portals are the only shortcut.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable

from ..core.actions import FORWARD, JUMP, TURN_LEFT, TURN_RIGHT, ExplorerAction
from ..core.geometry import Direction, Pos, in_bounds, manhattan


@dataclass
class PathResult:
    found: bool
    start: Pos
    goal: Pos | None = None
    actions: list[ExplorerAction] = field(default_factory=list)
    cells: list[Pos] = field(default_factory=list)          # cell after each action
    facings: list[Direction] = field(default_factory=list)  # facing after each action
    cost: float = 0.0
    expanded: int = 0
    heuristic_start: float = 0.0

    def explain(self) -> list[str]:
        if not self.found:
            return [f"A* from {self.start}: no path ({self.expanded} nodes expanded)"]
        L = [f"A* {self.start} -> {self.goal}: cost {self.cost:.2f}, {len(self.actions)} actions, "
             f"{self.expanded} nodes expanded",
             f"h(start) = {self.heuristic_start:.2f} (admissible: portal-aware Manhattan)",
             "Costs: turn = 1, forward = 1 + risk_weight*risk, jump = jump_cost + risk_weight*risk"]
        L += [f"  {i + 1}. {a:<10} -> {c} facing {f.label}"
              for i, (a, c, f) in enumerate(zip(map(str, self.actions), self.cells, self.facings))]
        return L


def portal_exit_distances(goals: set[Pos], portals: dict[Pos, Pos]) -> dict[Pos, float]:
    """Relaxed distance from every portal cell to the nearest goal (Bellman-Ford, <= 2k nodes)."""
    D = {q: min(manhattan(q, g) for g in goals) for q in portals}
    for _ in range(len(portals) + 1):
        changed = False
        for q in portals:
            best = min([D[q]] + [manhattan(q, p) + D[portals[p]] for p in portals])
            if best < D[q]:
                D[q], changed = best, True
        if not changed:
            break
    return D


def portal_aware_distance(x: Pos, goals: set[Pos], portals: dict[Pos, Pos], D: dict[Pos, float] | None = None) -> float:
    D = D if D is not None else portal_exit_distances(goals, portals)
    best = min(manhattan(x, g) for g in goals)
    for p, q in portals.items():
        best = min(best, manhattan(x, p) + D[q])
    return best


def astar(start: Pos, facing: Direction, goals: set[Pos], size: int, cell_cost: Callable[[Pos], float | None],
          jump: bool = False, portals: dict[Pos, Pos] | None = None, jump_cost: float = 3.0,
          jump_distance: int = 3, use_heuristic: bool = True) -> PathResult:
    goals, portals = set(goals), dict(portals or {})
    if not goals:
        return PathResult(False, start)
    D = portal_exit_distances(goals, portals)

    def h(p: Pos) -> float:
        return portal_aware_distance(p, goals, portals, D) if use_heuristic else 0.0

    start_state = (start, facing, jump)
    g_cost = {start_state: 0.0}
    parent: dict[tuple, tuple] = {}
    heap = [(h(start), 0.0, 0, start_state)]
    counter, expanded, closed = 0, 0, set()
    while heap:
        _, _, _, state = heapq.heappop(heap)
        if state in closed:
            continue
        closed.add(state)
        pos, f, j = state
        if pos in goals:
            acts, cells, facings, s = [], [], [], state
            while s in parent:
                prev, a = parent[s]
                acts.append(a), cells.append(s[0]), facings.append(s[1])
                s = prev
            return PathResult(True, start, pos, acts[::-1], cells[::-1], facings[::-1], g_cost[state],
                              expanded, h(start))
        expanded += 1
        edges = [((pos, f.left(), j), TURN_LEFT, 1.0), ((pos, f.right(), j), TURN_RIGHT, 1.0)]
        for action, dist, base, nj in ((FORWARD, 1, 1.0, j), (JUMP, jump_distance, jump_cost, False)):
            if action is JUMP and not j:
                continue
            raw = pos.step(f, dist)
            if not in_bounds(raw, size):
                continue
            if raw in portals and cell_cost(raw) is None:
                continue
            land = portals.get(raw, raw)
            extra = cell_cost(land)
            if extra is None:
                continue
            edges.append(((land, f, nj), action, base + extra))
        for ns, action, step_cost in edges:
            ng = g_cost[state] + step_cost
            if ng < g_cost.get(ns, float("inf")) - 1e-12:
                g_cost[ns] = ng
                parent[ns] = (state, action)
                counter += 1
                heapq.heappush(heap, (ng + h(ns[0]), h(ns[0]), counter, ns))
    return PathResult(False, start, expanded=expanded, heuristic_start=h(start))
