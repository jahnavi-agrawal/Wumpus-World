"""A* search for the explorer.

State: (cell, jumps_left). Edges:
  MOVE d : cost 1 + risk_cost(landing)
  JUMP d : cost jump_cost + risk_cost(landing)      (middle cell skipped, uses one jump)
Landing on a teleport pad continues at the paired pad (RULES.md), so pads are non-local edges.
`cell_cost(c)` returns None for impassable cells, else a non-negative extra cost.

Heuristic: min over goals of min(|p-g|, |p-A|+|B-g|, |p-B|+|A-g|) — Manhattan distance with
the pad shortcut. Admissible because every edge costs ≥ 1 per cell of Manhattan progress
(jump_cost ≥ 2 for 2 cells) and the pad shortcut is the only way to beat Manhattan distance.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable

from ..core.actions import ExplorerAction
from ..core.geometry import DIRECTIONS, Pos, in_bounds, manhattan


@dataclass
class PathResult:
    found: bool
    start: Pos
    goal: Pos | None = None
    actions: list[ExplorerAction] = field(default_factory=list)
    cells: list[Pos] = field(default_factory=list)      # cell after each action
    cost: float = 0.0
    expanded: int = 0
    heuristic_start: float = 0.0

    def explain(self) -> list[str]:
        if not self.found:
            return [f"A* from {self.start}: no path ({self.expanded} nodes expanded)"]
        L = [f"A* {self.start} -> {self.goal}: cost {self.cost:.2f}, {len(self.actions)} actions, "
             f"{self.expanded} nodes expanded",
             f"h(start) = {self.heuristic_start:.2f} (admissible lower bound, pad-aware Manhattan)",
             "Step costs: move = 1 + risk_weight·risk, jump = jump_cost + risk_weight·risk"]
        L += [f"  {i + 1}. {a}  -> {c}" for i, (a, c) in enumerate(zip(self.actions, self.cells))]
        return L


def pad_aware_distance(p: Pos, g: Pos, pads: tuple[Pos, Pos] | None) -> int:
    d = manhattan(p, g)
    if pads:
        a, b = pads
        d = min(d, manhattan(p, a) + manhattan(b, g), manhattan(p, b) + manhattan(a, g))
    return d


def astar(start: Pos, goals: set[Pos], size: int, cell_cost: Callable[[Pos], float | None],
          jumps: int = 0, pads: tuple[Pos, Pos] | None = None, jump_cost: float = 2.5,
          use_heuristic: bool = True) -> PathResult:
    goals = set(goals)
    if not goals:
        return PathResult(False, start)

    def h(p: Pos) -> float:
        return min(pad_aware_distance(p, g, pads) for g in goals) if use_heuristic else 0.0

    def land(c: Pos) -> Pos:
        if pads and c in pads:
            return pads[1] if c == pads[0] else pads[0]
        return c

    start_state = (start, jumps)
    g_cost = {start_state: 0.0}
    parent: dict[tuple, tuple] = {}
    counter = 0
    heap = [(h(start), 0, counter, start_state)]
    closed = set()
    expanded = 0
    while heap:
        f, _, _, state = heapq.heappop(heap)
        if state in closed:
            continue
        closed.add(state)
        pos, jl = state
        if pos in goals:
            acts, cells, s = [], [], state
            while s in parent:
                s_prev, a = parent[s]
                acts.append(a)
                cells.append(s[0])
                s = s_prev
            return PathResult(True, start, pos, acts[::-1], cells[::-1], g_cost[state], expanded, h(start))
        expanded += 1
        for d in DIRECTIONS:
            edges = [(pos.step(d), ExplorerAction.move(d), 1.0, jl)]
            if jl > 0:
                edges.append((pos.step(d, 2), ExplorerAction.jump(d), jump_cost, jl - 1))
            for raw, action, base, njl in edges:
                if not in_bounds(raw, size):
                    continue
                if pads and raw in pads and cell_cost(raw) is None:
                    continue
                c = land(raw)
                extra = cell_cost(c)
                if extra is None:
                    continue
                ns = (c, njl)
                ng = g_cost[state] + base + extra
                if ng < g_cost.get(ns, float("inf")) - 1e-12:
                    g_cost[ns] = ng
                    parent[ns] = (state, action)
                    counter += 1
                    heapq.heappush(heap, (ng + h(c), h(c), counter, ns))
    return PathResult(False, start, expanded=expanded, heuristic_start=h(start))
