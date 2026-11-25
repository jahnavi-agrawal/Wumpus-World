"""A* search for the explorer (rules: facing, turns, 3-cell jump, portal pairs).

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
    mode: str = ""
    labels: list[str] = field(default_factory=list)             # per cell: proven pit-free / UNPROVEN
    rejected_cells: dict = field(default_factory=dict)          # reason -> cells the search refused

    def to_dict(self) -> dict:
        return {"found": self.found, "start": tuple(self.start), "goal": tuple(self.goal) if self.goal else None,
                "actions": [str(a) for a in self.actions], "cells": [tuple(c) for c in self.cells],
                "facings": [f.label for f in self.facings], "labels": list(self.labels), "cost": self.cost,
                "expanded": self.expanded, "mode": self.mode,
                "rejected_cells": {k: [tuple(c) for c in v] for k, v in self.rejected_cells.items()}}

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


# ============================================================================ planning layer
from enum import Enum
from types import MappingProxyType


class PlanningMode(Enum):
    STRICT = "strict-safe"          # every intermediate cell proven pit-free; only the goal may be unproven
    TOLERANT = "risk-tolerant"      # unproven cells allowed when their death risk <= tolerance


@dataclass(frozen=True)
class RiskModel:
    """Everything the planner needs, built from the explorer's knowledge (never the map)."""
    size: int
    proven_safe: frozenset          # proven pit-free cells (visited, rules, entailment)
    risk: MappingProxyType          # cell -> P(death) estimate
    blocked: MappingProxyType       # cell -> reason (proven pit, collapsed, about to collapse)
    avoid: MappingProxyType         # cell -> reason, passable only as the goal (unknown-destination portal)
    portals: MappingProxyType       # known portal pairs
    known_limit: float = 0.5        # max risk accepted even on proven pit-free cells (Wumpus)


def _label(c: Pos, model: RiskModel) -> str:
    r = model.risk.get(c, 0.0)
    if c in model.portals:
        return "known portal"
    if c in model.proven_safe:
        return f"proven pit-free, risk {r:.0%}"
    return f"UNPROVEN, risk {r:.0%}"            # never label an unknown cell as safe


def plan(start: Pos, facing: Direction, goals: set[Pos], model: RiskModel, mode: PlanningMode,
         tolerance: float, jump: bool, risk_weight: float = 25.0, jump_cost: float = 3.0,
         jump_distance: int = 3, use_portals: bool = True) -> PathResult:
    rejected: dict[str, set[Pos]] = {}

    def reject(c: Pos, why: str):
        rejected.setdefault(why, set()).add(c)
        return None

    def cost(c: Pos):
        if c in model.blocked:
            return reject(c, model.blocked[c])
        if c in model.portals:
            return 0.0 if use_portals else reject(c, "portals disabled for this variant")
        if c in model.avoid and c not in goals:
            return reject(c, model.avoid[c])
        r = model.risk.get(c, 0.0)
        if c in goals or c in model.proven_safe:
            limit = max(tolerance, model.known_limit)
            return risk_weight * r if r <= limit else reject(c, f"risk above {limit:.0%}")
        if mode is PlanningMode.TOLERANT:
            return risk_weight * r if r <= tolerance else reject(c, f"unproven and risk above tolerance {tolerance:.0%}")
        return reject(c, "unproven cell (strict mode)")

    res = astar(start, facing, goals, model.size, cost, jump, dict(model.portals) if use_portals else {},
                jump_cost, jump_distance)
    res.mode = mode.value
    res.labels = [_label(c, model) for c in res.cells]
    res.rejected_cells = {k: sorted(v) for k, v in rejected.items()}
    return res


@dataclass
class RoutePlan:
    best: PathResult
    mode: PlanningMode
    alternatives: list[tuple[str, PathResult]] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)          # (variant, reason)

    def to_dict(self) -> dict:
        return {"mode": self.mode.value, "best": self.best.to_dict(),
                "alternatives": [{"variant": n, **r.to_dict()} for n, r in self.alternatives],
                "rejected": [{"variant": n, "reason": why} for n, why in self.rejected]}

    def explain(self) -> list[str]:
        L = [f"Planning mode: {self.mode.value}"] + self.best.explain()
        if self.best.found:
            L += [f"     step {i + 1} cell {c}: {lab}" for i, (c, lab) in enumerate(zip(self.best.cells, self.best.labels))]
        if self.alternatives:
            L.append("Alternative routes:")
            L += [f"  {n}: cost {r.cost:.2f}, {len(r.actions)} actions: " + " ".join(map(str, r.actions))
                  for n, r in self.alternatives]
        if self.rejected:
            L.append("Rejected routes:")
            L += [f"  {n}: {why}" for n, why in self.rejected]
        if self.best.rejected_cells:
            L.append("Cells the search could not use:")
            L += [f"  {why}: {', '.join(map(str, cells[:8]))}{' ...' if len(cells) > 8 else ''}"
                  for why, cells in self.best.rejected_cells.items()]
        return L


def plan_routes(start: Pos, facing: Direction, goals: set[Pos], model: RiskModel, mode: PlanningMode,
                tolerance: float, jump: bool, **kw) -> RoutePlan:
    """Best route in `mode`, plus alternative variants and why the others were rejected."""
    best = plan(start, facing, goals, model, mode, tolerance, jump, **kw)
    out = RoutePlan(best, mode)
    variants = [("strict-safe", dict(mode=PlanningMode.STRICT)),
                ("risk-tolerant", dict(mode=PlanningMode.TOLERANT)),
                ("without jump", dict(mode=mode, jump=False)),
                ("without portals", dict(mode=mode, use_portals=False))]
    seen = {tuple(best.actions)} if best.found else set()
    for name, v in variants:
        if name == mode.value:
            continue
        if name == "without jump" and not jump:
            continue
        if name == "without portals" and not model.portals:
            continue
        args = dict(mode=mode, jump=jump, **kw)
        args.update(v)
        r = plan(start, facing, goals, model, args.pop("mode"), tolerance, args.pop("jump"), **args)
        if not r.found:
            reasons = "; ".join(f"{len(c)} {why}" for why, c in r.rejected_cells.items()) or "no connection"
            out.rejected.append((name, f"no path ({reasons})"))
        elif tuple(r.actions) in seen:
            continue
        elif best.found and r.cost > best.cost + 1e-9:
            seen.add(tuple(r.actions))
            out.alternatives.append((name, r))
            out.rejected.append((name, f"costlier: {r.cost:.2f} vs {best.cost:.2f}"))
        else:
            seen.add(tuple(r.actions))
            out.alternatives.append((name, r))
    return out
