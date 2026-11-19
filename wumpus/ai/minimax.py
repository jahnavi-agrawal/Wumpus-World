"""Minimax with alpha-beta pruning, plus the explorer-vs-Wumpus "duel" model it searches.

The search core (`alphabeta`, `minimax`) is game-agnostic: any object with
`actions(s)`, `result(s, a)`, `is_terminal(s)`, `evaluate(s)`, `is_max(s)` works.

Partial observability is handled by *determinization* (DECISIONS.md D5): the explorer does
not know where the Wumpus is, so we run a separate perfect-information search for each of the
top-k Wumpus positions from the Bayesian filter and score each root action by the
belief-weighted average of its exact minimax values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from ..core.actions import WAIT, ExplorerAction, ExplorerActionType
from ..core.geometry import DIRECTIONS, Pos, in_bounds, manhattan, ray

INF = math.inf


@dataclass
class SearchStats:
    nodes: int = 0
    cutoffs: int = 0
    leaves: int = 0


def minimax(game, s, depth: int, stats: SearchStats | None = None) -> float:
    """Plain minimax (reference implementation, used in tests)."""
    stats = stats or SearchStats()
    stats.nodes += 1
    if depth == 0 or game.is_terminal(s):
        stats.leaves += 1
        return game.evaluate(s)
    vals = [minimax(game, game.result(s, a), depth - 1, stats) for a in game.actions(s)]
    return max(vals) if game.is_max(s) else min(vals)


def alphabeta(game, s, depth: int, alpha: float = -INF, beta: float = INF,
              stats: SearchStats | None = None, pv: list | None = None) -> float:
    stats = stats if stats is not None else SearchStats()
    stats.nodes += 1
    if depth == 0 or game.is_terminal(s):
        stats.leaves += 1
        return game.evaluate(s)
    best_line: list = []
    if game.is_max(s):
        value = -INF
        for a in game.actions(s):
            line: list = []
            v = alphabeta(game, game.result(s, a), depth - 1, alpha, beta, stats, line)
            if v > value:
                value, best_line = v, [a] + line
            alpha = max(alpha, value)
            if alpha >= beta:
                stats.cutoffs += 1
                break
    else:
        value = INF
        for a in game.actions(s):
            line = []
            v = alphabeta(game, game.result(s, a), depth - 1, alpha, beta, stats, line)
            if v < value:
                value, best_line = v, [a] + line
            beta = min(beta, value)
            if alpha >= beta:
                stats.cutoffs += 1
                break
    if pv is not None:
        pv[:] = best_line
    return value


# ------------------------------------------------------------------------- duel model
@dataclass(frozen=True)
class DuelState:
    explorer: Pos
    wumpus: Pos
    wumpus_alive: bool
    arrows: int
    jumps: int
    explorer_to_move: bool = True
    explorer_dead: bool = False
    risk: float = 0.0          # accumulated pit risk along the explorer's line


@dataclass(frozen=True)
class WumpusStep:
    """A Wumpus move in the search tree (`direction` None = stay)."""
    direction: object = None

    def __str__(self) -> str:
        return f"W:{self.direction.name}" if self.direction else "W:STAY"


@dataclass
class DuelGame:
    size: int
    pads: tuple[Pos, Pos] | None
    blocked: frozenset[Pos]                # known pits + collapsed tiles
    pit_risk: dict[Pos, float]             # P(pit) for unknown cells (0 for known safe)
    goal_dist: dict[Pos, float]            # A*/BFS distance-to-goal over passable cells
    max_step_risk: float = 0.30
    w_death: float = 1000.0
    w_goal: float = 4.0
    w_dist: float = 6.0
    w_arrow: float = 15.0
    w_kill: float = 300.0

    def _land(self, c: Pos) -> Pos:
        if self.pads and c in self.pads:
            return self.pads[1] if c == self.pads[0] else self.pads[0]
        return c

    def is_max(self, s: DuelState) -> bool:
        return s.explorer_to_move

    def is_terminal(self, s: DuelState) -> bool:
        return s.explorer_dead or not s.wumpus_alive

    def actions(self, s: DuelState) -> list:
        if s.explorer_to_move:
            acts: list = []
            if s.arrows > 0 and s.wumpus_alive:
                acts += [ExplorerAction.shoot(d) for d in DIRECTIONS if ray(s.explorer, d, self.size)]
            for d in DIRECTIONS:
                for a, c in ((ExplorerAction.move(d), s.explorer.step(d)),
                             (ExplorerAction.jump(d), s.explorer.step(d, 2))):
                    if a.kind is ExplorerActionType.JUMP and s.jumps == 0:
                        continue
                    if in_bounds(c, self.size):
                        c = self._land(c)
                        if c not in self.blocked and self.pit_risk.get(c, 0.0) <= self.max_step_risk:
                            acts.append(a)
            acts.append(WAIT)
            # move ordering: shots first, then by goal distance (better pruning)
            return sorted(acts, key=lambda a: (a.kind is not ExplorerActionType.SHOOT, self._order(s, a)))
        acts = [WumpusStep(d) for d in DIRECTIONS
                if in_bounds(s.wumpus.step(d), self.size) and s.wumpus.step(d) not in self.blocked]
        acts.append(WumpusStep(None))
        return sorted(acts, key=lambda a: manhattan(self._land(s.wumpus.step(a.direction)) if a.direction else s.wumpus, s.explorer))

    def _order(self, s: DuelState, a: ExplorerAction) -> float:
        if a.direction is None:
            return self.goal_dist.get(s.explorer, 99)
        n = 2 if a.kind is ExplorerActionType.JUMP else 1
        return self.goal_dist.get(self._land(s.explorer.step(a.direction, n)), 99)

    def result(self, s: DuelState, a) -> DuelState:
        if s.explorer_to_move:
            if a.kind is ExplorerActionType.SHOOT:
                hit = s.wumpus in ray(s.explorer, a.direction, self.size)
                return replace(s, arrows=s.arrows - 1, wumpus_alive=not hit, explorer_to_move=False)
            if a.kind in (ExplorerActionType.MOVE, ExplorerActionType.JUMP):
                n = 2 if a.kind is ExplorerActionType.JUMP else 1
                c = self._land(s.explorer.step(a.direction, n))
                return replace(s, explorer=c, jumps=s.jumps - (n == 2), explorer_to_move=False,
                               explorer_dead=(c == s.wumpus), risk=s.risk + self.pit_risk.get(c, 0.0))
            return replace(s, explorer_to_move=False)
        w = self._land(s.wumpus.step(a.direction)) if a.direction else s.wumpus
        return replace(s, wumpus=w, explorer_to_move=True, explorer_dead=(w == s.explorer))

    def evaluate(self, s: DuelState) -> float:
        if s.explorer_dead:
            return -self.w_death
        v = -self.w_death * s.risk - self.w_goal * self.goal_dist.get(s.explorer, 30) + self.w_arrow * s.arrows
        if not s.wumpus_alive:
            return v + self.w_kill
        d = manhattan(s.explorer, s.wumpus)
        if d == 1 and not s.explorer_to_move:
            return -self.w_death            # the Wumpus moves next and will eat us
        if d == 1:
            v -= 0.2 * self.w_death         # adjacent with our move: dangerous but escapable
        return v + self.w_dist * min(d, 4)


@dataclass
class MinimaxResult:
    best_action: ExplorerAction
    expected_value: float
    root_values: dict[ExplorerAction, float]
    hypotheses: list[tuple[Pos, float, dict[ExplorerAction, float], list]] = field(default_factory=list)
    stats: SearchStats = field(default_factory=SearchStats)
    depth: int = 0

    def explain(self) -> list[str]:
        L = [f"Determinized Minimax, alpha-beta, depth {self.depth} plies, "
             f"{len(self.hypotheses)} Wumpus hypotheses",
             f"Nodes searched {self.stats.nodes}, alpha-beta cutoffs {self.stats.cutoffs}, leaves {self.stats.leaves}",
             "Expected value per root action  E[V(a)] = Σ_h P(h)·V_h(a):"]
        for a, v in sorted(self.root_values.items(), key=lambda kv: -kv[1]):
            mark = "  <== chosen" if a == self.best_action else ""
            L.append(f"  {str(a):<9} {v:9.1f}{mark}")
        for pos, p, vals, pv in self.hypotheses:
            L.append(f"Hypothesis W at {pos}  (belief {p:.1%}, renormalised)")
            L.append("  V: " + ", ".join(f"{a}={v:.0f}" for a, v in sorted(vals.items(), key=lambda kv: -kv[1])[:5]))
            L.append("  principal variation: " + " -> ".join(map(str, pv)))
        L.append("Assumptions: Wumpus plays optimally against us; its position is one of the "
                 "hypotheses; powers (collapse/shuffle) are ignored inside the search.")
        return L


def determinized_decision(game: DuelGame, root: DuelState, hypotheses: list[tuple[Pos, float]],
                          depth: int) -> MinimaxResult:
    """Exact minimax value of every root action under each hypothesis, combined by belief."""
    total_p = sum(p for _, p in hypotheses)
    stats = SearchStats()
    combined: dict[ExplorerAction, float] = {}
    hyp_out = []
    root_actions = None
    for pos, p in hypotheses:
        s0 = replace(root, wumpus=pos)
        root_actions = game.actions(s0)
        vals, best_pv, best_v = {}, [], -INF
        for a in root_actions:
            line: list = []
            v = alphabeta(game, game.result(s0, a), depth - 1, -INF, INF, stats, line)
            vals[a] = v
            if v > best_v:
                best_v, best_pv = v, [a] + line
            combined[a] = combined.get(a, 0.0) + (p / total_p) * v
        hyp_out.append((pos, p / total_p, vals, best_pv))
    best = max(root_actions, key=lambda a: (combined[a], -root_actions.index(a)))
    return MinimaxResult(best, combined[best], combined, hyp_out, stats, depth)
