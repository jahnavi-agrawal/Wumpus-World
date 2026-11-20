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

from ..core.actions import FORWARD, JUMP, SHOOT, TURN_LEFT, TURN_RIGHT, ExplorerAction
from ..core.geometry import DIRECTIONS, Direction, Pos, in_bounds, manhattan, ray

INF = math.inf


MAX_PRUNED_RECORDED = 40


@dataclass
class SearchStats:
    nodes: int = 0
    cutoffs: int = 0
    leaves: int = 0
    pruned: list = field(default_factory=list)      # recorded cutoffs (first MAX_PRUNED_RECORDED)
    pruned_moves: int = 0                           # sibling moves never searched because of cutoffs


def minimax(game, s, depth: int, stats: SearchStats | None = None) -> float:
    """Plain minimax (reference implementation, used in tests)."""
    stats = stats or SearchStats()
    stats.nodes += 1
    if depth == 0 or game.is_terminal(s):
        stats.leaves += 1
        return game.evaluate(s)
    vals = [minimax(game, game.result(s, a), depth - 1, stats) for a in game.actions(s)]
    return max(vals) if game.is_max(s) else min(vals)


def _cut(stats: SearchStats, path: tuple, actions: list, i: int, kind: str, alpha: float, beta: float) -> None:
    stats.cutoffs += 1
    skipped = actions[i + 1:]
    stats.pruned_moves += len(skipped)
    if len(stats.pruned) < MAX_PRUNED_RECORDED and skipped:
        stats.pruned.append({"after": [str(a) for a in path], "cut": kind, "alpha": alpha, "beta": beta,
                             "skipped": [str(a) for a in skipped]})


def alphabeta(game, s, depth: int, alpha: float = -INF, beta: float = INF,
              stats: SearchStats | None = None, pv: list | None = None, path: tuple = ()) -> float:
    stats = stats if stats is not None else SearchStats()
    stats.nodes += 1
    if depth == 0 or game.is_terminal(s):
        stats.leaves += 1
        return game.evaluate(s)
    best_line: list = []
    if game.is_max(s):
        value = -INF
        acts = game.actions(s)
        for i, a in enumerate(acts):
            line: list = []
            v = alphabeta(game, game.result(s, a), depth - 1, alpha, beta, stats, line, path + (a,))
            if v > value:
                value, best_line = v, [a] + line
            alpha = max(alpha, value)
            if alpha >= beta:
                _cut(stats, path + (a,), acts, i, "beta cutoff (MIN will avoid this branch)", alpha, beta)
                break
    else:
        value = INF
        acts = game.actions(s)
        for i, a in enumerate(acts):
            line = []
            v = alphabeta(game, game.result(s, a), depth - 1, alpha, beta, stats, line, path + (a,))
            if v < value:
                value, best_line = v, [a] + line
            beta = min(beta, value)
            if alpha >= beta:
                _cut(stats, path + (a,), acts, i, "alpha cutoff (MAX already has better)", alpha, beta)
                break
    if pv is not None:
        pv[:] = best_line
    return value


# ------------------------------------------------------------------------- duel model
@dataclass(frozen=True)
class DuelState:
    explorer: Pos
    facing: Direction
    wumpus: Pos
    wumpus_alive: bool
    arrows: int
    jump: bool
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
    portals: dict[Pos, Pos]                # known portal pairs (symmetric)
    blocked: frozenset[Pos]                # known pits + collapsed tiles (+ cell about to collapse)
    pit_risk: dict[Pos, float]             # P(pit) for unknown cells (0 for known safe)
    goal_dist: dict[Pos, float]            # BFS distance-to-goal over usable cells
    exit_dist: dict[Pos, float] = field(default_factory=dict)   # BFS distance to the entrance
    max_step_risk: float = 0.30
    jump_distance: int = 3
    w_death: float = 1000.0
    w_goal: float = 4.0
    w_dist: float = 6.0
    w_arrow: float = 60.0           # shooting pays only when P(hit) is roughly > w_arrow / w_kill
    w_kill: float = 300.0
    w_exit: float = 1.0             # keep a way back to the entrance
    w_no_exit: float = 40.0         # penalty when the entrance is unreachable from here
    w_jump: float = 20.0            # value of an unused jump (escape tool)
    w_escape: float = 8.0           # per escape route (up to 3) not adjacent to the Wumpus

    def _land(self, c: Pos) -> Pos:
        return self.portals.get(c, c)

    def is_max(self, s: DuelState) -> bool:
        return s.explorer_to_move

    def is_terminal(self, s: DuelState) -> bool:
        return s.explorer_dead or not s.wumpus_alive

    def _landing(self, s: DuelState, a: ExplorerAction) -> Pos | None:
        n = self.jump_distance if a is JUMP else 1
        c = s.explorer.step(s.facing, n)
        if not in_bounds(c, self.size):
            return None
        c = self._land(c)
        if c in self.blocked or self.pit_risk.get(c, 0.0) > self.max_step_risk:
            return None
        return c

    def actions(self, s: DuelState) -> list:
        if s.explorer_to_move:
            acts: list = []
            if s.arrows > 0 and s.wumpus_alive and ray(s.explorer, s.facing, self.size):
                acts.append(SHOOT)
            if self._landing(s, FORWARD) is not None:
                acts.append(FORWARD)
            if s.jump and self._landing(s, JUMP) is not None:
                acts.append(JUMP)
            acts += [TURN_LEFT, TURN_RIGHT]
            return acts
        acts = [WumpusStep(d) for d in DIRECTIONS
                if in_bounds(s.wumpus.step(d), self.size) and s.wumpus.step(d) not in self.blocked]
        acts.append(WumpusStep(None))
        return sorted(acts, key=lambda a: manhattan(self._land(s.wumpus.step(a.direction)) if a.direction
                                                    else s.wumpus, s.explorer))

    def result(self, s: DuelState, a) -> DuelState:
        if s.explorer_to_move:
            if a is SHOOT:
                hit = s.wumpus in ray(s.explorer, s.facing, self.size)
                return replace(s, arrows=s.arrows - 1, wumpus_alive=not hit, explorer_to_move=False)
            if a is FORWARD or a is JUMP:
                c = self._landing(s, a)
                return replace(s, explorer=c, jump=s.jump and a is not JUMP, explorer_to_move=False,
                               explorer_dead=(c == s.wumpus), risk=s.risk + self.pit_risk.get(c, 0.0))
            f = s.facing.left() if a is TURN_LEFT else s.facing.right()
            return replace(s, facing=f, explorer_to_move=False)
        w = self._land(s.wumpus.step(a.direction)) if a.direction else s.wumpus
        return replace(s, wumpus=w, explorer_to_move=True, explorer_dead=(w == s.explorer))

    def facing_goal_cost(self, s: DuelState) -> int:
        """Turns needed before the next step can make progress: 0 ahead, 1 sideways, 2 behind."""
        here = self.goal_dist.get(s.explorer, 30)
        good = [d for d in DIRECTIONS if self.goal_dist.get(self._land(s.explorer.step(d)), 99) < here]
        if not good or s.facing in good:
            return 0
        return 2 if all(d == s.facing.left().left() for d in good) else 1

    def escape_routes(self, s: DuelState) -> int:
        """Neighbouring cells we could step to that are usable and not next to the Wumpus."""
        n = 0
        for d in DIRECTIONS:
            c = s.explorer.step(d)
            if in_bounds(c, self.size) and c not in self.blocked and self.pit_risk.get(c, 0.0) <= self.max_step_risk \
                    and (not s.wumpus_alive or manhattan(self._land(c), s.wumpus) >= 2):
                n += 1
        return n

    def evaluate(self, s: DuelState) -> float:
        if s.explorer_dead:
            return -self.w_death
        goal = self.goal_dist.get(s.explorer, 30) + self.facing_goal_cost(s)
        v = -self.w_death * s.risk - self.w_goal * goal + self.w_arrow * s.arrows + self.w_jump * s.jump
        if self.exit_dist:
            d_exit = self.exit_dist.get(s.explorer)
            v -= self.w_no_exit if d_exit is None else self.w_exit * d_exit
        if not s.wumpus_alive:
            return v + self.w_kill
        if s.explorer_to_move and s.arrows > 0 and s.wumpus in ray(s.explorer, s.facing, self.size):
            return v - self.w_arrow + self.w_kill      # shot is ready: next ply SHOOT kills it
        d = manhattan(s.explorer, s.wumpus)
        if d == 1 and not s.explorer_to_move:
            return -self.w_death            # the Wumpus moves next and will eat us
        v += self.w_escape * min(3, self.escape_routes(s))
        if d == 1:
            # adjacent, our move, and it is NOT on our arrow line: turning to face it gets us eaten,
            # so this is close to lost (only running forward helps). Beyond-horizon knowledge.
            v -= 0.6 * self.w_death
        return v + self.w_dist * min(d, 4)


@dataclass
class MinimaxResult:
    best_action: ExplorerAction
    expected_value: float
    root_values: dict[ExplorerAction, float]
    hypotheses: list[tuple[Pos, float, dict[ExplorerAction, float], list]] = field(default_factory=list)
    stats: SearchStats = field(default_factory=SearchStats)
    depth: int = 0
    replies: dict = field(default_factory=dict)     # (hypothesis pos, action) -> predicted Wumpus reply

    def predicted_reply(self, action=None):
        """Wumpus reply to `action` (default: the chosen one) under the most likely hypothesis."""
        a = action or self.best_action
        pos = max(self.hypotheses, key=lambda h: h[1])[0]
        return pos, self.replies.get((pos, a))

    def to_dict(self) -> dict:
        return {
            "depth": self.depth, "selected": str(self.best_action), "expected_value": self.expected_value,
            "candidates": [{"action": str(a), "expected_value": v,
                            "predicted_replies": {str(pos): str(self.replies.get((pos, a)))
                                                  for pos, _, _, _ in self.hypotheses}}
                           for a, v in sorted(self.root_values.items(), key=lambda kv: -kv[1])],
            "hypotheses": [{"wumpus": tuple(pos), "belief": p, "values": {str(a): v for a, v in vals.items()},
                            "principal_variation": [str(x) for x in pv]} for pos, p, vals, pv in self.hypotheses],
            "nodes": self.stats.nodes, "cutoffs": self.stats.cutoffs, "pruned_moves": self.stats.pruned_moves,
            "pruned_branches": self.stats.pruned,
        }

    def explain(self) -> list[str]:
        L = [f"Determinized Minimax, alpha-beta, depth {self.depth} plies, "
             f"{len(self.hypotheses)} Wumpus hypotheses",
             f"Nodes searched {self.stats.nodes}, alpha-beta cutoffs {self.stats.cutoffs}, leaves {self.stats.leaves}",
             "Expected value per root action  E[V(a)] = Σ_h P(h)·V_h(a):"]
        for a, v in sorted(self.root_values.items(), key=lambda kv: -kv[1]):
            mark = "  <== chosen" if a == self.best_action else ""
            L.append(f"  {str(a):<11} {v:9.1f}{mark}")
        L.append("Predicted Wumpus reply to each candidate (most likely hypothesis):")
        top = max(self.hypotheses, key=lambda h: h[1])[0] if self.hypotheses else None
        L += [f"  {str(a):<11} -> {self.replies.get((top, a))}" for a in self.root_values]
        for pos, p, vals, pv in self.hypotheses:
            L.append(f"Hypothesis W at {pos}  (belief {p:.1%}, renormalised)")
            L.append("  V: " + ", ".join(f"{a}={v:.0f}" for a, v in sorted(vals.items(), key=lambda kv: -kv[1])[:5]))
            L.append("  principal variation: " + " -> ".join(map(str, pv)))
        L.append(f"Pruned: {self.stats.cutoffs} cutoffs skipped {self.stats.pruned_moves} sibling moves, e.g.:")
        for b in self.stats.pruned[:5]:
            L.append(f"  after {' '.join(b['after'])}: {b['cut']} -> skipped {', '.join(b['skipped'])}")
        L.append("Assumptions: the Wumpus plays optimally against us; its position is one of the "
                 "hypotheses (determinization); pit risk is charged, not branched on.")
        return L


def determinized_decision(game: DuelGame, root: DuelState, hypotheses: list[tuple[Pos, float]],
                          depth: int) -> MinimaxResult:
    """Exact minimax value of every root action under each hypothesis, combined by belief."""
    total_p = sum(p for _, p in hypotheses)
    stats = SearchStats()
    combined: dict[ExplorerAction, float] = {}
    hyp_out = []
    root_actions = None
    replies: dict = {}
    for pos, p in hypotheses:
        s0 = replace(root, wumpus=pos)
        root_actions = game.actions(s0)
        vals, best_pv, best_v = {}, [], -INF
        for a in root_actions:
            line: list = []
            v = alphabeta(game, game.result(s0, a), depth - 1, -INF, INF, stats, line, (a,))
            vals[a] = v
            replies[(pos, a)] = line[0] if line else None
            if v > best_v:
                best_v, best_pv = v, [a] + line
            combined[a] = combined.get(a, 0.0) + (p / total_p) * v
        hyp_out.append((pos, p / total_p, vals, best_pv))
    # tie-break: prefer actions that do not die on the spot (a human may not play the worst case)
    instant = {a: sum((p / total_p) * game.result(replace(root, wumpus=pos), a).explorer_dead
                      for pos, p in hypotheses) for a in root_actions}
    best = max(root_actions, key=lambda a: (combined[a], -instant[a], -root_actions.index(a)))
    return MinimaxResult(best, combined[best], combined, hyp_out, stats, depth, replies)
