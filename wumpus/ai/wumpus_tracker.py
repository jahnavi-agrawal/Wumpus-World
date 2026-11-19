"""Bayesian forward filter (discrete HMM) over the position of the moving Wumpus.

    belief_t(w) ∝ P(e_t | w) · Σ_w' T(w' -> w) · belief_{t-1}(w')

Evidence used (all deterministic likelihoods 0/1, derived from public rules):
  * explorer survived entering cell p          -> W ≠ p
  * arrow missed along a ray                   -> W ∉ ray
  * COLLAPSE(c) event                          -> W stayed, manhattan(W, c) ≤ collapse_range
  * SHUFFLE event                              -> W stayed
  * explorer not eaten after the Wumpus turn   -> W ≠ explorer
  * stench / no stench at explorer cell p      -> W ∈ N(p) / W ∉ N(p)
Motion model T (an ASSUMPTION about the human, see DECISIONS.md D3):
  stay with p_stay; otherwise with p_pursuit move uniformly among moves that reduce the
  Manhattan distance to the explorer, else uniformly among all legal moves. Moves into a
  cell are weighted by (1 - P(pit)) because the Wumpus cannot enter pits; known pits and
  collapsed tiles are excluded. Pads teleport the Wumpus to the paired pad.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.geometry import Pos, all_cells, manhattan, neighbors
from ..core.percepts import PublicInfo


@dataclass
class FilterStep:
    label: str
    mass_kept: float           # probability mass that survived this step (before renormalising)


@dataclass
class TrackerReport:
    turn: int
    alive: bool
    steps: list[FilterStep] = field(default_factory=list)
    top: list[tuple[Pos, float]] = field(default_factory=list)
    reset: str | None = None

    def explain(self) -> list[str]:
        if not self.alive:
            return ["Wumpus confirmed dead (scream heard). Belief cleared."]
        L = ["HMM forward filter over Wumpus position:"]
        for s in self.steps:
            L.append(f"  {s.label:<52} mass kept {s.mass_kept:.3f}")
        if self.reset:
            L.append(f"  !! all mass eliminated: {self.reset}")
        L.append("Posterior (top cells):")
        L += [f"  {c}: {p:.1%}" for c, p in self.top]
        return L


class WumpusTracker:
    def __init__(self, info: PublicInfo, stay_prob: float = 0.15, pursuit_prob: float = 0.5):
        self.info = info
        self.size = info.size
        self.stay_prob = stay_prob
        self.pursuit_prob = pursuit_prob
        self.alive = True
        self.belief: dict[Pos, float] = self._initial()
        self.report = TrackerReport(0, True)

    def _initial(self, exclude: set[Pos] = frozenset()) -> dict[Pos, float]:
        pads = set(self.info.pads or ())
        cells = [c for c in all_cells(self.size)
                 if manhattan(c, self.info.start) >= self.info.wumpus_min_start_distance
                 and c not in pads and c not in exclude]
        return {c: 1.0 / len(cells) for c in cells}

    # ------------------------------------------------------------------ bookkeeping
    def begin(self, turn: int) -> None:
        self.report = TrackerReport(turn, self.alive)

    def _apply(self, label: str, keep) -> None:
        before = sum(self.belief.values())
        self.belief = {c: p for c, p in self.belief.items() if keep(c)}
        after = sum(self.belief.values())
        self.report.steps.append(FilterStep(label, after / before if before else 0.0))

    def normalize(self, fallback_exclude: set[Pos]) -> None:
        total = sum(self.belief.values())
        if total <= 1e-12:
            # Model mismatch (the human did something our motion model gave probability 0).
            # Recover by restarting from a uniform belief over cells consistent with this turn.
            self.report.reset = "evidence contradicted the motion model; belief reset to uniform"
            pads = set(self.info.pads or ())
            cells = [c for c in all_cells(self.size) if c not in fallback_exclude and c not in pads]
            self.belief = {c: 1.0 / len(cells) for c in cells}
        else:
            self.belief = {c: p / total for c, p in self.belief.items() if p > 0}
        self.report.top = self.top(5)

    def kill(self) -> None:
        self.alive = False
        self.belief = {}
        self.report = TrackerReport(self.report.turn, False)

    # ------------------------------------------------------------------ evidence
    def observe_survived_at(self, p: Pos) -> None:
        self._apply(f"explorer survived entering {p} -> W≠{p}", lambda c: c != p)

    def observe_arrow_missed(self, cells: list[Pos]) -> None:
        s = set(cells)
        self._apply(f"arrow missed along {len(cells)} cells", lambda c: c not in s)

    def observe_collapse(self, target: Pos, rng: int) -> None:
        self._apply(f"COLLAPSE {target}: W stayed, within {rng} of it",
                    lambda c: manhattan(c, target) <= rng)

    def observe_stench(self, p: Pos, stench: bool) -> None:
        nb = set(neighbors(p, self.size))
        if stench:
            self._apply(f"stench at {p} -> W adjacent", lambda c: c in nb)
        else:
            self._apply(f"no stench at {p} -> W not adjacent", lambda c: c not in nb)

    # ------------------------------------------------------------------ motion
    def transition(self, w: Pos, explorer: Pos, blocked: set[Pos], pit_prob: dict[Pos, float]) -> dict[Pos, float]:
        """T(w -> ·): distribution of the Wumpus' next cell."""
        moves = [n for n in neighbors(w, self.size) if n not in blocked and pit_prob.get(n, 0.0) < 1.0]
        out: dict[Pos, float] = {}
        if not moves:
            return {w: 1.0}
        out[w] = self.stay_prob
        move_mass = 1.0 - self.stay_prob
        closer = [n for n in moves if manhattan(n, explorer) < manhattan(w, explorer)]
        parts = [(moves, 1.0 - self.pursuit_prob if closer else 1.0)]
        if closer:
            parts.append((closer, self.pursuit_prob))
        for cand, share in parts:
            weights = {n: 1.0 - pit_prob.get(n, 0.0) for n in cand}
            z = sum(weights.values())
            if z <= 0:
                out[w] += move_mass * share
                continue
            for n, wt in weights.items():
                dest = self._pad(n)
                out[dest] = out.get(dest, 0.0) + move_mass * share * wt / z
        return out

    def _pad(self, n: Pos) -> Pos:
        if self.info.pads and n in self.info.pads:
            a, b = self.info.pads
            return b if n == a else a
        return n

    def predict(self, explorer: Pos, blocked: set[Pos], pit_prob: dict[Pos, float]) -> None:
        new: dict[Pos, float] = {}
        for w, p in self.belief.items():
            for n, t in self.transition(w, explorer, blocked, pit_prob).items():
                new[n] = new.get(n, 0.0) + p * t
        self.belief = new
        self.report.steps.append(FilterStep("predict: Wumpus motion model (stay/pursue/wander)", 1.0))

    # ------------------------------------------------------------------ queries
    def prob(self, c: Pos) -> float:
        return self.belief.get(c, 0.0)

    def prob_within(self, p: Pos, r: int) -> float:
        return sum(v for c, v in self.belief.items() if manhattan(c, p) <= r)

    def entry_risk(self, c: Pos, blocked: set[Pos], pit_prob: dict[Pos, float]) -> float:
        """P(death by Wumpus | explorer steps into c now) ≈ P(W at c) + P(W moves into c next)."""
        if not self.alive:
            return 0.0
        risk = self.prob(c)
        for n in neighbors(c, self.size):
            pn = self.prob(n)
            if pn > 0:
                risk += pn * self.transition(n, c, blocked, pit_prob).get(c, 0.0)
        return min(risk, 1.0)

    def top(self, k: int) -> list[tuple[Pos, float]]:
        return sorted(self.belief.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
