"""Propositional knowledge representation and inference.

* Literals are non-zero ints (+v / -v); a clause is a frozenset of literals (CNF).
* `dpll` decides satisfiability (unit propagation, pure literals, branching).
* `KnowledgeBase.ask(q)` decides KB |= q by refutation: KB ∧ ¬q is UNSAT.
* `DungeonKB` encodes the public Wumpus-world rules about pits/breezes over symbols
  P_x_y and B_x_y. Every clause remembers *why* it was added (its source).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..core.geometry import Pos, all_cells, neighbors
from ..core.percepts import PublicInfo

Clause = frozenset


@dataclass
class SolverStats:
    nodes: int = 0          # branching decisions
    propagations: int = 0   # unit propagations


def dpll(clauses: list[frozenset[int]], assignment: dict[int, bool] | None = None,
         stats: SolverStats | None = None) -> dict[int, bool] | None:
    """Return a satisfying (partial) assignment or None if unsatisfiable."""
    stats = stats or SolverStats()
    assignment = dict(assignment or {})
    work = [list(c) for c in clauses]
    while True:  # simplify + unit propagation until fixpoint
        changed, reduced = False, []
        for c in work:
            rest, sat = [], False
            for lit in c:
                val = assignment.get(abs(lit))
                if val is None:
                    rest.append(lit)
                elif val == (lit > 0):
                    sat = True
                    break
            if sat:
                continue
            if not rest:
                return None                      # conflict: clause falsified
            if len(rest) == 1:
                assignment[abs(rest[0])] = rest[0] > 0
                stats.propagations += 1
                changed = True
                continue
            reduced.append(rest)
        work = reduced
        if not changed:
            break
    if not work:
        return assignment
    # pure literal elimination
    polarity: dict[int, set[bool]] = {}
    for c in work:
        for lit in c:
            polarity.setdefault(abs(lit), set()).add(lit > 0)
    pures = {v: next(iter(p)) for v, p in polarity.items() if len(p) == 1}
    if pures:
        assignment.update(pures)
        return dpll([frozenset(c) for c in work], assignment, stats)
    # branch on the most frequent variable among the shortest clauses
    shortest = min(len(c) for c in work)
    counts: dict[int, int] = {}
    for c in work:
        if len(c) == shortest:
            for lit in c:
                counts[abs(lit)] = counts.get(abs(lit), 0) + 1
    var = max(sorted(counts), key=lambda v: counts[v])
    frozen = [frozenset(c) for c in work]
    for value in (True, False):
        stats.nodes += 1
        result = dpll(frozen, {**assignment, var: value}, stats)
        if result is not None:
            return result
    return None


@dataclass
class InferenceResult:
    query: str
    entailed: bool
    nodes: int
    propagations: int
    relevant: list[tuple[str, str]] = field(default_factory=list)   # (clause text, source)

    def explain(self) -> list[str]:
        verdict = "UNSAT -> entailed" if self.entailed else "SAT -> not entailed"
        lines = [f"Query: KB |= {self.query}",
                 f"Method: DPLL refutation, check KB ∧ ¬({self.query})",
                 f"Result: {verdict}  (branch nodes={self.nodes}, unit propagations={self.propagations})",
                 f"Relevant clauses ({len(self.relevant)}, connected to the query symbol):"]
        lines += [f"  {c}    <- {s}" for c, s in self.relevant]
        return lines


class KnowledgeBase:
    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._names: list[str] = [""]
        self.clauses: list[frozenset[int]] = []
        self.sources: list[str] = []
        self._seen: set[frozenset[int]] = set()

    def var(self, name: str) -> int:
        if name not in self._ids:
            self._ids[name] = len(self._names)
            self._names.append(name)
        return self._ids[name]

    def lit(self, name: str, positive: bool = True) -> int:
        v = self.var(name)
        return v if positive else -v

    def tell(self, literals: Iterable[int], source: str) -> None:
        c = frozenset(literals)
        if any(-l in c for l in c) or c in self._seen:   # tautology / duplicate
            return
        self._seen.add(c)
        self.clauses.append(c)
        self.sources.append(source)

    def clause_text(self, c: frozenset[int]) -> str:
        return " ∨ ".join(("" if l > 0 else "¬") + self._names[abs(l)] for l in sorted(c, key=abs))

    def _component(self, var: int) -> list[int]:
        """Indices of clauses connected to `var` through shared symbols."""
        by_var: dict[int, list[int]] = {}
        for i, c in enumerate(self.clauses):
            for l in c:
                by_var.setdefault(abs(l), []).append(i)
        seen_v, seen_c, stack = {var}, set(), [var]
        while stack:
            v = stack.pop()
            for i in by_var.get(v, []):
                if i not in seen_c:
                    seen_c.add(i)
                    for l in self.clauses[i]:
                        if abs(l) not in seen_v:
                            seen_v.add(abs(l))
                            stack.append(abs(l))
        return sorted(seen_c)

    def satisfiable(self) -> bool:
        return dpll(self.clauses) is not None

    def ask(self, name: str, positive: bool = True) -> InferenceResult:
        """KB |= literal ?  Only the connected component of the symbol is needed (sound when
        the KB itself is satisfiable, which the agent checks after each rebuild)."""
        q = f"{'' if positive else '¬'}{name}"
        if name not in self._ids:
            return InferenceResult(q, False, 0, 0)
        lit = self.lit(name, positive)
        idx = self._component(abs(lit))
        stats = SolverStats()
        sat = dpll([self.clauses[i] for i in idx] + [frozenset([-lit])], stats=stats)
        rel = [(self.clause_text(self.clauses[i]), self.sources[i]) for i in idx]
        return InferenceResult(q, sat is None, stats.nodes, stats.propagations, rel)


# ----------------------------------------------------------------------- domain encoding
def P(c: Pos) -> str:
    return f"P{c.x}{c.y}"


def B(c: Pos) -> str:
    return f"B{c.x}{c.y}"


def O(c: Pos) -> str:
    return f"O{c.x}{c.y}"


def S(c: Pos) -> str:
    return f"S{c.x}{c.y}"


@dataclass(frozen=True)
class CellStatus:
    cell: Pos
    status: str                       # pits: "safe" | "pit" | "unknown";  portals: "portal" | "none" | "unknown"
    proof: InferenceResult | None     # the entailment that established it


class DungeonKB:
    """Builds a KnowledgeBase from the explorer's (valid) evidence.

    Symbols: Pxy (original pit), Bxy (breeze sensed), Oxy (portal), Sxy (portal signal sensed).
    Rules: Bc <=> OR(P of neighbours), Sc <=> OR(O of neighbours), for every sensed cell c.
    """

    def __init__(self, info: PublicInfo):
        self.info = info
        self.kb = KnowledgeBase()

    @staticmethod
    def _biconditional(kb: KnowledgeBase, sensed: str, causes: list[str], value: bool, turn: int,
                       c: Pos, what: str, rule: str) -> None:
        kb.tell([kb.lit(sensed, value)], f"percept turn {turn}: {'' if value else 'no '}{what} at {c}")
        kb.tell([kb.lit(sensed, False)] + [kb.lit(n) for n in causes], f"{rule} rule at {c}")
        for n in causes:
            kb.tell([kb.lit(sensed), kb.lit(n, False)], f"{rule} rule at {c}")

    def rebuild(self, visited: set[Pos], collapsed, breeze_obs: dict[Pos, tuple[bool, int]],
                portal_cells: set[Pos] = frozenset(), signal_obs: dict[Pos, tuple[bool, int]] | None = None
                ) -> KnowledgeBase:
        kb = KnowledgeBase()
        info = self.info
        for c in sorted(info.safe_start_zone):
            kb.tell([kb.lit(P(c), False)], "public rule: start zone has no pits")
            kb.tell([kb.lit(O(c), False)], "public rule: start zone has no portals")
        for c in sorted(visited):
            kb.tell([kb.lit(P(c), False)], f"visited {c} and survived")
            if c not in portal_cells:
                kb.tell([kb.lit(O(c), False)], f"stood on {c} without teleporting")
        for c in sorted(portal_cells):
            kb.tell([kb.lit(O(c))], f"teleported via {c}")
            kb.tell([kb.lit(P(c), False)], "public rule: portals are never pits")
        for c in sorted(collapsed):
            kb.tell([kb.lit(P(c), False)], f"{c} was an unstable tile (not an original pit)")
            kb.tell([kb.lit(O(c), False)], f"{c} was an unstable tile (not a portal)")
        for c, (b, turn) in sorted(breeze_obs.items()):
            self._biconditional(kb, B(c), [P(n) for n in neighbors(c, info.size)], b, turn, c, "breeze", "breeze")
        for c, (sig, turn) in sorted((signal_obs or {}).items()):
            self._biconditional(kb, S(c), [O(n) for n in neighbors(c, info.size)], sig, turn, c,
                                "portal signal", "portal-signal")
        self.kb = kb
        return kb

    def classify(self, cells: Iterable[Pos]) -> dict[Pos, CellStatus]:
        """Pit status of each cell, by entailment."""
        out = {}
        for c in cells:
            safe = self.kb.ask(P(c), positive=False)
            if safe.entailed:
                out[c] = CellStatus(c, "safe", safe)
                continue
            pit = self.kb.ask(P(c), positive=True)
            out[c] = CellStatus(c, "pit", pit) if pit.entailed else CellStatus(c, "unknown", safe)
        return out

    def classify_portals(self, cells: Iterable[Pos]) -> dict[Pos, CellStatus]:
        out = {}
        for c in cells:
            yes = self.kb.ask(O(c), positive=True)
            if yes.entailed:
                out[c] = CellStatus(c, "portal", yes)
                continue
            no = self.kb.ask(O(c), positive=False)
            out[c] = CellStatus(c, "none", no) if no.entailed else CellStatus(c, "unknown", yes)
        return out

    def all_cells(self) -> list[Pos]:
        return all_cells(self.info.size)
