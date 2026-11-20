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

    def because(self) -> list[str]:
        """Distinct evidence (percepts and rules) behind the clauses the solver used."""
        seen, out = set(), []
        for _, src in self.relevant:
            if src not in seen:
                seen.add(src)
                out.append(src)
        return out

    def explain(self) -> list[str]:
        verdict = "UNSAT -> entailed" if self.entailed else "SAT -> not entailed"
        lines = [f"Query: KB |= {self.query}",
                 f"Method: DPLL refutation, check KB ∧ ¬({self.query})",
                 f"Result: {verdict}  (branch nodes={self.nodes}, unit propagations={self.propagations})",
                 f"Relevant clauses ({len(self.relevant)}, connected to the query symbol):"]
        lines += [f"  {c}    <- {s}" for c, s in self.relevant]
        if self.entailed:
            lines.append("In words: " + "; ".join(self.because()))
        return lines


@dataclass
class ConsistencyReport:
    consistent: bool
    core: list[tuple[str, str]] = field(default_factory=list)   # minimal conflicting clauses (text, source)

    def explain(self) -> list[str]:
        if self.consistent:
            return ["KB is satisfiable: the evidence is consistent."]
        return (["CONTRADICTION: the evidence cannot all be true at once.",
                 f"Minimal conflicting set ({len(self.core)} clauses; removing any one restores consistency):"]
                + [f"  {c}    <- {s}" for c, s in self.core])


class KnowledgeBase:
    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._names: list[str] = [""]
        self.clauses: list[frozenset[int]] = []
        self.sources: list[str] = []
        self.valid_turn: list[int | None] = []      # None = permanent; t = only valid at turn t
        self._seen: set[frozenset[int]] = set()

    def var(self, name: str) -> int:
        if name not in self._ids:
            self._ids[name] = len(self._names)
            self._names.append(name)
        return self._ids[name]

    def lit(self, name: str, positive: bool = True) -> int:
        v = self.var(name)
        return v if positive else -v

    def tell(self, literals: Iterable[int], source: str, valid_turn: int | None = None) -> None:
        c = frozenset(literals)
        if any(-l in c for l in c) or c in self._seen:   # tautology / duplicate
            return
        self._seen.add(c)
        self.clauses.append(c)
        self.sources.append(source)
        self.valid_turn.append(valid_turn)

    def time_sensitive(self) -> list[tuple[str, str, int]]:
        return [(self.clause_text(c), s, t) for c, s, t in zip(self.clauses, self.sources, self.valid_turn)
                if t is not None]

    def consistency(self) -> ConsistencyReport:
        """Satisfiability check; if UNSAT, shrink to a minimal unsatisfiable subset by deletion."""
        if dpll(self.clauses) is not None:
            return ConsistencyReport(True)
        core = list(range(len(self.clauses)))
        i = 0
        while i < len(core):
            trial = core[:i] + core[i + 1:]
            if dpll([self.clauses[j] for j in trial]) is None:
                core = trial            # still contradictory without clause i -> not needed
            else:
                i += 1
        return ConsistencyReport(False, [(self.clause_text(self.clauses[j]), self.sources[j]) for j in core])

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
def PIT(c: Pos) -> str:
    return f"PIT_{c.x}_{c.y}"


def BREEZE(c: Pos) -> str:
    return f"BREEZE_{c.x}_{c.y}"


def WUMPUS(c: Pos) -> str:
    return f"WUMPUS_{c.x}_{c.y}"


def STENCH(c: Pos) -> str:
    return f"STENCH_{c.x}_{c.y}"


def SAFE(c: Pos) -> str:
    return f"SAFE_{c.x}_{c.y}"


def PORTAL(c: Pos) -> str:
    return f"PORTAL_{c.x}_{c.y}"


def SIGNAL(c: Pos) -> str:
    return f"SIGNAL_{c.x}_{c.y}"


P, B, O, S = PIT, BREEZE, PORTAL, SIGNAL      # short aliases


@dataclass(frozen=True)
class CellStatus:
    cell: Pos
    status: str                       # pits: safe|pit|unknown   portals: portal|none|unknown
                                      # wumpus: wumpus|none|unknown   safety: safe|unknown
    proof: InferenceResult | None     # the entailment that established it
    explanation: str = ""             # human-readable sentence


@dataclass(frozen=True)
class WumpusEvidence:
    """Time-sensitive Wumpus observation: valid only at `turn`, because the Wumpus moves."""
    turn: int
    position: Pos                     # where the explorer stands (and survived)
    stench: bool
    alive: bool = True


class DungeonKB:
    """Builds a KnowledgeBase from the explorer's (valid) evidence.

    Permanent facts: pits (static, except shuffles, which drop old breeze evidence), portals, rules.
    Time-sensitive facts: everything about WUMPUS_x_y, valid only for the turn it was observed.
    Rules (CNF of the biconditionals):
      BREEZE_c <=> OR PIT_n          (breeze => some adjacent pit; no breeze => none adjacent)
      SIGNAL_c <=> OR PORTAL_n
      STENCH_c <=> OR WUMPUS_n       (stench => possible adjacent Wumpus; none => ruled out)
      SAFE_c   <=> ¬PIT_c ∧ ¬WUMPUS_c
      ¬PIT_c ∨ ¬WUMPUS_c             (the Wumpus never stands on a pit)
    """

    def __init__(self, info: PublicInfo):
        self.info = info
        self.kb = KnowledgeBase()

    @staticmethod
    def _biconditional(kb: KnowledgeBase, sensed: str, causes: list[str], value: bool, turn: int,
                       c: Pos, what: str, rule: str, valid: int | None = None) -> None:
        kb.tell([kb.lit(sensed, value)], f"percept turn {turn}: {'' if value else 'no '}{what} at {c}", valid)
        kb.tell([kb.lit(sensed, False)] + [kb.lit(n) for n in causes], f"{rule} rule at {c}", valid)
        for n in causes:
            kb.tell([kb.lit(sensed), kb.lit(n, False)], f"{rule} rule at {c}", valid)

    def rebuild(self, visited: set[Pos], collapsed, breeze_obs: dict[Pos, tuple[bool, int]],
                portal_cells: set[Pos] = frozenset(), signal_obs: dict[Pos, tuple[bool, int]] | None = None,
                wumpus: WumpusEvidence | None = None, safe_queries: Iterable[Pos] = ()) -> KnowledgeBase:
        kb = KnowledgeBase()
        info = self.info
        for c in sorted(info.safe_start_zone):
            kb.tell([kb.lit(PIT(c), False)], "public rule: start zone has no pits")
            kb.tell([kb.lit(PORTAL(c), False)], "public rule: start zone has no portals")
        for c in sorted(visited):
            kb.tell([kb.lit(PIT(c), False)], f"visited {c} and survived")
            if c not in portal_cells:
                kb.tell([kb.lit(PORTAL(c), False)], f"stood on {c} without teleporting")
        for c in sorted(portal_cells):
            kb.tell([kb.lit(PORTAL(c))], f"teleported via {c}")
            kb.tell([kb.lit(PIT(c), False)], "public rule: portals are never pits")
        for c in sorted(collapsed):
            kb.tell([kb.lit(PIT(c), False)], f"{c} was an unstable tile (not an original pit)")
            kb.tell([kb.lit(PORTAL(c), False)], f"{c} was an unstable tile (not a portal)")
        for c, (b, turn) in sorted(breeze_obs.items()):
            self._biconditional(kb, BREEZE(c), [PIT(n) for n in neighbors(c, info.size)], b, turn, c,
                                "breeze", "breeze")
        for c, (sig, turn) in sorted((signal_obs or {}).items()):
            self._biconditional(kb, SIGNAL(c), [PORTAL(n) for n in neighbors(c, info.size)], sig, turn, c,
                                "portal signal", "portal-signal")
        if wumpus is not None:
            self._wumpus_facts(kb, wumpus, set(collapsed))
        for c in safe_queries:
            kb.tell([kb.lit(SAFE(c), False), kb.lit(PIT(c), False)], f"safety rule at {c}")
            kb.tell([kb.lit(SAFE(c), False), kb.lit(WUMPUS(c), False)], f"safety rule at {c}")
            kb.tell([kb.lit(SAFE(c)), kb.lit(PIT(c)), kb.lit(WUMPUS(c))], f"safety rule at {c}")
            kb.tell([kb.lit(PIT(c), False), kb.lit(WUMPUS(c), False)], "public rule: the Wumpus never stands on a pit")
        self.kb = kb
        return kb

    def _wumpus_facts(self, kb: KnowledgeBase, ev: WumpusEvidence, collapsed: set[Pos]) -> None:
        size, t = self.info.size, ev.turn
        if not ev.alive:
            for c in all_cells(size):
                kb.tell([kb.lit(WUMPUS(c), False)], "scream heard: the Wumpus is dead")
            return
        kb.tell([kb.lit(WUMPUS(ev.position), False)], f"turn {t}: we stand on {ev.position} alive", t)
        nbrs = neighbors(ev.position, size)
        self._biconditional(kb, STENCH(ev.position), [WUMPUS(n) for n in nbrs], ev.stench, t, ev.position,
                            "stench", "stench", valid=t)
        for n in nbrs:
            kb.tell([kb.lit(PIT(n), False), kb.lit(WUMPUS(n), False)], "public rule: the Wumpus never stands on a pit")
            if n in collapsed:
                kb.tell([kb.lit(WUMPUS(n), False)], f"{n} collapsed: the Wumpus cannot stand there")

    # ------------------------------------------------------------------ queries
    def _three_way(self, c: Pos, sym: str, yes: str, no: str, yes_text: str, no_text: str) -> CellStatus:
        r_no = self.kb.ask(sym, positive=False)
        if r_no.entailed:
            return CellStatus(c, no, r_no, f"{c} {no_text} because: " + "; ".join(r_no.because()))
        r_yes = self.kb.ask(sym, positive=True)
        if r_yes.entailed:
            return CellStatus(c, yes, r_yes, f"{c} {yes_text} because: " + "; ".join(r_yes.because()))
        return CellStatus(c, "unknown", r_no, f"{c}: the evidence neither proves nor rules it out (ambiguous)")

    def classify(self, cells: Iterable[Pos]) -> dict[Pos, CellStatus]:
        """Pit status of each cell, by entailment."""
        return {c: self._three_way(c, PIT(c), "pit", "safe", "certainly contains a pit", "certainly has no pit")
                for c in cells}

    def classify_portals(self, cells: Iterable[Pos]) -> dict[Pos, CellStatus]:
        return {c: self._three_way(c, PORTAL(c), "portal", "none", "is certainly a portal",
                                   "is certainly not a portal") for c in cells}

    def classify_wumpus(self, cells: Iterable[Pos]) -> dict[Pos, CellStatus]:
        """Wumpus status NOW (time-sensitive: only valid for the current turn)."""
        return {c: self._three_way(c, WUMPUS(c), "wumpus", "none", "certainly holds the Wumpus right now",
                                   "certainly does not hold the Wumpus right now") for c in cells}

    def classify_safe(self, cells: Iterable[Pos]) -> dict[Pos, CellStatus]:
        out = {}
        for c in cells:
            r = self.kb.ask(SAFE(c), positive=True)
            out[c] = (CellStatus(c, "safe", r, f"{c} is safe right now (no pit, no Wumpus) because: "
                                 + "; ".join(r.because())) if r.entailed
                      else CellStatus(c, "unknown", r, f"{c}: not proven safe"))
        return out

    def all_cells(self) -> list[Pos]:
        return all_cells(self.info.size)
