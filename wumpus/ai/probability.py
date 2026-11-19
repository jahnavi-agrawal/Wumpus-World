"""Bayesian pit probability estimation.

Model: each unknown cell independently contains a pit with prior δ. Evidence: every valid
breeze observation (B_c ⇔ some neighbour is a pit). Cells proven by the KB are fixed.
For the remaining unknown cells that appear in a breeze constraint ("frontier"), we
enumerate every assignment consistent with all constraints — exactly as in AIMA §12.7:

    P(P_x | e) = α · Σ_{frontier models m consistent with e, m[x]=pit} δ^{#pits(m)} (1-δ)^{#safe(m)}

The frontier is split into independent connected components (cells linked by a shared
constraint); each is enumerated separately with backtracking that prunes a partial model
as soon as some breezy cell can no longer get a pit. Cells outside every constraint keep δ.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.geometry import Pos, neighbors

SAMPLE_WORLDS = 8


@dataclass
class CellRiskReport:
    cell: Pos
    p_pit: float
    method: str                                   # kb-safe | kb-pit | enumeration | prior | approx
    prior: float
    component: tuple[Pos, ...] = ()
    constraints: list[tuple[Pos, tuple[Pos, ...]]] = field(default_factory=list)
    worlds_consistent: int = 0
    weight_pit: float = 0.0
    weight_total: float = 0.0
    sample_worlds: list[tuple[tuple[Pos, ...], float]] = field(default_factory=list)
    kb_proof: object = None                        # InferenceResult for kb-safe / kb-pit

    def explain(self) -> list[str]:
        L = [f"Cell {self.cell}: P(pit) = {self.p_pit:.1%}   method = {self.method}",
             f"Prior δ = {self.prior:.4f}  (= remaining pits / remaining unknown cells)"]
        if self.method in ("kb-safe", "kb-pit"):
            L.append("Fixed by logical entailment (see proof below).")
            if self.kb_proof is not None:
                L += self.kb_proof.explain()
        elif self.method == "prior":
            L.append("No valid breeze observation touches this cell -> posterior = prior.")
        else:
            L.append(f"Frontier component ({len(self.component)} cells): " + " ".join(map(str, self.component)))
            L.append("Constraints (breezy cell -> at least one pit among):")
            L += [f"  B{b} -> {{{', '.join(map(str, u))}}}" for b, u in self.constraints]
            if self.method == "enumeration":
                L.append(f"Consistent models enumerated: {self.worlds_consistent}")
                L.append(f"Σ weight(models with pit at {self.cell}) = {self.weight_pit:.6g}")
                L.append(f"Σ weight(all consistent models)       = {self.weight_total:.6g}")
                L.append(f"P = {self.weight_pit:.6g} / {self.weight_total:.6g} = {self.p_pit:.4f}")
                L.append(f"Heaviest models (pits in component, weight = δ^k(1-δ)^(n-k)):")
                L += [f"  pits={{{', '.join(map(str, pits)) or '-'}}}  w={w:.6g}" for pits, w in self.sample_worlds]
            else:
                L.append("Component exceeded the enumeration cap: approximated with the "
                         "single-constraint posterior δ / (1-(1-δ)^|U|), max over constraints.")
        return L


def _components(constraints: dict[Pos, tuple[Pos, ...]]) -> list[tuple[list[Pos], list[Pos]]]:
    """Group constraint cells by shared unknown cells. Returns [(cells, constraint keys)]."""
    parent: dict[Pos, Pos] = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u in constraints.values():
        for c in u:
            parent.setdefault(c, c)
        for c in u[1:]:
            parent[find(c)] = find(u[0])
    groups: dict[Pos, tuple[list, list]] = {}
    for c in sorted(parent):
        groups.setdefault(find(c), ([], []))[0].append(c)
    for b, u in sorted(constraints.items()):
        groups[find(u[0])][1].append(b)
    return list(groups.values())


def _enumerate(cells: list[Pos], cons: list[tuple[Pos, ...]], prior: float):
    """Backtracking enumeration. Returns (count, total, per-cell pit weight, heaviest models)."""
    idx = {c: i for i, c in enumerate(cells)}
    cons_idx = [[idx[c] for c in u] for u in cons]
    last = [max(u) for u in cons_idx]             # constraint decided after its last cell
    by_last: dict[int, list[list[int]]] = {}
    for u, l in zip(cons_idx, last):
        by_last.setdefault(l, []).append(u)
    n = len(cells)
    assign = [False] * n
    pit_w = [0.0] * n
    stats = {"count": 0, "total": 0.0}
    samples: list[tuple[float, tuple[Pos, ...]]] = []

    def rec(i: int, k: int):
        if i == n:
            w = prior ** k * (1 - prior) ** (n - k)
            stats["count"] += 1
            stats["total"] += w
            for j in range(n):
                if assign[j]:
                    pit_w[j] += w
            samples.append((w, tuple(c for c, a in zip(cells, assign) if a)))
            if len(samples) > 4 * SAMPLE_WORLDS:
                samples.sort(key=lambda s: -s[0])
                del samples[SAMPLE_WORLDS:]
            return
        for val in (True, False):
            assign[i] = val
            if all(any(assign[j] for j in u) for u in by_last.get(i, [])):
                rec(i + 1, k + val)
        assign[i] = False

    rec(0, 0)
    samples.sort(key=lambda s: (-s[0], s[1]))
    return stats["count"], stats["total"], pit_w, [(p, w) for w, p in samples[:SAMPLE_WORLDS]]


def estimate_pits(size: int, statuses: dict, breeze_obs: dict[Pos, tuple[bool, int]],
                  prior: float, max_component_cells: int = 18) -> dict[Pos, CellRiskReport]:
    """`statuses`: Pos -> CellStatus from DungeonKB.classify for every non-visited cell of interest."""
    out: dict[Pos, CellRiskReport] = {}
    known_pit = {c for c, s in statuses.items() if s.status == "pit"}
    unknown = {c for c, s in statuses.items() if s.status == "unknown"}
    for c, s in statuses.items():
        if s.status != "unknown":
            out[c] = CellRiskReport(c, 1.0 if s.status == "pit" else 0.0, f"kb-{s.status}", prior, kb_proof=s.proof)
    constraints: dict[Pos, tuple[Pos, ...]] = {}
    for b, (breezy, _) in breeze_obs.items():
        if not breezy:
            continue
        nb = neighbors(b, size)
        if any(n in known_pit for n in nb):
            continue                              # already satisfied in every consistent model
        u = tuple(n for n in nb if n in unknown)
        if u:
            constraints[b] = u
    in_frontier: set[Pos] = set()
    for cells, keys in _components(constraints):
        in_frontier.update(cells)
        cons = [constraints[k] for k in keys]
        cons_view = [(k, constraints[k]) for k in keys]
        if len(cells) > max_component_cells:
            for c in cells:
                p = max(prior / (1 - (1 - prior) ** len(u)) for u in cons if c in u)
                out[c] = CellRiskReport(c, min(p, 1.0), "approx", prior, tuple(cells), cons_view)
            continue
        count, total, pit_w, samples = _enumerate(cells, cons, prior)
        for i, c in enumerate(cells):
            p = pit_w[i] / total if total > 0 else prior
            out[c] = CellRiskReport(c, p, "enumeration", prior, tuple(cells), cons_view, count,
                                    pit_w[i], total, samples)
    for c in unknown - in_frontier:
        out[c] = CellRiskReport(c, prior, "prior", prior)
    return out
