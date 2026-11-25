"""Bayesian hazard estimation and A* planning (modes, alternatives, structured output)."""
import json
from types import MappingProxyType

import pytest

from wumpus.agents.bayesian import _enumerate, binary_entropy, estimate_hazard
from wumpus.agents.pathfinding import PlanningMode, RiskModel, plan, plan_routes
from wumpus.core.actions import FORWARD, JUMP
from wumpus.core.geometry import Direction as D, Pos, all_cells


# ------------------------------------------------------------------ hazard estimates
def test_known_safe_cell_has_zero_hazard():
    h = estimate_hazard(Pos(1, 1), None, 0.4, 0.0, 0.0, known_pit_free=True, confirmed_hazard=None,
                        wumpus_status="none", wumpus_evidence=[])
    assert (h.p_pit, h.p_wumpus, h.p_death, h.confidence) == (0.0, 0.0, 0.0, 1.0)


def test_confirmed_hazard_has_probability_one():
    h = estimate_hazard(Pos(2, 0), None, 0.3, 0.1, 0.0, known_pit_free=False, confirmed_hazard="proven pit",
                        wumpus_status=None, wumpus_evidence=[])
    assert h.p_pit == 1.0 and h.p_death == 1.0 and h.confidence == 1.0
    w = estimate_hazard(Pos(0, 2), None, 0.0, 0.3, 0.2, known_pit_free=True, confirmed_hazard=None,
                        wumpus_status="wumpus", wumpus_evidence=[])
    assert w.p_wumpus == 1.0 and w.p_death == 1.0


def test_separate_and_combined_probabilities():
    h = estimate_hazard(Pos(3, 2), None, 0.2, 0.1, 0.05, known_pit_free=False, confirmed_hazard=None,
                        wumpus_status=None, wumpus_evidence=["filter"])
    assert h.p_pit == 0.2 and h.p_wumpus == 0.1 and h.p_wumpus_in == 0.05
    assert h.p_death == pytest.approx(0.2 + 0.8 * 0.15)       # pit and Wumpus are mutually exclusive
    assert h.confidence == pytest.approx(1 - binary_entropy(h.p_death))
    assert json.dumps(h.to_dict()) and "Evidence:" in "\n".join(h.explain())


def test_kb_wumpus_verdict_overrides_filter():
    h = estimate_hazard(Pos(1, 2), None, 0.0, 0.4, 0.0, known_pit_free=True, confirmed_hazard=None,
                        wumpus_status="none", wumpus_evidence=[])
    assert h.p_wumpus == 0.0


def test_binary_entropy():
    assert binary_entropy(0.5) == pytest.approx(1.0) and binary_entropy(0.0) == binary_entropy(1.0) == 0.0


def test_neighbouring_pits_are_not_treated_as_independent():
    """One breeze, two candidate cells: the joint P(both) differs from the product of marginals."""
    cells = [Pos(2, 0), Pos(1, 1)]
    count, total, pit_w, samples = _enumerate(cells, [tuple(cells)], 0.2)
    pa, pb = pit_w[0] / total, pit_w[1] / total
    both = sum(w for pits, w in samples if set(pits) == set(cells)) / total
    assert count == 3 and both == pytest.approx(0.04 / 0.36)
    assert both != pytest.approx(pa * pb)                      # dependency captured by enumeration


# ------------------------------------------------------------------ planning
def model(proven, risk=None, blocked=(), portals=None, avoid=(), default=0.0):
    cells = all_cells(8)
    return RiskModel(8, frozenset(Pos(*c) for c in proven),
                     MappingProxyType({c: (risk or {}).get(c, default) for c in cells}),
                     MappingProxyType({Pos(*c): "collapsed tile" for c in blocked}),
                     MappingProxyType({Pos(*c): "portal with unknown destination" for c in avoid}),
                     MappingProxyType(portals or {}))


ROW0 = [(x, 0) for x in range(8)]


def test_strict_mode_only_uses_proven_cells():
    m = model(ROW0[:3], risk={Pos(3, 0): 0.1})
    r = plan(Pos(0, 0), D.E, {Pos(5, 0)}, m, PlanningMode.STRICT, 0.3, False)
    assert not r.found and "unproven cell (strict mode)" in r.rejected_cells
    t = plan(Pos(0, 0), D.E, {Pos(5, 0)}, m, PlanningMode.TOLERANT, 0.3, False)
    assert t.found and t.cells == [Pos(1, 0), Pos(2, 0), Pos(3, 0), Pos(4, 0), Pos(5, 0)]
    assert t.labels[0].startswith("proven pit-free") and t.labels[2] == "UNPROVEN, risk 10%"
    assert not any(lab.startswith("proven") for lab in t.labels[2:])        # never call unknown safe


def test_risk_tolerance_is_configurable():
    m = model(ROW0[:3], risk={Pos(0, 0): 0, Pos(1, 0): 0, Pos(2, 0): 0, Pos(3, 0): 0.25, Pos(4, 0): 0.0},
              default=0.9)                                     # no cheap detour
    assert not plan(Pos(0, 0), D.E, {Pos(4, 0)}, m, PlanningMode.TOLERANT, 0.2, False).found
    assert plan(Pos(0, 0), D.E, {Pos(4, 0)}, m, PlanningMode.TOLERANT, 0.3, False).found


def test_collapsed_cells_blocked_and_jump_crosses_them():
    m = model(ROW0, blocked=[(1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (2, 2)] + [(1, y) for y in range(3, 8)]
              + [(2, y) for y in range(3, 8)])
    assert not plan(Pos(0, 0), D.E, {Pos(3, 0)}, m, PlanningMode.STRICT, 0.3, False).found
    r = plan(Pos(0, 0), D.E, {Pos(3, 0)}, m, PlanningMode.STRICT, 0.3, True)
    assert r.actions == [JUMP] and "collapsed tile" not in r.labels


def test_teleport_edges_and_turn_costs():
    portals = {Pos(1, 0): Pos(6, 6), Pos(6, 6): Pos(1, 0)}
    m = model(all_cells(8), portals=portals)
    r = plan(Pos(0, 0), D.E, {Pos(6, 7)}, m, PlanningMode.STRICT, 0.3, False)
    assert r.cells[0] == Pos(6, 6) and r.labels[0] == "known portal" and r.cost == 3   # forward, turn, forward


def test_unknown_destination_portal_avoided_unless_goal():
    m = model(ROW0 + [(x, 1) for x in range(8)], avoid=[(1, 0)])
    r = plan(Pos(0, 0), D.E, {Pos(2, 0)}, m, PlanningMode.STRICT, 0.3, False)
    assert Pos(1, 0) not in r.cells and r.cost > 2


def test_alternative_routes_and_rejection_reasons():
    risk = {Pos(x, 0): 0.0 for x in range(8)}
    risk.update({Pos(2, 0): 0.2})
    m = model([(x, y) for x in range(8) for y in (0, 1) if (x, y) != (2, 0)] + [(2, 0)], risk=risk)
    rp = plan_routes(Pos(0, 0), D.E, {Pos(4, 0)}, m, PlanningMode.STRICT, 0.3, True)
    assert rp.best.found and Pos(2, 0) not in rp.best.cells                 # avoids the 20% cell
    names = [n for n, _ in rp.alternatives] + [n for n, _ in rp.rejected]
    assert "without jump" in names
    d = rp.to_dict()
    json.dumps(d)
    assert d["best"]["labels"] and d["mode"] == "strict-safe"
    text = "\n".join(rp.explain())
    assert "Planning mode: strict-safe" in text and "step 1 cell" in text


def test_rejected_variant_reports_why():
    m = model(ROW0[:2], default=0.9)                      # everything unproven is very risky
    rp = plan_routes(Pos(0, 0), D.E, {Pos(3, 0)}, m, PlanningMode.STRICT, 0.3, False)
    assert not rp.best.found and "unproven cell (strict mode)" in rp.best.rejected_cells
    reasons = dict(rp.rejected)
    assert "no path" in reasons["risk-tolerant"] and "above tolerance" in reasons["risk-tolerant"]


def test_strict_failure_with_tolerant_alternative():
    m = model(ROW0[:2], risk={Pos(2, 0): 0.9})            # tolerant mode can detour through row 1
    rp = plan_routes(Pos(0, 0), D.E, {Pos(3, 0)}, m, PlanningMode.STRICT, 0.3, False)
    assert not rp.best.found
    alt = dict(rp.alternatives)["risk-tolerant"]
    assert alt.found and Pos(2, 0) not in alt.cells


def test_astar_is_optimal_with_turns():
    m = model(all_cells(8))
    r = plan(Pos(0, 0), D.N, {Pos(3, 3)}, m, PlanningMode.STRICT, 0.3, False)
    assert r.cost == 7          # 6 moves + exactly 1 turn


# ------------------------------------------------------------------ dynamic replanning in the agent
def test_agent_replans_when_path_risk_rises():
    from wumpus.agents.explorer import ExplorerAgent, Plan
    from wumpus.config import GameConfig
    from wumpus.core.percepts import Percept
    from wumpus.environment.engine import GameEngine
    a = ExplorerAgent(GameEngine(GameConfig(seed=1)).public_info())
    a.observe(Percept(turn=1, step=0, position=Pos(0, 0), facing=D.N))
    cells = [Pos(0, 1), Pos(0, 2)]
    a.plan = Plan(Pos(1, 2), "explore (1,2)", [FORWARD, FORWARD], cells, [D.N, D.N], 2.0,
                  [a.cell_risk(c) for c in cells])
    assert a._plan_invalid_reason(Pos(1, 2)) is None
    a.pit_prob[Pos(0, 2)] = 0.4                            # new evidence makes a path cell risky
    a.statuses.pop(Pos(0, 2), None)
    reason = a._plan_invalid_reason(Pos(1, 2))
    assert reason and reason.startswith("risk of (0,2) rose")
    a._ensure_plan(Pos(1, 2), "explore (1,2)")
    rp = a.log.tagged("REPLAN")[-1]
    assert rp.summary.startswith("risk of (0,2) rose") and a.log.get(rp.refs[0]).tag == "A*_PLANNER"
