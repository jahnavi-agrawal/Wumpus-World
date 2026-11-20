"""Integration tests: real engine + real explorer on fixed dungeons (evaluation/scenarios.py)."""
import json

import pytest

from evaluation.scenarios import SCENARIOS, run_scenario, world
from wumpus.agents.explorer import ExplorerAgent
from wumpus.config import GameConfig
from wumpus.core.actions import FORWARD, JUMP, TURN_LEFT
from wumpus.core.geometry import Direction as D, Pos
from wumpus.core.percepts import Percept
from wumpus.environment.engine import GameEngine


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_scenario(scenario):
    r = run_scenario(scenario)
    failed = [d for d, ok in r.checks if not ok]
    assert not failed, f"{scenario.name}: {failed}"


def test_at_least_ten_distinct_scenarios():
    names = {s.name for s in SCENARIOS}
    assert len(names) >= 10
    assert {"safe_exploration", "logical_pit_deduction", "ambiguous_hazards", "moving_wumpus",
            "teleporter_discovery", "collapsing_route", "shuffle_event", "jump_opportunity",
            "adversarial_trap", "gold_retrieval_and_escape"} <= names


def test_scenarios_are_deterministic():
    a, b = run_scenario(SCENARIOS[1]), run_scenario(SCENARIOS[1])
    assert [str(e) for e in a.engine.events] == [str(e) for e in b.engine.events]


def test_every_decision_has_a_complete_report():
    r = run_scenario([s for s in SCENARIOS if s.name == "adversarial_trap"][0])
    reports = r.agent.reports
    assert len(reports) == len(r.agent.log.tagged("DECISION"))
    for rep in reports:
        assert rep.observations["position"] and rep.selected and rep.explanation
        assert {a for a, _ in rep.candidates} == {"FORWARD", "JUMP", "TURN_LEFT", "TURN_RIGHT", "GRAB",
                                                    "SHOOT", "CLIMB"}
        json.dumps({"obs": rep.observations, "facts": rep.new_facts, "hazards": dict(rep.hazards),
                    "astar": rep.astar, "minimax": rep.minimax})
    assert any(rep.minimax for rep in reports)            # Minimax evaluation recorded when it ran
    assert any(rep.astar for rep in reports)              # A* path recorded
    assert any(rep.new_facts for rep in reports)          # newly inferred facts recorded
    shot = [rep for rep in reports if rep.selected == "SHOOT"][0]
    assert "P(hit)" in dict(shot.candidates)["SHOOT"]


def _agent():
    return ExplorerAgent(GameEngine(GameConfig(seed=1)).public_info())


def P(t, pos, facing=D.E, **kw):
    return Percept(turn=t, step=t - 1, position=Pos(*pos), facing=facing, **kw)


def test_guard_never_enters_a_confirmed_hazard():
    a = _agent()
    a.observe(P(1, (0, 0)))
    a.last_action = FORWARD
    a.observe(P(2, (1, 0), unstable=True))
    a.last_action = FORWARD
    a.observe(P(3, (2, 0)))                                # (1,0) collapsed behind us
    a.facing = D.W
    assert a._guard(FORWARD) is TURN_LEFT                  # would step into the collapsed tile
    assert any("guard: refused FORWARD into (1,0)" in e.summary for e in a.log.tagged("WARNING"))
    assert a._guard(JUMP) is JUMP                          # a jump into the wall is a bump, not a hazard


def test_candidate_actions_explain_each_option():
    a = _agent()
    a.observe(P(1, (0, 0)))
    c = dict(a.candidate_actions())
    assert "proven pit-free" in c["FORWARD"] and "P(death) 0%" in c["FORWARD"]
    assert c["CLIMB"].startswith("exit") and c["GRAB"] == "nothing to grab"
    assert "P(hit)" in c["SHOOT"]


def test_escape_route_counts_jump_only_if_not_spent_getting_there():
    a = _agent()
    a.observe(P(1, (0, 0)))
    a.last_action = FORWARD
    a.observe(P(2, (1, 0)))
    a.last_action = FORWARD
    a.observe(P(3, (2, 0), unstable=True))
    walk, jump = a._exit_connected(use_jump=False), a._exit_connected()
    assert Pos(3, 0) not in walk                           # (2,0) will collapse: no walking route back
    assert Pos(3, 0) in jump                               # but the jump from (3,0) lands on (0,0)
