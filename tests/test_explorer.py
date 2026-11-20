"""Integration tests: the full agent, through the engine and with hand-fed percepts."""
import pytest

from wumpus.agents.explorer import ExplorerAgent
from wumpus.config import GameConfig
from wumpus.control.wumpus_controllers import ScriptedWumpusController
from wumpus.core.actions import FORWARD, JUMP, STAY, TURN_LEFT, TURN_RIGHT
from wumpus.core.geometry import Direction as D, Pos
from wumpus.core.percepts import Percept
from wumpus.environment.engine import Outcome, Phase
from wumpus.runner import play_game
from tests.helpers import make_engine


def percept(turn, pos, facing=D.E, **kw):
    return Percept(turn=turn, step=turn - 1, position=Pos(*pos), facing=facing, **kw)


def info():
    return make_engine().public_info()


# --------------------------------------------------------------- full games
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("mode", ["idle", "random", "hunter"])
def test_full_games_terminate_with_traceable_log(seed, mode):
    engine, agent, stats = play_game(GameConfig(seed=seed), ScriptedWumpusController(mode, seed))
    assert engine.phase is Phase.OVER
    assert stats["max_decide_s"] < 2.0
    for e in agent.log.entries:                     # every reference points to an earlier entry
        assert all(0 <= r < e.id for r in e.refs) and e.details


def test_wins_majority_against_idle_wumpus():
    outcomes = [play_game(GameConfig(seed=s), ScriptedWumpusController("idle", s))[0].result.outcome
                for s in range(20)]
    assert sum(o is Outcome.EXPLORER_WIN for o in outcomes) >= 10
    assert sum(o is Outcome.WUMPUS_WIN for o in outcomes) <= 10


# --------------------------------------------------------------- traceability
def test_first_turn_reasoning_is_logged_with_evidence():
    eng = make_engine(pits=[(3, 0)], gold=(5, 5), wumpus=(6, 6))
    agent = ExplorerAgent(eng.public_info())
    agent.act(eng.percept())
    tags = {e.tag for e in agent.log.entries}
    assert {"PERCEPT", "KB_TELL", "INFERENCE", "WUMPUS_BELIEF", "TARGET", "A*_PLANNER", "DECISION"} <= tags
    assert "DPLL" in "\n".join(agent.log.tagged("INFERENCE")[0].details)


def test_risk_calc_entry_contains_real_calculation():
    eng = make_engine(pits=[(2, 0)], gold=(6, 6), wumpus=(6, 5))
    agent = ExplorerAgent(eng.public_info())
    eng.apply_explorer_action(agent.act(eng.percept()))      # FORWARD -> (1,0): breeze
    eng.apply_wumpus_action(STAY)
    agent.observe(eng.percept())
    calc = {e.cell: e for e in agent.log.tagged("RISK_CALC")}
    assert Pos(2, 0) in calc and Pos(1, 1) in calc
    text = "\n".join(calc[Pos(2, 0)].details)
    assert "Consistent models enumerated: 3" in text and "B(1,0)" in text and "P = " in text
    assert calc[Pos(2, 0)].summary.startswith("Cell (2,0):")


# --------------------------------------------------------------- v2 mechanics knowledge
def test_unstable_floor_becomes_known_pit_and_explains_breeze():
    a = ExplorerAgent(info())
    a.observe(percept(1, (0, 0)))
    a.last_action = FORWARD
    a.observe(percept(2, (1, 0), unstable=True))
    assert a.unstable_here == Pos(1, 0)
    a.last_action = FORWARD
    a.observe(percept(3, (2, 0), breeze=True))               # breeze caused by the collapsed tile
    assert a.collapsed == {Pos(1, 0): 3} and a.pit_prob[Pos(1, 0)] == 1.0
    assert Pos(2, 0) not in a.valid_breeze()                 # explained -> no pit inference from it
    assert any("collapsed behind us" in e.summary for e in a.log.tagged("EVENT"))
    assert a.cell_risk(Pos(1, 0)) == 1.0


def test_teleport_learns_portal_pair():
    a = ExplorerAgent(info())
    a.observe(percept(1, (0, 0)))
    a.last_action = FORWARD
    a.observe(percept(2, (1, 0)))
    a.last_action, a.last_pos, a.last_facing = FORWARD, Pos(1, 0), D.E
    a.observe(percept(3, (5, 5), teleported=True))           # stepped onto (2,0)
    assert a.portals == {Pos(2, 0): Pos(5, 5), Pos(5, 5): Pos(2, 0)}
    assert a.tracker.portals == a.portals
    assert a.dkb.classify_portals([Pos(2, 0)])[Pos(2, 0)].status == "portal"


def test_portal_signal_inference():
    a = ExplorerAgent(info())
    a.observe(percept(1, (0, 0)))
    a.last_action = FORWARD
    a.observe(percept(2, (1, 0), portal_signal=True))   # neighbours (0,0) and (1,1) are in the start zone/
    st = a.dkb.classify_portals([Pos(2, 0), Pos(1, 1)])  # (1,1): unknown; (2,0): unknown -> ambiguous
    assert {s.status for s in st.values()} == {"unknown"}
    a.last_action, a.last_pos, a.last_facing = TURN_LEFT, Pos(1, 0), D.E
    a.observe(percept(3, (1, 0), facing=D.N, portal_signal=True))
    a.last_action = FORWARD
    a.observe(percept(4, (1, 1), facing=D.N))           # stood on (1,1): not a portal
    assert a.dkb.classify_portals([Pos(2, 0)])[Pos(2, 0)].status == "portal"


def test_shuffle_invalidates_breeze_evidence():
    a = ExplorerAgent(info())
    a.observe(percept(1, (0, 0)))
    a.last_action = FORWARD
    a.observe(percept(2, (1, 0), breeze=True))
    a.last_action = FORWARD
    a.observe(percept(3, (2, 0), shuffled=True))
    assert list(a.breeze_obs) == [Pos(2, 0)]
    assert any("SHUFFLE" in e.summary for e in a.log.tagged("EVENT"))


# --------------------------------------------------------------- replanning (incl. the 3 fixes)
def test_unexpected_teleport_triggers_replan():
    a = ExplorerAgent(info())
    for t, pos in enumerate([(0, 0), (1, 0), (2, 0)], start=1):
        a.last_action = FORWARD
        a.observe(percept(t, pos))
    a.observe(percept(4, (2, 0), facing=D.W, has_gold=True))
    assert a.decide() is FORWARD and a.plan.goal == Pos(0, 0)        # heading home along row 0
    a.observe(percept(5, (6, 6), facing=D.W, has_gold=True, teleported=True))   # (1,0) was a portal!
    a.decide()
    assert a.portals == {Pos(1, 0): Pos(6, 6), Pos(6, 6): Pos(1, 0)}
    assert any("unexpected state" in e.summary for e in a.log.tagged("REPLAN"))


def test_fix1_cut_off_from_exit_routes_through_unexplored_cells():
    """Collapsed tile behind us: the proven-safe network no longer reaches the exit."""
    a = ExplorerAgent(info())
    a.observe(percept(1, (0, 0), facing=D.N))
    a.last_action = FORWARD
    a.observe(percept(2, (0, 1), facing=D.N, unstable=True))
    a.last_action = FORWARD
    a.observe(percept(3, (0, 2), facing=D.N, has_gold=True))  # (0,1) collapsed behind us
    act = a.decide()
    path = [e for e in a.log.tagged("A*_PLANNER") if e.turn == 3][-1]
    assert "unexplored" in " ".join(e.summary for e in a.log.tagged("REPLAN")) or act is not TURN_LEFT
    assert a.plan is not None and a.plan.goal == Pos(0, 0) and Pos(0, 1) not in a.plan.cells
    assert "(0,0)" in path.summary


def test_fix2_minimax_does_not_dither_between_turns():
    """Idle Wumpus parked next to the corridor: no endless TURN_LEFT/TURN_RIGHT."""
    for seed in range(30):
        _, agent, _ = play_game(GameConfig(seed=seed), ScriptedWumpusController("idle", seed))
        d = [e.summary for e in agent.log.tagged("DECISION")]
        for i in range(len(d) - 8):
            window = d[i:i + 8]
            assert not all("TURN" in x and "Minimax" in x for x in window), (seed, window)


def test_fix3_tie_break_prefers_not_dying_immediately():
    from wumpus.agents.minimax import DuelGame, DuelState, determinized_decision
    from wumpus.core.geometry import all_cells, manhattan
    g = DuelGame(size=8, portals={}, blocked=frozenset(), pit_risk={},
                 goal_dist={c: manhattan(c, Pos(0, 0)) for c in all_cells(8)})
    s = DuelState(Pos(3, 3), D.N, Pos(3, 4), True, 0, False)
    r = determinized_decision(g, s, [(Pos(3, 4), 1.0)], depth=4)
    assert r.best_action is not FORWARD


def test_hunter_can_be_beaten_sometimes():
    wins = sum(play_game(GameConfig(seed=s), ScriptedWumpusController("hunter", s))[0].result.outcome
               is Outcome.EXPLORER_WIN for s in range(30))
    assert wins >= 1
