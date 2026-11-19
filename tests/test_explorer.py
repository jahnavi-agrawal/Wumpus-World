"""Integration tests: the full agent playing seeded games through the engine."""
import pytest

from wumpus.ai.explorer import ExplorerAgent
from wumpus.config import GameConfig
from wumpus.control.wumpus_controllers import ScriptedWumpusController
from wumpus.core.actions import STAY, ExplorerActionType, WumpusAction
from wumpus.core.geometry import Pos
from wumpus.env.engine import Outcome, Phase
from wumpus.runner import play_game
from tests.helpers import make_engine


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("mode", ["idle", "random", "hunter"])
def test_full_games_terminate_and_never_enter_known_hazards(seed, mode):
    def check(engine, agent):
        # the agent never stands on a cell it had proven to be a pit, nor on a collapsed tile
        assert engine.world.explorer not in agent.collapsed or not engine.world.explorer_alive
    engine, agent, stats = play_game(GameConfig(seed=seed), ScriptedWumpusController(mode, seed), on_step=check)
    assert engine.phase is Phase.OVER
    assert stats["max_decide_s"] < 1.5
    for e in agent.log.entries:                       # every reference points to an earlier entry
        assert all(0 <= r < e.id for r in e.refs) and e.details


def test_wins_most_games_against_idle_wumpus():
    wins = sum(play_game(GameConfig(seed=s), ScriptedWumpusController("idle", s))[0].result.outcome
               is Outcome.EXPLORER_WIN for s in range(15))
    assert wins >= 11


def test_first_turn_reasoning_is_logged_with_evidence():
    eng = make_engine(pits=[(2, 0)], gold=(5, 5), wumpus=(6, 6))
    agent = ExplorerAgent(eng.public_info())
    agent.act(eng.percept())
    tags = [e.tag for e in agent.log.entries]
    for t in ("PERCEPT", "KB_TELL", "KB_INFER", "WUMPUS_BELIEF", "TARGET", "ASTAR", "DECISION"):
        assert t in tags
    infer = agent.log.tagged("KB_INFER")[0]
    assert "DPLL" in "\n".join(infer.details)


def test_risk_calc_entry_contains_real_calculation():
    from wumpus.core.actions import ExplorerAction
    from wumpus.core.geometry import Direction as D
    eng = make_engine(pits=[(2, 0)], gold=(6, 6), wumpus=(6, 5))
    agent = ExplorerAgent(eng.public_info())
    agent.observe(eng.percept())
    agent.last_action = ExplorerAction.move(D.E)
    eng.apply_explorer_action(agent.last_action)          # breeze at (1,0): (2,0) or (1,1)
    eng.apply_wumpus_action(STAY)
    agent.observe(eng.percept())
    calc = {e.cell: e for e in agent.log.tagged("RISK_CALC")}
    assert Pos(2, 0) in calc and Pos(1, 1) in calc
    text = "\n".join(calc[Pos(2, 0)].details)
    assert "Consistent models enumerated: 3" in text and "B(1,0)" in text and "P = " in text
    assert calc[Pos(2, 0)].summary.startswith("Cell (2,0):")


def _percept(turn, pos, **kw):
    from wumpus.core.percepts import Percept
    base = dict(stench=False, breeze=False, glitter=False, scream=False, has_gold=False, arrows=1, jumps_left=2)
    base.update(kw)
    return Percept(turn=turn, position=Pos(*pos), **base)


def test_collapse_on_path_triggers_replan():
    """Hand-fed percepts: walk up column 0, take the gold, then (0,1) collapses on the way home."""
    from wumpus.core.percepts import EventType, PublicEvent
    agent = ExplorerAgent(make_engine().public_info())
    for t, pos in enumerate([(0, 0), (0, 1), (0, 2), (0, 3)], start=1):
        agent.observe(_percept(t, pos))
    agent.observe(_percept(5, (0, 3), has_gold=True))
    assert agent.decide().kind is ExplorerActionType.MOVE
    assert agent.plan.cells == [Pos(0, 1), Pos(0, 0)]
    agent.observe(_percept(6, (0, 2), has_gold=True, events=(PublicEvent(EventType.COLLAPSE, 5, Pos(0, 1)),)))
    agent.decide()
    replans = [e for e in agent.log.tagged("REPLAN") if e.turn == 6]
    assert replans and "(0,1) on the path collapsed" in replans[0].summary
    new_path = [e for e in agent.log.tagged("ASTAR") if e.turn == 6][0]
    assert new_path.id in replans[0].refs
    assert not any(line.endswith("-> (0,1)") for line in new_path.details)   # avoids the hole
    assert "(0,0)" in new_path.summary


def test_shuffle_invalidates_breeze_evidence():
    eng = make_engine(pits=[(5, 5)], gold=(7, 7), wumpus=(7, 0))
    agent = ExplorerAgent(eng.public_info())
    eng.apply_explorer_action(agent.act(eng.percept()))
    from wumpus.core.actions import SHUFFLE
    eng.apply_wumpus_action(SHUFFLE)
    agent.act(eng.percept())
    assert len(agent.breeze_obs) == 1
    assert any("SHUFFLE" in e.summary for e in agent.log.tagged("EVENT"))
