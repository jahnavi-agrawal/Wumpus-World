import pytest

from wumpus.config import GameConfig
from wumpus.control.wumpus_controllers import HumanWumpusController, ScriptedWumpusController
from wumpus.core.actions import CLIMB, GRAB, SHUFFLE, STAY, ExplorerAction, WumpusAction
from wumpus.core.geometry import Direction as D, Pos
from wumpus.core.percepts import EventType
from wumpus.env.engine import GameEngine, Outcome, Phase
from wumpus.env.mechanics import IllegalAction
from tests.helpers import make_engine


def test_turn_sequence_alternates():
    e = make_engine()
    assert e.phase is Phase.EXPLORER and e.turn == 1
    with pytest.raises(IllegalAction):
        e.apply_wumpus_action(STAY)
    e.apply_explorer_action(ExplorerAction.move(D.N))
    assert e.phase is Phase.WUMPUS
    e.apply_wumpus_action(STAY)
    assert e.phase is Phase.EXPLORER and e.turn == 2


def test_collapse_event_delivered_once():
    e = make_engine(wumpus=(4, 4))
    e.apply_explorer_action(ExplorerAction.move(D.N))
    e.apply_wumpus_action(WumpusAction.collapse(Pos(4, 3)))
    p = e.percept()
    assert [(ev.kind, ev.pos) for ev in p.events] == [(EventType.COLLAPSE, Pos(4, 3))]
    e.apply_explorer_action(ExplorerAction.move(D.S))
    e.apply_wumpus_action(STAY)
    assert e.percept().events == ()


def test_shuffle_event():
    e = make_engine(pits=[(5, 5)])
    e.apply_explorer_action(ExplorerAction.move(D.N))
    e.apply_wumpus_action(SHUFFLE)
    assert e.percept().events[0].kind is EventType.SHUFFLE


def test_scream_and_dead_wumpus_skips_phase():
    e = make_engine(wumpus=(6, 0))
    e.apply_explorer_action(ExplorerAction.shoot(D.E))
    assert e.phase is Phase.EXPLORER and e.turn == 2
    assert e.percept().scream
    e.apply_explorer_action(ExplorerAction.move(D.N))
    assert not e.percept().scream


def test_win_draw_loss():
    e = make_engine(gold=(0, 0))
    e.apply_explorer_action(GRAB)
    e.apply_wumpus_action(STAY)
    e.apply_explorer_action(CLIMB)
    assert e.result.outcome is Outcome.EXPLORER_WIN and e.phase is Phase.OVER
    e = make_engine()
    e.apply_explorer_action(CLIMB)
    assert e.result.outcome is Outcome.DRAW
    e = make_engine(pits=[(2, 0)], explorer=(1, 0), visited=[(0, 0), (1, 0)])
    e.apply_explorer_action(ExplorerAction.move(D.E))
    assert e.result.outcome is Outcome.WUMPUS_WIN


def test_turn_limit():
    e = make_engine(GameConfig(max_turns=3))
    for _ in range(3):
        e.apply_explorer_action(ExplorerAction.move(D.N) if e.turn % 2 else ExplorerAction.move(D.S))
        e.apply_wumpus_action(STAY)
    assert e.result.outcome is Outcome.WUMPUS_WIN and "turn limit" in e.result.reason


def test_percept_contains_no_hidden_layout():
    p = make_engine().percept()
    hidden = {"pits", "gold", "wumpus", "world"}
    assert not hidden & set(vars(p))


def test_human_controller_rejects_illegal():
    e = make_engine(pits=[(6, 0)], wumpus=(7, 0))
    e.apply_explorer_action(ExplorerAction.move(D.N))
    h = HumanWumpusController()
    assert h.choose(e.world, e.legal_wumpus_actions()) is None
    h.submit(WumpusAction.move(D.W))          # into a pit: illegal
    assert h.choose(e.world, e.legal_wumpus_actions()) is None
    h.submit(WumpusAction.move(D.N))
    assert h.choose(e.world, e.legal_wumpus_actions()) == WumpusAction.move(D.N)


def test_hunter_approaches():
    e = make_engine(wumpus=(5, 0))
    e.apply_explorer_action(ExplorerAction.move(D.E))
    a = ScriptedWumpusController("hunter", 0).choose(e.world, e.legal_wumpus_actions())
    assert a == WumpusAction.move(D.W)


def test_engine_seed_reproducible():
    a, b = GameEngine(GameConfig(seed=5)), GameEngine(GameConfig(seed=5))
    assert a.world.pits == b.world.pits and a.world.wumpus == b.world.wumpus
