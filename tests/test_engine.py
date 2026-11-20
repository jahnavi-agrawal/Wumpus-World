import dataclasses

import pytest

from wumpus.config import GameConfig
from wumpus.control.wumpus_controllers import HumanWumpusController, ScriptedWumpusController
from wumpus.core.actions import (CLIMB, FORWARD, GRAB, JUMP, SHOOT, STAY, TURN_LEFT, TURN_RIGHT,
                                 WumpusAction)
from wumpus.core.geometry import Direction as D, Pos
from wumpus.core.percepts import Percept
from wumpus.environment.engine import GameEngine, Outcome, Phase
from wumpus.environment.events import EventKind as K
from wumpus.environment.mechanics import IllegalAction
from tests.helpers import make_engine


def test_turn_sequence_alternates_and_status():
    e = make_engine()
    s = e.status()
    assert s.current_player == "explorer" and s.turn == 1 and s.step_count == 0 and not s.game_over
    with pytest.raises(IllegalAction):
        e.apply_wumpus_action(STAY)
    e.apply_explorer_action(FORWARD)
    assert e.status().current_player == "wumpus"
    with pytest.raises(IllegalAction):
        e.apply_explorer_action(FORWARD)
    e.apply_wumpus_action(STAY)
    s = e.status()
    assert s.current_player == "explorer" and s.turn == 2 and s.step_count == 1 and s.score == -1


def test_percepts_breeze_stench_glitter_portal_unstable():
    e = make_engine(pits=[(2, 1)], wumpus=(1, 2), gold=(1, 1), portals=[((0, 1 + 1), (6, 6))],
                    unstable=[(1, 1)], explorer=(1, 1), visited=[(0, 0), (1, 1)])
    p = e.percept()
    assert p.breeze and p.stench and p.glitter and p.unstable and not p.portal_signal
    e = make_engine(portals=[((0, 2), (6, 6))], explorer=(0, 1), visited=[(0, 0), (0, 1)])
    assert e.percept().portal_signal


def test_bump_teleport_shuffle_scream_flags_last_one_turn():
    e = make_engine(facing=D.W)
    e.apply_explorer_action(FORWARD)
    e.apply_wumpus_action(STAY)
    assert e.percept().bump
    e.apply_explorer_action(TURN_RIGHT)
    e.apply_wumpus_action(STAY)
    assert not e.percept().bump
    e = make_engine(portals=[((1, 0), (5, 5))])
    e.apply_explorer_action(FORWARD)
    e.apply_wumpus_action(STAY)
    p = e.percept()
    assert p.teleported and p.position == Pos(5, 5)
    e = make_engine(shuffle=[(1, 0)], pits=[(6, 6)])
    e.apply_explorer_action(FORWARD)
    e.apply_wumpus_action(STAY)
    assert e.percept().shuffled
    e = make_engine(wumpus=(5, 0))
    e.apply_explorer_action(SHOOT)
    assert e.phase is Phase.EXPLORER and e.turn == 2       # dead Wumpus skips its turn
    assert e.percept().scream


def test_scoring_win_with_gold():
    e = make_engine(gold=(1, 0))
    for a in (FORWARD, GRAB, TURN_LEFT, TURN_LEFT, FORWARD):
        e.apply_explorer_action(a)
        e.apply_wumpus_action(STAY)
    e.apply_explorer_action(CLIMB)
    s = e.status()
    assert s.game_over and s.result.outcome is Outcome.EXPLORER_WIN
    assert s.score == -6 + 1000 and s.inventory["gold"]
    assert e.events[-1].kind is K.GAME_OVER and e.events[-1].data["score"] == 994


def test_shoot_costs_ten_extra():
    e = make_engine(wumpus=(5, 3))
    e.apply_explorer_action(SHOOT)
    assert e.score == -11 and e.status().arrows == 0


def test_draw_and_death_scores():
    e = make_engine()
    e.apply_explorer_action(CLIMB)
    assert e.result.outcome is Outcome.DRAW and e.score == -1
    e = make_engine(pits=[(1, 0)])
    e.apply_explorer_action(FORWARD)
    assert e.result.outcome is Outcome.WUMPUS_WIN and e.score == -1001
    e = make_engine(wumpus=(2, 0))
    e.apply_explorer_action(FORWARD)
    e.apply_wumpus_action(WumpusAction.move(D.W))
    assert e.result.outcome is Outcome.WUMPUS_WIN and e.result.reason == "EATEN"


def test_escape_after_collapsed_return_path_needs_alternative():
    e = make_engine(unstable=[(1, 0)], gold=(2, 0))
    e.apply_explorer_action(FORWARD); e.apply_wumpus_action(STAY)
    e.apply_explorer_action(FORWARD); e.apply_wumpus_action(STAY)
    p = e.percept()
    assert p.glitter and p.breeze             # the collapsed tile behind us is now a pit
    e.apply_explorer_action(GRAB); e.apply_wumpus_action(STAY)
    for a in (TURN_LEFT, FORWARD, TURN_LEFT, FORWARD, FORWARD, TURN_LEFT, FORWARD):
        e.apply_explorer_action(a); e.apply_wumpus_action(STAY)
    assert e.world.explorer == Pos(0, 0)
    e.apply_explorer_action(CLIMB)
    assert e.result.outcome is Outcome.EXPLORER_WIN


def test_turn_limit():
    e = make_engine(GameConfig(max_turns=3))
    for _ in range(3):
        e.apply_explorer_action(TURN_LEFT)
        if e.phase is Phase.WUMPUS:
            e.apply_wumpus_action(STAY)
    assert e.result.outcome is Outcome.WUMPUS_WIN and "turn limit" in e.result.reason


def test_percept_contains_no_hidden_layout():
    fields = {f.name for f in dataclasses.fields(Percept)}
    hidden = {"pits", "gold", "wumpus", "portals", "shuffle_tiles", "collapsed", "world", "route", "explorer"}
    assert not hidden & fields
    p = GameEngine(GameConfig(seed=2)).percept()
    assert all(isinstance(getattr(p, f), (bool, int, tuple, D)) for f in fields)


def test_human_controller_rejects_illegal():
    e = make_engine(pits=[(6, 0)], wumpus=(7, 0))
    e.apply_explorer_action(TURN_LEFT)
    h = HumanWumpusController()
    assert h.choose(e.world, e.legal_wumpus_actions()) is None
    h.submit(WumpusAction.move(D.W))          # into a pit: illegal
    assert h.choose(e.world, e.legal_wumpus_actions()) is None
    h.submit(WumpusAction.move(D.N))
    assert h.choose(e.world, e.legal_wumpus_actions()) == WumpusAction.move(D.N)


def test_hunter_approaches():
    e = make_engine(wumpus=(5, 0))
    e.apply_explorer_action(FORWARD)
    a = ScriptedWumpusController("hunter", 0).choose(e.world, e.legal_wumpus_actions())
    assert a == WumpusAction.move(D.W)


def test_engine_seed_reproducible():
    a, b = GameEngine(GameConfig(seed=5)), GameEngine(GameConfig(seed=5))
    assert a.world.pits == b.world.pits and a.world.portals == b.world.portals
