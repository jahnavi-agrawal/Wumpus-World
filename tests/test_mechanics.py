"""One test per state-transition rule of RULES.md."""
import random

import pytest

from wumpus.config import GameConfig
from wumpus.core.actions import CLIMB, GRAB, SHUFFLE, STAY, ExplorerAction, WumpusAction
from wumpus.core.geometry import Direction as D, Pos
from wumpus.env import mechanics as M
from tests.helpers import make_world

CFG = GameConfig()


def test_move_marks_visited():
    w = make_world()
    M.apply_explorer_action(w, ExplorerAction.move(D.N))
    assert w.explorer == Pos(0, 1) and Pos(0, 1) in w.visited and w.explorer_alive


def test_move_out_of_bounds_illegal():
    with pytest.raises(M.IllegalAction):
        M.apply_explorer_action(make_world(), ExplorerAction.move(D.S))


def test_fall_into_pit():
    w = make_world(pits=[(0, 2)], explorer=(0, 1))
    M.apply_explorer_action(w, ExplorerAction.move(D.N))
    assert not w.explorer_alive


def test_jump_skips_middle_pit_and_consumes_charge():
    w = make_world(pits=[(0, 2)], explorer=(0, 1))
    msgs, _ = M.apply_explorer_action(w, ExplorerAction.jump(D.N))
    assert w.explorer == Pos(0, 3) and w.explorer_alive and w.jumps_left == 1
    assert Pos(0, 2) not in w.visited


def test_jump_requires_charges():
    w = make_world(jumps=0)
    assert ExplorerAction.jump(D.N) not in M.legal_explorer_actions(w)


def test_jump_onto_pit_kills():
    w = make_world(pits=[(0, 3)], explorer=(0, 1))
    M.apply_explorer_action(w, ExplorerAction.jump(D.N))
    assert not w.explorer_alive


def test_teleport_pad_explorer_and_no_chaining():
    w = make_world(pads=[(0, 2), (5, 5)], explorer=(0, 1))
    msgs, _ = M.apply_explorer_action(w, ExplorerAction.move(D.N))
    assert w.explorer == Pos(5, 5) and Pos(5, 5) in w.visited
    assert any("teleported" in m for m in msgs)


def test_teleport_onto_wumpus_is_death():
    w = make_world(pads=[(0, 2), (5, 5)], explorer=(0, 1), wumpus=(5, 5))
    M.apply_explorer_action(w, ExplorerAction.move(D.N))
    assert not w.explorer_alive


def test_wumpus_uses_pad():
    w = make_world(pads=[(4, 4), (1, 6)], wumpus=(4, 3))
    M.apply_wumpus_action(w, WumpusAction.move(D.N), CFG, random.Random(0))
    assert w.wumpus == Pos(1, 6)


def test_shoot_hit_and_miss():
    w = make_world(wumpus=(5, 0))
    _, scream = M.apply_explorer_action(w, ExplorerAction.shoot(D.E))
    assert scream and not w.wumpus_alive and w.arrows == 0
    w = make_world(wumpus=(5, 1))
    _, scream = M.apply_explorer_action(w, ExplorerAction.shoot(D.E))
    assert not scream and w.wumpus_alive
    assert all(a.kind.value != "SHOOT" for a in M.legal_explorer_actions(w))


def test_arrow_flies_over_pits():
    w = make_world(pits=[(2, 0)], wumpus=(4, 0))
    _, scream = M.apply_explorer_action(w, ExplorerAction.shoot(D.E))
    assert scream


def test_grab_and_climb_legality():
    w = make_world(gold=(0, 0))
    assert GRAB in M.legal_explorer_actions(w) and CLIMB in M.legal_explorer_actions(w)
    M.apply_explorer_action(w, GRAB)
    assert w.has_gold and w.gold is None
    w = make_world(explorer=(1, 0))
    assert CLIMB not in M.legal_explorer_actions(w)


def test_walk_into_wumpus():
    w = make_world(wumpus=(1, 0))
    M.apply_explorer_action(w, ExplorerAction.move(D.E))
    assert not w.explorer_alive


def test_wumpus_cannot_enter_pit_or_collapsed():
    w = make_world(pits=[(6, 0)], wumpus=(7, 0))
    w.collapsed.add(Pos(7, 1))
    legal = M.legal_wumpus_actions(w, CFG)
    assert WumpusAction.move(D.W) not in legal and WumpusAction.move(D.N) not in legal
    assert STAY in legal


def test_wumpus_eats_explorer():
    w = make_world(wumpus=(2, 0), explorer=(1, 0))
    M.apply_wumpus_action(w, WumpusAction.move(D.W), CFG, random.Random(0))
    assert not w.explorer_alive


def test_collapse_rules():
    w = make_world(wumpus=(4, 4), pits=[(4, 5)], gold=(5, 4), pads=[(3, 4), (7, 7)], explorer=(4, 2))
    ok = lambda c: M.is_valid_collapse_target(w, Pos(*c), CFG)
    assert ok((4, 3)) and ok((2, 4)) and ok((6, 4))
    assert not ok((4, 5))      # pit
    assert not ok((5, 4))      # gold
    assert not ok((3, 4))      # pad
    assert not ok((4, 2))      # explorer
    assert not ok((4, 4))      # wumpus itself
    assert not ok((7, 4))      # out of range (distance 3)
    M.apply_wumpus_action(w, WumpusAction.collapse(Pos(4, 3)), CFG, random.Random(0))
    assert Pos(4, 3) in w.collapsed and w.collapse_charges == 2 and w.wumpus == Pos(4, 4)
    assert w.collapse_cooldown == CFG.collapse_cooldown
    assert not any(a.kind.value == "COLLAPSE" for a in M.legal_wumpus_actions(w, CFG))
    for _ in range(CFG.collapse_cooldown):
        M.apply_wumpus_action(w, STAY, CFG, random.Random(0))
    assert any(a.kind.value == "COLLAPSE" for a in M.legal_wumpus_actions(w, CFG))


def test_start_cell_never_collapses():
    w = make_world(wumpus=(1, 1), explorer=(3, 3), visited=[(0, 0), (3, 3)])
    assert not M.is_valid_collapse_target(w, Pos(0, 0), CFG)


def test_collapsed_tile_kills_explorer():
    w = make_world()
    w.collapsed.add(Pos(1, 0))
    M.apply_explorer_action(w, ExplorerAction.move(D.E))
    assert not w.explorer_alive


def test_shuffle_preserves_count_and_avoids_visited():
    for seed in range(20):
        w = make_world(pits=[(3, 3), (4, 4), (5, 5)], visited=[(0, 0), (0, 1), (0, 2), (1, 2)],
                       explorer=(1, 2), wumpus=(6, 6), pads=[(2, 5), (6, 1)])
        w.collapsed.add(Pos(2, 2))
        M.apply_wumpus_action(w, SHUFFLE, CFG, random.Random(seed))
        assert len(w.pits) == 3 and w.shuffle_charges == 0
        forbidden = w.visited | {w.explorer, w.wumpus, w.gold, Pos(2, 2), Pos(2, 5), Pos(6, 1),
                                 Pos(0, 0), Pos(1, 0), Pos(0, 1)}
        assert not (w.pits & forbidden)
    assert SHUFFLE not in M.legal_wumpus_actions(w, CFG)


def test_dead_wumpus_has_no_actions():
    w = make_world()
    w.wumpus_alive = False
    assert M.legal_wumpus_actions(w, CFG) == []


def test_percepts():
    w = make_world(pits=[(2, 1)], wumpus=(1, 2), explorer=(1, 1))
    assert M.breeze_at(w, Pos(1, 1)) and M.stench_at(w, Pos(1, 1))
    assert not M.breeze_at(w, Pos(0, 0)) and not M.stench_at(w, Pos(0, 0))
    w.collapsed.add(Pos(0, 0))
    assert not M.breeze_at(w, Pos(0, 1))   # collapsed tiles do not cause breeze
