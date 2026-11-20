"""One or more tests per state-transition rule of RULES.md (v2)."""
import pytest

from wumpus.config import GameConfig
from wumpus.core.actions import (CLIMB, FORWARD, GRAB, JUMP, SHOOT, STAY, TURN_LEFT, TURN_RIGHT,
                                 WumpusAction)
from wumpus.core.geometry import Direction as D, Pos
from wumpus.environment import mechanics as M
from wumpus.environment.events import EventKind as K
from tests.helpers import make_world

CFG = GameConfig()


def act(w, a, step=1):
    return M.apply_explorer_action(w, a, 1, step, CFG)


def kinds(evs):
    return [e.kind for e in evs]


# ------------------------------------------------------------- movement
def test_forward_moves_in_facing_direction():
    w = make_world()
    evs, f = act(w, FORWARD)
    assert w.explorer == Pos(1, 0) and Pos(1, 0) in w.visited and kinds(evs) == [K.MOVED]


def test_turns_rotate_facing():
    w = make_world()
    act(w, TURN_LEFT)
    assert w.facing is D.N
    act(w, TURN_LEFT)
    assert w.facing is D.W
    act(w, TURN_RIGHT)
    act(w, TURN_RIGHT)
    act(w, TURN_RIGHT)
    assert w.facing is D.S
    evs, _ = act(w, TURN_RIGHT)
    assert evs[0].kind is K.TURNED and evs[0].data == {"frm": "DOWN", "to": "LEFT"}


def test_bump_into_wall():
    w = make_world(facing=D.W)
    evs, f = act(w, FORWARD)
    assert f.bump and w.explorer == Pos(0, 0) and kinds(evs) == [K.BUMPED]


# ------------------------------------------------------------- hazards
def test_fall_into_pit():
    w = make_world(pits=[(1, 0)])
    evs, _ = act(w, FORWARD)
    assert not w.explorer_alive and K.FELL_INTO_PIT in kinds(evs)


def test_walk_into_wumpus():
    w = make_world(wumpus=(1, 0))
    evs, _ = act(w, FORWARD)
    assert not w.explorer_alive and K.EATEN in kinds(evs)


# ------------------------------------------------------------- jump
def test_jump_lands_three_ahead_crossing_two():
    w = make_world(pits=[(1, 0), (2, 0)], wumpus=(2, 0 + 7))
    evs, _ = act(w, JUMP)
    assert w.explorer == Pos(3, 0) and w.explorer_alive and not w.jump_available
    jumped = [e for e in evs if e.kind is K.JUMPED][0]
    assert jumped.data["crossed"] == ((1, 0), (2, 0))
    assert Pos(1, 0) not in w.visited and Pos(2, 0) not in w.visited


def test_jump_over_wumpus_is_safe_but_landing_hazards_apply():
    w = make_world(wumpus=(1, 0))
    act(w, JUMP)
    assert w.explorer_alive and w.explorer == Pos(3, 0)
    w = make_world(pits=[(3, 0)])
    evs, _ = act(w, JUMP)
    assert not w.explorer_alive and K.FELL_INTO_PIT in kinds(evs)
    w = make_world(wumpus=(3, 0))
    act(w, JUMP)
    assert not w.explorer_alive


def test_jump_is_one_use_and_bumps_at_wall():
    w = make_world(explorer=(5, 0))
    evs, f = act(w, JUMP)
    assert f.bump and w.jump_available and w.explorer == Pos(5, 0)      # not consumed
    w = make_world()
    act(w, JUMP)
    evs, _ = act(w, JUMP)
    assert kinds(evs) == [K.JUMP_UNAVAILABLE] and w.explorer == Pos(3, 0)


def test_jump_skips_intermediate_portal_and_unstable():
    w = make_world(portals=[((1, 0), (5, 5))], unstable=[(2, 0)])
    act(w, JUMP)
    assert w.explorer == Pos(3, 0) and Pos(2, 0) in w.unstable


# ------------------------------------------------------------- teleport
def test_teleport_both_pairs_and_no_chaining():
    w = make_world(portals=[((1, 0), (5, 5)), ((5, 6), (2, 7))])
    evs, f = act(w, FORWARD)
    assert f.teleported and w.explorer == Pos(5, 5) and K.TELEPORTED in kinds(evs)
    w.facing = D.N
    act(w, FORWARD)                       # onto (5,6) -> second pair -> (2,7)
    assert w.explorer == Pos(2, 7)


def test_teleport_destination_hazard_is_evaluated():
    w = make_world(portals=[((1, 0), (5, 5))], wumpus=(5, 5))
    act(w, FORWARD)
    assert not w.explorer_alive


def test_wumpus_uses_portal():
    w = make_world(portals=[((6, 0), (3, 3))], wumpus=(7, 0))
    evs = M.apply_wumpus_action(w, WumpusAction.move(D.W), 1, 1)
    assert w.wumpus == Pos(3, 3) and K.TELEPORTED in kinds(evs)


# ------------------------------------------------------------- collapsing tiles
def test_unstable_tile_collapses_after_leaving():
    w = make_world(unstable=[(1, 0)])
    act(w, FORWARD)
    assert w.explorer_alive and Pos(1, 0) in w.unstable       # standing on it is fine
    evs, _ = act(w, FORWARD)
    assert Pos(1, 0) in w.pits and Pos(1, 0) in w.collapsed and K.TILE_COLLAPSED in kinds(evs)
    w.facing = D.W
    evs, _ = act(w, FORWARD)                                  # walking back = falling
    assert not w.explorer_alive


def test_turning_on_unstable_tile_does_not_collapse_it():
    w = make_world(unstable=[(1, 0)])
    act(w, FORWARD)
    act(w, TURN_LEFT)
    assert Pos(1, 0) in w.unstable and Pos(1, 0) not in w.pits


def test_jump_off_unstable_tile_collapses_it():
    w = make_world(unstable=[(1, 0)])
    act(w, FORWARD)
    act(w, JUMP)
    assert Pos(1, 0) in w.pits and w.explorer == Pos(4, 0)


def test_collapsed_tile_blocks_wumpus():
    w = make_world(unstable=[(6, 0)], wumpus=(7, 1))
    w.unstable.discard(Pos(6, 0)); w.pits.add(Pos(6, 0)); w.collapsed.add(Pos(6, 0))
    w.wumpus = Pos(7, 0)
    assert WumpusAction.move(D.W) not in M.legal_wumpus_actions(w)


# ------------------------------------------------------------- shuffle tiles
def test_shuffle_tile_swaps_one_pit_deterministically():
    def run():
        w = make_world(pits=[(5, 5), (6, 6)], shuffle=[(1, 0)], route=[(0, 0), (1, 0)], seed=9)
        evs, f = act(w, FORWARD, step=4)
        return w, evs, f
    w1, evs, f = run()
    w2, _, _ = run()
    assert f.shuffled and w1.pits == w2.pits and len(w1.pits) == 2
    ev = [e for e in evs if e.kind is K.CELLS_SHUFFLED][0]
    a, b = map(lambda t: Pos(*t), ev.data["swapped"])
    assert a in {Pos(5, 5), Pos(6, 6)} and b in w1.pits and a not in w1.pits
    assert Pos(1, 0) not in w1.shuffle_tiles and Pos(1, 0) in w1.used_shuffle   # one use
    w1.facing = D.W
    act(w1, FORWARD); w1.facing = D.E
    _, f2 = act(w1, FORWARD)
    assert not f2.shuffled


def test_shuffle_never_touches_visited_route_zone_or_specials():
    for seed in range(40):
        route = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
        w = make_world(pits=[(4, 4), (5, 5), (6, 1)], shuffle=[(2, 2)], route=route, explorer=(1, 2),
                       visited=[(0, 0), (0, 1), (0, 2), (1, 2), (3, 3)], unstable=[(4, 5)],
                       portals=[((6, 6), (7, 3))], gold=(7, 7), wumpus=(5, 0), seed=seed)
        before = set(w.pits)
        act(w, FORWARD)
        forbidden = {Pos(*c) for c in route} | w.visited | {Pos(4, 5), Pos(6, 6), Pos(7, 3), Pos(7, 7),
                                                           Pos(5, 0), Pos(1, 0)}
        assert len(w.pits) == 3 and not (w.pits - before) & forbidden


# ------------------------------------------------------------- gold, shooting, escape
def test_grab_gold_and_nothing_to_grab():
    w = make_world(gold=(1, 0))
    evs, _ = act(w, GRAB)
    assert kinds(evs) == [K.NOTHING_TO_GRAB] and not w.has_gold
    act(w, FORWARD)
    evs, _ = act(w, GRAB)
    assert kinds(evs) == [K.GRABBED_GOLD] and w.has_gold and w.gold is None


def test_shoot_along_facing_hit_and_miss():
    w = make_world(wumpus=(6, 0))
    evs, f = act(w, SHOOT)
    assert f.scream and not w.wumpus_alive and w.arrows == 0 and kinds(evs) == [K.ARROW_HIT]
    w = make_world(wumpus=(0, 5))                      # north of us, but we face east
    evs, f = act(w, SHOOT)
    assert not f.scream and w.wumpus_alive and kinds(evs) == [K.ARROW_MISSED]
    evs, _ = act(w, SHOOT)
    assert kinds(evs) == [K.NO_ARROWS]


def test_arrow_flies_over_pits():
    w = make_world(pits=[(2, 0)], wumpus=(4, 0))
    _, f = act(w, SHOOT)
    assert f.scream


def test_climb_only_at_start():
    w = make_world(explorer=(1, 0))
    evs, _ = act(w, CLIMB)
    assert kinds(evs) == [K.CANNOT_CLIMB]
    w = make_world()
    evs, _ = act(w, CLIMB)
    assert kinds(evs) == [K.CLIMBED_OUT] and evs[0].data["with_gold"] is False


# ------------------------------------------------------------- wumpus rules
def test_wumpus_legal_moves_walls_and_pits():
    w = make_world(pits=[(6, 0)], wumpus=(7, 0))
    legal = M.legal_wumpus_actions(w)
    assert legal == [WumpusAction.move(D.N), STAY]       # E is a wall, S is a wall, W is a pit


def test_wumpus_eats_explorer_and_illegal_rejected():
    w = make_world(wumpus=(2, 0), explorer=(1, 0))
    evs = M.apply_wumpus_action(w, WumpusAction.move(D.W), 1, 1)
    assert not w.explorer_alive and K.EATEN in kinds(evs)
    w = make_world(wumpus=(7, 0))
    with pytest.raises(M.IllegalAction):
        M.apply_wumpus_action(w, WumpusAction.move(D.E), 1, 1)


def test_dead_wumpus_has_no_actions():
    w = make_world()
    w.wumpus_alive = False
    assert M.legal_wumpus_actions(w) == []


def test_every_action_emits_structured_events():
    for a in (FORWARD, TURN_LEFT, TURN_RIGHT, JUMP, GRAB, SHOOT, CLIMB):
        evs, _ = act(make_world(), a)
        assert evs and all(e.actor and e.kind and isinstance(e.to_dict(), dict) for e in evs)
