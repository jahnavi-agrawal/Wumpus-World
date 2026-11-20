import random
from dataclasses import dataclass

import pytest

from wumpus.agents.minimax import (DuelGame, DuelState, SearchStats, alphabeta, determinized_decision,
                               minimax)
from wumpus.core.actions import FORWARD, JUMP, SHOOT, TURN_LEFT
from wumpus.core.geometry import Direction as D, Pos, all_cells, manhattan


@dataclass(frozen=True)
class TreeState:
    path: tuple = ()


class RandomTree:
    """Uniform random game tree with seeded leaf values."""

    def __init__(self, seed, b=3):
        self.seed, self.b = seed, b

    def is_max(self, s):
        return len(s.path) % 2 == 0

    def is_terminal(self, s):
        return False

    def actions(self, s):
        return list(range(self.b))

    def result(self, s, a):
        return TreeState(s.path + (a,))

    def evaluate(self, s):
        return random.Random(hash((self.seed, s.path))).randint(-100, 100)


@pytest.mark.parametrize("seed", range(20))
def test_alphabeta_equals_minimax_and_prunes(seed):
    g = RandomTree(seed, b=3 + seed % 3)
    s1, s2 = SearchStats(), SearchStats()
    assert alphabeta(g, TreeState(), 5, stats=s1) == minimax(g, TreeState(), 5, stats=s2)
    assert s1.nodes < s2.nodes and s1.cutoffs > 0


def test_textbook_tree():
    """AIMA Fig 5.2: MAX over three MIN nodes with leaves (3,12,8) (2,4,6) (14,5,2) -> 3."""
    leaves = [[3, 12, 8], [2, 4, 6], [14, 5, 2]]

    class G(RandomTree):
        def evaluate(self, s):
            return leaves[s.path[0]][s.path[1]]
    stats = SearchStats()
    assert alphabeta(G(0), TreeState(), 2, stats=stats) == 3
    assert stats.cutoffs >= 1   # second MIN node is cut after seeing 2


def duel(**kw):
    dist = {c: manhattan(c, Pos(0, 0)) for c in all_cells(8)}
    return DuelGame(size=8, portals={}, blocked=frozenset(kw.get("blocked", ())), pit_risk=kw.get("risk", {}),
                    goal_dist=dist)


def state(e, f, w, arrows=1, jump=False):
    return DuelState(explorer=Pos(*e), facing=f, wumpus=Pos(*w), wumpus_alive=True, arrows=arrows, jump=jump)


def test_shoots_wumpus_in_line_of_fire():
    r = determinized_decision(duel(), state((3, 3), D.N, (3, 4)), [(Pos(3, 4), 1.0)], depth=4)
    assert r.best_action is SHOOT and r.stats.cutoffs > 0


def test_never_walks_into_the_wumpus():
    r = determinized_decision(duel(), state((3, 3), D.N, (3, 4), arrows=0), [(Pos(3, 4), 1.0)], depth=4)
    assert r.best_action is not FORWARD and r.root_values[FORWARD] == -1000


def test_belief_weighted_shot_needs_facing():
    # 80% ahead (north), 20% east: facing north -> shoot
    r = determinized_decision(duel(), state((3, 3), D.N, (0, 0)), [(Pos(3, 4), 0.8), (Pos(4, 3), 0.2)], depth=2)
    assert r.best_action is SHOOT
    text = "\n".join(r.explain())
    assert "Hypothesis W at (3,4)" in text and "chosen" in text


def test_jump_escape_lands_three_ahead():
    g = duel()
    s = g.result(state((1, 3), D.E, (0, 3), arrows=0, jump=True), JUMP)
    assert s.explorer == Pos(4, 3) and not s.jump


def test_risky_cells_and_blocked_excluded():
    g = duel(blocked={Pos(3, 4)}, risk={Pos(3, 6): 0.9})
    acts = g.actions(state((3, 3), D.N, (7, 7), arrows=0, jump=True))
    assert FORWARD not in acts and JUMP not in acts and TURN_LEFT in acts


def test_facing_matters_for_leaf_value():
    g = duel()
    toward = g.evaluate(state((3, 3), D.S, (7, 7), arrows=0))
    away = g.evaluate(state((3, 3), D.N, (7, 7), arrows=0))
    assert toward > away                      # facing the goal direction is worth more
