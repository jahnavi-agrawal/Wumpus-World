import random
from dataclasses import dataclass

import pytest

from wumpus.ai.minimax import (DuelGame, DuelState, SearchStats, alphabeta, determinized_decision,
                               minimax)
from wumpus.core.actions import ExplorerAction, ExplorerActionType
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
    return DuelGame(size=8, pads=None, blocked=frozenset(kw.get("blocked", ())), pit_risk=kw.get("risk", {}),
                    goal_dist=dist)


def test_shoots_wumpus_in_line():
    g = duel()
    root = DuelState(explorer=Pos(3, 3), wumpus=Pos(3, 4), wumpus_alive=True, arrows=1, jumps=0)
    r = determinized_decision(g, root, [(Pos(3, 4), 1.0)], depth=4)
    assert r.best_action == ExplorerAction.shoot(D.N)
    assert r.stats.cutoffs > 0


def test_flees_without_arrow_never_steps_into_wumpus():
    g = duel()
    root = DuelState(explorer=Pos(3, 3), wumpus=Pos(3, 4), wumpus_alive=True, arrows=0, jumps=0)
    r = determinized_decision(g, root, [(Pos(3, 4), 1.0)], depth=4)
    assert r.best_action != ExplorerAction.move(D.N)
    assert r.root_values[ExplorerAction.move(D.N)] == -1000


def test_belief_weighted_shot_direction():
    g = duel()
    root = DuelState(explorer=Pos(3, 3), wumpus=Pos(0, 0), wumpus_alive=True, arrows=1, jumps=0)
    r = determinized_decision(g, root, [(Pos(3, 4), 0.8), (Pos(4, 3), 0.2)], depth=2)
    assert r.best_action == ExplorerAction.shoot(D.N)
    text = "\n".join(r.explain())
    assert "Hypothesis W at (3,4)" in text and "chosen" in text


def test_risky_cells_excluded_and_blocked_respected():
    g = duel(blocked={Pos(2, 3)}, risk={Pos(3, 2): 0.9})
    root = DuelState(explorer=Pos(3, 3), wumpus=Pos(7, 7), wumpus_alive=True, arrows=0, jumps=0)
    acts = g.actions(root)
    assert ExplorerAction.move(D.W) not in acts and ExplorerAction.move(D.S) not in acts
    assert all(a.kind is not ExplorerActionType.JUMP for a in acts)
