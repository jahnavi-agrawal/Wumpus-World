import itertools
import random

from wumpus.ai.logic import DungeonKB, KnowledgeBase, P, dpll
from wumpus.config import GameConfig
from wumpus.core.geometry import Pos
from wumpus.env.engine import GameEngine


def brute_force_sat(clauses, n):
    for bits in itertools.product([False, True], repeat=n):
        if all(any(bits[abs(l) - 1] == (l > 0) for l in c) for c in clauses):
            return True
    return False


def test_dpll_matches_truth_table_on_random_3cnf():
    rng = random.Random(1)
    for _ in range(300):
        n = 7
        clauses = [frozenset(rng.choice([1, -1]) * v for v in rng.sample(range(1, n + 1), 3))
                   for _ in range(rng.randint(5, 40))]
        model = dpll(clauses)
        assert (model is not None) == brute_force_sat(clauses, n)
        if model is not None:   # returned model really satisfies the formula
            full = {v: model.get(v, False) for v in range(1, n + 1)}
            assert all(any(full[abs(l)] == (l > 0) for l in c) for c in clauses)


def test_entailment_modus_ponens_and_non_entailment():
    kb = KnowledgeBase()
    a, b, c = kb.lit("A"), kb.lit("B"), kb.lit("C")
    kb.tell([a], "fact")
    kb.tell([-a, b], "A => B")
    assert kb.ask("B").entailed
    assert not kb.ask("C").entailed and not kb.ask("C", False).entailed
    assert not kb.ask("B", False).entailed


def _info():
    return GameEngine(GameConfig(seed=1)).public_info()


def test_dungeon_kb_classic_inference():
    """(0,0),(1,0),(0,1) visited; breeze only at (1,0)  =>  pit at (2,0), (1,1) & (0,2) safe."""
    dkb = DungeonKB(_info())
    obs = {Pos(0, 0): (False, 1), Pos(1, 0): (True, 2), Pos(0, 1): (False, 3)}
    kb = dkb.rebuild({Pos(0, 0), Pos(1, 0), Pos(0, 1)}, set(), obs)
    assert kb.satisfiable()
    st = dkb.classify([Pos(2, 0), Pos(1, 1), Pos(0, 2), Pos(3, 3)])
    assert st[Pos(2, 0)].status == "pit"
    assert st[Pos(1, 1)].status == "safe" and st[Pos(0, 2)].status == "safe"
    assert st[Pos(3, 3)].status == "unknown"
    proof = st[Pos(2, 0)].proof
    assert proof.entailed and any("breeze at (1,0)" in s for _, s in proof.relevant)
    assert any("UNSAT" in line for line in proof.explain())


def test_ambiguous_breeze_is_unknown():
    dkb = DungeonKB(_info())
    # breeze at (1,1) (pretend visited), two unknown neighbours (2,1),(1,2)
    vis = {Pos(0, 0), Pos(1, 0), Pos(0, 1), Pos(1, 1)}
    obs = {p: (p == Pos(1, 1), 1) for p in vis}
    dkb.rebuild(vis, set(), obs)
    st = dkb.classify([Pos(2, 1), Pos(1, 2)])
    assert st[Pos(2, 1)].status == "unknown" and st[Pos(1, 2)].status == "unknown"


def test_collapsed_cell_known_not_pit():
    dkb = DungeonKB(_info())
    dkb.rebuild({Pos(0, 0)}, {Pos(4, 4)}, {})
    assert dkb.classify([Pos(4, 4)])[Pos(4, 4)].status == "safe"   # safe from pits (still deadly!)


def test_symbol_names():
    assert P(Pos(3, 2)) == "P32"
