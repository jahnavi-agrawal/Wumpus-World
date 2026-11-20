import pytest

from wumpus.ai.wumpus_tracker import WumpusTracker
from wumpus.config import GameConfig
from wumpus.core.geometry import Pos, manhattan, neighbors
from wumpus.environment.engine import GameEngine

INFO = GameEngine(GameConfig(seed=1)).public_info()


def tracker():
    t = WumpusTracker(INFO)
    t.begin(1)
    return t


def test_initial_uniform_respects_public_rules():
    t = tracker()
    assert sum(t.belief.values()) == pytest.approx(1.0)
    assert all(manhattan(c, Pos(0, 0)) >= 3 for c in t.belief) and len(t.belief) == 64 - 6


def test_stench_concentrates_on_neighbours():
    t = tracker()
    t.observe_stench(Pos(4, 4), True)
    t.normalize(set())
    assert set(t.belief) <= set(neighbors(Pos(4, 4), 8))
    assert sum(t.belief.values()) == pytest.approx(1.0)


def test_no_stench_and_survival_remove_cells():
    t = tracker()
    t.observe_survived_at(Pos(3, 3))
    t.observe_stench(Pos(3, 3), False)
    t.normalize(set())
    assert Pos(3, 3) not in t.belief and not set(neighbors(Pos(3, 3), 8)) & set(t.belief)


def test_predict_conserves_mass_and_respects_blocked():
    t = tracker()
    t.belief = {Pos(5, 5): 1.0}
    t.predict(Pos(0, 0), blocked={Pos(5, 6)}, pit_prob={})
    assert sum(t.belief.values()) == pytest.approx(1.0)
    assert Pos(5, 6) not in t.belief and t.belief[Pos(5, 5)] == pytest.approx(0.15)
    # pursuit: cells closer to the explorer get more mass than farther ones
    assert t.belief[Pos(4, 5)] > t.belief[Pos(6, 5)]


def test_known_pit_excluded_and_pit_risk_downweights():
    t = tracker()
    tr = t.transition(Pos(5, 5), Pos(5, 0), set(), {Pos(5, 6): 1.0, Pos(6, 5): 0.5})
    assert Pos(5, 6) not in tr and tr[Pos(6, 5)] < tr[Pos(4, 5)]
    assert sum(tr.values()) == pytest.approx(1.0)


def test_learned_portal_teleports_mass():
    a, b = Pos(4, 4), Pos(1, 6)
    t = tracker()
    t.portals = {a: b, b: a}
    t.belief = {Pos(4, 3): 1.0}
    t.predict(Pos(0, 0), set(), {})
    assert t.belief.get(b, 0) > 0 and a not in t.belief


def test_arrow_evidence():
    t = tracker()
    t.observe_arrow_missed([Pos(5, y) for y in range(8)])
    t.normalize(set())
    assert all(c.x != 5 for c in t.belief) and sum(t.belief.values()) == pytest.approx(1.0)


def test_contradiction_resets():
    t = tracker()
    t.observe_stench(Pos(0, 0), True)   # impossible at start (Wumpus ≥ 3 away)
    t.normalize({Pos(0, 0)})
    assert t.report.reset and sum(t.belief.values()) == pytest.approx(1.0)


def test_entry_risk():
    t = tracker()
    t.belief = {Pos(3, 3): 1.0}
    assert t.entry_risk(Pos(3, 3), set(), {}) == 1.0
    r = t.entry_risk(Pos(3, 2), set(), {})
    assert 0.3 < r < 1.0          # adjacent: Wumpus likely pursues into the cell
    assert t.entry_risk(Pos(0, 7), set(), {}) == 0.0
    t.kill()
    assert t.entry_risk(Pos(3, 3), set(), {}) == 0.0
