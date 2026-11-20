"""Prompt 1: knowledge representation and inference, on manually constructed scenarios."""
import pytest

from wumpus.agents.explorer import ExplorerAgent
from wumpus.agents.knowledge_base import P, DungeonKB, WumpusEvidence
from wumpus.core.actions import FORWARD
from wumpus.core.geometry import Direction as D, Pos
from wumpus.core.percepts import Percept
from tests.helpers import make_engine


def info():
    from wumpus.config import GameConfig
    from wumpus.environment.engine import GameEngine
    return GameEngine(GameConfig(seed=1)).public_info()      # real public info: 7 pits


def kb_with(visited, breezy=(), stench_at=None, stench=False, turn=5, alive=True, safe_queries=()):
    dkb = DungeonKB(info())
    vis = {Pos(*v) for v in visited}
    obs = {v: (v in {Pos(*b) for b in breezy}, 1) for v in vis}
    ev = WumpusEvidence(turn, Pos(*stench_at), stench, alive) if stench_at else None
    dkb.rebuild(vis, set(), obs, wumpus=ev, safe_queries=[Pos(*c) for c in safe_queries])
    return dkb


# ---------------------------------------------------------------- rules
def test_stench_implies_possible_adjacent_wumpus_not_certain():
    dkb = kb_with([(0, 0), (1, 0), (1, 1)], stench_at=(1, 1), stench=True)
    st = dkb.classify_wumpus([Pos(1, 2), Pos(2, 1), Pos(0, 1)])
    assert all(x.status == "unknown" for x in st.values())        # possible, not confirmed


def test_no_stench_rules_out_adjacent_wumpus():
    dkb = kb_with([(0, 0), (1, 0), (1, 1)], stench_at=(1, 1), stench=False)
    st = dkb.classify_wumpus([Pos(1, 2), Pos(2, 1), Pos(0, 1), Pos(1, 0)])
    assert all(x.status == "none" for x in st.values())
    assert "no stench at (1,1)" in st[Pos(1, 2)].explanation


def test_stench_plus_proven_pit_narrows_the_wumpus():
    # breeze only at (1,1) and none at (0,2): (1,2) pit-free -> pit at (2,1); stench at (1,1)
    dkb = DungeonKB(info())
    vis = {Pos(0, 0), Pos(0, 1), Pos(1, 1), Pos(0, 2)}
    obs = {v: (v == Pos(1, 1), 1) for v in vis}
    dkb.rebuild(vis, set(), obs, wumpus=WumpusEvidence(5, Pos(1, 1), True))
    assert dkb.classify([Pos(2, 1)])[Pos(2, 1)].status == "pit"
    w = dkb.classify_wumpus([Pos(2, 1), Pos(1, 2), Pos(1, 0), Pos(0, 1)])
    assert w[Pos(2, 1)].status == "none"                    # the Wumpus never stands on a pit
    assert w[Pos(1, 2)].status == "unknown"


def test_wumpus_certain_when_only_one_candidate_left():
    # stench at (0,1): neighbours (0,2),(1,1),(0,0). (0,0) we... is not this turn's cell -> unknown.
    # Make (1,1) and (0,0) impossible: (1,1) proven pit, (0,0) collapsed.
    dkb = DungeonKB(info())
    vis = {Pos(0, 1), Pos(1, 0), Pos(2, 1), Pos(1, 2)}
    obs = {Pos(1, 0): (True, 1), Pos(2, 1): (True, 1), Pos(1, 2): (True, 1), Pos(0, 1): (True, 1)}
    # make (1,1) the only pit explanation for (1,0): (2,0) and (0,0) pit-free
    obs[Pos(1, 0)] = (True, 1)
    dkb.rebuild(vis | {Pos(2, 0)}, {Pos(0, 0)}, {Pos(1, 0): (True, 1), Pos(2, 0): (False, 1)},
                wumpus=WumpusEvidence(5, Pos(0, 1), True))
    assert dkb.classify([Pos(1, 1)])[Pos(1, 1)].status == "pit"
    w = dkb.classify_wumpus([Pos(0, 2)])
    assert w[Pos(0, 2)].status == "wumpus" and "right now" in w[Pos(0, 2)].explanation


def test_safe_requires_no_pit_and_no_wumpus():
    dkb = kb_with([(0, 0), (1, 0), (1, 1)], stench_at=(1, 1), stench=False, safe_queries=[(1, 2), (5, 5)])
    s = dkb.classify_safe([Pos(1, 2), Pos(5, 5)])
    assert s[Pos(1, 2)].status == "safe" and "no stench" in s[Pos(1, 2)].explanation
    assert s[Pos(5, 5)].status == "unknown"                 # pit and Wumpus both unknown far away
    dkb = kb_with([(0, 0), (1, 0), (1, 1)], stench_at=(1, 1), stench=True, safe_queries=[(1, 2)])
    assert dkb.classify_safe([Pos(1, 2)])[Pos(1, 2)].status == "unknown"   # pit-free, Wumpus possible


def test_breeze_does_not_make_every_neighbour_a_pit():
    dkb = kb_with([(0, 0), (1, 0), (1, 1)], breezy=[(1, 1)])
    assert {x.status for x in dkb.classify([Pos(2, 1), Pos(1, 2)]).values()} == {"unknown"}


def test_combines_constraints_from_multiple_observations():
    # breeze at (1,1) -> (2,1) or (1,2); no breeze at (0,2) -> (1,2) safe; therefore (2,1) is a pit
    dkb = kb_with([(0, 0), (0, 1), (1, 1), (0, 2)], breezy=[(1, 1)])
    st = dkb.classify([Pos(2, 1), Pos(1, 2)])
    assert st[Pos(1, 2)].status == "safe" and st[Pos(2, 1)].status == "pit"
    assert "breeze at (1,1)" in st[Pos(2, 1)].explanation and "no breeze at (0,2)" in st[Pos(2, 1)].explanation


def test_explanations_are_human_readable():
    st = kb_with([(0, 0), (1, 0), (0, 1)], breezy=[(1, 0)]).classify([Pos(2, 0)])[Pos(2, 0)]
    assert st.explanation.startswith("(2,0) certainly contains a pit because:")
    assert any(line.startswith("In words:") for line in st.proof.explain())


# ---------------------------------------------------------------- time and revision
def test_wumpus_facts_are_time_sensitive_and_expire():
    dkb = kb_with([(0, 0), (1, 0), (1, 1)], stench_at=(1, 1), stench=False, turn=7)
    ts = dkb.kb.time_sensitive()
    assert ts and all(v == 7 for _, _, v in ts)
    assert all("WUMPUS" in c or "STENCH" in c for c, _, _ in ts)
    dkb2 = kb_with([(0, 0), (1, 0), (1, 1)], stench_at=(0, 0), stench=False, turn=8)
    assert dkb2.classify_wumpus([Pos(1, 2)])[Pos(1, 2)].status == "unknown"   # may have moved there


def test_dead_wumpus_ruled_out_everywhere_permanently():
    dkb = kb_with([(0, 0)], stench_at=(0, 0), alive=False)
    assert dkb.classify_wumpus([Pos(5, 5)])[Pos(5, 5)].status == "none"
    assert not dkb.kb.time_sensitive()


def test_contradiction_detected_with_minimal_core():
    kb = kb_with([(0, 0), (1, 0)], breezy=[(1, 0)]).kb
    kb.tell([kb.lit(P(Pos(2, 0)), False)], "fake: told (2,0) has no pit")
    kb.tell([kb.lit(P(Pos(1, 1)), False)], "fake: told (1,1) has no pit")
    rep = kb.consistency()
    assert not rep.consistent and "CONTRADICTION" in rep.explain()[0]
    srcs = {s for _, s in rep.core}
    assert {"fake: told (2,0) has no pit", "fake: told (1,1) has no pit",
            "percept turn 1: breeze at (1,0)"} <= srcs
    assert "visited (0,0) and survived" in srcs or "public rule: start zone has no pits" in srcs
    from wumpus.agents.knowledge_base import dpll
    for i in range(len(rep.core)):                         # minimal: every clause is needed
        texts = [c for j, (c, _) in enumerate(rep.core) if j != i]
        assert texts != [c for c, _ in rep.core]


# ---------------------------------------------------------------- the agent's knowledge
def pct(turn, pos, facing=D.E, **kw):
    return Percept(turn=turn, step=turn - 1, position=Pos(*pos), facing=facing, **kw)


def test_agent_snapshot_is_read_only_and_complete():
    a = ExplorerAgent(info())
    a.observe(pct(1, (0, 0)))
    a.last_action = FORWARD
    a.observe(pct(2, (1, 0), breeze=True, stench=True))
    k = a.knowledge()
    with pytest.raises(TypeError):
        k.confirmed_hazards[Pos(7, 7)] = "x"
    with pytest.raises(AttributeError):
        k.visited.add(Pos(5, 5))
    with pytest.raises(Exception):
        k.turn = 99
    assert k.visited == {Pos(0, 0), Pos(1, 0)} and k.percepts[2].breeze
    assert Pos(2, 0) in k.suspected_pits and Pos(1, 1) in k.suspected_pits     # ambiguous breeze
    assert Pos(2, 0) in k.ambiguous
    assert set(k.suspected_wumpus) <= {Pos(2, 0), Pos(1, 1), Pos(0, 0)}       # stench -> adjacent only
    assert k.time_sensitive_facts and all(v == 2 for _, _, v in k.time_sensitive_facts)
    assert Pos(1, 0) in k.known_safe and Pos(0, 1) in k.pit_free


def test_agent_knows_special_tiles_and_portal_connections():
    a = ExplorerAgent(info())
    a.observe(pct(1, (1, 0)))
    a.last_action, a.last_pos, a.last_facing = FORWARD, Pos(1, 0), D.E
    a.observe(pct(2, (5, 5), teleported=True))                # stepped onto (2,0)
    a.last_action = FORWARD
    a.observe(pct(3, (6, 5), unstable=True))
    a.last_action = FORWARD
    a.observe(pct(4, (7, 5), shuffled=True))
    k = a.knowledge()
    assert k.portal_connections[Pos(2, 0)] == Pos(5, 5)
    assert k.special_tiles[Pos(6, 5)] == "collapsed" and k.special_tiles[Pos(7, 5)] == "used shuffle tile"
    assert k.confirmed_hazards[Pos(6, 5)] == "collapsed tile"


def test_agent_belief_revision_on_contradiction():
    a = ExplorerAgent(info())
    a.observe(pct(1, (0, 0)))
    a.last_action = FORWARD
    a.observe(pct(2, (1, 0), breeze=True))            # pit at (2,0) or (1,1)
    a.visited |= {Pos(2, 0), Pos(1, 1)}               # (as if) both were later visited safely
    a.last_action = FORWARD
    a.observe(pct(3, (2, 0)))
    w = [e for e in a.log.tagged("WARNING") if "CONTRADICTION" in e.summary]
    assert w and "breeze observation at (1,0)" in w[0].summary
    assert Pos(1, 0) not in a.breeze_obs and a.dkb.kb.consistency().consistent


def test_agent_never_imports_the_map():
    import wumpus.agents.explorer as ex
    assert "environment" not in ex.__dict__ and not any("environment" in str(v) for v in vars(ex).values()
                                                         if hasattr(v, "__module__") and v.__module__)
