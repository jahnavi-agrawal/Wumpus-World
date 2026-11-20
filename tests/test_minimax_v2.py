"""Prompt 3: Minimax evaluation terms, structured output, opponent replies, AI Wumpus, prediction checks."""
import json

from wumpus.agents.minimax import DuelGame, DuelState, WumpusStep, alphabeta, determinized_decision
from wumpus.config import GameConfig
from wumpus.control.wumpus_controllers import MinimaxWumpusController, make_controller
from wumpus.core.actions import FORWARD, SHOOT, STAY, TURN_LEFT, TURN_RIGHT, WumpusAction
from wumpus.core.geometry import Direction as D, Pos, all_cells, manhattan
from wumpus.environment.engine import Outcome, Phase
from wumpus.runner import play_game
from tests.helpers import make_engine

DIST = {c: manhattan(c, Pos(0, 0)) for c in all_cells(8)}


def game(**kw):
    return DuelGame(size=8, portals={}, blocked=frozenset(kw.pop("blocked", ())), pit_risk=kw.pop("risk", {}),
                    goal_dist=kw.pop("goal", DIST), exit_dist=kw.pop("exit", DIST), **kw)


def st(e, f, w, arrows=1, jump=False, alive=True, to_move=True):
    return DuelState(Pos(*e), f, Pos(*w), alive, arrows, jump, explorer_to_move=to_move)


# ------------------------------------------------------------- terminal states and evaluation
def test_terminal_states():
    g = game()
    dead = DuelState(Pos(3, 3), D.N, Pos(3, 3), True, 1, False, explorer_dead=True)
    assert g.is_terminal(dead) and g.evaluate(dead) == -1000
    killed = st((3, 3), D.N, (3, 5), alive=False)
    assert g.is_terminal(killed) and g.evaluate(killed) > g.evaluate(st((3, 3), D.N, (7, 7)))


def test_evaluation_values_survival_abilities_and_exit():
    g = game()
    far = st((3, 3), D.S, (7, 7), arrows=0)
    assert g.evaluate(st((3, 3), D.S, (7, 7), arrows=0, jump=True)) > g.evaluate(far)          # jump kept
    assert g.evaluate(st((3, 3), D.S, (7, 7), arrows=1)) > g.evaluate(far)                     # arrow kept
    no_exit = game(exit={c: d for c, d in DIST.items() if c != Pos(3, 3)})
    assert no_exit.evaluate(far) < g.evaluate(far)                                             # cut off
    cornered = game(blocked={Pos(2, 3), Pos(4, 3), Pos(3, 2)})
    assert cornered.escape_routes(far) == 1 and g.escape_routes(far) == 4
    assert cornered.evaluate(far) < g.evaluate(far)


# ------------------------------------------------------------- opponent responses and selection
def test_min_player_eats_when_it_can():
    g = game()
    r = determinized_decision(g, st((3, 3), D.E, (3, 4), arrows=0), [(Pos(3, 4), 1.0)], depth=2)
    assert r.replies[(Pos(3, 4), TURN_LEFT)] == WumpusStep(D.S)           # it moves onto us
    assert r.root_values[TURN_LEFT] == -1000


def test_selects_shot_and_predicts_reply():
    r = determinized_decision(game(), st((3, 3), D.N, (3, 5)), [(Pos(3, 5), 1.0)], depth=4)
    assert r.best_action is SHOOT
    hyp, reply = r.predicted_reply()
    assert hyp == Pos(3, 5)


def test_structured_output_and_pruned_branches():
    r = determinized_decision(game(), st((3, 3), D.N, (3, 6)), [(Pos(3, 6), 0.7), (Pos(5, 3), 0.3)], depth=4)
    d = r.to_dict()
    json.dumps(d)
    assert d["depth"] == 4 and d["selected"] == str(r.best_action)
    assert {c["action"] for c in d["candidates"]} == {str(a) for a in r.root_values}
    assert all(set(c["predicted_replies"]) == {"(3,6)", "(5,3)"} for c in d["candidates"])
    assert d["cutoffs"] > 0 and d["pruned_branches"]
    assert d["pruned_moves"] >= sum(len(b["skipped"]) for b in d["pruned_branches"])
    text = "\n".join(r.explain())
    assert "Predicted Wumpus reply" in text and "Pruned:" in text


def test_alphabeta_root_value_unchanged_by_path_recording():
    from wumpus.agents.minimax import minimax
    g, s = game(), st((2, 2), D.E, (5, 5))
    assert alphabeta(g, s, 4) == minimax(g, s, 4)


# ------------------------------------------------------------- AI-controlled Wumpus
def test_minimax_wumpus_eats_adjacent_explorer():
    e = make_engine(wumpus=(2, 0))
    e.apply_explorer_action(FORWARD)                      # explorer to (1,0), adjacent
    a = MinimaxWumpusController().choose(e.world, e.legal_wumpus_actions())
    assert a == WumpusAction.move(D.W)


def test_minimax_wumpus_dodges_the_arrow_line():
    """Explorer at (0,1) facing E with its arrow; the Wumpus on that row steps off the line."""
    e = make_engine(wumpus=(4, 1), explorer=(0, 1), facing=D.E, visited=[(0, 0), (0, 1)])
    e.apply_explorer_action(TURN_LEFT)                    # now facing N
    e.apply_wumpus_action(STAY)
    e.apply_explorer_action(TURN_RIGHT)                   # facing E again: Wumpus is in the line of fire
    a = MinimaxWumpusController().choose(e.world, e.legal_wumpus_actions())
    assert a in (WumpusAction.move(D.N), WumpusAction.move(D.S))


def test_ai_vs_ai_games_run_with_legal_moves():
    for seed in range(4):
        engine, agent, _ = play_game(GameConfig(seed=seed, max_turns=60), make_controller("minimax", seed))
        assert engine.phase is Phase.OVER


# ------------------------------------------------------------- prediction discarded
def test_refuted_prediction_triggers_replan():
    from wumpus.agents.explorer import ExplorerAgent
    from wumpus.core.percepts import Percept
    from wumpus.environment.engine import GameEngine
    a = ExplorerAgent(GameEngine(GameConfig(seed=1)).public_info())
    a.observe(Percept(turn=1, step=0, position=Pos(0, 0), facing=D.E))
    a.last_action = FORWARD
    a.prediction = (1, Pos(6, 6), WumpusStep(None), Pos(6, 6))      # "it will stay at (6,6)"
    a.observe(Percept(turn=2, step=1, position=Pos(1, 0), facing=D.E, stench=True))   # it is next to us!
    rp = [e for e in a.log.tagged("REPLAN") if "did not act as predicted" in e.summary]
    assert rp and "(6,6)" in rp[0].summary and a.plan is None


def test_consistent_prediction_is_kept_silently():
    from wumpus.agents.explorer import ExplorerAgent
    from wumpus.core.percepts import Percept
    from wumpus.environment.engine import GameEngine
    a = ExplorerAgent(GameEngine(GameConfig(seed=1)).public_info())
    a.observe(Percept(turn=1, step=0, position=Pos(0, 0), facing=D.E))
    a.last_action = FORWARD
    a.prediction = (1, Pos(1, 1), WumpusStep(None), Pos(1, 1))
    a.observe(Percept(turn=2, step=1, position=Pos(1, 0), facing=D.E, stench=True))
    assert not [e for e in a.log.tagged("REPLAN") if "predicted" in e.summary]
