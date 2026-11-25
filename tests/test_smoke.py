"""Smoke tests: one check per component. Run with `python -m pytest`."""
import ast
import os
import pathlib
import random
from dataclasses import dataclass

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
import pytest

from evaluation.scenarios import SCENARIOS, run_scenario
from wumpus.agents.bayesian import estimate_pits
from wumpus.agents.knowledge_base import DungeonKB
from wumpus.agents.minimax import SearchStats, alphabeta
from wumpus.agents.pathfinding import astar
from wumpus.config import GameConfig
from wumpus.control.wumpus_controllers import ScriptedWumpusController
from wumpus.core.actions import CLIMB, FORWARD, GRAB, JUMP, SHOOT, STAY, TURN_LEFT, TURN_RIGHT
from wumpus.core.geometry import Direction as D, Pos, all_cells
from wumpus.environment import mechanics as M
from wumpus.environment.engine import GameEngine, Phase
from wumpus.environment.mechanics import IllegalAction
from wumpus.environment.world import generate_world
from wumpus.runner import play_game
from tests.helpers import make_engine, make_world

AI_DIR = pathlib.Path(__file__).resolve().parents[1] / "wumpus" / "agents"
INFO = GameEngine(GameConfig(seed=1)).public_info()


# ------------------------------------------------------------- hidden-information separation
def test_ai_package_never_imports_hidden_state():
    forbidden = ("env", "environment", "ui", "control", "runner", "pygame")
    files = list(AI_DIR.glob("*.py"))
    assert len(files) >= 7
    for f in files:
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not any(p in forbidden for p in name.split(".")), f"{f.name} imports {name}"


# ------------------------------------------------------------- world, engine, rules
def test_world_generation_is_deterministic():
    a = generate_world(GameConfig(seed=42), random.Random(42))
    b = generate_world(GameConfig(seed=42), random.Random(42))
    assert (a.pits, a.gold, a.wumpus, a.portals, a.unstable, a.shuffle_tiles) == \
           (b.pits, b.gold, b.wumpus, b.portals, b.unstable, b.shuffle_tiles)
    c = generate_world(GameConfig(seed=43), random.Random(43))
    assert (a.pits, a.gold) != (c.pits, c.gold)


def test_turns_alternate_and_illegal_moves_are_rejected():
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


def test_every_explorer_action_emits_structured_events():
    for a in (FORWARD, TURN_LEFT, TURN_RIGHT, JUMP, GRAB, SHOOT, CLIMB):
        evs, _ = M.apply_explorer_action(make_world(), a, 1, 1, GameConfig())
        assert evs and all(e.actor and e.kind and isinstance(e.to_dict(), dict) for e in evs)


# ------------------------------------------------------------- AI components
def test_logic_deduces_pit_and_safe_cells():
    """(0,0),(1,0),(0,1) visited; breeze only at (1,0)  =>  pit at (2,0), (1,1) & (0,2) safe."""
    dkb = DungeonKB(INFO)
    obs = {Pos(0, 0): (False, 1), Pos(1, 0): (True, 2), Pos(0, 1): (False, 3)}
    assert dkb.rebuild({Pos(0, 0), Pos(1, 0), Pos(0, 1)}, set(), obs).satisfiable()
    st = dkb.classify([Pos(2, 0), Pos(1, 1), Pos(0, 2), Pos(3, 3)])
    assert st[Pos(2, 0)].status == "pit"
    assert st[Pos(1, 1)].status == "safe" and st[Pos(0, 2)].status == "safe"
    assert st[Pos(3, 3)].status == "unknown"


def test_pit_probabilities_match_textbook():
    """AIMA 12.7: breeze in (1,0) and (0,1), prior 0.2 -> P(1,1)=0.86, P(2,0)=P(0,2)=0.31."""
    visited, breezy = [Pos(0, 0), Pos(1, 0), Pos(0, 1)], {Pos(1, 0), Pos(0, 1)}
    obs = {v: (v in breezy, 1) for v in visited}
    dkb = DungeonKB(INFO)
    dkb.rebuild(set(visited), set(), obs)
    cells = [c for c in all_cells(8) if c not in visited]
    r = estimate_pits(8, dkb.classify(cells), obs, 0.2)
    assert r[Pos(1, 1)].p_pit == pytest.approx(0.2 / 0.232, abs=1e-9)
    assert r[Pos(2, 0)].p_pit == pytest.approx(0.072 / 0.232, abs=1e-9)
    assert r[Pos(0, 2)].p_pit == pytest.approx(0.072 / 0.232, abs=1e-9)


@pytest.mark.parametrize("seed", range(3))
def test_astar_is_optimal(seed):
    rng = random.Random(seed)
    costs = {c: (None if rng.random() < 0.2 else rng.choice([0.0, 0.0, 1.5, 5.0])) for c in all_cells(8)}
    cells = rng.sample(all_cells(8), 4)
    portals = {cells[0]: cells[1], cells[1]: cells[0], cells[2]: cells[3], cells[3]: cells[2]}
    for p in portals:
        costs[p] = 0.0
    costs[Pos(0, 0)] = 0.0
    goal = rng.choice([c for c in all_cells(8) if costs[c] is not None and c not in portals])
    f = rng.choice(list(D))
    a = astar(Pos(0, 0), f, {goal}, 8, costs.get, jump=True, portals=portals)
    d = astar(Pos(0, 0), f, {goal}, 8, costs.get, jump=True, portals=portals, use_heuristic=False)
    assert a.found == d.found
    if a.found:
        assert a.cost == pytest.approx(d.cost)


@dataclass(frozen=True)
class _Node:
    path: tuple = ()


class _TextbookTree:
    """AIMA Fig 5.2: MAX over three MIN nodes with leaves (3,12,8) (2,4,6) (14,5,2) -> 3."""
    leaves = [[3, 12, 8], [2, 4, 6], [14, 5, 2]]

    def is_max(self, s):
        return len(s.path) % 2 == 0

    def is_terminal(self, s):
        return False

    def actions(self, s):
        return [0, 1, 2]

    def result(self, s, a):
        return _Node(s.path + (a,))

    def evaluate(self, s):
        return self.leaves[s.path[0]][s.path[1]]


def test_alphabeta_textbook_tree():
    stats = SearchStats()
    assert alphabeta(_TextbookTree(), _Node(), 2, stats=stats) == 3
    assert stats.cutoffs >= 1


# ------------------------------------------------------------- whole game
def test_full_game_terminates():
    engine, agent, stats = play_game(GameConfig(seed=1), ScriptedWumpusController("random", 1))
    assert engine.phase is Phase.OVER
    assert stats["max_decide_s"] < 2.0
    assert agent.log.entries


def test_scenario_logical_pit_deduction():
    scenario = next(s for s in SCENARIOS if s.name == "logical_pit_deduction")
    failed = [d for d, ok in run_scenario(scenario).checks if not ok]
    assert not failed, failed


def test_terminal_game_runs(capsys):
    import main
    main.main(["--terminal", "--seed", "5", "--wumpus", "idle", "--no-board", "--max-turns", "30"])
    assert "GAME OVER" in capsys.readouterr().out


def test_gui_runs_and_waits_for_the_human_wumpus():
    from wumpus.ui.app import App
    app = App(seed=3, wumpus_mode="human", headless=True)
    try:
        for _ in range(600):
            app.frame()
            if app.session.human_turn:
                break
        assert app.session.human_turn
        turn = app.session.reality.status.turn
        app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_x))      # X = stay
        assert app.session.phase is Phase.EXPLORER and app.session.reality.status.turn == turn + 1
    finally:
        pygame.quit()
