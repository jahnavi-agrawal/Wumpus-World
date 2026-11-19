"""The explorer must never read the hidden dungeon layout."""
import ast
import pathlib

from wumpus.config import GameConfig
from wumpus.control.wumpus_controllers import ScriptedWumpusController
from wumpus.ai.explorer import ExplorerAgent
from wumpus.runner import play_game

AI_DIR = pathlib.Path(__file__).resolve().parents[1] / "wumpus" / "ai"
FORBIDDEN = ("env", "environment", "ui", "control", "runner", "pygame")


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield ("." * node.level) + (node.module or "")


def test_ai_package_never_imports_hidden_state():
    files = list(AI_DIR.glob("*.py"))
    assert len(files) >= 7
    for f in files:
        for name in _imports(f):
            parts = name.lstrip(".").split(".")
            assert not any(p in FORBIDDEN for p in parts), f"{f.name} imports {name}"


def test_decisions_are_a_function_of_percepts_only():
    """Replaying the recorded percept stream into a fresh agent reproduces every action."""
    percepts, actions = [], []
    def record(engine, agent):
        pass
    cfg = GameConfig(seed=3)
    engine, agent, _ = play_game(cfg, ScriptedWumpusController("random", 3))
    # collect percept/action pairs from the log of a second, instrumented run
    from wumpus.env.engine import GameEngine, Phase
    e = GameEngine(cfg)
    a = ExplorerAgent(e.public_info())
    ctrl = ScriptedWumpusController("random", 3)
    while e.phase is not Phase.OVER:
        if e.phase is Phase.EXPLORER:
            p = e.percept()
            act = a.act(p)
            percepts.append(p)
            actions.append(act)
            e.apply_explorer_action(act)
        else:
            e.apply_wumpus_action(ctrl.choose(e.world, e.legal_wumpus_actions()))
    fresh = ExplorerAgent(e.public_info())
    assert [fresh.act(p) for p in percepts] == actions
    assert e.result == engine.result      # and the whole game is seed-reproducible
