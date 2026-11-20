"""Terminal version: rendering, command parsing and full games driven by scripted keystrokes."""
import pytest

from wumpus.config import GameConfig
from wumpus.core.actions import STAY, WumpusAction
from wumpus.core.geometry import Direction as D
from wumpus.environment.engine import Phase
from wumpus.terminal import (parse_command, render_belief, render_board, render_entry, render_knowledge,
                             render_status, run)
from tests.helpers import make_engine


class Keyboard:
    """Feeds scripted lines to the game; raises EOFError when exhausted."""

    def __init__(self, lines):
        self.lines, self.prompts = list(lines), 0

    def __call__(self, prompt=""):
        self.prompts += 1
        if not self.lines:
            raise EOFError
        return self.lines.pop(0)


def capture():
    out = []
    return out, lambda *a: out.append(" ".join(map(str, a)))


# ------------------------------------------------------------------ parsing
@pytest.mark.parametrize("text,action", [("w", WumpusAction.move(D.N)), ("A", WumpusAction.move(D.W)),
                                         ("down", WumpusAction.move(D.S)), (" d ", WumpusAction.move(D.E)),
                                         ("x", STAY), ("stay", STAY)])
def test_parse_moves(text, action):
    c = parse_command(text)
    assert c.kind == "move" and c.action == action


def test_parse_other_commands():
    assert parse_command("i 12").arg == 12 and parse_command("i#7").arg == 7
    assert parse_command("i abc").kind == "invalid"
    assert [parse_command(t).kind for t in "bkmlhq"] == ["brain", "knowledge", "belief", "legal", "help", "quit"]
    assert parse_command("fly").kind == "invalid" and parse_command("").kind == "invalid"


# ------------------------------------------------------------------ rendering
def test_board_shows_hidden_layout_for_the_human():
    e = make_engine(pits=[(2, 0)], wumpus=(5, 5), gold=(3, 3), portals=[((4, 0), (6, 6))],
                    unstable=[(1, 1)], shuffle=[(0, 2)])
    text = render_board(e.world)
    rows = text.splitlines()
    assert rows[7].startswith("0 |E> |")
    assert " P " in rows[7] and "oA" in rows[7] and "oA" in rows[1]        # both ends of the pair
    assert " W " in rows[2] and " G " in rows[4] and " u " in rows[6] and " s " in rows[5]


def test_knowledge_and_belief_views_use_agent_beliefs_only():
    from wumpus.agents.explorer import ExplorerAgent
    e = make_engine(pits=[(2, 0)], wumpus=(6, 6))
    a = ExplorerAgent(e.public_info())
    e.apply_explorer_action(a.act(e.percept()))
    e.apply_wumpus_action(STAY)
    a.observe(e.percept())                       # breeze at (1,0)
    k = render_knowledge(a)
    row0 = k.splitlines()[8]                     # y = 0
    p20 = a.pit_prob[(2, 0)]
    assert row0.startswith("0 |") and f"{p20:.0%}" in row0 and 0 < p20 < 1
    assert " P " not in k                        # never the true layout
    assert "P(Wumpus is here)" in render_belief(a)


def test_status_line():
    e = make_engine()
    assert "Turn 1" in render_status(e) and "arrows: 1" in render_status(e) and "to move: explorer" in render_status(e)


# ------------------------------------------------------------------ full games
def test_full_game_against_scripted_wumpus():
    out, write = capture()
    engine, agent = run(GameConfig(seed=2), "random", inp=Keyboard([]), out=write)
    assert engine.phase is Phase.OVER
    assert any(line.startswith("GAME OVER") for line in out)
    assert any("Brain Activity:" in line for line in out)


def test_human_turns_with_inspection_and_illegal_moves():
    # inspect entries, show views, try a wall move, then stay forever
    keys = ["h", "i 0", "i 9999", "k", "m", "b", "l", "fly"] + ["x"] * 200
    out, write = capture()
    engine, agent = run(GameConfig(seed=2), "human", inp=Keyboard(keys), out=write)
    text = "\n".join(out)
    assert engine.phase is Phase.OVER
    assert "#0  turn 1  [PERCEPT]" in text                     # inspected entry with its details
    assert "No brain entry #9999" in text
    assert "Explorer knowledge" in text and "P(Wumpus is here)" in text
    assert "Legal:" in text and "unknown command" in text


def test_illegal_wumpus_move_is_rejected_and_reprompted():
    e = make_engine(wumpus=(7, 0))
    # drive the human prompt directly through the loop helper
    from wumpus.agents.explorer import ExplorerAgent
    from wumpus.terminal import _ask_human
    a = ExplorerAgent(e.public_info())
    e.apply_explorer_action(a.act(e.percept()))
    out, write = capture()
    kb = Keyboard(["d", "s", "w"])                             # E wall, S wall, then N
    act = _ask_human(e, a, e.legal_wumpus_actions(), kb, write)
    assert act == WumpusAction.move(D.N) and kb.prompts == 3
    assert sum("Illegal move" in line for line in out) == 2


def test_quit_and_eof_end_the_game_cleanly():
    for keys in (["q"], []):
        out, write = capture()
        engine, _ = run(GameConfig(seed=4), "human", inp=Keyboard(keys), out=write)
        assert engine.phase is not Phase.OVER and "Game abandoned." in out


def test_inspect_shows_dependencies():
    out, write = capture()
    engine, agent = run(GameConfig(seed=2), "idle", inp=Keyboard([]), out=write)
    decision = agent.log.tagged("DECISION")[0]
    text = render_entry(agent, decision.id)
    assert "[DECISION]" in text and ("depends on:" in text or not decision.refs)


def test_main_cli_runs_headless(capsys):
    import main
    main.main(["--terminal", "--seed", "5", "--wumpus", "idle", "--no-board", "--max-turns", "30"])
    assert "GAME OVER" in capsys.readouterr().out
