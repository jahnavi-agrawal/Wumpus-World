"""GUI tests with the real game and AI modules (SDL dummy video driver, no window)."""
import os
import time

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
import pytest

from wumpus.agents.explorer import ExplorerAgent
from wumpus.config import GameConfig
from wumpus.core.actions import STAY, WumpusAction
from wumpus.core.geometry import Direction as D, Pos
from wumpus.environment.engine import GameEngine, Phase
from wumpus.ui import theme as T
from wumpus.ui.app import App
from wumpus.ui.session import mind_snapshot


@pytest.fixture
def app():
    a = App(seed=3, wumpus_mode="human", headless=True)
    yield a
    pygame.quit()


def key(app, k):
    app.handle(pygame.event.Event(pygame.KEYDOWN, key=k))


def run_until(app, cond, max_frames=600):
    for _ in range(max_frames):
        app.frame()
        if cond():
            return True
    return False


def test_explorer_moves_automatically_then_waits_for_human_wumpus(app):
    assert run_until(app, lambda: app.session.human_turn)
    turn = app.session.reality.status.turn
    for _ in range(30):
        app.frame()
    assert app.session.human_turn and app.session.reality.status.turn == turn      # waits for you
    legal = app.session.reality.legal_wumpus
    k = {D.N: pygame.K_w, D.E: pygame.K_d, D.S: pygame.K_s, D.W: pygame.K_a}
    move = next((a for a in legal if a.direction is not None), STAY)
    key(app, k[move.direction] if move.direction else pygame.K_x)
    assert app.session.phase is Phase.EXPLORER and app.session.reality.status.turn == turn + 1


def test_mouse_click_moves_the_wumpus(app):
    assert run_until(app, lambda: app.session.human_turn)
    w = app.session.reality.world.wumpus
    target = next(w.step(a.direction) for a in app.session.reality.legal_wumpus if a.direction is not None)
    app.on_click(T.cell_rect(T.REALITY, target.x, target.y).center)
    world = app.session.reality.world
    assert world.wumpus == target or world.wumpus in world.portals.values()


def test_illegal_wumpus_move_rejected(app):
    assert run_until(app, lambda: app.session.human_turn)
    w = app.session.reality.world
    illegal = [d for d in D if WumpusAction.move(d) not in app.session.reality.legal_wumpus]
    if illegal:
        assert not app.submit_wumpus(WumpusAction.move(illegal[0]))
        assert app.session.human_turn


def test_pause_and_single_step(app):
    key(app, pygame.K_p)
    assert app.paused
    for _ in range(60):
        app.frame()
    assert app.session.log_count == 0                         # nothing happens while paused
    key(app, pygame.K_SPACE)
    assert run_until(app, lambda: app.session.log_count > 0 and not app.session.busy)
    assert app.session.reality.status.step_count == 1          # exactly one explorer step
    for _ in range(60):
        app.frame()
    assert app.session.reality.status.step_count == 1


def test_restart_new_map_and_speed(app):
    run_until(app, lambda: app.session.log_count > 0)
    seed = app.session.seed
    key(app, pygame.K_r)
    assert app.session.seed == seed and app.session.log_count == 0
    key(app, pygame.K_n)
    assert app.session.log_count == 0
    s0 = app.speed
    key(app, pygame.K_EQUALS)
    assert app.speed > s0


def test_escape_menu(app):
    key(app, pygame.K_ESCAPE)
    assert app.menu_open
    app.frame()
    app.menu_action(3)                                         # cycle Wumpus controller
    assert app.session.wumpus_mode != "human"
    key(app, pygame.K_ESCAPE)
    assert not app.menu_open


def test_brain_rows_are_real_agent_entries_and_inspector_works(app):
    assert run_until(app, lambda: app.session.human_turn)
    app.frame()
    log = app.session.agent.log
    assert app.brain.rows
    for _, eid in app.brain.rows:                               # every displayed row is a real entry
        assert 0 <= eid < len(log.entries) and log.get(eid).tag in T.KEY_TAGS
    rect, eid = app.brain.rows[-1]
    app.on_click(rect.center)
    assert app.inspector is not None and app.inspector.entry_id == eid
    app.frame()
    entry = log.get(eid)
    if entry.refs:
        row, rid = app.inspector.ref_rows[0]
        app.on_click(row.center)
        assert app.inspector.entry_id == rid                    # followed the dependency
    key(app, pygame.K_ESCAPE)
    assert app.inspector is None


def test_ai_runs_in_background_without_blocking_frames(app):
    agent = app.session.agent
    real = agent.act

    def slow(p):
        time.sleep(0.4)
        return real(p)
    agent.act = slow
    t = time.perf_counter()
    app.frame()                                    # starts the worker
    assert app.session.busy
    slowest = 0.0
    while app.session.busy and time.perf_counter() - t < 3:
        f0 = time.perf_counter()
        app.frame()
        slowest = max(slowest, time.perf_counter() - f0)
    assert not app.session.busy and slowest < 0.25              # frames kept flowing while it thought


def test_mind_panel_draws_from_agent_knowledge_only():
    """The Agent's Mind can be drawn from an explorer that never had access to an engine."""
    from wumpus.ui.anim import Animator
    from wumpus.ui.board_view import draw_mind
    pygame.init()
    T.reset_fonts()
    surf = pygame.Surface((T.WIDTH, T.HEIGHT))
    agent = ExplorerAgent(GameEngine(GameConfig(seed=1)).public_info())
    from wumpus.core.percepts import Percept
    agent.act(Percept(turn=1, step=0, position=Pos(0, 0), facing=D.E, breeze=True))
    snap = mind_snapshot(agent)
    assert not hasattr(snap, "world")
    draw_mind(surf, T.MIND, snap, Animator())


def test_ai_vs_ai_game_completes_in_the_gui():
    a = App(seed=2, wumpus_mode="random", headless=True)
    a.speed_i = 4                                                   # fastest speed
    assert run_until(a, lambda: a.session.phase is Phase.OVER, max_frames=6000)
    a.frame()                                                       # draws the game-over banner
    pygame.quit()


def test_layout_matches_spec():
    assert (T.WIDTH, T.HEIGHT) == (1280, 800) and T.HEADER.h == 56
    assert T.REALITY.size == T.MIND.size and 180 <= T.BRAIN.h <= 220
    assert 48 <= T.TILE_SIZE <= 52 and 5 <= T.TILE_GAP <= 7
    assert (T.BG, T.PANEL, T.TILE, T.BORDER) == ((15, 23, 42), (30, 41, 59), (40, 53, 72), (51, 65, 85))
    assert T.TAG_COLORS["INFERENCE"] == (0x10, 0xB9, 0x81) and T.TAG_COLORS["DECISION"] == (0xF8, 0x71, 0x71)
    assert T.cell_rect(T.REALITY, 3, 2).topleft[1] == T.cell_rect(T.MIND, 3, 2).topleft[1]   # same coords
