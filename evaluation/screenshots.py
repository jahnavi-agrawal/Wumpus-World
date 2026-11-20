"""Capture screenshots of the real interface (headless, deterministic seeds).

    python -m evaluation.screenshots          ->  docs/screenshots/*.png
"""
from __future__ import annotations

import os
import pathlib

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from wumpus.environment.engine import Phase
from wumpus.ui.app import App

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "screenshots"


def _run(app: App, cond, max_frames=4000):
    for _ in range(max_frames):
        app.frame()
        if cond():
            break
    for _ in range(40):                      # let animations settle
        if app.session.busy:
            app.frame()
            continue
        app.paused = True
        app.frame()


def capture() -> list[pathlib.Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    shots = []

    def save(app, name):
        p = OUT / name
        pygame.image.save(app.screen, str(p))
        shots.append(p)

    # 1-3: human Wumpus game, a few turns in
    app = App(seed=12, wumpus_mode="human", headless=True)
    app.frame()
    save(app, "01_start.png")
    app.paused = False
    _run(app, lambda: app.session.human_turn and app.session.reality.status.turn >= 1)
    turns = 0
    while turns < 9 and app.session.phase is not Phase.OVER:
        if app.session.human_turn:
            legal = app.session.reality.legal_wumpus
            app.submit_wumpus(legal[-1])                           # STAY keeps the demo readable
            turns += 1
        app.paused = False
        _run(app, lambda: app.session.human_turn or app.session.phase is Phase.OVER)
    save(app, "02_midgame_your_turn.png")
    risk = [e for e in app.session.agent.log.entries
            if e.tag == "RISK_CALC" and any("Consistent models" in d for d in e.details)]
    if risk:
        from wumpus.ui.brain_panel import Inspector
        app.inspector = Inspector(app.session.agent.log, risk[-1].id)
        app.frame()
        save(app, "03_inspector_evidence.png")
        app.inspector = None
    app.menu_open = True
    app.frame()
    save(app, "04_menu.png")
    pygame.quit()

    # 5: AI-vs-AI (Minimax Wumpus) to the end
    app = App(seed=5, wumpus_mode="random", headless=True)
    app.speed_i = 4
    _run(app, lambda: app.session.phase is Phase.OVER, max_frames=8000)
    save(app, "05_game_over.png")
    pygame.quit()
    return shots


if __name__ == "__main__":
    for p in capture():
        print(p)
