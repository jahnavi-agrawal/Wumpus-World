"""WUMPUS: THE MIND GAMES - Pygame interface (1280x800, dark neon).

Controls:  SPACE step | P pause/resume | R restart | N new map | ESC menu | +/- speed | F log filter
           Wumpus (your turn): WASD / arrow keys, X = stay, or click a highlighted cell
           Brain Activity: mouse wheel to scroll, click an entry to inspect its evidence
"""
from __future__ import annotations

import os
import random
import time

import pygame

from ..core.actions import STAY, WumpusAction
from ..core.geometry import Direction, Pos
from ..environment.engine import Phase
from . import icons
from . import theme as T
from .anim import Animator
from .board_view import draw_mind, draw_reality
from .brain_panel import BrainPanel, Inspector
from .session import GameSession

KEY_DIRS = {pygame.K_w: Direction.N, pygame.K_UP: Direction.N, pygame.K_d: Direction.E, pygame.K_RIGHT: Direction.E,
            pygame.K_s: Direction.S, pygame.K_DOWN: Direction.S, pygame.K_a: Direction.W, pygame.K_LEFT: Direction.W}
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)
MENU_ITEMS = ("Resume", "Restart (same map)", "New map", "Wumpus: {mode}", "Speed: {speed}x", "Quit")


class App:
    def __init__(self, seed: int = 7, wumpus_mode: str = "human", headless: bool = False):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        pygame.init()
        T.reset_fonts()
        pygame.display.set_caption("WUMPUS: THE MIND GAMES")
        self.screen = pygame.display.set_mode((T.WIDTH, T.HEIGHT))
        self.clock = pygame.time.Clock()
        self.session = GameSession(seed, wumpus_mode)
        self.anim = Animator()
        self.brain = BrainPanel(T.BRAIN)
        self.inspector: Inspector | None = None
        self.paused = False
        self.speed_i = 2
        self.menu_open = False
        self.menu_rows: list[tuple[pygame.Rect, int]] = []
        self.menu_sel = 0
        self.running = True
        self.step_requests = 0            # SPACE presses waiting to be executed (single-step)
        self.next_auto = 0.0
        self.t0 = time.perf_counter()
        self.frames = 0

    # ------------------------------------------------------------ helpers
    @property
    def speed(self) -> float:
        return SPEEDS[self.speed_i]

    def _delay(self) -> float:
        return 0.55 / self.speed

    def restart(self, seed=None):
        self.session.restart(seed)
        self.anim.clear()
        self.inspector = None
        self.brain.scroll = 0
        self.step_requests = 0

    def _apply(self, events, before_e: Pos, before_w: Pos, n_before: int):
        if events is None:
            return
        self.anim.speed = self.speed
        self.anim.on_events(events, before_e, before_w)
        self.anim.on_entries(self.session.agent.log.entries[n_before:self.session.log_count])
        self.next_auto = time.perf_counter() + self._delay()

    def submit_wumpus(self, action: WumpusAction) -> bool:
        s = self.session
        w = s.reality.world
        ev = s.submit_wumpus(action)
        self._apply(ev, w.explorer, w.wumpus, s.log_count)
        return ev is not None

    # ------------------------------------------------------------ input
    def handle(self, ev):
        if ev.type == pygame.QUIT:
            self.running = False
        elif ev.type == pygame.KEYDOWN:
            self.on_key(ev.key)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.on_click(ev.pos)
        elif ev.type == pygame.MOUSEWHEEL:
            if self.inspector:
                self.inspector.on_wheel(-ev.y)
            else:
                self.brain.on_wheel(ev.y, self.session.entries())

    def on_key(self, key):
        if self.inspector is not None:
            if key == pygame.K_ESCAPE:
                self.inspector = None
            return
        if self.menu_open:
            if key == pygame.K_ESCAPE:
                self.menu_open = False
            elif key in (pygame.K_UP, pygame.K_w):
                self.menu_sel = (self.menu_sel - 1) % len(MENU_ITEMS)
            elif key in (pygame.K_DOWN, pygame.K_s):
                self.menu_sel = (self.menu_sel + 1) % len(MENU_ITEMS)
            elif key in (pygame.K_RETURN, pygame.K_SPACE):
                self.menu_action(self.menu_sel)
            return
        if key == pygame.K_ESCAPE:
            self.menu_open = True
        elif key == pygame.K_SPACE:
            self.step_requests += 1
        elif key == pygame.K_p:
            self.paused = not self.paused
        elif key == pygame.K_r:
            self.restart()
        elif key == pygame.K_n:
            self.restart(random.randrange(10_000))
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.speed_i = min(len(SPEEDS) - 1, self.speed_i + 1)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed_i = max(0, self.speed_i - 1)
        elif key == pygame.K_f:
            self.brain.key_only = not self.brain.key_only
            self.brain.scroll = 0
        elif self.session.human_turn and key in KEY_DIRS:
            self.submit_wumpus(WumpusAction.move(KEY_DIRS[key]))
        elif self.session.human_turn and key == pygame.K_x:
            self.submit_wumpus(STAY)

    def on_click(self, pos):
        if self.inspector is not None:
            if self.inspector.click(pos) == "close":
                self.inspector = None
            return
        if self.menu_open:
            for r, i in self.menu_rows:
                if r.collidepoint(pos):
                    self.menu_action(i)
            return
        eid = self.brain.click(pos)
        if eid is not None:
            self.inspector = Inspector(self.session.agent.log, eid)
            return
        cell = T.cell_at(T.REALITY, *pos)
        if cell is not None and self.session.human_turn:
            w = self.session.reality.world.wumpus
            target = Pos(*cell)
            if target == w:
                self.submit_wumpus(STAY)
            else:
                for d in Direction:
                    if w.step(d) == target:
                        self.submit_wumpus(WumpusAction.move(d))

    def menu_action(self, i: int):
        name = MENU_ITEMS[i]
        if name == "Resume":
            self.menu_open = False
        elif name.startswith("Restart"):
            self.restart()
            self.menu_open = False
        elif name == "New map":
            self.restart(random.randrange(10_000))
            self.menu_open = False
        elif name.startswith("Wumpus"):
            modes = GameSession.WUMPUS_MODES
            self.session.set_wumpus_mode(modes[(modes.index(self.session.wumpus_mode) + 1) % len(modes)])
        elif name.startswith("Speed"):
            self.speed_i = (self.speed_i + 1) % len(SPEEDS)
        elif name == "Quit":
            self.running = False

    # ------------------------------------------------------------ game flow
    def update(self):
        s = self.session
        if s.busy:
            w = s.reality.world
            n0 = s.log_count
            ev = s.poll()
            if ev is not None:
                self._apply(ev, w.explorer, w.wumpus, n0)
            return
        if self.inspector or self.menu_open or s.phase is Phase.OVER or self.anim.busy:
            return
        now = time.perf_counter()
        want_step = self.step_requests > 0 or (not self.paused and now >= self.next_auto)
        if not want_step:
            return
        if s.phase is Phase.EXPLORER:
            if s.request_explorer_step() and self.step_requests:
                self.step_requests -= 1
        elif s.phase is Phase.WUMPUS and s.controller is not None:
            w = s.reality.world
            ev = s.wumpus_ai_step()
            self._apply(ev, w.explorer, w.wumpus, s.log_count)
            if self.step_requests:
                self.step_requests -= 1
        else:
            self.step_requests = 0          # human Wumpus: waits for your move

    # ------------------------------------------------------------ drawing
    def draw_header(self):
        s = self.session
        st = s.reality.status
        surf = self.screen
        pygame.draw.rect(surf, T.PANEL, T.HEADER)
        pygame.draw.line(surf, T.BORDER, (0, T.HEADER.bottom - 1), (T.WIDTH, T.HEADER.bottom - 1))
        title = T.font(20, mono=False, bold=True).render("WUMPUS:", True, T.CYAN)
        surf.blit(title, (T.MARGIN, 15))
        surf.blit(T.font(20, mono=False, bold=True).render("THE MIND GAMES", True, T.VIOLET),
                  (T.MARGIN + title.get_width() + 8, 15))
        t = time.perf_counter() - self.t0
        x = T.WIDTH - T.MARGIN
        items = [("TURN", f"{st.turn:03d}"), ("ARROWS", str(st.arrows)), ("SCORE", str(st.score)),
                 ("SEED", str(s.seed)), ("SPEED", f"{self.speed:g}x"), ("WUMPUS", s.wumpus_mode.upper())]
        live = not self.paused and s.phase is not Phase.OVER
        label = "LIVE" if live else ("PAUSED" if self.paused else "ENDED")
        col = T.EMERALD if live else (T.AMBER if self.paused else T.CORAL)
        img = T.font(13, bold=True).render(label, True, col)
        x -= img.get_width()
        surf.blit(img, (x, 20))
        icons.live_dot(surf, (x - 14, 28), t if live else 0.0, col)
        x -= 44
        for k, v in items:
            vi = T.font(16, bold=True).render(v, True, T.TEXT)
            ki = T.font(10, bold=True).render(k, True, T.TEXT_FAINT)
            wdt = max(vi.get_width(), ki.get_width())
            x -= wdt
            surf.blit(ki, (x, 10))
            surf.blit(vi, (x, 24))
            x -= 26

    def draw_banner(self):
        s = self.session
        if s.phase is not Phase.OVER:
            return
        res = s.engine.result
        col = T.EMERALD if res.outcome.name == "EXPLORER_WIN" else (T.AMBER if res.outcome.name == "DRAW" else T.CORAL)
        reason = res.reason.replace("_", " ").lower()
        title = res.outcome.value if reason in res.outcome.value.lower() or reason.startswith("climbed")             else f"{res.outcome.value} ({reason})"
        a = T.font(18, bold=True).render(title, True, col)
        b = T.font(12).render(f"final score {s.engine.score}   |   R restart   N new map   ESC menu", True, T.TEXT_DIM)
        r = pygame.Rect(0, 0, max(a.get_width(), b.get_width()) + 48, 64)
        r.center = (T.WIDTH // 2, T.REALITY.bottom - 46)
        pygame.draw.rect(self.screen, T.BG, r, border_radius=12)
        pygame.draw.rect(self.screen, col, r, 2, border_radius=12)
        self.screen.blit(a, a.get_rect(center=(r.centerx, r.top + 22)))
        self.screen.blit(b, b.get_rect(center=(r.centerx, r.top + 45)))

    def draw_menu(self):
        dim = pygame.Surface((T.WIDTH, T.HEIGHT), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 170))
        self.screen.blit(dim, (0, 0))
        r = pygame.Rect(0, 0, 460, 96 + 46 * len(MENU_ITEMS))
        r.center = (T.WIDTH // 2, T.HEIGHT // 2)
        pygame.draw.rect(self.screen, T.PANEL, r, border_radius=T.RADIUS)
        pygame.draw.rect(self.screen, T.VIOLET, r, 2, border_radius=T.RADIUS)
        self.screen.blit(T.font(18, bold=True).render("MENU", True, T.VIOLET), (r.left + 24, r.top + 18))
        self.menu_rows = []
        mouse = pygame.mouse.get_pos()
        for i, name in enumerate(MENU_ITEMS):
            label = name.format(mode=self.session.wumpus_mode, speed=f"{self.speed:g}")
            row = pygame.Rect(r.left + 20, r.top + 58 + i * 46, r.w - 40, 38)
            hot = row.collidepoint(mouse) or i == self.menu_sel
            pygame.draw.rect(self.screen, T.TILE if hot else T.BG, row, border_radius=8)
            self.screen.blit(T.font(15, bold=hot).render(label, True, T.CYAN if hot else T.TEXT), (row.left + 14, row.top + 9))
            self.menu_rows.append((row, i))
        hint = T.font(11).render("SPACE step  P pause  R restart  N new map  +/- speed  F log filter", True, T.TEXT_FAINT)
        self.screen.blit(hint, hint.get_rect(center=(r.centerx, r.bottom - 20)))

    def draw(self):
        s = self.session
        self.screen.fill(T.BG)
        self.draw_header()
        hover = T.cell_at(T.REALITY, *pygame.mouse.get_pos())
        draw_reality(self.screen, T.REALITY, s.reality, self.anim, Pos(*hover) if hover else None)
        draw_mind(self.screen, T.MIND, s.mind, self.anim)
        self.brain.draw(self.screen, s.entries(), s.mind.report, s.last_think_s, s.busy)
        self.draw_banner()
        if self.menu_open:
            self.draw_menu()
        if self.inspector:
            self.inspector.draw(self.screen)
        pygame.display.flip()

    def frame(self):
        for ev in pygame.event.get():
            self.handle(ev)
        self.update()
        self.draw()
        self.clock.tick(T.FPS)
        self.frames += 1

    def run(self, max_frames: int | None = None):
        while self.running and (max_frames is None or self.frames < max_frames):
            self.frame()
        pygame.quit()


def run_gui(seed: int = 7, wumpus_mode: str = "human") -> None:
    App(seed, wumpus_mode).run()
