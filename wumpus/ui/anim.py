"""Small animation system: movement tweens and cell flashes, driven by structured GameEvents."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import pygame

from ..core.geometry import Pos
from ..environment.events import EventKind as K
from . import theme as T


@dataclass
class Tween:
    entity: str          # "explorer" | "wumpus"
    path: list           # list of Pos (cells), visited in order
    start: float
    duration: float
    arc: bool = False    # jump: hop above the line

    def pos(self, now: float):
        k = min(1.0, (now - self.start) / self.duration) if self.duration > 0 else 1.0
        k = k * k * (3 - 2 * k)                       # smoothstep
        n = len(self.path) - 1
        if n <= 0:
            return self.path[0], 0.0
        f = k * n
        i = min(int(f), n - 1)
        a, b = self.path[i], self.path[i + 1]
        u = f - i
        lift = math.sin(k * math.pi) * 0.8 if self.arc else 0.0
        return (a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u), lift


@dataclass
class Flash:
    panel: str           # "reality" | "mind"
    kind: str            # teleport | collapse | shuffle | gold | infer | route | hit | death
    cell: Pos
    start: float
    duration: float
    color: tuple = T.CYAN


@dataclass
class Animator:
    speed: float = 1.0
    tweens: list = field(default_factory=list)
    flashes: list = field(default_factory=list)

    def now(self) -> float:
        return time.perf_counter()

    @property
    def busy(self) -> bool:
        t = self.now()
        return any(t < tw.start + tw.duration for tw in self.tweens)

    def clear(self):
        self.tweens.clear()
        self.flashes.clear()

    def entity_pos(self, entity: str):
        t = self.now()
        live = [tw for tw in self.tweens if tw.entity == entity and t < tw.start + tw.duration]
        return live[-1].pos(t) if live else None

    def _flash(self, panel, kind, cell, color, dur=0.9):
        self.flashes.append(Flash(panel, kind, cell, self.now(), dur / self.speed, color))

    # ------------------------------------------------------------ events -> animations
    def on_events(self, events, before_explorer: Pos, before_wumpus: Pos):
        t, d = self.now(), 0.28 / self.speed
        exp_path, w_path, jump = [before_explorer], [before_wumpus], False
        for ev in events:
            k = ev.kind
            if ev.actor == "explorer" and k is K.MOVED:
                exp_path.append(ev.end)
            elif ev.actor == "explorer" and k is K.JUMPED:
                exp_path.append(ev.end)
                jump = True
            elif k is K.TELEPORTED:
                self._flash("reality", "teleport", ev.start, T.VIOLET)
                self._flash("reality", "teleport", ev.end, T.VIOLET)
                if ev.actor == "explorer":
                    exp_path = [ev.end]                       # instant relocation after the ring
                else:
                    w_path = [ev.end]
            elif k is K.WUMPUS_MOVED:
                w_path.append(ev.end)
            elif k is K.TILE_COLLAPSED:
                self._flash("reality", "collapse", ev.start, T.RED, 1.2)
            elif k is K.CELLS_SHUFFLED:
                self._flash("reality", "shuffle", ev.start, (0x2D, 0xD4, 0xBF), 1.2)
                if ev.data.get("swapped"):
                    for c in ev.data["swapped"]:
                        self._flash("reality", "shuffle", Pos(*c), (0x2D, 0xD4, 0xBF), 1.2)
            elif k is K.GRABBED_GOLD:
                self._flash("reality", "gold", ev.start, T.GOLD, 1.2)
            elif k is K.ARROW_HIT:
                self._flash("reality", "hit", ev.end, T.RED, 1.0)
            elif k in (K.EATEN, K.FELL_INTO_PIT):
                self._flash("reality", "death", ev.end, T.RED, 1.5)
        if len(exp_path) > 1:
            self.tweens.append(Tween("explorer", exp_path, t, d * (len(exp_path) - 1) * (1.6 if jump else 1), jump))
        if len(w_path) > 1:
            self.tweens.append(Tween("wumpus", w_path, t, d))

    def on_entries(self, entries):
        for e in entries:
            if e.tag == "INFERENCE" and e.cell is not None:
                self._flash("mind", "infer", e.cell, T.EMERALD, 1.4)
            elif e.tag == "A*_PLANNER" and e.cell is not None:
                self._flash("mind", "route", e.cell, T.CYAN, 1.0)

    # ------------------------------------------------------------ drawing
    def draw_flashes(self, surf, panel: str, card: pygame.Rect):
        t = self.now()
        alive = []
        for f in self.flashes:
            k = (t - f.start) / f.duration
            if k >= 1:
                continue
            alive.append(f)
            if f.panel != panel:
                continue
            r = T.cell_rect(card, f.cell.x, f.cell.y)
            a = int(255 * (1 - k))
            g = pygame.Surface((r.w + 24, r.h + 24), pygame.SRCALPHA)
            c = (r.w // 2 + 12, r.h // 2 + 12)
            if f.kind in ("teleport", "shuffle", "gold"):
                pygame.draw.circle(g, (*f.color, a), c, int(8 + k * (r.w * 0.7)), 3)
            elif f.kind in ("collapse", "death", "hit"):
                pygame.draw.rect(g, (*f.color, a // 2), pygame.Rect(12, 12, r.w, r.h), border_radius=8)
                pygame.draw.rect(g, (*f.color, a), pygame.Rect(12, 12, r.w, r.h), 3, border_radius=8)
            else:                                            # infer / route: soft pulse
                pygame.draw.rect(g, (*f.color, int(a * 0.55)), pygame.Rect(12, 12, r.w, r.h), 3, border_radius=8)
            surf.blit(g, (r.x - 12, r.y - 12))
        self.flashes = alive
