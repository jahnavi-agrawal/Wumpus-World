"""How-to-play screens shown before the game starts (and with H at any time)."""
from __future__ import annotations

import time

import pygame

from . import icons
from . import theme as T
from .brain_panel import _wrap

# Each page: (title, [(icon, heading, text), ...]). Icons are drawn with ui/icons.py.
PAGES = [
    ("YOU ARE THE WUMPUS", [
        ("wumpus", "Your goal: stop the explorer",
         "An AI explorer enters at the bottom-left corner, looking for the gold. "
         "YOU WIN if it falls into a pit, if it walks into you or you step onto it, or if it has not "
         "escaped after 150 turns. IT WINS if it grabs the gold and climbs out at the entrance. If it "
         "climbs out without the gold, the game is a draw."),
        ("move", "Your turn",
         "After every explorer action you move ONE cell up, down, left or right, or stay. Use "
         "W A S D or the arrow keys (X = stay), or click one of the cyan-outlined cells. You cannot "
         "walk into walls or pits. Stepping onto a portal teleports you to its twin."),
        ("stench", "What the explorer can sense",
         "It cannot see the map. It smells a STENCH only when you are directly next to it (up, down, "
         "left or right; diagonal does not count). It feels a BREEZE next to a pit and sees GLITTER "
         "on the gold."),
        ("explorer", "What it can do to you",
         "It has ONE arrow, which flies straight in the direction its chevron points. If you are anywhere "
         "on that line, you die. It also has ONE jump that lands 3 cells ahead."),
        ("tips", "Tips",
         "Approach diagonally: it cannot smell you there. Never wait in a straight line in front of it. "
         "Use portals to cut across the map. Block its way home once it has the gold."),
    ]),
    ("READING THE AI'S MIND", [
        ("reality", "Panel 01 - Reality",
         "The real dungeon. Only you see it: pits (vortex), gold (octagon), portal pairs A/B, cracked "
         "tiles that collapse after the explorer leaves them, shuffle tiles that move a pit. Small B/S "
         "hexagons show where a breeze or stench can be felt."),
        ("mind", "Panel 02 - Agent's Mind",
         "Only what the explorer knows. Dark green: visited. Green: proven safe by logic. Red tint: "
         "suspected pit. Amber outline: where it thinks YOU are. Red X: confirmed danger. Percentages: "
         "its estimated chance of dying there. Cyan dashed line: its planned route to the GOAL cell."),
        ("brain", "Panel 03 - Brain Activity",
         "Its real reasoning, step by step: INFERENCE (logical proofs), RISK_CALC (probabilities), "
         "A*_PLANNER (routes), MINIMAX (thinking about your next move), DECISION (what it does). "
         "Click any line to see the exact evidence and calculation."),
        ("back", "Why does it walk back over visited cells?",
         "It only learns what is in a cell by standing in it. It steps forward to probe a new cell. If it "
         "feels a breeze, any unexplored neighbour might be a pit, so it steps back to safe ground. "
         "Cells it has already visited are the only ones it knows are safe, so it walks back through "
         "them to reach the next unexplored area or the exit. The log then says 'backtracking ...' "
         "and gives the reason."),
    ]),
    ("CONTROLS", [
        ("move", "Moving the Wumpus", "W A S D or arrow keys, X to stay, or click a cyan-outlined cell."),
        ("keys", "Game", "SPACE single step  |  P pause / resume  |  R restart same map  |  N new map  |  "
                         "+ / - speed  |  ESC menu  |  H this help"),
        ("brain", "Brain Activity", "Mouse wheel scrolls the log. Click a line to inspect it, then click a "
                                    "'depends on' line to follow the chain of reasoning. F shows all or "
                                    "only the key log lines."),
        ("tips", "Watch the AI instead", "In the ESC menu, set 'Wumpus' to random, hunter or minimax to "
                                         "let an AI play the Wumpus."),
    ]),
]


class HelpScreen:
    def __init__(self):
        self.page = 0
        self.rect = pygame.Rect(90, 60, T.WIDTH - 180, T.HEIGHT - 120)
        self.buttons: dict[str, pygame.Rect] = {}

    @property
    def last(self) -> bool:
        return self.page == len(PAGES) - 1

    def key(self, k) -> str:
        """Returns 'close' when the help should close."""
        if k in (pygame.K_RIGHT, pygame.K_d):
            self.page = min(len(PAGES) - 1, self.page + 1)
        elif k in (pygame.K_LEFT, pygame.K_a):
            self.page = max(0, self.page - 1)
        elif k in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE, pygame.K_ESCAPE, pygame.K_h):
            return "close"
        return "none"

    def click(self, pos) -> str:
        for name, r in self.buttons.items():
            if r.collidepoint(pos):
                if name == "start":
                    return "close"
                if name == "next":
                    self.page = min(len(PAGES) - 1, self.page + 1)
                elif name == "back":
                    self.page = max(0, self.page - 1)
                elif name.startswith("tab"):
                    self.page = int(name[3:])
        return "none"

    # ------------------------------------------------------------ drawing
    def _icon(self, surf, kind, rect, t):
        if kind == "wumpus":
            icons.wumpus(surf, rect)
        elif kind == "explorer":
            icons.explorer(surf, rect, "E")
        elif kind == "stench":
            icons.hex_letter(surf, (rect.centerx - 10, rect.centery), 11, "S", T.AMBER)
            icons.hex_letter(surf, (rect.centerx + 12, rect.centery), 11, "B", T.CYAN)
        elif kind == "move":
            for d, (dx, dy) in {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}.items():
                r = pygame.Rect(0, 0, 14, 14)
                r.center = (rect.centerx + dx * 15, rect.centery + dy * 15)
                pygame.draw.rect(surf, T.CYAN, r, 2, border_radius=4)
            icons.wumpus(surf, rect.inflate(-24, -24))
        elif kind == "reality":
            icons.pit(surf, rect, t)
        elif kind == "mind":
            for i, col in enumerate((T.EMERALD_DARK, T.EMERALD, T.CRIMSON, T.AMBER)):
                pygame.draw.rect(surf, col, pygame.Rect(rect.left + 4 + (i % 2) * 20, rect.top + 4 + (i // 2) * 20, 18, 18),
                                 border_radius=4)
        elif kind == "brain":
            for i, col in enumerate((T.EMERALD, T.YELLOW, T.CYAN, T.VIOLET, T.CORAL)):
                pygame.draw.line(surf, col, (rect.left + 4, rect.top + 6 + i * 8), (rect.right - 4, rect.top + 6 + i * 8), 3)
        elif kind == "back":
            a, b = (rect.left + 8, rect.centery), (rect.right - 8, rect.centery)
            pygame.draw.rect(surf, T.EMERALD_DARK, pygame.Rect(rect.left + 2, rect.top + 12, 20, 20), border_radius=4)
            pygame.draw.rect(surf, T.CRIMSON, pygame.Rect(rect.right - 22, rect.top + 12, 20, 20), border_radius=4)
            pygame.draw.line(surf, T.CYAN, a, b, 2)
            pygame.draw.polygon(surf, T.CYAN, [(a[0], a[1]), (a[0] + 8, a[1] - 5), (a[0] + 8, a[1] + 5)])
        elif kind == "keys":
            for i, ch in enumerate("SPR"):
                r = pygame.Rect(rect.left + 2 + i * 15, rect.centery - 8, 14, 16)
                pygame.draw.rect(surf, T.BORDER, r, border_radius=3)
                img = T.font(10, bold=True).render(ch, True, T.TEXT)
                surf.blit(img, img.get_rect(center=r.center))
        elif kind == "tips":
            icons.gold(surf, rect, t)

    def _button(self, surf, name, label, rect, primary=False):
        hot = rect.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(surf, (T.EMERALD if primary else T.TILE) if not hot else T.CYAN, rect, border_radius=10)
        img = T.font(15, bold=True).render(label, True, T.BG if (primary or hot) else T.TEXT)
        surf.blit(img, img.get_rect(center=rect.center))
        self.buttons[name] = rect

    def draw(self, surf):
        t = time.perf_counter()
        surf.fill(T.BG)
        r = self.rect
        pygame.draw.rect(surf, T.PANEL, r, border_radius=T.RADIUS)
        pygame.draw.rect(surf, T.VIOLET, r, 2, border_radius=T.RADIUS)
        a = T.font(26, mono=False, bold=True).render("WUMPUS:", True, T.CYAN)
        surf.blit(a, (r.left + 32, r.top + 22))
        surf.blit(T.font(26, mono=False, bold=True).render("THE MIND GAMES", True, T.VIOLET), (r.left + 42 + a.get_width(), r.top + 22))
        self.buttons = {}
        # tabs
        x = r.left + 32
        for i, (title, _) in enumerate(PAGES):
            img = T.font(13, bold=True).render(f"{i + 1}. {title}", True, T.CYAN if i == self.page else T.TEXT_FAINT)
            tab = pygame.Rect(x - 8, r.top + 68, img.get_width() + 16, 26)
            if i == self.page:
                pygame.draw.rect(surf, T.BG, tab, border_radius=8)
            surf.blit(img, (x, r.top + 73))
            self.buttons[f"tab{i}"] = tab
            x += tab.w + 10
        # items
        title, items = PAGES[self.page]
        y = r.top + 112
        text_w = r.w - 64 - 70
        for kind, head, text in items:
            box = pygame.Rect(r.left + 32, y, 48, 48)
            pygame.draw.rect(surf, T.BG, box, border_radius=10)
            self._icon(surf, kind, box, t)
            surf.blit(T.font(16, mono=False, bold=True).render(head, True, T.TEXT), (box.right + 22, y))
            yy = y + 24
            for line in _wrap(text, T.font(15, mono=False), text_w):
                surf.blit(T.font(15, mono=False).render(line, True, (0xCB, 0xD5, 0xE1)), (box.right + 22, yy))
                yy += 21
            y = max(yy, box.bottom) + 14
        # buttons
        by = r.bottom - 62
        if self.page > 0:
            self._button(surf, "back", "< Back", pygame.Rect(r.left + 32, by, 130, 40))
        if not self.last:
            self._button(surf, "next", "Next >", pygame.Rect(r.right - 32 - 360, by, 130, 40))
        self._button(surf, "start", "Start the game", pygame.Rect(r.right - 32 - 210, by, 210, 40), primary=True)
        hint = T.font(11).render("LEFT / RIGHT: pages   ENTER: start   H: open this help again during the game",
                                 True, T.TEXT_FAINT)
        surf.blit(hint, hint.get_rect(center=(r.centerx, r.bottom - 14)))
