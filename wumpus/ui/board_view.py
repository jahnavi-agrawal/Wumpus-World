"""PANEL 01 / REALITY (hidden dungeon) and PANEL 02 / AGENT'S MIND (explorer knowledge).

`draw_reality` reads only a RealitySnapshot; `draw_mind` reads only a MindSnapshot, which is built
from the explorer's own knowledge. The two boards share the coordinate layout, not the data.
"""
from __future__ import annotations

import time

import pygame

from ..core.geometry import Pos, neighbors
from ..core.percepts import PERCEPT_FLAGS
from ..environment.engine import Phase
from . import icons
from . import theme as T


def _blend(a, b, k):
    return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3))


def draw_card(surf, rect: pygame.Rect, number: str, title: str, subtitle: str = ""):
    pygame.draw.rect(surf, T.PANEL, rect, border_radius=T.RADIUS)
    pygame.draw.rect(surf, T.BORDER, rect, 1, border_radius=T.RADIUS)
    surf.blit(T.font(13, bold=True).render(f"PANEL {number}", True, T.CYAN), (rect.left + 18, rect.top + 12))
    surf.blit(T.font(15, mono=False, bold=True).render(title, True, T.TEXT), (rect.left + 100, rect.top + 10))
    if subtitle:
        img = T.font(12).render(subtitle, True, T.TEXT_DIM)
        surf.blit(img, (rect.right - 18 - img.get_width(), rect.top + 13))


def _float_rect(card, fx, fy, lift=0.0):
    ox, oy = T.grid_origin(card)
    step = T.TILE_SIZE + T.TILE_GAP
    return pygame.Rect(int(ox + fx * step), int(oy + (7 - fy) * step - lift * step), T.TILE_SIZE, T.TILE_SIZE)


def _side_x(card):
    return T.grid_origin(card)[0] + T.GRID_PX + 18


def _section(surf, x, y, text, color=T.TEXT_DIM):
    surf.blit(T.font(11, bold=True).render(text, True, color), (x, y))
    return y + 18


# =================================================================== reality
def draw_reality(surf, card: pygame.Rect, snap, anim, hover=None):
    t = time.perf_counter()
    w = snap.world
    draw_card(surf, card, "01", "REALITY", "hidden dungeon - only you see this")
    labels, n = {}, 0
    for a in sorted(w.portals):
        if a not in labels:
            labels[a] = labels[w.portals[a]] = "AB"[n] if n < 2 else str(n)
            n += 1
    legal_cells = set()
    if snap.phase is Phase.WUMPUS:
        for a in snap.legal_wumpus:
            if a.direction is not None:
                legal_cells.add(w.wumpus.step(a.direction))
    for y in range(8):
        for x in range(8):
            p = Pos(x, y)
            r = T.cell_rect(card, x, y)
            base = _blend(T.TILE, T.EMERALD_DARK, 0.35) if p in w.visited else T.TILE
            pygame.draw.rect(surf, base, r, border_radius=8)
            if p in w.collapsed:
                icons.cracked(surf, r, collapsed=True)
            elif p in w.pits:
                icons.pit(surf, r, t)
            elif p in w.portals:
                icons.portal(surf, r, labels[p], t)
            elif p in w.unstable:
                icons.cracked(surf, r)
            elif p in w.shuffle_tiles or p in w.used_shuffle:
                icons.shuffle(surf, r, used=p in w.used_shuffle)
            if p == w.gold:
                icons.gold(surf, r, t)
            if p in legal_cells:
                pygame.draw.rect(surf, T.CYAN, r, 2, border_radius=8)
            if hover == p and p in legal_cells:
                pygame.draw.rect(surf, (*T.CYAN,), r.inflate(4, 4), 1, border_radius=10)
    # percept badges at every cell where a percept would be felt (dim) - explorer cell bright
    for y in range(8):
        for x in range(8):
            p = Pos(x, y)
            r = T.cell_rect(card, x, y)
            here = p == w.explorer
            if any(n in w.pits for n in neighbors(p, 8)):
                icons.hex_letter(surf, (r.left + 8, r.top + 8), 6.5, "B", T.CYAN, lit=here)
            if w.wumpus_alive and any(n == w.wumpus for n in neighbors(p, 8)):
                icons.hex_letter(surf, (r.right - 8, r.top + 8), 6.5, "S", T.AMBER, lit=here)
    anim.draw_flashes(surf, "reality", card)
    # Wumpus and explorer (animated)
    wp = anim.entity_pos("wumpus")
    wr = _float_rect(card, *wp[0], wp[1]) if wp else T.cell_rect(card, w.wumpus.x, w.wumpus.y)
    icons.wumpus(surf, wr, alive=w.wumpus_alive)
    if w.explorer_alive:
        ep = anim.entity_pos("explorer")
        er = _float_rect(card, *ep[0], ep[1]) if ep else T.cell_rect(card, w.explorer.x, w.explorer.y)
        icons.explorer(surf, er, w.facing.name)
    _reality_side(surf, card, snap)


def _reality_side(surf, card, snap):
    x, y = _side_x(card), T.grid_origin(card)[1]
    p = snap.percept
    y = _section(surf, x, y, "EXPLORER PERCEPTS")
    names = {"breeze": "Breeze", "stench": "Stench", "glitter": "Glitter", "bump": "Bump", "scream": "Scream",
             "portal_signal": "Portal", "unstable": "Unstable", "shuffled": "Shuffle", "teleported": "Teleport"}
    for f in PERCEPT_FLAGS:
        on = getattr(p, f)
        col = {"breeze": T.CYAN, "stench": T.AMBER, "glitter": T.GOLD}.get(f, T.VIOLET) if on else T.TEXT_FAINT
        pygame.draw.circle(surf, col, (x + 5, y + 7), 4 if on else 3)
        surf.blit(T.font(12).render(names[f], True, T.TEXT if on else T.TEXT_FAINT), (x + 14, y))
        y += 17
    y += 8
    y = _section(surf, x, y, "WUMPUS (YOU)")
    if snap.phase is Phase.WUMPUS:
        lines = [("YOUR TURN", T.EMERALD), ("WASD / arrows", T.TEXT_DIM), ("X = stay", T.TEXT_DIM),
                 ("or click a", T.TEXT_DIM), ("cyan cell", T.TEXT_DIM)]
    elif snap.phase is Phase.OVER:
        lines = [("game over", T.CORAL)]
    else:
        lines = [("explorer's turn", T.TEXT_DIM)]
    for txt, col in lines:
        surf.blit(T.font(12, bold=txt == "YOUR TURN").render(txt, True, col), (x, y))
        y += 16


# =================================================================== agent's mind
def draw_mind(surf, card: pygame.Rect, snap, anim):
    """Explorer's knowledge only (MindSnapshot): never the hidden world."""
    k = snap.knowledge
    draw_card(surf, card, "02", "AGENT'S MIND", f"as perceived at turn {k.turn}")
    belief = snap.wumpus_belief
    frontier = {n for v in k.visited for n in neighbors(v, 8)} - set(k.visited)
    for y in range(8):
        for x in range(8):
            p = Pos(x, y)
            r = T.cell_rect(card, x, y)
            if p in k.confirmed_hazards:
                fill = (0x45, 0x0A, 0x0A)
            elif p in k.visited:
                fill = T.EMERALD_DARK
            elif p in k.known_safe or p in k.pit_free:
                fill = _blend(T.TILE, T.EMERALD, 0.45)
            else:
                fill = T.SLATE
            pygame.draw.rect(surf, fill, r, border_radius=8)
            pp = k.suspected_pits.get(p, 0.0)
            if pp > 0 and p not in k.confirmed_hazards:                 # pit suspicion: crimson
                ov = pygame.Surface(r.size, pygame.SRCALPHA)
                pygame.draw.rect(ov, (*T.CRIMSON, int(40 + 150 * pp)), ov.get_rect(), border_radius=8)
                surf.blit(ov, r.topleft)
            wb = belief.get(p, 0.0)
            if wb >= 0.03:                                              # Wumpus suspicion: amber
                pygame.draw.rect(surf, T.AMBER, r.inflate(-4, -4), max(1, int(1 + 3 * min(1, wb * 2))),
                                 border_radius=7)
            if p in k.confirmed_hazards:                                # confirmed: red indicator
                pygame.draw.rect(surf, T.RED, r, 2, border_radius=8)
                pygame.draw.line(surf, T.RED, (r.left + 12, r.top + 12), (r.right - 12, r.bottom - 12), 3)
                pygame.draw.line(surf, T.RED, (r.right - 12, r.top + 12), (r.left + 12, r.bottom - 12), 3)
            if p in k.portal_connections:
                icons.portal(surf, r.inflate(-14, -14), "", 0.0)
            h = snap.hazards.get(p)
            if h is not None and p in frontier and p not in k.confirmed_hazards and h.p_death >= 0.005:
                col = T.YELLOW if h.p_death < 0.30 else T.ORANGE
                img = T.font(12, bold=True).render(f"{h.p_death:.0%}", True, col)
                surf.blit(img, img.get_rect(center=(r.centerx, r.bottom - 10)))
    anim.draw_flashes(surf, "mind", card)
    # A* route: 2px cyan dashed line through cell centres
    if snap.plan_cells:
        pts = [T.cell_rect(card, snap.explorer.x, snap.explorer.y).center] + \
              [T.cell_rect(card, c.x, c.y).center for c in snap.plan_cells]
        phase = time.perf_counter() * 20
        for a, b in zip(pts, pts[1:]):
            if a != b:
                icons.dashed_line(surf, T.CYAN, a, b, 2, phase=phase)
        if snap.plan_goal is not None:
            pygame.draw.rect(surf, T.CYAN, T.cell_rect(card, snap.plan_goal.x, snap.plan_goal.y), 2, border_radius=8)
    icons.explorer(surf, T.cell_rect(card, snap.explorer.x, snap.explorer.y), snap.facing.name)
    _mind_side(surf, card, snap)


def _mind_side(surf, card, snap):
    k = snap.knowledge
    x, y = _side_x(card), T.grid_origin(card)[1]
    y = _section(surf, x, y, "LEGEND")
    for col, txt in ((T.EMERALD_DARK, "visited"), (_blend(T.TILE, T.EMERALD, 0.45), "proven safe"),
                     (T.SLATE, "unknown"), (T.CRIMSON, "pit suspicion"), (T.AMBER, "Wumpus suspicion"),
                     ((0x45, 0x0A, 0x0A), "confirmed hazard")):
        pygame.draw.rect(surf, col, pygame.Rect(x, y + 2, 12, 12), border_radius=3)
        surf.blit(T.font(12).render(txt, True, T.TEXT_DIM), (x + 18, y))
        y += 17
    surf.blit(T.font(12, bold=True).render("12%", True, T.YELLOW), (x, y))
    surf.blit(T.font(12).render("death risk", True, T.TEXT_DIM), (x + 34, y))
    y += 17
    icons.dashed_line(surf, T.CYAN, (x, y + 7), (x + 26, y + 7), 2)
    surf.blit(T.font(12).render("A* route", True, T.TEXT_DIM), (x + 34, y))
    y += 26
    y = _section(surf, x, y, "KNOWLEDGE")
    for txt in (f"visited   {len(k.visited)}", f"pit-free  {len(k.pit_free)}", f"ambiguous {len(k.ambiguous)}",
                f"hazards   {len(k.confirmed_hazards)}", f"portals   {len(k.portal_connections) // 2}",
                "Wumpus    " + ("alive" if k.wumpus_alive else "dead")):
        surf.blit(T.font(12).render(txt, True, T.TEXT), (x, y))
        y += 16
