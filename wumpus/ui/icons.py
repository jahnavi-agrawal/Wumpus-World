"""Vector icons drawn with pygame primitives (no image assets, no emoji)."""
from __future__ import annotations

import math

import pygame
import pygame.gfxdraw

from . import theme as T

_ANGLE = {"N": -90, "E": 0, "S": 90, "W": 180}       # screen angles (y grows downwards)


def _rot(cx, cy, pts, deg):
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    return [(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in pts]


def _aapoly(surf, pts, color, filled=True):
    ipts = [(int(round(x)), int(round(y))) for x, y in pts]
    if filled:
        pygame.gfxdraw.filled_polygon(surf, ipts, color)
    pygame.gfxdraw.aapolygon(surf, ipts, color)


def explorer(surf, rect: pygame.Rect, facing_name: str, color=T.CYAN, glow=True):
    """Cyan directional chevron pointing in the facing direction."""
    cx, cy, s = rect.centerx, rect.centery, rect.w * 0.36
    if glow:
        g = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.circle(g, (*color, 40), (rect.w // 2, rect.h // 2), int(rect.w * 0.46))
        surf.blit(g, rect.topleft)
    pts = [(s, 0), (-s * 0.8, -s * 0.85), (-s * 0.35, 0), (-s * 0.8, s * 0.85)]
    _aapoly(surf, _rot(cx, cy, pts, _ANGLE[facing_name]), color)


def wumpus(surf, rect: pygame.Rect, alive=True):
    """Crimson horned silhouette: head with two horns and glowing eyes."""
    c = T.CRIMSON if alive else T.TEXT_FAINT
    cx, cy, r = rect.centerx, rect.centery + rect.h * 0.06, rect.w * 0.27
    horn_l = [(cx - r * 0.9, cy - r * 0.4), (cx - r * 1.35, cy - r * 1.55), (cx - r * 0.25, cy - r * 0.85)]
    horn_r = [(cx + r * 0.9, cy - r * 0.4), (cx + r * 1.35, cy - r * 1.55), (cx + r * 0.25, cy - r * 0.85)]
    _aapoly(surf, horn_l, c)
    _aapoly(surf, horn_r, c)
    pygame.gfxdraw.filled_circle(surf, int(cx), int(cy), int(r), c)
    pygame.gfxdraw.aacircle(surf, int(cx), int(cy), int(r), c)
    jaw = [(cx - r * 0.8, cy + r * 0.4), (cx + r * 0.8, cy + r * 0.4), (cx, cy + r * 1.35)]
    _aapoly(surf, jaw, c)
    if alive:
        for dx in (-0.38, 0.38):
            pygame.draw.circle(surf, T.YELLOW, (int(cx + r * dx), int(cy - r * 0.1)), max(2, int(r * 0.17)))
    else:
        pygame.draw.line(surf, T.RED, (rect.left + 8, rect.top + 8), (rect.right - 8, rect.bottom - 8), 3)
        pygame.draw.line(surf, T.RED, (rect.right - 8, rect.top + 8), (rect.left + 8, rect.bottom - 8), 3)


def pit(surf, rect: pygame.Rect, t: float = 0.0):
    """Violet/blue vortex: concentric rings with a rotating spiral."""
    cx, cy = rect.center
    R = rect.w * 0.40
    pygame.gfxdraw.filled_circle(surf, cx, cy, int(R), (0x0B, 0x10, 0x24))
    for i, col in enumerate((T.PIT_BLUE, T.VIOLET, (0x4C, 0x1D, 0x95))):
        pygame.gfxdraw.aacircle(surf, cx, cy, int(R * (1 - i * 0.28)), col)
    for k in range(3):
        a0 = t * 2.0 + k * 2 * math.pi / 3
        pts = [(cx + math.cos(a0 + j * 0.35) * R * (0.95 - j * 0.11),
                cy + math.sin(a0 + j * 0.35) * R * (0.95 - j * 0.11)) for j in range(8)]
        pygame.draw.aalines(surf, T.VIOLET, False, pts)


def gold(surf, rect: pygame.Rect, t: float = 0.0):
    """Faceted yellow octagon."""
    cx, cy, r = rect.centerx, rect.centery, rect.w * 0.30 * (1 + 0.04 * math.sin(t * 4))
    outer = [(cx + r * math.cos(math.pi / 8 + k * math.pi / 4), cy + r * math.sin(math.pi / 8 + k * math.pi / 4))
             for k in range(8)]
    inner = [(cx + (x - cx) * 0.55, cy + (y - cy) * 0.55) for x, y in outer]
    _aapoly(surf, outer, (0xCA, 0x8A, 0x04))
    _aapoly(surf, inner, T.GOLD)
    for (x1, y1), (x2, y2) in zip(outer, inner):
        pygame.draw.aaline(surf, (0xFE, 0xF0, 0x8A), (x1, y1), (x2, y2))


def hex_letter(surf, center, r: float, letter: str, color, lit=True):
    """Hexagonal badge with a letter (amber S = stench, cyan B = breeze)."""
    cx, cy = center
    pts = [(cx + r * math.cos(math.pi / 6 + k * math.pi / 3), cy + r * math.sin(math.pi / 6 + k * math.pi / 3))
           for k in range(6)]
    col = color if lit else T.BORDER
    _aapoly(surf, pts, (*T.BG, ), True)
    pygame.draw.aalines(surf, col, True, pts)
    f = T.font(max(9, int(r * 1.25)), mono=True, bold=True)
    img = f.render(letter, True, col)
    surf.blit(img, img.get_rect(center=(cx, cy + 1)))


def portal(surf, rect: pygame.Rect, label: str = "", t: float = 0.0, color=T.VIOLET):
    """Violet portal: pulsing double ellipse with a label."""
    cx, cy = rect.center
    k = 1 + 0.06 * math.sin(t * 5)
    for i, a in enumerate((0.42, 0.30)):
        e = pygame.Rect(0, 0, rect.w * a * 2 * k, rect.h * a * 1.5 * k)
        e.center = (cx, cy)
        pygame.draw.ellipse(surf, color if i == 0 else (0xC4, 0xB5, 0xFD), e, 2)
    if label:
        img = T.font(12, bold=True).render(label, True, T.TEXT)
        surf.blit(img, img.get_rect(center=(cx, cy)))


def cracked(surf, rect: pygame.Rect, collapsed=False):
    """Cracked floor (unstable) or a broken hole (collapsed)."""
    cx, cy = rect.center
    if collapsed:
        pygame.draw.rect(surf, (0x0B, 0x0F, 0x1A), rect.inflate(-8, -8), border_radius=6)
        pygame.draw.rect(surf, T.RED, rect.inflate(-8, -8), 2, border_radius=6)
    col = T.RED if collapsed else T.ORANGE
    lines = [[(cx - 16, cy - 14), (cx - 4, cy - 3), (cx - 9, cy + 7), (cx + 2, cy + 16)],
             [(cx - 4, cy - 3), (cx + 9, cy - 9), (cx + 17, cy - 16)],
             [(cx - 9, cy + 7), (cx + 8, cy + 5), (cx + 15, cy + 13)]]
    for ln in lines:
        pygame.draw.lines(surf, col, False, ln, 2)


def shuffle(surf, rect: pygame.Rect, used=False):
    """Two interlocking curved arrows."""
    cx, cy = rect.center
    col = T.TEXT_FAINT if used else (0x2D, 0xD4, 0xBF)
    r = rect.w * 0.26
    for sign in (1, -1):
        box = pygame.Rect(0, 0, r * 2, r * 2)
        box.center = (cx, cy)
        start = 0.3 if sign == 1 else math.pi + 0.3
        pygame.draw.arc(surf, col, box, start, start + 2.4, 3)
        a = start + 2.4
        tip = (cx + r * math.cos(a), cy - r * math.sin(a))
        d = a + math.pi / 2
        head = [tip, (tip[0] - 7 * math.cos(d - 0.5), tip[1] + 7 * math.sin(d - 0.5)),
                (tip[0] - 7 * math.cos(d + 0.5), tip[1] + 7 * math.sin(d + 0.5))]
        _aapoly(surf, head, col)


def live_dot(surf, center, t: float, color=T.EMERALD):
    k = 0.5 + 0.5 * math.sin(t * 4)
    g = pygame.Surface((30, 30), pygame.SRCALPHA)
    pygame.draw.circle(g, (*color, int(60 + 80 * k)), (15, 15), int(8 + 5 * k))
    surf.blit(g, (center[0] - 15, center[1] - 15))
    pygame.draw.circle(surf, color, center, 5)


def dashed_line(surf, color, p1, p2, width=2, dash=7, gap=5, phase=0.0):
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    pos = -phase % (dash + gap)
    while pos < length:
        a = max(0.0, pos)
        b = min(length, pos + dash)
        if b > a:
            pygame.draw.line(surf, color, (x1 + dx * a, y1 + dy * a), (x1 + dx * b, y1 + dy * b), width)
        pos += dash + gap
