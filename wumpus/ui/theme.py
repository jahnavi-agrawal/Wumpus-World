"""Design tokens and layout for the 1280x800 neon interface (exact values from the UI spec)."""
from __future__ import annotations

import pygame

WIDTH, HEIGHT = 1280, 800
FPS = 60

# ---------------------------------------------------------------- palette
BG = (0x0F, 0x17, 0x2A)            # #0F172A
PANEL = (0x1E, 0x29, 0x3B)         # #1E293B
TILE = (0x28, 0x35, 0x48)          # #283548
BORDER = (0x33, 0x41, 0x55)        # #334155
TEXT = (0xE2, 0xE8, 0xF0)
TEXT_DIM = (0x94, 0xA3, 0xB8)
TEXT_FAINT = (0x64, 0x74, 0x8B)
CYAN = (0x38, 0xBD, 0xF8)          # #38BDF8  (also [A*_PLANNER])
VIOLET = (0xA7, 0x8B, 0xFA)        # #A78BFA  (also [MINIMAX])
EMERALD = (0x10, 0xB9, 0x81)       # #10B981  (also [INFERENCE])
EMERALD_DARK = (0x06, 0x4E, 0x3B)
YELLOW = (0xFA, 0xCC, 0x15)        # #FACC15  (also [RISK_CALC])
ORANGE = (0xFB, 0x92, 0x3C)
CORAL = (0xF8, 0x71, 0x71)         # #F87171  (also [DECISION])
CRIMSON = (0xDC, 0x26, 0x26)
RED = (0xEF, 0x44, 0x44)
AMBER = (0xF5, 0x9E, 0x0B)
GOLD = (0xFA, 0xCC, 0x15)
PIT_BLUE = (0x63, 0x66, 0xF1)
SLATE = TILE

TAG_COLORS = {
    "INFERENCE": EMERALD, "RISK_CALC": YELLOW, "A*_PLANNER": CYAN, "MINIMAX": VIOLET, "DECISION": CORAL,
    "EVENT": (0x7D, 0xD3, 0xFC), "REPLAN": ORANGE, "WARNING": RED, "TARGET": (0x67, 0xE8, 0xF9),
    "WUMPUS_BELIEF": AMBER, "PERCEPT": TEXT_DIM, "KB_TELL": TEXT_FAINT,
}
KEY_TAGS = ("INFERENCE", "RISK_CALC", "A*_PLANNER", "MINIMAX", "DECISION", "EVENT", "REPLAN", "WARNING")

RADIUS = 14

# ---------------------------------------------------------------- layout
MARGIN = 16
HEADER = pygame.Rect(0, 0, WIDTH, 56)
BRAIN_H = 204
BRAIN = pygame.Rect(MARGIN, HEIGHT - MARGIN - BRAIN_H, WIDTH - 2 * MARGIN, BRAIN_H)
_card_top = HEADER.bottom + 12
_card_h = BRAIN.top - 12 - _card_top
_card_w = (WIDTH - 3 * MARGIN) // 2
REALITY = pygame.Rect(MARGIN, _card_top, _card_w, _card_h)
MIND = pygame.Rect(2 * MARGIN + _card_w, _card_top, _card_w, _card_h)

TILE_SIZE = 50
TILE_GAP = 6
GRID_PX = 8 * TILE_SIZE + 7 * TILE_GAP          # 442


def grid_origin(card: pygame.Rect) -> tuple[int, int]:
    """Top-left of the 8x8 grid inside a card (same offset in both cards: coordinates match)."""
    return card.left + 18, card.top + 40


def cell_rect(card: pygame.Rect, x: int, y: int) -> pygame.Rect:
    ox, oy = grid_origin(card)
    row = 7 - y                                    # (0,0) is bottom-left
    return pygame.Rect(ox + x * (TILE_SIZE + TILE_GAP), oy + row * (TILE_SIZE + TILE_GAP), TILE_SIZE, TILE_SIZE)


def cell_at(card: pygame.Rect, px: int, py: int) -> tuple[int, int] | None:
    for y in range(8):
        for x in range(8):
            if cell_rect(card, x, y).collidepoint(px, py):
                return x, y
    return None


# ---------------------------------------------------------------- fonts
_fonts: dict = {}
MONO = "consolas,cascadiamono,menlo,dejavusansmono,couriernew,monospace"
SANS = "segoeui,inter,helveticaneue,arial,dejavusans"


def font(size: int, mono: bool = True, bold: bool = False) -> pygame.font.Font:
    key = (size, mono, bold)
    if key not in _fonts:
        _fonts[key] = pygame.font.SysFont(MONO if mono else SANS, size, bold=bold)
    return _fonts[key]


# Logic/maths symbols used in explanations, mapped to ASCII so every font renders them.
_ASCII = {"∨": " v ", "∧": " ^ ", "¬": "~", "⇔": "<=>", "Σ": "SUM", "≤": "<=", "≥": ">=", "≠": "!=",
          "∉": " not in ", "∈": " in ", "∝": "~", "→": "->", "−": "-", "…": "...", "·": "*", "δ": "d",
          "⊨": "|=", "—": "-", "≈": "~="}


def ascii_safe(text: str) -> str:
    return "".join(_ASCII.get(ch, ch) for ch in text)


def reset_fonts() -> None:
    """Fonts become invalid after pygame.quit(); call after every pygame.init()."""
    _fonts.clear()
