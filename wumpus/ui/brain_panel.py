"""PANEL 03 / BRAIN ACTIVITY: timestamped, colour-coded log of the explorer's real reasoning.

Rows are the agent's TraceEntry objects (structured: id, turn, tag, summary, details, refs, cell,
timestamp). Clicking a row opens the Inspector with the stored evidence; clicking a dependency in
the inspector jumps to that entry.
"""
from __future__ import annotations

import time

import pygame

from . import theme as T

ROW_H = 16


def _wrap(text: str, fnt, width: int) -> list[str]:
    words, lines, cur = T.ascii_safe(text).split(" "), [], ""
    for w in words:
        trial = (cur + " " + w) if cur else w
        if fnt.size(trial)[0] <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            while fnt.size(w)[0] > width and len(w) > 1:      # hard-break very long tokens
                cut = len(w)
                while cut > 1 and fnt.size(w[:cut])[0] > width:
                    cut -= 1
                lines.append(w[:cut])
                w = w[cut:]
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


class BrainPanel:
    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.log_rect = pygame.Rect(rect.left + 14, rect.top + 34, int(rect.w * 0.64), rect.h - 44)
        self.summary_rect = pygame.Rect(self.log_rect.right + 14, rect.top + 34,
                                        rect.right - self.log_rect.right - 28, rect.h - 44)
        self.scroll = 0                 # rows scrolled up from the bottom (0 = follow newest)
        self.key_only = True
        self.rows: list[tuple[pygame.Rect, int]] = []

    def visible(self, entries):
        return [e for e in entries if not self.key_only or e.tag in T.KEY_TAGS]

    def on_wheel(self, dy: int, entries):
        n = len(self.visible(entries))
        cap = max(0, n - self.log_rect.h // ROW_H)
        self.scroll = max(0, min(cap, self.scroll + dy * 3))

    def click(self, pos) -> int | None:
        for r, eid in self.rows:
            if r.collidepoint(pos):
                return eid
        return None

    def draw(self, surf, entries, report, think_s: float, busy: bool):
        rect = self.rect
        pygame.draw.rect(surf, T.PANEL, rect, border_radius=T.RADIUS)
        pygame.draw.rect(surf, T.BORDER, rect, 1, border_radius=T.RADIUS)
        surf.blit(T.font(13, bold=True).render("PANEL 03", True, T.CYAN), (rect.left + 14, rect.top + 10))
        surf.blit(T.font(15, mono=False, bold=True).render("BRAIN ACTIVITY", True, T.TEXT), (rect.left + 96, rect.top + 8))
        hint = f"{'key tags' if self.key_only else 'all tags'} (F)  |  wheel: scroll  |  click a line: inspect evidence"
        img = T.font(11).render(hint, True, T.TEXT_FAINT)
        surf.blit(img, (self.log_rect.right - img.get_width(), rect.top + 12))
        # ---- log
        pygame.draw.rect(surf, T.BG, self.log_rect, border_radius=8)
        vis = self.visible(entries)
        n_rows = self.log_rect.h // ROW_H
        end = len(vis) - self.scroll
        shown = vis[max(0, end - n_rows):end]
        self.rows = []
        f = T.font(12)
        mx, my = pygame.mouse.get_pos()
        y = self.log_rect.top + 2
        for e in shown:
            row = pygame.Rect(self.log_rect.left, y, self.log_rect.w, ROW_H)
            if row.collidepoint(mx, my):
                pygame.draw.rect(surf, T.TILE, row)
            ts = time.strftime("%H:%M:%S", time.localtime(e.created)) + f".{int(e.created * 1000) % 1000:03d}"
            x = row.left + 6
            for txt, col in ((ts, T.TEXT_FAINT), (f" T{e.turn:03d} ", T.TEXT_FAINT),
                             (f"[{e.tag}]", T.TAG_COLORS.get(e.tag, T.TEXT_DIM))):
                img = f.render(txt, True, col)
                surf.blit(img, (x, y + 1))
                x += img.get_width()
            room = row.right - x - 8
            text = " " + T.ascii_safe(e.summary)
            while text and f.size(text)[0] > room:
                text = text[:-2]
            surf.blit(f.render(text, True, T.TEXT), (x, y + 1))
            self.rows.append((row, e.id))
            y += ROW_H
        if self.scroll:
            surf.blit(T.font(11).render(f"-- scrolled {self.scroll} rows (wheel down to follow) --", True, T.AMBER),
                      (self.log_rect.left + 6, self.log_rect.bottom - 14))
        # ---- current decision summary
        sr = self.summary_rect
        pygame.draw.rect(surf, T.BG, sr, border_radius=8)
        pygame.draw.rect(surf, T.CORAL, sr, 1, border_radius=8)
        surf.blit(T.font(11, bold=True).render("CURRENT DECISION", True, T.CORAL), (sr.left + 10, sr.top + 8))
        if busy:
            surf.blit(T.font(20, bold=True).render("thinking...", True, T.VIOLET), (sr.left + 10, sr.top + 26))
            return
        if report is None:
            surf.blit(T.font(13).render("waiting for the first turn", True, T.TEXT_DIM), (sr.left + 10, sr.top + 30))
            return
        surf.blit(T.font(22, bold=True).render(report.selected, True, T.TEXT), (sr.left + 10, sr.top + 24))
        meta = f"turn {report.turn}  |  thought {think_s * 1000:.0f} ms"
        if report.minimax:
            meta += f"  |  Minimax E[V] {report.minimax['expected_value']:.0f}"
        surf.blit(T.font(11).render(meta, True, T.TEXT_DIM), (sr.left + 10, sr.top + 54))
        y = sr.top + 72
        for line in _wrap(report.explanation, T.font(12), sr.w - 20)[:4]:
            surf.blit(T.font(12).render(line, True, T.TEXT), (sr.left + 10, y))
            y += 15
        if report.new_facts:
            surf.blit(T.font(11).render(f"+ {len(report.new_facts)} new inferred facts", True, T.EMERALD),
                      (sr.left + 10, sr.bottom - 18))


class Inspector:
    """Modal showing one entry's stored evidence (details) and its dependencies."""

    def __init__(self, log, entry_id: int):
        self.log = log
        self.entry_id = entry_id
        self.scroll = 0
        self.rect = pygame.Rect(140, 90, T.WIDTH - 280, T.HEIGHT - 180)
        self.ref_rows: list[tuple[pygame.Rect, int]] = []
        self.close_rect = pygame.Rect(self.rect.right - 40, self.rect.top + 10, 28, 24)

    def open(self, entry_id: int):
        self.entry_id, self.scroll = entry_id, 0

    def click(self, pos) -> str:
        if self.close_rect.collidepoint(pos) or not self.rect.collidepoint(pos):
            return "close"
        for r, eid in self.ref_rows:
            if r.collidepoint(pos):
                self.open(eid)
                return "navigate"
        return "none"

    def on_wheel(self, dy):
        self.scroll = max(0, self.scroll + dy * 3)

    def draw(self, surf):
        e = self.log.get(self.entry_id)
        dim = pygame.Surface((T.WIDTH, T.HEIGHT), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 160))
        surf.blit(dim, (0, 0))
        r = self.rect
        pygame.draw.rect(surf, T.PANEL, r, border_radius=T.RADIUS)
        pygame.draw.rect(surf, T.TAG_COLORS.get(e.tag, T.BORDER), r, 2, border_radius=T.RADIUS)
        col = T.TAG_COLORS.get(e.tag, T.TEXT)
        surf.blit(T.font(16, bold=True).render(f"#{e.id}  [{e.tag}]  turn {e.turn}", True, col), (r.left + 20, r.top + 14))
        surf.blit(T.font(12).render("ESC / click outside to close  |  wheel to scroll  |  click a dependency to follow it",
                                    True, T.TEXT_FAINT), (r.left + 20, r.top + 38))
        pygame.draw.rect(surf, T.TILE, self.close_rect, border_radius=6)
        surf.blit(T.font(14, bold=True).render("X", True, T.TEXT), (self.close_rect.left + 9, self.close_rect.top + 3))
        f = T.font(13)
        lines = _wrap(e.summary, T.font(14, bold=True), r.w - 40)
        y = r.top + 62
        for ln in lines:
            surf.blit(T.font(14, bold=True).render(ln, True, T.TEXT), (r.left + 20, y))
            y += 18
        y += 6
        body = []
        for d in e.details:
            indent = len(d) - len(d.lstrip(" "))
            for i, ln in enumerate(_wrap(d.strip(), f, r.w - 50 - indent * 7)):
                body.append(" " * indent + ("" if i == 0 else "  ") + ln)
        bottom = r.bottom - (30 + 18 * min(4, len(e.refs)) if e.refs else 16)
        max_rows = max(1, (bottom - y) // 16)
        self.scroll = min(self.scroll, max(0, len(body) - max_rows))
        for ln in body[self.scroll:self.scroll + max_rows]:
            surf.blit(f.render(ln, True, T.TEXT_DIM), (r.left + 24, y))
            y += 16
        if len(body) > max_rows:
            surf.blit(T.font(11).render(f"lines {self.scroll + 1}-{min(len(body), self.scroll + max_rows)} of {len(body)}",
                                        True, T.AMBER), (r.right - 170, r.top + 40))
        self.ref_rows = []
        if e.refs:
            y = bottom + 6
            surf.blit(T.font(12, bold=True).render("DEPENDS ON", True, T.CYAN), (r.left + 20, y))
            y += 18
            for rid in e.refs[:4]:
                ref = self.log.get(rid)
                row = pygame.Rect(r.left + 20, y, r.w - 40, 17)
                if row.collidepoint(pygame.mouse.get_pos()):
                    pygame.draw.rect(surf, T.TILE, row)
                txt = T.ascii_safe(f"#{ref.id} [{ref.tag}] {ref.summary}")[:140]
                surf.blit(T.font(12).render(txt, True, T.TAG_COLORS.get(ref.tag, T.TEXT)), (row.left + 4, y + 1))
                self.ref_rows.append((row, rid))
                y += 18
