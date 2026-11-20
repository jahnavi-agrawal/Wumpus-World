"""State-transition rules (RULES.md v2). Functions mutate the WorldState and return the
structured events they caused plus the percept flags the explorer will feel."""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..config import GameConfig
from ..core.actions import STAY, ExplorerAction, WumpusAction, WumpusActionType
from ..core.geometry import DIRECTIONS, Pos, in_bounds, ray
from .events import EventKind as K, GameEvent
from .world import WorldState, safe_start_zone


class IllegalAction(ValueError):
    pass


@dataclass
class ActionFlags:
    bump: bool = False
    scream: bool = False
    teleported: bool = False
    shuffled: bool = False


class _Ctx:
    def __init__(self, turn: int, step: int):
        self.turn, self.step, self.events = turn, step, []

    def emit(self, kind, actor, start=None, end=None, **data):
        self.events.append(GameEvent(kind, actor, self.turn, self.step, start, end, data))


# ------------------------------------------------------------------ explorer
def shuffle_swap(w: WorldState, tile: Pos, step: int) -> tuple[Pos, Pos] | None:
    """Seeded deterministic swap of one pit with one empty eligible cell (RULES.md)."""
    zone = safe_start_zone(w.start, w.size)
    route = set(w.route)
    forbidden = (w.visited | {w.explorer, w.wumpus} | w.special_cells() | zone | route
                 | ({w.gold} if w.gold else set()))
    cells = [Pos(x, y) for y in range(w.size) for x in range(w.size) if Pos(x, y) not in forbidden]
    pits = sorted(c for c in cells if c in w.pits)
    empty = sorted(c for c in cells if c not in w.pits)
    if not pits or not empty:
        return None
    rng = random.Random(f"{w.seed}:shuffle:{tile.x},{tile.y}:{step}")
    a, b = rng.choice(pits), rng.choice(empty)
    w.pits.remove(a)
    w.pits.add(b)
    return a, b


def _leave(w: WorldState, c: Pos, ctx: _Ctx) -> None:
    if c in w.unstable:
        w.unstable.remove(c)
        w.collapsed.add(c)
        w.pits.add(c)
        ctx.emit(K.TILE_COLLAPSED, "world", c, c)


def _arrive(w: WorldState, c: Pos, ctx: _Ctx, flags: ActionFlags) -> None:
    pair = w.portal_pair(c)
    if pair is not None:
        ctx.emit(K.TELEPORTED, "explorer", c, pair)
        flags.teleported = True
        c = pair
    w.explorer = c
    if c in w.pits:
        w.explorer_alive = False
        ctx.emit(K.FELL_INTO_PIT, "explorer", c, c)
    elif w.wumpus_alive and w.wumpus == c:
        w.explorer_alive = False
        ctx.emit(K.EATEN, "explorer", c, c)
    else:
        w.visited.add(c)
        if c in w.shuffle_tiles:
            w.shuffle_tiles.remove(c)
            w.used_shuffle.add(c)
            swapped = shuffle_swap(w, c, ctx.step)
            flags.shuffled = True
            ctx.emit(K.CELLS_SHUFFLED, "world", c, c,
                     swapped=(tuple(map(tuple, swapped)) if swapped else None))


def apply_explorer_action(w: WorldState, a: ExplorerAction, turn: int, step: int,
                          config: GameConfig) -> tuple[list[GameEvent], ActionFlags]:
    ctx, flags = _Ctx(turn, step), ActionFlags()
    pos = w.explorer
    if a is ExplorerAction.TURN_LEFT or a is ExplorerAction.TURN_RIGHT:
        old = w.facing
        w.facing = old.left() if a is ExplorerAction.TURN_LEFT else old.right()
        ctx.emit(K.TURNED, "explorer", pos, pos, frm=old.label, to=w.facing.label)
    elif a is ExplorerAction.FORWARD:
        target = pos.step(w.facing)
        if not in_bounds(target, w.size):
            flags.bump = True
            ctx.emit(K.BUMPED, "explorer", pos, pos, facing=w.facing.label)
        else:
            ctx.emit(K.MOVED, "explorer", pos, target)
            _leave(w, pos, ctx)
            _arrive(w, target, ctx, flags)
    elif a is ExplorerAction.JUMP:
        target = pos.step(w.facing, config.jump_distance)
        if not w.jump_available:
            ctx.emit(K.JUMP_UNAVAILABLE, "explorer", pos, pos)
        elif not in_bounds(target, w.size):
            flags.bump = True
            ctx.emit(K.BUMPED, "explorer", pos, pos, facing=w.facing.label, jump=True)
        else:
            w.jump_available = False
            crossed = [pos.step(w.facing, i) for i in range(1, config.jump_distance)]
            ctx.emit(K.JUMPED, "explorer", pos, target, crossed=tuple(map(tuple, crossed)))
            _leave(w, pos, ctx)
            _arrive(w, target, ctx, flags)
    elif a is ExplorerAction.GRAB:
        if w.gold is not None and w.gold == pos:
            w.has_gold, w.gold = True, None
            ctx.emit(K.GRABBED_GOLD, "explorer", pos, pos)
        else:
            ctx.emit(K.NOTHING_TO_GRAB, "explorer", pos, pos)
    elif a is ExplorerAction.SHOOT:
        if w.arrows <= 0:
            ctx.emit(K.NO_ARROWS, "explorer", pos, pos)
        else:
            w.arrows -= 1
            path = ray(pos, w.facing, w.size)
            if w.wumpus_alive and w.wumpus in path:
                w.wumpus_alive = False
                flags.scream = True
                ctx.emit(K.ARROW_HIT, "explorer", pos, w.wumpus, facing=w.facing.label)
            else:
                ctx.emit(K.ARROW_MISSED, "explorer", pos, path[-1] if path else pos, facing=w.facing.label)
    elif a is ExplorerAction.CLIMB:
        if pos == w.start:
            ctx.emit(K.CLIMBED_OUT, "explorer", pos, pos, with_gold=w.has_gold)
        else:
            ctx.emit(K.CANNOT_CLIMB, "explorer", pos, pos)
    else:
        raise IllegalAction(f"unknown explorer action {a}")
    return ctx.events, flags


# ------------------------------------------------------------------ wumpus
def wumpus_can_enter(w: WorldState, c: Pos) -> bool:
    return in_bounds(c, w.size) and c not in w.pits


def legal_wumpus_actions(w: WorldState) -> list[WumpusAction]:
    if not w.wumpus_alive:
        return []
    return [WumpusAction.move(d) for d in DIRECTIONS if wumpus_can_enter(w, w.wumpus.step(d))] + [STAY]


def apply_wumpus_action(w: WorldState, a: WumpusAction, turn: int, step: int) -> list[GameEvent]:
    if a not in legal_wumpus_actions(w):
        raise IllegalAction(f"illegal wumpus action {a}")
    ctx = _Ctx(turn, step)
    if a.kind is WumpusActionType.STAY:
        ctx.emit(K.WUMPUS_STAYED, "wumpus", w.wumpus, w.wumpus)
        return ctx.events
    start, c = w.wumpus, w.wumpus.step(a.direction)
    ctx.emit(K.WUMPUS_MOVED, "wumpus", start, c, direction=a.direction.name)
    pair = w.portal_pair(c)
    if pair is not None:
        ctx.emit(K.TELEPORTED, "wumpus", c, pair)
        c = pair
    w.wumpus = c
    if w.explorer_alive and c == w.explorer:
        w.explorer_alive = False
        ctx.emit(K.EATEN, "explorer", c, c)
    return ctx.events
