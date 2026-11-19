"""State-transition rules (RULES.md). Every function mutates the given WorldState and
returns a short list of human-readable resolution messages."""
from __future__ import annotations

import random

from ..config import GameConfig
from ..core.actions import (CLIMB, GRAB, STAY, SHUFFLE, WAIT, ExplorerAction, ExplorerActionType,
                            WumpusAction, WumpusActionType)
from ..core.geometry import DIRECTIONS, Direction, Pos, all_cells, in_bounds, manhattan, neighbors, ray
from .world import WorldState, safe_start_zone


class IllegalAction(ValueError):
    pass


# ---------------------------------------------------------------- explorer
def legal_explorer_actions(s: WorldState) -> list[ExplorerAction]:
    acts: list[ExplorerAction] = []
    for d in DIRECTIONS:
        if in_bounds(s.explorer.step(d), s.size):
            acts.append(ExplorerAction.move(d))
    if s.jumps_left > 0:
        acts += [ExplorerAction.jump(d) for d in DIRECTIONS if in_bounds(s.explorer.step(d, 2), s.size)]
    if s.arrows > 0:
        acts += [ExplorerAction.shoot(d) for d in DIRECTIONS if ray(s.explorer, d, s.size)]
    if s.gold is not None and s.gold == s.explorer:
        acts.append(GRAB)
    if s.explorer == s.start:
        acts.append(CLIMB)
    acts.append(WAIT)
    return acts


def resolve_explorer_landing(s: WorldState, c: Pos) -> list[str]:
    msgs = []
    pair = s.pad_pair(c)
    if pair is not None:
        msgs.append(f"explorer teleported {c} -> {pair}")
        c = pair
    s.explorer = c
    if c in s.pits:
        s.explorer_alive = False
        msgs.append(f"explorer fell into a pit at {c}")
    elif c in s.collapsed:
        s.explorer_alive = False
        msgs.append(f"explorer fell through collapsed tile {c}")
    elif s.wumpus_alive and s.wumpus == c:
        s.explorer_alive = False
        msgs.append(f"explorer walked into the Wumpus at {c}")
    else:
        s.visited.add(c)
    return msgs


def apply_explorer_action(s: WorldState, a: ExplorerAction) -> tuple[list[str], bool]:
    """Returns (messages, scream)."""
    if a not in legal_explorer_actions(s):
        raise IllegalAction(f"illegal explorer action {a}")
    k = a.kind
    if k is ExplorerActionType.MOVE:
        return [f"explorer moves {a.direction.name}"] + resolve_explorer_landing(s, s.explorer.step(a.direction)), False
    if k is ExplorerActionType.JUMP:
        s.jumps_left -= 1
        mid = s.explorer.step(a.direction)
        return [f"explorer jumps {a.direction.name} over {mid}"] + resolve_explorer_landing(
            s, s.explorer.step(a.direction, 2)), False
    if k is ExplorerActionType.SHOOT:
        s.arrows -= 1
        if s.wumpus_alive and s.wumpus in ray(s.explorer, a.direction, s.size):
            s.wumpus_alive = False
            return [f"arrow {a.direction.name} hits the Wumpus at {s.wumpus}"], True
        return [f"arrow {a.direction.name} misses"], False
    if k is ExplorerActionType.GRAB:
        s.has_gold, s.gold = True, None
        return ["explorer grabs the gold"], False
    if k is ExplorerActionType.CLIMB:
        return ["explorer climbs out" + (" with the gold" if s.has_gold else " empty-handed")], False
    return ["explorer waits"], False


# ---------------------------------------------------------------- wumpus
def wumpus_can_enter(s: WorldState, c: Pos) -> bool:
    return in_bounds(c, s.size) and not s.is_blocked_for_wumpus(c)


def is_valid_collapse_target(s: WorldState, c: Pos, config: GameConfig) -> bool:
    return (s.wumpus_alive and s.collapse_charges > 0 and s.collapse_cooldown == 0
            and in_bounds(c, s.size) and manhattan(s.wumpus, c) <= config.collapse_range
            and c != s.start and c != s.explorer and c != s.wumpus and c not in s.pits
            and not (s.pads and c in s.pads) and c != s.gold and c not in s.collapsed)


def legal_wumpus_actions(s: WorldState, config: GameConfig) -> list[WumpusAction]:
    if not s.wumpus_alive:
        return []
    acts = [WumpusAction.move(d) for d in DIRECTIONS if wumpus_can_enter(s, s.wumpus.step(d))]
    acts.append(STAY)
    if s.collapse_charges > 0 and s.collapse_cooldown == 0:
        acts += [WumpusAction.collapse(c) for c in all_cells(s.size) if is_valid_collapse_target(s, c, config)]
    if s.shuffle_charges > 0:
        acts.append(SHUFFLE)
    return acts


def shuffle_eligible_cells(s: WorldState) -> list[Pos]:
    zone = safe_start_zone(s.start, s.size)
    return [c for c in all_cells(s.size)
            if c not in s.visited and c != s.explorer and c != s.wumpus and c not in s.collapsed
            and not (s.pads and c in s.pads) and c != s.gold and c not in zone]


def apply_wumpus_action(s: WorldState, a: WumpusAction, config: GameConfig, rng: random.Random) -> list[str]:
    if a not in legal_wumpus_actions(s, config):
        raise IllegalAction(f"illegal wumpus action {a}")
    msgs: list[str] = []
    if a.kind is WumpusActionType.MOVE:
        c = s.wumpus.step(a.direction)
        msgs.append(f"wumpus moves {a.direction.name}")
        pair = s.pad_pair(c)
        if pair is not None:
            msgs.append(f"wumpus teleported {c} -> {pair}")
            c = pair
        s.wumpus = c
        if s.explorer_alive and c == s.explorer:
            s.explorer_alive = False
            msgs.append(f"the Wumpus devours the explorer at {c}")
    elif a.kind is WumpusActionType.COLLAPSE:
        s.collapsed.add(a.target)
        s.collapse_charges -= 1
        s.collapse_cooldown = config.collapse_cooldown + 1  # decremented below at end of this turn
        msgs.append(f"wumpus collapses tile {a.target}")
    elif a.kind is WumpusActionType.SHUFFLE:
        eligible = shuffle_eligible_cells(s)
        n = min(len(s.pits), len(eligible))
        s.pits = set(rng.sample(eligible, n))
        s.shuffle_charges -= 1
        msgs.append("wumpus shuffles the pits")
    else:
        msgs.append("wumpus stays")
    if s.collapse_cooldown > 0:
        s.collapse_cooldown -= 1
    return msgs


# ---------------------------------------------------------------- percepts
def stench_at(s: WorldState, p: Pos) -> bool:
    return s.wumpus_alive and manhattan(s.wumpus, p) == 1


def breeze_at(s: WorldState, p: Pos) -> bool:
    return any(n in s.pits for n in neighbors(p, s.size))
