"""Builds the explorer's Percept from the hidden world. This is the ONLY place where hidden
state is turned into explorer-visible information; it exposes exactly the RULES.md percepts."""
from __future__ import annotations

from ..core.geometry import Pos, neighbors
from ..core.percepts import Percept
from .mechanics import ActionFlags
from .world import WorldState


def breeze_at(w: WorldState, p: Pos) -> bool:
    return any(n in w.pits for n in neighbors(p, w.size))


def stench_at(w: WorldState, p: Pos) -> bool:
    return w.wumpus_alive and any(n == w.wumpus for n in neighbors(p, w.size))


def portal_signal_at(w: WorldState, p: Pos) -> bool:
    return any(n in w.portals for n in neighbors(p, w.size))


def build_percept(w: WorldState, turn: int, step: int, score: int, flags: ActionFlags | None) -> Percept:
    f = flags or ActionFlags()
    p = w.explorer
    return Percept(turn=turn, step=step, position=p, facing=w.facing,
                   breeze=breeze_at(w, p), stench=stench_at(w, p), glitter=w.gold == p,
                   bump=f.bump, scream=f.scream, portal_signal=portal_signal_at(w, p),
                   unstable=p in w.unstable, shuffled=f.shuffled, teleported=f.teleported,
                   has_gold=w.has_gold, arrows=w.arrows, jump_available=w.jump_available, score=score)
