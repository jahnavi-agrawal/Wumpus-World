"""The ONLY information channel from the hidden environment to the explorer AI.

`PublicInfo` is given once at game start, a `Percept` at the start of every explorer turn.
Both are frozen and contain no hidden layout data (no pit, portal, tile, gold or Wumpus
locations), only public rules/counts and the explorer's knowledge of its own state.
(`wumpus/environment/percepts.py` *builds* these from the hidden world.)
"""
from __future__ import annotations

from dataclasses import dataclass

from .geometry import Direction, Pos

PERCEPT_FLAGS = ("breeze", "stench", "glitter", "bump", "scream", "portal_signal", "unstable",
                 "shuffled", "teleported")


@dataclass(frozen=True)
class PublicInfo:
    size: int
    start: Pos
    start_facing: Direction
    pit_count: int
    portal_pairs: int
    unstable_tiles: int
    shuffle_tiles: int
    safe_start_zone: frozenset[Pos]
    wumpus_min_start_distance: int
    arrows: int
    jumps: int
    jump_distance: int
    max_turns: int


@dataclass(frozen=True)
class Percept:
    turn: int
    step: int
    position: Pos
    facing: Direction
    breeze: bool = False
    stench: bool = False
    glitter: bool = False
    bump: bool = False
    scream: bool = False
    portal_signal: bool = False
    unstable: bool = False
    shuffled: bool = False
    teleported: bool = False
    has_gold: bool = False
    arrows: int = 1
    jump_available: bool = True
    score: int = 0

    def flags(self) -> str:
        names = [n for n in PERCEPT_FLAGS if getattr(self, n)]
        return ", ".join(names) if names else "nothing"
