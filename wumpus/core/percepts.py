"""The ONLY information channel from the hidden environment to the explorer AI.

`PublicInfo` is given once at game start, a `Percept` at the start of every
explorer turn. Both are frozen; neither contains hidden layout data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .geometry import Pos


class EventType(Enum):
    COLLAPSE = "COLLAPSE"   # pos = collapsed cell
    SHUFFLE = "SHUFFLE"     # pits were re-placed on unvisited cells


@dataclass(frozen=True)
class PublicEvent:
    kind: EventType
    turn: int
    pos: Pos | None = None

    def __str__(self) -> str:
        return f"{self.kind.value}" + (f" {self.pos}" if self.pos else "")


@dataclass(frozen=True)
class PublicInfo:
    size: int
    start: Pos
    pit_count: int
    pads: tuple[Pos, Pos] | None
    safe_start_zone: frozenset[Pos]
    wumpus_min_start_distance: int
    collapse_range: int
    arrows: int
    jumps: int
    max_turns: int


@dataclass(frozen=True)
class Percept:
    turn: int
    position: Pos
    stench: bool
    breeze: bool
    glitter: bool
    scream: bool
    has_gold: bool
    arrows: int
    jumps_left: int
    events: tuple[PublicEvent, ...] = field(default_factory=tuple)

    def flags(self) -> str:
        names = [n for n in ("stench", "breeze", "glitter", "scream") if getattr(self, n)]
        return ", ".join(names) if names else "nothing"
