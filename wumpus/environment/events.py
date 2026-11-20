"""Structured event output. Every action produces one or more GameEvents.

Events are environment-internal (they may mention hidden facts, e.g. which cells a
shuffle swapped) and are used by the engine's history, the terminal/GUI god view and tests.
The explorer never sees them; it only receives a Percept.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..core.geometry import Pos


class EventKind(Enum):
    MOVED = "MOVED"
    TURNED = "TURNED"
    BUMPED = "BUMPED"
    JUMPED = "JUMPED"
    TELEPORTED = "TELEPORTED"
    FELL_INTO_PIT = "FELL_INTO_PIT"
    EATEN = "EATEN"
    GRABBED_GOLD = "GRABBED_GOLD"
    NOTHING_TO_GRAB = "NOTHING_TO_GRAB"
    ARROW_HIT = "ARROW_HIT"
    ARROW_MISSED = "ARROW_MISSED"
    NO_ARROWS = "NO_ARROWS"
    CLIMBED_OUT = "CLIMBED_OUT"
    CANNOT_CLIMB = "CANNOT_CLIMB"
    JUMP_UNAVAILABLE = "JUMP_UNAVAILABLE"
    TILE_COLLAPSED = "TILE_COLLAPSED"
    CELLS_SHUFFLED = "CELLS_SHUFFLED"
    WUMPUS_MOVED = "WUMPUS_MOVED"
    WUMPUS_STAYED = "WUMPUS_STAYED"
    GAME_OVER = "GAME_OVER"


@dataclass(frozen=True)
class GameEvent:
    kind: EventKind
    actor: str                       # "explorer" | "wumpus" | "world"
    turn: int
    step: int
    start: Pos | None = None
    end: Pos | None = None
    data: dict = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "actor": self.actor, "turn": self.turn, "step": self.step,
                "start": tuple(self.start) if self.start else None,
                "end": tuple(self.end) if self.end else None, **self.data}

    def __str__(self) -> str:
        where = ""
        if self.start and self.end and self.start != self.end:
            where = f" {self.start}->{self.end}"
        elif self.end or self.start:
            where = f" {self.end or self.start}"
        extra = " " + " ".join(f"{k}={v}" for k, v in self.data.items()) if self.data else ""
        return f"[{self.actor}] {self.kind.value}{where}{extra}"
