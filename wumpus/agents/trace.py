"""Brain Activity log. Each entry's `details` are produced by the algorithm's own result
object (`explain()`), never written by hand, so every line in the UI can be inspected."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..core.geometry import Pos

TAGS = ("PERCEPT", "EVENT", "KB_TELL", "INFERENCE", "RISK_CALC", "WUMPUS_BELIEF", "TARGET",
        "A*_PLANNER", "REPLAN", "MINIMAX", "DECISION", "WARNING")


@dataclass
class TraceEntry:
    id: int
    turn: int
    tag: str
    summary: str
    details: list[str] = field(default_factory=list)
    refs: list[int] = field(default_factory=list)      # ids of entries this one depends on
    cell: Pos | None = None                            # board cell to highlight
    created: float = field(default_factory=time.time, compare=False)   # wall-clock timestamp

    def line(self) -> str:
        return f"[{self.tag}] {self.summary}"


class BrainLog:
    def __init__(self) -> None:
        self.entries: list[TraceEntry] = []

    def add(self, turn: int, tag: str, summary: str, details: list[str] | None = None,
            refs: list[int] | None = None, cell: Pos | None = None) -> TraceEntry:
        assert tag in TAGS, tag
        e = TraceEntry(len(self.entries), turn, tag, summary, list(details or []), list(refs or []), cell)
        self.entries.append(e)
        return e

    def get(self, entry_id: int) -> TraceEntry:
        return self.entries[entry_id]

    def for_turn(self, turn: int) -> list[TraceEntry]:
        return [e for e in self.entries if e.turn == turn]

    def tagged(self, tag: str) -> list[TraceEntry]:
        return [e for e in self.entries if e.tag == tag]
