"""Action value types for both players."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .geometry import Direction, Pos


class ExplorerActionType(Enum):
    MOVE = "MOVE"
    JUMP = "JUMP"
    SHOOT = "SHOOT"
    GRAB = "GRAB"
    CLIMB = "CLIMB"
    WAIT = "WAIT"


@dataclass(frozen=True)
class ExplorerAction:
    kind: ExplorerActionType
    direction: Direction | None = None

    @staticmethod
    def move(d: Direction) -> "ExplorerAction":
        return ExplorerAction(ExplorerActionType.MOVE, d)

    @staticmethod
    def jump(d: Direction) -> "ExplorerAction":
        return ExplorerAction(ExplorerActionType.JUMP, d)

    @staticmethod
    def shoot(d: Direction) -> "ExplorerAction":
        return ExplorerAction(ExplorerActionType.SHOOT, d)

    def __str__(self) -> str:
        return self.kind.value + (f" {self.direction.name}" if self.direction else "")


GRAB = ExplorerAction(ExplorerActionType.GRAB)
CLIMB = ExplorerAction(ExplorerActionType.CLIMB)
WAIT = ExplorerAction(ExplorerActionType.WAIT)


class WumpusActionType(Enum):
    MOVE = "MOVE"
    STAY = "STAY"
    COLLAPSE = "COLLAPSE"
    SHUFFLE = "SHUFFLE"


@dataclass(frozen=True)
class WumpusAction:
    kind: WumpusActionType
    direction: Direction | None = None
    target: Pos | None = None

    @staticmethod
    def move(d: Direction) -> "WumpusAction":
        return WumpusAction(WumpusActionType.MOVE, d)

    @staticmethod
    def collapse(target: Pos) -> "WumpusAction":
        return WumpusAction(WumpusActionType.COLLAPSE, target=Pos(*target))

    def __str__(self) -> str:
        if self.kind is WumpusActionType.MOVE:
            return f"MOVE {self.direction.name}"
        if self.kind is WumpusActionType.COLLAPSE:
            return f"COLLAPSE {self.target}"
        return self.kind.value


STAY = WumpusAction(WumpusActionType.STAY)
SHUFFLE = WumpusAction(WumpusActionType.SHUFFLE)
