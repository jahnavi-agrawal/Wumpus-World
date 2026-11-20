"""Action value types for both players."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .geometry import Direction


class ExplorerAction(Enum):
    FORWARD = "FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    JUMP = "JUMP"
    GRAB = "GRAB"
    SHOOT = "SHOOT"
    CLIMB = "CLIMB"

    def __str__(self) -> str:
        return self.value


FORWARD = ExplorerAction.FORWARD
TURN_LEFT = ExplorerAction.TURN_LEFT
TURN_RIGHT = ExplorerAction.TURN_RIGHT
JUMP = ExplorerAction.JUMP
GRAB = ExplorerAction.GRAB
SHOOT = ExplorerAction.SHOOT
CLIMB = ExplorerAction.CLIMB


class WumpusActionType(Enum):
    MOVE = "MOVE"
    STAY = "STAY"


@dataclass(frozen=True)
class WumpusAction:
    kind: WumpusActionType
    direction: Direction | None = None

    @staticmethod
    def move(d: Direction) -> "WumpusAction":
        return WumpusAction(WumpusActionType.MOVE, d)

    def __str__(self) -> str:
        return f"MOVE {self.direction.name}" if self.kind is WumpusActionType.MOVE else "STAY"


STAY = WumpusAction(WumpusActionType.STAY)
