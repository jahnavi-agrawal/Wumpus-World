"""Grid geometry shared by environment, AI and UI. Origin (0,0) is bottom-left."""
from __future__ import annotations

from enum import Enum
from typing import NamedTuple


class Pos(NamedTuple):
    x: int
    y: int

    def step(self, d: "Direction", n: int = 1) -> "Pos":
        return Pos(self.x + d.dx * n, self.y + d.dy * n)

    def __str__(self) -> str:  # (3,2)
        return f"({self.x},{self.y})"


class Direction(Enum):
    N = (0, 1)
    E = (1, 0)
    S = (0, -1)
    W = (-1, 0)

    @property
    def dx(self) -> int:
        return self.value[0]

    @property
    def dy(self) -> int:
        return self.value[1]


DIRECTIONS: tuple[Direction, ...] = (Direction.N, Direction.E, Direction.S, Direction.W)


def in_bounds(p: Pos, size: int) -> bool:
    return 0 <= p.x < size and 0 <= p.y < size


def neighbors(p: Pos, size: int) -> list[Pos]:
    """Orthogonal neighbours in deterministic N, E, S, W order."""
    return [q for q in (p.step(d) for d in DIRECTIONS) if in_bounds(q, size)]


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def ray(p: Pos, d: Direction, size: int) -> list[Pos]:
    """Cells along direction d starting next to p, up to the wall."""
    out = []
    q = p.step(d)
    while in_bounds(q, size):
        out.append(q)
        q = q.step(d)
    return out


def all_cells(size: int) -> list[Pos]:
    return [Pos(x, y) for y in range(size) for x in range(size)]


def direction_between(a: Pos, b: Pos) -> Direction | None:
    """Direction from a to b if they are on the same row/column."""
    if a == b:
        return None
    if a.x == b.x:
        return Direction.N if b.y > a.y else Direction.S
    if a.y == b.y:
        return Direction.E if b.x > a.x else Direction.W
    return None
