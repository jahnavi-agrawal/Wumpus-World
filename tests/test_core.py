from wumpus.core.geometry import Direction, Pos, direction_between, manhattan, neighbors, ray
from wumpus.core.actions import JUMP, WumpusAction


def test_neighbors_corner_and_order():
    assert neighbors(Pos(0, 0), 8) == [Pos(0, 1), Pos(1, 0)]
    assert neighbors(Pos(3, 3), 8) == [Pos(3, 4), Pos(4, 3), Pos(3, 2), Pos(2, 3)]


def test_ray_and_direction():
    assert ray(Pos(5, 2), Direction.E, 8) == [Pos(6, 2), Pos(7, 2)]
    assert ray(Pos(0, 0), Direction.S, 8) == []
    assert direction_between(Pos(1, 1), Pos(1, 5)) is Direction.N
    assert direction_between(Pos(1, 1), Pos(2, 2)) is None


def test_step_and_manhattan():
    assert Pos(1, 1).step(Direction.N, 2) == Pos(1, 3)
    assert manhattan(Pos(0, 0), Pos(3, 4)) == 7


def test_action_str():
    assert str(JUMP) == "JUMP"
    assert str(WumpusAction.move(Direction.N)) == "MOVE N"


def test_turning():
    assert Direction.E.left() is Direction.N and Direction.E.right() is Direction.S
    assert Direction.N.left() is Direction.W and Direction.W.right() is Direction.N
    assert Direction.E.label == "RIGHT"
