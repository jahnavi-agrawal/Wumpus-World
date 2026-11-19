from wumpus.config import GameConfig
from wumpus.core.geometry import Pos
from wumpus.env.engine import GameEngine
from wumpus.env.world import WorldState


def make_world(pits=(), gold=(7, 7), wumpus=(7, 0), explorer=(0, 0), pads=None, size=8,
               arrows=1, jumps=2, collapse_charges=3, shuffle_charges=1, visited=None) -> WorldState:
    explorer = Pos(*explorer)
    return WorldState(size=size, start=Pos(0, 0), pits={Pos(*p) for p in pits},
                      gold=Pos(*gold) if gold else None, wumpus=Pos(*wumpus), explorer=explorer,
                      pads=tuple(Pos(*p) for p in pads) if pads else None, arrows=arrows, jumps_left=jumps,
                      collapse_charges=collapse_charges, shuffle_charges=shuffle_charges,
                      visited=set(map(lambda p: Pos(*p), visited)) if visited else {explorer})


def make_engine(config: GameConfig | None = None, **world_kwargs) -> GameEngine:
    return GameEngine(config or GameConfig(), world=make_world(**world_kwargs))
