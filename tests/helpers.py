from wumpus.config import GameConfig
from wumpus.core.geometry import Direction, Pos
from wumpus.environment.engine import GameEngine
from wumpus.environment.world import WorldState


def make_world(pits=(), gold=(7, 7), wumpus=(7, 0), explorer=(0, 0), facing=Direction.E, portals=(),
               unstable=(), shuffle=(), route=(), size=8, arrows=1, jump=True, visited=None, seed=0) -> WorldState:
    P = lambda t: Pos(*t)
    pmap = {}
    for a, b in portals:
        pmap[P(a)], pmap[P(b)] = P(b), P(a)
    explorer = P(explorer)
    return WorldState(size=size, start=Pos(0, 0), pits={P(p) for p in pits}, gold=P(gold) if gold else None,
                      wumpus=P(wumpus), explorer=explorer, facing=facing, portals=pmap,
                      unstable={P(c) for c in unstable}, shuffle_tiles={P(c) for c in shuffle},
                      route=tuple(P(c) for c in route), arrows=arrows, jump_available=jump, seed=seed,
                      visited={P(v) for v in visited} if visited else {explorer})


def make_engine(config: GameConfig | None = None, **world_kwargs) -> GameEngine:
    return GameEngine(config or GameConfig(), world=make_world(**world_kwargs))
