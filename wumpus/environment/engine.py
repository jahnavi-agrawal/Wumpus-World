"""GameEngine: owns the hidden WorldState, runs the turn sequence, keeps score and events."""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from ..config import GameConfig
from ..core.actions import ExplorerAction, WumpusAction
from ..core.geometry import Direction
from ..core.percepts import Percept, PublicInfo
from . import mechanics as M
from .events import EventKind as K, GameEvent
from .percepts import build_percept
from .world import WorldState, generate_world, safe_start_zone


class Outcome(Enum):
    EXPLORER_WIN = "Explorer escaped with the gold"
    DRAW = "Explorer escaped empty-handed"
    WUMPUS_WIN = "The Wumpus wins"


class Phase(Enum):
    EXPLORER = "explorer"
    WUMPUS = "wumpus"
    OVER = "over"


@dataclass(frozen=True)
class GameResult:
    outcome: Outcome
    reason: str


@dataclass(frozen=True)
class GameStatus:
    turn: int
    step_count: int
    score: int
    current_player: str          # "explorer" | "wumpus" | "none"
    game_over: bool
    result: GameResult | None
    inventory: dict              # {"gold": bool, "arrows": int, "jump": bool}
    arrows: int


class GameEngine:
    def __init__(self, config: GameConfig | None = None, world: WorldState | None = None):
        self.config = config or GameConfig()
        self.rng = random.Random(self.config.seed)
        self.world = world if world is not None else generate_world(self.config, self.rng)
        self.turn = 1
        self.step_count = 0
        self.score = 0
        self.phase = Phase.EXPLORER
        self.result: GameResult | None = None
        self.events: list[GameEvent] = []
        self._flags: M.ActionFlags | None = None

    # ------------------------------------------------------------ public channel
    def public_info(self) -> PublicInfo:
        c, w = self.config, self.world
        return PublicInfo(size=w.size, start=w.start, start_facing=Direction.E,
                          pit_count=len(w.pits - w.collapsed), portal_pairs=c.portal_pairs,
                          unstable_tiles=c.unstable_tiles, shuffle_tiles=c.shuffle_tiles,
                          safe_start_zone=safe_start_zone(w.start, w.size),
                          wumpus_min_start_distance=c.wumpus_min_start_distance, arrows=c.arrows,
                          jumps=c.jumps, jump_distance=c.jump_distance, max_turns=c.max_turns)

    def percept(self) -> Percept:
        return build_percept(self.world, self.turn, self.step_count, self.score, self._flags)

    def status(self) -> GameStatus:
        w = self.world
        player = {"explorer": "explorer", "wumpus": "wumpus"}.get(self.phase.value, "none")
        return GameStatus(self.turn, self.step_count, self.score, player, self.phase is Phase.OVER,
                          self.result, {"gold": w.has_gold, "arrows": w.arrows, "jump": w.jump_available},
                          w.arrows)

    # ------------------------------------------------------------ turn sequence
    def legal_explorer_actions(self) -> list[ExplorerAction]:
        return list(ExplorerAction) if self.phase is Phase.EXPLORER else []

    def legal_wumpus_actions(self) -> list[WumpusAction]:
        return M.legal_wumpus_actions(self.world) if self.phase is Phase.WUMPUS else []

    def _finish(self, outcome: Outcome, reason: str, actor: str) -> GameEvent:
        self.result = GameResult(outcome, reason)
        self.phase = Phase.OVER
        ev = GameEvent(K.GAME_OVER, actor, self.turn, self.step_count, data={
            "outcome": outcome.name, "reason": reason, "score": self.score})
        self.events.append(ev)
        return ev

    def apply_explorer_action(self, a: ExplorerAction) -> list[GameEvent]:
        if self.phase is not Phase.EXPLORER:
            raise M.IllegalAction("not the explorer's turn")
        c, w = self.config, self.world
        self.step_count += 1
        self.score += c.score_step
        evs, self._flags = M.apply_explorer_action(w, a, self.turn, self.step_count, c)
        if any(e.kind is K.ARROW_HIT or e.kind is K.ARROW_MISSED for e in evs):
            self.score += c.score_shoot
        self.events += evs
        if not w.explorer_alive:
            self.score += c.score_death
            evs.append(self._finish(Outcome.WUMPUS_WIN, evs[-1].kind.value, "explorer"))
        elif any(e.kind is K.CLIMBED_OUT for e in evs):
            if w.has_gold:
                self.score += c.score_escape_with_gold
            evs.append(self._finish(Outcome.EXPLORER_WIN if w.has_gold else Outcome.DRAW,
                                    "climbed out" + (" with the gold" if w.has_gold else ""), "explorer"))
        elif not w.wumpus_alive:
            evs += self._end_turn()
        else:
            self.phase = Phase.WUMPUS
        return evs

    def apply_wumpus_action(self, a: WumpusAction) -> list[GameEvent]:
        if self.phase is not Phase.WUMPUS:
            raise M.IllegalAction("not the Wumpus' turn")
        evs = M.apply_wumpus_action(self.world, a, self.turn, self.step_count)
        self.events += evs
        if not self.world.explorer_alive:
            self.score += self.config.score_death
            evs.append(self._finish(Outcome.WUMPUS_WIN, "EATEN", "wumpus"))
        else:
            evs += self._end_turn()
        return evs

    def _end_turn(self) -> list[GameEvent]:
        if self.turn >= self.config.max_turns:
            return [self._finish(Outcome.WUMPUS_WIN, "turn limit reached", "world")]
        self.turn += 1
        self.phase = Phase.EXPLORER
        return []
