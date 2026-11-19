"""GameEngine: owns the hidden WorldState and runs the turn sequence (ARCHITECTURE.md §5)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from ..config import GameConfig
from ..core.actions import ExplorerAction, ExplorerActionType, WumpusAction, WumpusActionType
from ..core.percepts import EventType, Percept, PublicEvent, PublicInfo
from . import mechanics as M
from .world import WorldState, generate_world, safe_start_zone


class Outcome(Enum):
    EXPLORER_WIN = "Explorer escaped with the gold"
    DRAW = "Explorer retreated empty-handed"
    WUMPUS_WIN = "The Wumpus wins"


class Phase(Enum):
    EXPLORER = "explorer"
    WUMPUS = "wumpus"
    OVER = "over"


@dataclass(frozen=True)
class GameResult:
    outcome: Outcome
    reason: str


class GameEngine:
    def __init__(self, config: GameConfig | None = None, world: WorldState | None = None):
        self.config = config or GameConfig()
        self.rng = random.Random(self.config.seed)
        self.world = world if world is not None else generate_world(self.config, self.rng)
        self.turn = 1
        self.phase = Phase.EXPLORER
        self.result: GameResult | None = None
        self.history: list[tuple[int, str]] = []
        self._events: list[PublicEvent] = []
        self._scream = False

    # ------------------------------------------------------------ public channel
    def public_info(self) -> PublicInfo:
        c, w = self.config, self.world
        return PublicInfo(size=w.size, start=w.start, pit_count=len(w.pits), pads=w.pads,
                          safe_start_zone=safe_start_zone(w.start, w.size),
                          wumpus_min_start_distance=c.wumpus_min_start_distance,
                          collapse_range=c.collapse_range, arrows=c.arrows, jumps=c.jumps,
                          max_turns=c.max_turns)

    def percept(self) -> Percept:
        w = self.world
        return Percept(turn=self.turn, position=w.explorer, stench=M.stench_at(w, w.explorer),
                       breeze=M.breeze_at(w, w.explorer), glitter=w.gold == w.explorer,
                       scream=self._scream, has_gold=w.has_gold, arrows=w.arrows,
                       jumps_left=w.jumps_left, events=tuple(self._events))

    # ------------------------------------------------------------ turn sequence
    def legal_explorer_actions(self) -> list[ExplorerAction]:
        return M.legal_explorer_actions(self.world) if self.phase is Phase.EXPLORER else []

    def legal_wumpus_actions(self) -> list[WumpusAction]:
        return M.legal_wumpus_actions(self.world, self.config) if self.phase is Phase.WUMPUS else []

    def _log(self, msgs: list[str]) -> None:
        self.history += [(self.turn, m) for m in msgs]

    def _finish(self, outcome: Outcome, reason: str) -> None:
        self.result = GameResult(outcome, reason)
        self.phase = Phase.OVER
        self._log([f"GAME OVER: {outcome.value} ({reason})"])

    def apply_explorer_action(self, a: ExplorerAction) -> list[str]:
        if self.phase is not Phase.EXPLORER:
            raise M.IllegalAction("not the explorer's turn")
        self._events, self._scream = [], False   # percept of this turn has been consumed
        msgs, self._scream = M.apply_explorer_action(self.world, a)
        self._log(msgs)
        w = self.world
        if not w.explorer_alive:
            self._finish(Outcome.WUMPUS_WIN, msgs[-1])
        elif a.kind is ExplorerActionType.CLIMB:
            self._finish(Outcome.EXPLORER_WIN if w.has_gold else Outcome.DRAW, msgs[-1])
        elif not w.wumpus_alive:
            self._end_turn()            # dead Wumpus: skip its phase
        else:
            self.phase = Phase.WUMPUS
        return msgs

    def apply_wumpus_action(self, a: WumpusAction) -> list[str]:
        if self.phase is not Phase.WUMPUS:
            raise M.IllegalAction("not the Wumpus' turn")
        msgs = M.apply_wumpus_action(self.world, a, self.config, self.rng)
        self._log(msgs)
        if a.kind is WumpusActionType.COLLAPSE:
            self._events.append(PublicEvent(EventType.COLLAPSE, self.turn, a.target))
        elif a.kind is WumpusActionType.SHUFFLE:
            self._events.append(PublicEvent(EventType.SHUFFLE, self.turn))
        if not self.world.explorer_alive:
            self._finish(Outcome.WUMPUS_WIN, msgs[-1])
        else:
            self._end_turn()
        return msgs

    def _end_turn(self) -> None:
        if self.turn >= self.config.max_turns:
            self._finish(Outcome.WUMPUS_WIN, "turn limit reached")
            return
        self.turn += 1
        self.phase = Phase.EXPLORER
