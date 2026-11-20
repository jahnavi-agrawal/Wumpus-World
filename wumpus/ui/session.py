"""Game session for the GUI: engine + explorer + Wumpus controller, with the AI in a worker thread.

Two independent data sources feed the two boards:
  * RealitySnapshot - a copy of the hidden WorldState (for the human Wumpus / Reality panel)
  * MindSnapshot    - built ONLY from the explorer (`agent.knowledge()`, hazards, plan, report)
The engine is only touched on the main thread; the worker thread only runs `agent.act(percept)`.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from ..agents.explorer import DecisionReport, ExplorerAgent, KnowledgeSnapshot
from ..config import GameConfig
from ..control.wumpus_controllers import make_controller
from ..core.actions import WumpusAction
from ..core.geometry import Direction, Pos
from ..core.percepts import Percept
from ..environment.engine import GameEngine, GameStatus, Phase
from ..environment.events import GameEvent
from ..environment.world import WorldState


@dataclass(frozen=True)
class RealitySnapshot:
    world: WorldState               # copy of the hidden world
    percept: Percept                # what the explorer will perceive next
    status: GameStatus
    legal_wumpus: tuple
    phase: Phase


@dataclass(frozen=True)
class MindSnapshot:
    knowledge: KnowledgeSnapshot
    hazards: dict                   # Pos -> HazardEstimate
    plan_cells: tuple               # A* route (cells after each remaining action)
    plan_goal: Pos | None
    explorer: Pos                   # where the explorer expects to be after its decided action
    facing: Direction
    wumpus_belief: dict             # Pos -> probability
    report: DecisionReport | None
    observed: Pos | None = None     # where it last perceived itself (before that action)
    purpose: str = ""               # current goal in words


def mind_snapshot(agent: ExplorerAgent) -> MindSnapshot:
    plan = agent.plan
    pos, facing = agent.predicted_state()
    return MindSnapshot(agent.knowledge(), dict(agent.hazards),
                        tuple(plan.cells) if plan else (), plan.goal if plan else None,
                        pos, facing, dict(agent.tracker.belief) if agent.tracker.alive else {},
                        agent.reports[-1] if agent.reports else None, agent.pos,
                        plan.purpose if plan else "")


class GameSession:
    WUMPUS_MODES = ("human", "random", "hunter", "minimax", "idle")

    def __init__(self, seed: int = 7, wumpus_mode: str = "human", max_turns: int = 150):
        self.max_turns = max_turns
        self.wumpus_mode = wumpus_mode
        self.restart(seed)

    # ------------------------------------------------------------ lifecycle
    def restart(self, seed: int | None = None) -> None:
        self.seed = self.seed if seed is None else seed
        self.engine = GameEngine(GameConfig(seed=self.seed, max_turns=self.max_turns))
        self.agent = ExplorerAgent(self.engine.public_info())
        self.controller = None if self.wumpus_mode == "human" else make_controller(self.wumpus_mode, self.seed)
        self._thread: threading.Thread | None = None
        self._result = None
        self._error: BaseException | None = None
        self.log_count = 0
        self.last_think_s = 0.0
        self.last_events: list[GameEvent] = []
        self.reality = self._reality()
        self.mind = mind_snapshot(self.agent)

    def set_wumpus_mode(self, mode: str) -> None:
        self.wumpus_mode = mode
        self.controller = None if mode == "human" else make_controller(mode, self.seed)

    # ------------------------------------------------------------ state
    @property
    def phase(self) -> Phase:
        return self.engine.phase

    @property
    def busy(self) -> bool:
        return self._thread is not None

    @property
    def human_turn(self) -> bool:
        return self.phase is Phase.WUMPUS and self.controller is None and not self.busy

    def entries(self):
        return self.agent.log.entries[:self.log_count]

    def _reality(self) -> RealitySnapshot:
        e = self.engine
        return RealitySnapshot(e.world.copy(), e.percept(), e.status(), tuple(e.legal_wumpus_actions()), e.phase)

    def _sync(self) -> None:
        self.reality = self._reality()
        self.mind = mind_snapshot(self.agent)
        self.log_count = len(self.agent.log.entries)

    # ------------------------------------------------------------ explorer (background)
    def request_explorer_step(self) -> bool:
        if self.phase is not Phase.EXPLORER or self.busy:
            return False
        percept = self.engine.percept()

        def work():
            t0 = time.perf_counter()
            try:
                self._result = self.agent.act(percept)
            except BaseException as exc:          # surfaced on the main thread in poll()
                self._error = exc
            self.last_think_s = time.perf_counter() - t0

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        return True

    def poll(self) -> list[GameEvent] | None:
        """Main thread: if the explorer finished thinking, apply its action. Returns the events."""
        if self._thread is None or self._thread.is_alive():
            return None
        self._thread.join()
        self._thread = None
        if self._error is not None:
            err, self._error = self._error, None
            raise err
        events = self.engine.apply_explorer_action(self._result)
        self._result = None
        self._sync()
        self.last_events = events
        return events

    def wait(self, timeout: float = 10.0) -> list[GameEvent] | None:
        if self._thread is not None:
            self._thread.join(timeout)
        return self.poll()

    # ------------------------------------------------------------ Wumpus
    def submit_wumpus(self, action: WumpusAction) -> list[GameEvent] | None:
        """Human move. Returns None if it is not the Wumpus' turn or the move is illegal."""
        if self.phase is not Phase.WUMPUS or self.busy or action not in self.engine.legal_wumpus_actions():
            return None
        events = self.engine.apply_wumpus_action(action)
        self._sync()
        self.last_events = events
        return events

    def wumpus_ai_step(self) -> list[GameEvent] | None:
        if self.phase is not Phase.WUMPUS or self.controller is None or self.busy:
            return None
        a = self.controller.choose(self.engine.world, self.engine.legal_wumpus_actions())
        return self.submit_wumpus(a)
