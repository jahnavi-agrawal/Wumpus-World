"""Evaluation metrics measured from OUTSIDE the agent: each turn its beliefs are compared with the
hidden world. Used by the benchmark and to record human games in the GUI. The agent never sees this.

    rec = GameRecorder(agent)
    ... action = agent.act(engine.percept()); rec.before_action(engine.world, action); engine.apply...
    metrics = rec.finish(engine)
"""
from __future__ import annotations

from .core.actions import ExplorerAction
from .core.geometry import manhattan, neighbors
from .environment.events import EventKind as K

BINS = 10                      # calibration bins of width 0.1 on P(pit)


class GameRecorder:
    def __init__(self, agent):
        self.agent = agent
        info = agent.info
        could_hold_pit = info.size ** 2 - 3 - 2 * info.portal_pairs - info.unstable_tiles - info.shuffle_tiles
        self.base_rate = info.pit_count / could_hold_pit     # baseline: "knows only how many pits exist"
        self.cal = [[0, 0.0, 0] for _ in range(BINS)]       # per bin: cells, sum of predicted P, actual pits
        self.brier = self.brier_base = 0.0
        self.w_turns = self.w_top1 = 0
        self.w_mass = 0.0
        self.danger_turns = self.minimax_turns = self.minimax_danger_turns = 0
        self._danger_pending = False
        self._reports_seen = 0

    # ------------------------------------------------------------ per decision
    def before_action(self, world, action: ExplorerAction) -> None:
        """Call after agent.act() and before the engine applies the action."""
        a, size = self.agent, world.size
        # 1. pit-probability calibration on the frontier (unvisited cells next to visited ones)
        known_pits = set(a.collapsed)
        frontier = {n for v in a.visited for n in neighbors(v, size)} - a.visited - known_pits
        for c in frontier:
            p = a.pit_prob.get(c)
            if p is None:
                continue
            pit = c in world.pits
            b = self.cal[min(int(p * BINS), BINS - 1)]
            b[0] += 1
            b[1] += p
            b[2] += pit
            self.brier += (p - pit) ** 2
            self.brier_base += (self.base_rate - pit) ** 2
        # 2. Wumpus tracker: belief placed on the Wumpus' true cell
        if world.wumpus_alive and a.tracker.alive and a.tracker.belief:
            self.w_turns += 1
            self.w_mass += a.tracker.belief.get(world.wumpus, 0.0)
            self.w_top1 += max(a.tracker.belief, key=a.tracker.belief.get) == world.wumpus
        # 3. Minimax use, and danger = the real Wumpus is adjacent when the explorer decides
        used_minimax = False
        if len(a.reports) > self._reports_seen:
            used_minimax = a.reports[-1].minimax is not None
            self._reports_seen = len(a.reports)
        self.minimax_turns += used_minimax
        danger = world.wumpus_alive and manhattan(world.explorer, world.wumpus) == 1
        self.danger_turns += danger
        self.minimax_danger_turns += danger and used_minimax
        self._danger_pending = danger

    # ------------------------------------------------------------ end of game
    def finish(self, engine) -> dict:
        r = engine.result
        kinds = [e.kind for e in engine.events]
        return {
            "cal": [list(b) for b in self.cal],
            "cal_cells": sum(b[0] for b in self.cal), "brier": self.brier, "brier_base": self.brier_base,
            "w_turns": self.w_turns, "w_mass": self.w_mass, "w_top1": self.w_top1,
            "danger_turns": self.danger_turns,
            # eaten in the same turn as a danger decision (the game ends right there)
            "danger_deaths": int(self._danger_pending and r is not None and r.reason == "EATEN"),
            "minimax_turns": self.minimax_turns, "minimax_danger_turns": self.minimax_danger_turns,
            "shots": kinds.count(K.ARROW_HIT) + kinds.count(K.ARROW_MISSED), "hits": kinds.count(K.ARROW_HIT),
        }
