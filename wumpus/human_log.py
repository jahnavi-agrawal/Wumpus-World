"""Records games played by a human Wumpus (GUI or terminal) for evaluation/human_games.py.

Each finished game appends one JSON line to evaluation/results/human_games.jsonl. Only games where a
human controlled the Wumpus from the first to the last turn are recorded. Nothing is sent anywhere.
"""
from __future__ import annotations

import datetime
import json
import pathlib

from .metrics import GameRecorder

HUMAN_LOG = pathlib.Path(__file__).resolve().parents[1] / "evaluation" / "results" / "human_games.jsonl"


class HumanGameLog:
    def __init__(self, agent, interface: str, path: pathlib.Path | None = HUMAN_LOG):
        self.recorder = GameRecorder(agent)
        self.interface = interface
        self.path = path
        self.valid = path is not None          # set False if an AI takes over the Wumpus mid-game
        self.saved = False

    def before_action(self, world, action) -> None:
        self.recorder.before_action(world, action)

    def finish(self, engine) -> None:
        if not self.valid or self.saved or engine.result is None:
            return
        self.saved = True
        r = engine.result
        row = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "interface": self.interface,
               "seed": engine.config.seed, "outcome": r.outcome.name, "reason": r.reason, "score": engine.score,
               "steps": engine.step_count, "turns": engine.turn, "metrics": self.recorder.finish(engine)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
