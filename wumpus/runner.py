"""Headless game loop: wires the engine (hidden), the agent (percepts only) and a Wumpus controller."""
from __future__ import annotations

import time

from .ai.explorer import ExplorerAgent
from .config import AgentConfig, GameConfig
from .env.engine import GameEngine, Phase


def play_game(config: GameConfig, controller, agent_config: AgentConfig | None = None,
              on_step=None) -> tuple[GameEngine, ExplorerAgent, dict]:
    engine = GameEngine(config)
    agent = ExplorerAgent(engine.public_info(), agent_config)
    timings: list[float] = []
    while engine.phase is not Phase.OVER:
        if engine.phase is Phase.EXPLORER:
            t0 = time.perf_counter()
            action = agent.act(engine.percept())
            timings.append(time.perf_counter() - t0)
            engine.apply_explorer_action(action)
        else:
            engine.apply_wumpus_action(controller.choose(engine.world, engine.legal_wumpus_actions()))
        if on_step:
            on_step(engine, agent)
    stats = {"turns": engine.turn, "max_decide_s": max(timings, default=0.0),
             "mean_decide_s": sum(timings) / max(len(timings), 1)}
    return engine, agent, stats
