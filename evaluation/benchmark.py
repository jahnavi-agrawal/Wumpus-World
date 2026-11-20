"""Reproducible benchmark: 4 explorer variants x Wumpus opponents x N held-out maps.

    python -m evaluation.benchmark                 # 100 maps (seeds 1000-1099), random + minimax Wumpus
    python -m evaluation.benchmark --maps 20       # quicker

Agents
  random          baseline: random FORWARD/TURN (forward twice as likely), grabs on glitter, climbs at the
                  exit with the gold. No reasoning.
  logic+A*        KB inference + A*; only KB-proven cells are ever entered; no probabilities, no Minimax.
  logic+bayes+A*  + Bayesian pit enumeration and Wumpus filter (risk-aware exploration); no Minimax.
  full hybrid     + Minimax with alpha-beta when the Wumpus is near (the shipped agent).
Opponents (AI-vs-AI, automated; human games are never mixed in):
  random          Wumpus moves randomly.
  hunter          omniscient greedy chaser (shortest path to the explorer).
  minimax         omniscient Wumpus using the same DuelGame search from the MIN side.
Seeds 0-29 were used while developing the agent; the benchmark uses held-out seeds 1000+.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

AGENTS = ("random", "logic+A*", "logic+bayes+A*", "full hybrid")
OPPONENTS = ("random", "hunter", "minimax")
OUT = pathlib.Path(__file__).resolve().parent / "results"


class RandomAgent:
    def __init__(self, info, seed: int):
        self.info = info
        self.rng = random.Random(seed)

    def act(self, p):
        from wumpus.core.actions import CLIMB, FORWARD, GRAB, TURN_LEFT, TURN_RIGHT
        if p.glitter and not p.has_gold:
            return GRAB
        if p.has_gold and p.position == self.info.start:
            return CLIMB
        return self.rng.choice((FORWARD, FORWARD, TURN_LEFT, TURN_RIGHT))


def make_agent(name: str, info, seed: int):
    from wumpus.agents.explorer import ExplorerAgent
    from wumpus.config import AgentConfig
    if name == "random":
        return RandomAgent(info, seed)
    cfg = {"logic+A*": AgentConfig(use_bayes=False, use_minimax=False),
           "logic+bayes+A*": AgentConfig(use_minimax=False),
           "full hybrid": AgentConfig()}[name]
    return ExplorerAgent(info, cfg)


def play(task) -> dict:
    agent_name, opponent, seed = task
    from wumpus.config import GameConfig
    from wumpus.control.wumpus_controllers import make_controller
    from wumpus.environment.engine import GameEngine, Phase
    engine = GameEngine(GameConfig(seed=seed))
    agent = make_agent(agent_name, engine.public_info(), seed)
    wumpus = make_controller(opponent, seed)
    times = []
    while engine.phase is not Phase.OVER:
        if engine.phase is Phase.EXPLORER:
            t0 = time.perf_counter()
            a = agent.act(engine.percept())
            times.append(time.perf_counter() - t0)
            engine.apply_explorer_action(a)
        else:
            engine.apply_wumpus_action(wumpus.choose(engine.world, engine.legal_wumpus_actions()))
    r = engine.result
    return {"agent": agent_name, "opponent": opponent, "seed": seed, "outcome": r.outcome.name,
            "reason": r.reason, "score": engine.score, "steps": engine.step_count,
            "alive": engine.world.explorer_alive, "mean_ms": 1000 * statistics.mean(times),
            "max_ms": 1000 * max(times)}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"games": n}
    out["win_rate"] = sum(r["outcome"] == "EXPLORER_WIN" for r in rows) / n
    out["survival_rate"] = sum(r["alive"] for r in rows) / n
    out["draw_rate"] = sum(r["outcome"] == "DRAW" for r in rows) / n
    out["avg_score"] = statistics.mean(r["score"] for r in rows)
    out["avg_steps"] = statistics.mean(r["steps"] for r in rows)
    out["pit_deaths"] = sum(r["reason"] == "FELL_INTO_PIT" for r in rows)
    out["wumpus_deaths"] = sum(r["reason"] == "EATEN" for r in rows)
    out["timeout_rate"] = sum(r["reason"] == "turn limit reached" for r in rows) / n
    out["avg_planning_ms"] = statistics.mean(r["mean_ms"] for r in rows)
    out["max_planning_ms"] = max(r["max_ms"] for r in rows)
    return out


def table(summary: dict, maps: int, seeds: str) -> str:
    lines = [f"Maps per cell: {maps} (seeds {seeds}), max 150 turns, default GameConfig.", ""]
    for opp in OPPONENTS:
        lines += [f"### Opponent: {opp} Wumpus", "",
                  "| Agent | Win | Survival | Draw (retreat) | Avg score | Avg steps | Pit deaths | Wumpus deaths "
                  "| Timeout | Planning ms (avg / max) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for ag in AGENTS:
            s = summary[f"{ag} | {opp}"]
            lines.append(f"| {ag} | {s['win_rate']:.0%} | {s['survival_rate']:.0%} | {s['draw_rate']:.0%} | "
                         f"{s['avg_score']:.0f} | {s['avg_steps']:.1f} | {s['pit_deaths']} | {s['wumpus_deaths']} | "
                         f"{s['timeout_rate']:.0%} | {s['avg_planning_ms']:.1f} / {s['max_planning_ms']:.0f} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=100)
    ap.add_argument("--first-seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)
    seeds = range(args.first_seed, args.first_seed + args.maps)
    tasks = [(ag, opp, s) for opp in OPPONENTS for ag in AGENTS for s in seeds]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(play, tasks, chunksize=4))
    summary = {f"{ag} | {opp}": summarize([r for r in rows if r["agent"] == ag and r["opponent"] == opp])
               for opp in OPPONENTS for ag in AGENTS}
    OUT.mkdir(exist_ok=True)
    seed_txt = f"{seeds.start}-{seeds.stop - 1}"
    (OUT / "benchmark.json").write_text(json.dumps({"seeds": seed_txt, "summary": summary, "games": rows}, indent=1))
    md = table(summary, args.maps, seed_txt)
    (OUT / "benchmark.md").write_text(f"# Benchmark results\n\nGenerated by `python -m evaluation.benchmark "
                                      f"--maps {args.maps}` in {time.time() - t0:.0f} s.\n\n{md}")
    print(md)


if __name__ == "__main__":
    main()
