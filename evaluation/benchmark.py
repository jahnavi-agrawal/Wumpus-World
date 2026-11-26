"""Reproducible benchmark: explorer variants x Wumpus opponents x N held-out maps.

    python -m evaluation.benchmark                      # 500 maps (seeds 1000-1499), sweep on 200
    python -m evaluation.benchmark --maps 50 --sweep-maps 20     # quick check

Agents
  random          baseline: random FORWARD/TURN (forward twice as likely), grabs on glitter, climbs at the
                  exit with the gold. No reasoning.
  logic+A*        KB inference + A*; only KB-proven cells are ever entered; no probabilities, no Minimax.
  logic+bayes+A*  + Bayesian pit enumeration and Wumpus filter (risk-aware exploration); no Minimax.
  full hybrid     + Minimax with alpha-beta when the Wumpus is near (the shipped agent).
Opponents (AI-vs-AI, automated; human games are never mixed in):
  idle            Wumpus never moves (isolates hazard reasoning from the duel).
  random          Wumpus moves randomly.
  hunter          omniscient greedy chaser (shortest path to the explorer).
  minimax         omniscient Wumpus using the same DuelGame search from the MIN side.
Also
  oracle          omniscient explorer vs an idle Wumpus (evaluation/oracle.py): which maps are winnable at
                  all, and the fewest actions needed.
  sweep           full hybrid with other Minimax depths / hypothesis counts, vs hunter and minimax.
  belief quality  pit-probability calibration and Wumpus-tracker accuracy against the hidden world
                  (wumpus/metrics.py).
Seeds 0-29 were used while developing the agent; the benchmark uses held-out seeds 1000+.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

AGENTS = ("random", "logic+A*", "logic+bayes+A*", "full hybrid")
OPPONENTS = ("idle", "random", "hunter", "minimax")
ABLATION_STEPS = (("logic+A*", "logic+bayes+A*"), ("logic+bayes+A*", "full hybrid"), ("logic+A*", "full hybrid"))
SWEEP_OPPONENTS = ("hunter", "minimax")
SWEEP = {"depth 2": {"minimax_depth": 2}, "depth 6": {"minimax_depth": 6},
         "hypotheses 1": {"minimax_hypotheses": 1}, "hypotheses 8": {"minimax_hypotheses": 8}}
DEFAULT_LABEL = "depth 4, hypotheses 4 (default)"
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


def make_agent(name: str, info, seed: int, overrides: dict):
    from dataclasses import replace
    from wumpus.agents.explorer import ExplorerAgent
    from wumpus.config import AgentConfig
    if name == "random":
        return RandomAgent(info, seed)
    cfg = {"logic+A*": AgentConfig(use_bayes=False, use_minimax=False),
           "logic+bayes+A*": AgentConfig(use_minimax=False),
           "full hybrid": AgentConfig()}[name]
    return ExplorerAgent(info, replace(cfg, **overrides))


def run_task(task) -> dict:
    if task[0] == "oracle":
        from evaluation.oracle import shortest_win
        from wumpus.config import GameConfig
        return {"kind": "oracle", "seed": task[1], "shortest_win": shortest_win(GameConfig(seed=task[1]))}
    _, variant, agent_name, overrides, opponent, seed = task
    from wumpus.config import GameConfig
    from wumpus.control.wumpus_controllers import make_controller
    from wumpus.environment.engine import GameEngine, Phase
    from wumpus.metrics import GameRecorder
    engine = GameEngine(GameConfig(seed=seed))
    agent = make_agent(agent_name, engine.public_info(), seed, overrides)
    rec = GameRecorder(agent) if agent_name != "random" else None
    wumpus = make_controller(opponent, seed)
    times = []
    while engine.phase is not Phase.OVER:
        if engine.phase is Phase.EXPLORER:
            t0 = time.perf_counter()
            a = agent.act(engine.percept())
            times.append(time.perf_counter() - t0)
            if rec:
                rec.before_action(engine.world, a)
            engine.apply_explorer_action(a)
        else:
            engine.apply_wumpus_action(wumpus.choose(engine.world, engine.legal_wumpus_actions()))
    r = engine.result
    row = {"kind": "game", "variant": variant, "agent": agent_name, "opponent": opponent, "seed": seed,
           "outcome": r.outcome.name, "reason": r.reason, "score": engine.score, "steps": engine.step_count,
           "mean_ms": 1000 * statistics.mean(times), "max_ms": 1000 * max(times)}
    if rec:
        row["metrics"] = rec.finish(engine)
    return row


# ------------------------------------------------------------------ statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion (Wilson score interval)."""
    if n == 0:
        return 0.0, 1.0
    p, den = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar test on paired wins: P(a split at least this uneven | no difference)."""
    n, k = only_a + only_b, min(only_a, only_b)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def is_win(r: dict) -> bool:
    return r["outcome"] == "EXPLORER_WIN"


def summarize(rows: list[dict], oracle: dict) -> dict:
    n = len(rows)
    wins = sum(map(is_win, rows))
    winnable = [r for r in rows if oracle.get(r["seed"]) is not None]
    s = {"games": n, "wins": wins, "win_rate": wins / n, "win_ci": wilson(wins, n),
         "winnable": len(winnable), "win_rate_winnable": sum(map(is_win, winnable)) / max(len(winnable), 1),
         "draw": sum(r["outcome"] == "DRAW" for r in rows) / n,
         "pit_death": sum(r["reason"] == "FELL_INTO_PIT" for r in rows) / n,
         "eaten": sum(r["reason"] == "EATEN" for r in rows) / n,
         "timeout": sum(r["reason"] == "turn limit reached" for r in rows) / n,
         "median_score": statistics.median(r["score"] for r in rows),
         "avg_steps": statistics.mean(r["steps"] for r in rows),
         "avg_planning_ms": statistics.mean(r["mean_ms"] for r in rows),
         "max_planning_ms": max(r["max_ms"] for r in rows)}
    ratios = [r["steps"] / oracle[r["seed"]] for r in rows if is_win(r) and oracle.get(r["seed"])]
    s["steps_vs_oracle"] = statistics.median(ratios) if ratios else None
    ms = [r["metrics"] for r in rows if "metrics" in r]
    if ms:
        tot = lambda k: sum(m[k] for m in ms)
        s["cal"] = [[sum(m["cal"][b][i] for m in ms) for i in range(3)] for b in range(len(ms[0]["cal"]))]
        cells = tot("cal_cells")
        s["brier"] = tot("brier") / max(cells, 1)
        s["brier_base"] = tot("brier_base") / max(cells, 1)
        s["tracker_p_true"] = tot("w_mass") / max(tot("w_turns"), 1)
        s["tracker_top1"] = tot("w_top1") / max(tot("w_turns"), 1)
        s["danger_turns"], s["danger_deaths"] = tot("danger_turns"), tot("danger_deaths")
        s["minimax_turns_per_game"] = tot("minimax_turns") / n
        s["minimax_share_of_danger"] = tot("minimax_danger_turns") / max(tot("danger_turns"), 1)
        s["shots"], s["hits"] = tot("shots"), tot("hits")
    return s


def paired(a_rows: list[dict], b_rows: list[dict]) -> dict:
    a = {r["seed"]: is_win(r) for r in a_rows}
    b = {r["seed"]: is_win(r) for r in b_rows}
    only_a = sum(a[s] and not b[s] for s in a if s in b)
    only_b = sum(b[s] and not a[s] for s in a if s in b)
    return {"only_a": only_a, "only_b": only_b, "p": mcnemar(only_a, only_b)}


# ------------------------------------------------------------------ report
def pct(x: float) -> str:
    return f"{x:.0%}"


def ci(s: dict) -> str:
    lo, hi = s["win_ci"]
    return f"{s['win_rate']:.0%} [{lo:.0%}, {hi:.0%}]"


def report(summary: dict, pairs: dict, sweep: dict, oracle: dict, maps: int, sweep_maps: int,
           seed_txt: str) -> str:
    solved = [v for v in oracle.values() if v is not None]
    L = [f"Maps per cell: {maps} (seeds {seed_txt}), max 150 turns, default GameConfig. "
         "Win rates show a 95% Wilson confidence interval.", "",
         "### Winnability (oracle)", "",
         f"An omniscient explorer against an idle Wumpus wins **{len(solved)} of {len(oracle)}** maps "
         f"(fewest actions: median {statistics.median(solved):.0f}, max {max(solved)}). "
         "Unwinnable maps are excluded from \"win on winnable\".", ""]
    for opp in OPPONENTS:
        L += [f"### Opponent: {opp} Wumpus", "",
              "| Agent | Win [95% CI] | Win on winnable | Draw | Pit death | Eaten | Timeout | Median score "
              "| Avg steps | Steps vs oracle (wins) | Planning ms avg / max |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for ag in AGENTS:
            s = summary[f"{ag} | {opp}"]
            eff = f"{s['steps_vs_oracle']:.1f}x" if s["steps_vs_oracle"] else "-"
            L.append(f"| {ag} | {ci(s)} | {pct(s['win_rate_winnable'])} | {pct(s['draw'])} | {pct(s['pit_death'])} "
                     f"| {pct(s['eaten'])} | {pct(s['timeout'])} | {s['median_score']:.0f} | {s['avg_steps']:.1f} "
                     f"| {eff} | {s['avg_planning_ms']:.1f} / {s['max_planning_ms']:.0f} |")
        L.append("")
    L += ["### Does each component help? (paired on the same maps)", "",
          "Exact McNemar test on wins: only maps where exactly one of the two agents won count. "
          "p < 0.05 means the difference is unlikely to be chance.", "",
          "| Opponent | Comparison | Wins A / B | Maps only A won | Maps only B won | p |", "|---|---|---|---|---|---|"]
    for opp in OPPONENTS:
        for a, b in ABLATION_STEPS:
            p = pairs[f"{a} -> {b} | {opp}"]
            wa, wb = summary[f"{a} | {opp}"]["wins"], summary[f"{b} | {opp}"]["wins"]
            L.append(f"| {opp} | {a} -> {b} | {wa} / {wb} | {p['only_a']} | {p['only_b']} | "
                     f"{p['p']:.3f}{' *' if p['p'] < 0.05 else ''} |")
    L += ["", "### Belief quality (full hybrid, measured against the hidden world)", "",
          "**Pit-probability calibration** on frontier cells (unvisited cells next to visited ones), every "
          "decision, all opponents pooled. A calibrated agent's \"observed\" column matches \"predicted\".", "",
          "| Predicted P(pit) | Cells | Mean predicted | Observed pit rate |", "|---|---|---|---|"]
    cal = [[0, 0.0, 0] for _ in range(10)]
    for opp in OPPONENTS:
        for b, (c, sp, pits) in enumerate(summary[f"full hybrid | {opp}"]["cal"]):
            cal[b][0] += c
            cal[b][1] += sp
            cal[b][2] += pits
    for b, (c, sp, pits) in enumerate(cal):
        if c:
            L.append(f"| {b / 10:.1f}-{(b + 1) / 10:.1f} | {c} | {sp / c:.3f} | {pits / c:.3f} |")
    fh = [summary[f"full hybrid | {o}"] for o in OPPONENTS]
    w_cells = [sum(x for x, _, _ in s["cal"]) for s in fh]
    brier = sum(s["brier"] * w for s, w in zip(fh, w_cells)) / sum(w_cells)
    base = sum(s["brier_base"] * w for s, w in zip(fh, w_cells)) / sum(w_cells)
    L += ["", f"Brier score {brier:.4f} vs {base:.4f} for a baseline that only knows the pit density "
          f"(lower is better; {1 - brier / base:.0%} better).", "",
          "**Wumpus tracker** (HMM filter): probability it puts on the Wumpus' true cell, and how often its "
          "most likely cell is the true one. A uniform guess over the 64 cells gives 1.6%.", "",
          "| Opponent | Mean P(true cell) | Top cell correct |", "|---|---|---|"]
    for opp in OPPONENTS:
        s = summary[f"full hybrid | {opp}"]
        L.append(f"| {opp} | {pct(s['tracker_p_true'])} | {pct(s['tracker_top1'])} |")
    L += ["", "### Minimax where it acts", "",
          "A *danger turn* is a decision made with the real Wumpus in an adjacent cell. \"Eaten\" counts the "
          "danger turns that ended with the explorer eaten in that same turn.", "",
          "| Opponent | Agent | Danger turns | Eaten on a danger turn [95% CI] | Minimax ran on danger turns "
          "| Minimax turns / game | Shots (hits) |", "|---|---|---|---|---|---|---|"]
    for opp in OPPONENTS:
        for ag in ("logic+bayes+A*", "full hybrid"):
            s = summary[f"{ag} | {opp}"]
            k, n = s["danger_deaths"], s["danger_turns"]
            lo, hi = wilson(k, n)
            L.append(f"| {opp} | {ag} | {n} | {k / max(n, 1):.0%} [{lo:.0%}, {hi:.0%}] | "
                     f"{pct(s['minimax_share_of_danger'])} | {s['minimax_turns_per_game']:.1f} | "
                     f"{s['shots']} ({s['hits']}) |")
    L += ["", f"### Minimax settings sweep (full hybrid, first {sweep_maps} maps)", "",
          "| Setting | " + " | ".join(f"Win vs {o} [95% CI]" for o in SWEEP_OPPONENTS) +
          " | Eaten on a danger turn | Planning ms avg / max |", "|---|" + "---|" * (len(SWEEP_OPPONENTS) + 2)]
    for label in [DEFAULT_LABEL, *SWEEP]:
        cells = [sweep[f"{label} | {o}"] for o in SWEEP_OPPONENTS]
        k, n = sum(c["danger_deaths"] for c in cells), sum(c["danger_turns"] for c in cells)
        ms = statistics.mean(c["avg_planning_ms"] for c in cells)
        mx = max(c["max_planning_ms"] for c in cells)
        L.append(f"| {label} | " + " | ".join(ci(c) for c in cells) +
                 f" | {k / max(n, 1):.0%} of {n} | {ms:.1f} / {mx:.0f} |")
    L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=500)
    ap.add_argument("--sweep-maps", type=int, default=200)
    ap.add_argument("--first-seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)
    seeds = range(args.first_seed, args.first_seed + args.maps)
    sweep_seeds = seeds[:args.sweep_maps]
    tasks = [("oracle", s) for s in seeds]
    tasks += [("game", "main", ag, {}, opp, s) for opp in OPPONENTS for ag in AGENTS for s in seeds]
    tasks += [("game", label, "full hybrid", ov, opp, s)
              for label, ov in SWEEP.items() for opp in SWEEP_OPPONENTS for s in sweep_seeds]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(run_task, tasks, chunksize=4), 1):
            rows.append(r)
            if i % 250 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} tasks, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    oracle = {r["seed"]: r["shortest_win"] for r in rows if r["kind"] == "oracle"}
    games = [r for r in rows if r["kind"] == "game"]
    main_rows = [r for r in games if r["variant"] == "main"]
    cell = lambda ag, opp, rs: [r for r in rs if r["agent"] == ag and r["opponent"] == opp]
    summary = {f"{ag} | {opp}": summarize(cell(ag, opp, main_rows), oracle) for opp in OPPONENTS for ag in AGENTS}
    pairs = {f"{a} -> {b} | {opp}": paired(cell(a, opp, main_rows), cell(b, opp, main_rows))
             for opp in OPPONENTS for a, b in ABLATION_STEPS}
    sweep = {}
    in_sweep = set(sweep_seeds)
    for opp in SWEEP_OPPONENTS:
        default = [r for r in cell("full hybrid", opp, main_rows) if r["seed"] in in_sweep]
        sweep[f"{DEFAULT_LABEL} | {opp}"] = summarize(default, oracle)
        for label in SWEEP:
            sweep[f"{label} | {opp}"] = summarize([r for r in games if r["variant"] == label and r["opponent"] == opp],
                                                  oracle)
    for r in games:                                   # per-game rows stay small: drop the calibration bins
        r.get("metrics", {}).pop("cal", None)
    OUT.mkdir(exist_ok=True)
    seed_txt = f"{seeds.start}-{seeds.stop - 1}"
    (OUT / "benchmark.json").write_text(json.dumps(
        {"seeds": seed_txt, "oracle": oracle, "summary": summary, "paired": pairs, "sweep": sweep, "games": games},
        separators=(",", ":")))
    md = report(summary, pairs, sweep, oracle, args.maps, args.sweep_maps, seed_txt)
    (OUT / "benchmark.md").write_text(
        f"# Benchmark results\n\nGenerated by `python -m evaluation.benchmark --maps {args.maps} "
        f"--sweep-maps {args.sweep_maps}` in {time.time() - t0:.0f} s.\n\n{md}", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
