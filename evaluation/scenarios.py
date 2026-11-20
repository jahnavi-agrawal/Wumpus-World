"""Deterministic demonstration scenarios (hand-built dungeons, scripted Wumpus).

Each scenario builds a fixed hidden world, runs the real engine and the real explorer, and checks
expectations about what must happen (events in the engine, entries in the agent's Brain log).
Run:  python -m evaluation.scenarios            (prints a pass/fail table)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from wumpus.agents.explorer import ExplorerAgent
from wumpus.config import GameConfig
from wumpus.core.actions import STAY, WumpusAction
from wumpus.core.geometry import Direction, Pos
from wumpus.environment.engine import GameEngine, Outcome, Phase
from wumpus.environment.events import EventKind as K
from wumpus.environment.world import WorldState

D = Direction


def world(pits=(), gold=(7, 7), wumpus=(7, 0), portals=(), unstable=(), shuffle=(), route=(), seed=0) -> WorldState:
    P = lambda t: Pos(*t)
    pm = {}
    for a, b in portals:
        pm[P(a)], pm[P(b)] = P(b), P(a)
    return WorldState(size=8, start=Pos(0, 0), pits={P(p) for p in pits}, gold=P(gold), wumpus=P(wumpus),
                      explorer=Pos(0, 0), facing=D.E, portals=pm, unstable={P(c) for c in unstable},
                      shuffle_tiles={P(c) for c in shuffle}, route=tuple(P(c) for c in route),
                      arrows=1, jump_available=True, seed=seed, visited={Pos(0, 0)})


class ScriptedMoves:
    """Plays a fixed list of Wumpus moves (illegal or exhausted -> STAY)."""

    def __init__(self, moves=()):
        self.moves = list(moves)

    def choose(self, w, legal):
        a = self.moves.pop(0) if self.moves else STAY
        return a if a in legal else STAY


M = WumpusAction.move


@dataclass
class Scenario:
    name: str
    description: str
    build: Callable[[], WorldState]
    wumpus_moves: tuple = ()
    expect: list[tuple[str, Callable]] = field(default_factory=list)   # (description, check(engine, agent))
    max_turns: int = 150


@dataclass
class ScenarioResult:
    scenario: Scenario
    engine: GameEngine
    agent: ExplorerAgent
    checks: list[tuple[str, bool]]

    @property
    def passed(self) -> bool:
        return all(ok for _, ok in self.checks)


def run_scenario(sc: Scenario) -> ScenarioResult:
    w = sc.build()
    engine = GameEngine(GameConfig(seed=w.seed, max_turns=sc.max_turns), world=w)
    agent = ExplorerAgent(engine.public_info())
    ctrl = ScriptedMoves(sc.wumpus_moves)
    while engine.phase is not Phase.OVER:
        if engine.phase is Phase.EXPLORER:
            engine.apply_explorer_action(agent.act(engine.percept()))
        else:
            engine.apply_wumpus_action(ctrl.choose(engine.world, engine.legal_wumpus_actions()))
    checks = []
    for desc, fn in sc.expect:
        try:
            ok = bool(fn(engine, agent))
        except Exception:                       # a failed lookup counts as a failed expectation
            ok = False
        checks.append((desc, ok))
    return ScenarioResult(sc, engine, agent, checks)


# ------------------------------------------------------------------ expectation helpers
def won(e, a):
    return e.result.outcome is Outcome.EXPLORER_WIN


def event(kind):
    return lambda e, a: any(ev.kind is kind for ev in e.events)


def logged(tag, text=""):
    return lambda e, a: any(x.tag == tag and text in x.summary for x in a.log.entries)


def never_entered(*cells):
    cs = {Pos(*c) for c in cells}
    return lambda e, a: not any(ev.kind in (K.MOVED, K.JUMPED, K.TELEPORTED) and ev.end in cs
                                and ev.actor == "explorer" for ev in e.events)


def only_proven_steps(e, a):
    """Every FORWARD/JUMP landed on a cell the KB had proven pit-free at decision time."""
    for r in a.reports:
        if r.selected in ("FORWARD", "JUMP"):
            txt = dict(r.candidates)[r.selected]
            if "(proven pit-free)" not in txt:
                return False
    return True


def _risk(report):
    for act, txt in report.candidates:
        if act == report.selected and "P(death)" in txt:
            return int(txt.rsplit("P(death) ", 1)[1].rstrip("%")) / 100
    return 0.0


# ------------------------------------------------------------------ the scenarios
SCENARIOS: list[Scenario] = [
    Scenario(
        "safe_exploration",
        "No breeze near the start: every step is onto a cell the KB has proven pit-free.",
        lambda: world(pits=[(6, 6), (7, 5)], gold=(2, 2), wumpus=(7, 7)),
        expect=[("explorer wins", won),
                ("every step onto a KB-proven pit-free cell", only_proven_steps),
                ("cells proven safe by inference", logged("INFERENCE", "proven SAFE"))]),
    Scenario(
        "logical_pit_deduction",
        "Breeze at (2,0), none at (1,1) and (2,1)... the KB proves the pit at (3,0) and avoids it.",
        lambda: world(pits=[(3, 0)], gold=(4, 1), wumpus=(7, 7)),
        expect=[("pit (3,0) proven by DPLL", logged("INFERENCE", "Cell (3,0) proven PIT")),
                ("never entered (3,0)", never_entered((3, 0))),
                ("explorer wins", won)]),
    Scenario(
        "ambiguous_hazards",
        "A breeze with two unknown candidates: the KB leaves both ambiguous, Bayes rates them, and the "
        "explorer goes around instead of gambling.",
        lambda: world(pits=[(3, 1)], gold=(5, 0), wumpus=(7, 7)),
        expect=[("ambiguous cells got a probability", logged("RISK_CALC", "pit")),
                ("never entered the pit", never_entered((3, 1))),
                ("explorer wins", won)]),
    Scenario(
        "moving_wumpus",
        "The Wumpus walks down column 3 towards the explorer's path; stench changes the belief, the "
        "old Wumpus facts expire and Minimax takes over when it is close.",
        lambda: world(pits=[(6, 6)], gold=(4, 0), wumpus=(3, 4)),
        wumpus_moves=(M(D.S), M(D.S), M(D.S), STAY, STAY, M(D.N), M(D.N), M(D.N)),
        expect=[("Minimax consulted", logged("MINIMAX")),
                ("Wumpus facts expired between turns", logged("KB_TELL", "expired")),
                ("explorer survives", lambda e, a: e.world.explorer_alive)]),
    Scenario(
        "teleporter_discovery",
        "A hidden portal on the way: the explorer is teleported, learns the pair and replans.",
        lambda: world(portals=[((2, 0), (5, 5))], gold=(6, 5), wumpus=(7, 0), pits=[(0, 6)]),
        expect=[("teleported", event(K.TELEPORTED)),
                ("pair learned", lambda e, a: a.portals.get(Pos(2, 0)) == Pos(5, 5)),
                ("explorer wins", won)]),
    Scenario(
        "collapsing_route",
        "An unstable tile on the corridor collapses behind the explorer; the way home must change.",
        lambda: world(unstable=[(2, 0)], gold=(3, 0), wumpus=(7, 7), pits=[(6, 6)]),
        expect=[("tile collapsed", event(K.TILE_COLLAPSED)),
                ("explorer knows it", lambda e, a: Pos(2, 0) in a.collapsed),
                ("never stepped back onto it", lambda e, a: sum(1 for ev in e.events if ev.actor == "explorer"
                                                               and ev.kind is K.MOVED and ev.end == Pos(2, 0)) == 1),
                ("explorer wins", won)]),
    Scenario(
        "shuffle_event",
        "Stepping on a shuffle tile moves a pit; the explorer drops its stale breeze evidence.",
        lambda: world(shuffle=[(0, 2)], pits=[(3, 3), (5, 1)], gold=(0, 4), wumpus=(7, 7),
                      route=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], seed=4),
        expect=[("shuffle happened", event(K.CELLS_SHUFFLED)),
                ("stale evidence dropped", logged("EVENT", "SHUFFLE")),
                ("explorer wins", won)]),
    Scenario(
        "jump_opportunity",
        "Gold in a dead end (3,0) behind an unstable tile: once it collapses the only way home is to "
        "jump over it (3 cells ahead). The escape-route check counts the jump as a way back.",
        lambda: world(unstable=[(2, 0)], gold=(3, 0), wumpus=(7, 7), pits=[(3, 1), (4, 0)]),
        expect=[("tile collapsed behind the explorer", event(K.TILE_COLLAPSED)),
                ("jump used to get home", event(K.JUMPED)),
                ("explorer wins", won)]),
    Scenario(
        "adversarial_trap",
        "The Wumpus waits in the corridor ahead. Stench -> Minimax -> the explorer faces it and shoots.",
        lambda: world(gold=(5, 0), wumpus=(3, 0), pits=[(6, 6)]),
        expect=[("Minimax consulted", logged("MINIMAX")),
                ("Wumpus killed", event(K.ARROW_HIT)),
                ("explorer wins", won)]),
    Scenario(
        "gold_retrieval_and_escape",
        "Full run: explore, glitter, grab, plan home with A*, climb out with the gold (score > 0).",
        lambda: world(pits=[(4, 4), (6, 2)], gold=(3, 2), wumpus=(7, 7)),
        expect=[("grabbed", event(K.GRABBED_GOLD)),
                ("climbed out", event(K.CLIMBED_OUT)),
                ("positive score", lambda e, a: e.score > 0),
                ("explorer wins", won)]),
]


def run_all() -> list[ScenarioResult]:
    return [run_scenario(s) for s in SCENARIOS]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for r in run_all():
        e = r.engine
        print(f"{'PASS' if r.passed else 'FAIL'}  {r.scenario.name:<26} {e.result.outcome.name:<13} "
              f"turns={e.turn:<3} score={e.score}")
        for desc, ok in r.checks:
            print(f"        [{'x' if ok else ' '}] {desc}")
