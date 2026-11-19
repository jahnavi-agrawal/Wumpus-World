"""The explorer AI. Receives only PublicInfo + Percepts (never the hidden WorldState).

Per turn:  observe(percept)  -> belief update (KB, Bayes pits, Wumpus filter)
           decide()          -> goal selection, plan validation / A* replanning, Minimax if threatened
Everything noteworthy is written to `self.log` with the evidence that produced it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..config import AgentConfig
from ..core.actions import CLIMB, GRAB, WAIT, ExplorerAction, ExplorerActionType
from ..core.geometry import DIRECTIONS, Pos, all_cells, manhattan, neighbors, ray
from ..core.percepts import EventType, Percept, PublicInfo
from .logic import CellStatus, DungeonKB
from .minimax import DuelGame, DuelState, determinized_decision
from .pathfinding import PathResult, astar
from .probability import CellRiskReport, estimate_pits
from .trace import BrainLog
from .wumpus_tracker import FilterStep, WumpusTracker


@dataclass
class Plan:
    goal: Pos
    purpose: str
    actions: list[ExplorerAction]
    cells: list[Pos]
    cost: float
    step_risk: list[float] = field(default_factory=list)   # risk of each cell when planned


class ExplorerAgent:
    def __init__(self, info: PublicInfo, config: AgentConfig | None = None):
        self.info = info
        self.cfg = config or AgentConfig()
        self.log = BrainLog()
        self.turn = 0
        self.pos = info.start
        self.has_gold = False
        self.arrows = info.arrows
        self.jumps = info.jumps
        self.visited: set[Pos] = {info.start}
        self.collapsed: set[Pos] = set()
        self.breeze_obs: dict[Pos, tuple[bool, int]] = {}
        self.stench = False
        self.glitter = False
        self.dkb = DungeonKB(info)
        self.statuses: dict[Pos, CellStatus] = {}
        self.risk_reports: dict[Pos, CellRiskReport] = {}
        self.pit_prob: dict[Pos, float] = {}
        self.tracker = WumpusTracker(info, self.cfg.stay_prob, self.cfg.pursuit_prob)
        self.plan: Plan | None = None
        self.retreating = False
        self.last_action: ExplorerAction | None = None
        self.last_pos: Pos = info.start
        self.threat = 0.0
        self.plan_dropped: str | None = None       # why the plan was discarded outside _ensure_plan
        self.last_progress_turn = 1                # last turn a new cell was visited / gold taken
        self.turn_entries: dict[str, int] = {}     # tag -> entry id for this turn (for refs)

    # ================================================================ public API
    def act(self, percept: Percept) -> ExplorerAction:
        self.observe(percept)
        return self.decide()

    # ================================================================ belief update
    def observe(self, p: Percept) -> None:
        self.turn = p.turn
        self.turn_entries = {}
        L = self.log
        e = L.add(p.turn, "PERCEPT", f"at {p.position}: {p.flags()}" +
                  (f" | events: {', '.join(map(str, p.events))}" if p.events else ""),
                  [f"position {p.position}", f"stench={p.stench} breeze={p.breeze} glitter={p.glitter} "
                   f"scream={p.scream}", f"has_gold={p.has_gold} arrows={p.arrows} jumps_left={p.jumps_left}",
                   *(f"public event: {ev}" for ev in p.events)], cell=p.position)
        self.turn_entries["PERCEPT"] = e.id
        self.tracker.begin(p.turn)

        # --- evidence from our own previous action (before the Wumpus moved)
        la = self.last_action
        if la is not None and la.kind is ExplorerActionType.SHOOT and self.tracker.alive:
            if p.scream:
                self.tracker.kill()
                L.add(p.turn, "EVENT", "SCREAM: the arrow killed the Wumpus",
                      [f"arrow shot {la.direction.name} from {self.last_pos}", "scream percept received"])
            else:
                self.tracker.observe_arrow_missed(ray(self.last_pos, la.direction, self.info.size))
        if self.tracker.alive and la is not None and la.kind in (ExplorerActionType.MOVE, ExplorerActionType.JUMP):
            self.tracker.observe_survived_at(p.position)

        self.pos, self.has_gold, self.arrows, self.jumps = p.position, p.has_gold, p.arrows, p.jumps_left
        self.stench, self.glitter = p.stench, p.glitter
        if p.position not in self.visited or (p.has_gold and la is GRAB):
            self.last_progress_turn = p.turn
        self.visited.add(p.position)

        # --- public events
        stayed = False
        for ev in p.events:
            if ev.kind is EventType.COLLAPSE:
                self.collapsed.add(ev.pos)
                stayed = True
                L.add(p.turn, "EVENT", f"tile {ev.pos} collapsed - now impassable",
                      ["rule: collapsed tiles are deadly and impassable",
                       "rule: only non-pit cells collapse -> add ¬P to KB",
                       f"rule: Wumpus stood still within {self.info.collapse_range} of {ev.pos}"], cell=ev.pos)
            elif ev.kind is EventType.SHUFFLE:
                stale = len(self.breeze_obs)
                self.breeze_obs = {}
                stayed = True
                L.add(p.turn, "EVENT", f"SHUFFLE: {stale} breeze observations invalidated",
                      ["rule: pits were re-placed on cells never visited by the explorer",
                       "visited cells stay pit-free; all older breeze evidence is dropped",
                       "the Wumpus spent its turn and did not move"])
        self.breeze_obs[p.position] = (p.breeze, p.turn)

        self._update_pits()
        self._update_wumpus(p, stayed)

    def _update_pits(self) -> None:
        L, t = self.log, self.turn
        old_texts = {self.dkb.kb.clause_text(c) for c in self.dkb.kb.clauses}
        kb = self.dkb.rebuild(self.visited, self.collapsed, self.breeze_obs)
        if not kb.satisfiable():
            L.add(t, "WARNING", "KB inconsistent - keeping only the current observation",
                  ["This should be impossible under the public rules."])
            self.breeze_obs = {self.pos: self.breeze_obs[self.pos]}
            kb = self.dkb.rebuild(self.visited, self.collapsed, self.breeze_obs)
        new = [(kb.clause_text(c), s) for c, s in zip(kb.clauses, kb.sources) if kb.clause_text(c) not in old_texts]
        e = L.add(t, "KB_TELL", f"{len(kb.clauses)} clauses ({len(new)} new)",
                  [f"  {c}    <- {s}" for c, s in new] or ["no new clauses"])
        self.turn_entries["KB_TELL"] = e.id

        cells = [c for c in all_cells(self.info.size) if c not in self.visited and c not in self.collapsed]
        prev = self.statuses
        self.statuses = self.dkb.classify(cells)
        for c, st in sorted(self.statuses.items()):
            if st.status != "unknown" and (c not in prev or prev[c].status != st.status):
                L.add(t, "KB_INFER", f"Cell {c} proven {'SAFE (no pit)' if st.status == 'safe' else 'PIT'}",
                      st.proof.explain(), refs=[self.turn_entries["KB_TELL"]], cell=c)
        n_pit = sum(s.status == "pit" for s in self.statuses.values())
        n_unknown = sum(s.status == "unknown" for s in self.statuses.values())
        prior = max(0.0, min(1.0, (self.info.pit_count - n_pit) / n_unknown)) if n_unknown else 0.0
        old_prob = self.pit_prob
        self.risk_reports = estimate_pits(self.info.size, self.statuses, self.breeze_obs, prior,
                                          self.cfg.max_component_cells)
        self.pit_prob = {c: r.p_pit for c, r in self.risk_reports.items()}
        for c in self.visited:
            self.pit_prob[c] = 0.0
        for c, r in sorted(self.risk_reports.items()):
            if r.method in ("enumeration", "approx") and abs(old_prob.get(c, -1) - r.p_pit) >= 0.005:
                L.add(t, "RISK_CALC", f"Cell {c}: {r.p_pit:.0%} pit", r.explain(), cell=c)

    def _update_wumpus(self, p: Percept, stayed: bool) -> None:
        tr = self.tracker
        if not tr.alive:
            return
        if p.turn > 1:
            collapse = [ev for ev in p.events if ev.kind is EventType.COLLAPSE]
            if collapse:
                tr.observe_collapse(collapse[0].pos, self.info.collapse_range)
            elif stayed:
                tr.report.steps.append(FilterStep("SHUFFLE event: Wumpus spent its turn (no motion)", 1.0))
            else:
                tr.predict(p.position, self._blocked(), self.pit_prob)
            tr.observe_survived_at(p.position)
        tr.observe_stench(p.position, p.stench)
        nb = set(neighbors(p.position, self.info.size))
        fallback = [c for c in all_cells(self.info.size)
                    if c != p.position and (c in nb) == p.stench and c not in self._blocked()
                    and not (self.info.pads and c in self.info.pads)]
        tr.normalize(fallback_cells=fallback)
        self.threat = tr.prob_within(self.pos, 2)
        top = tr.top(1)
        e = self.log.add(p.turn, "WUMPUS_BELIEF",
                         f"most likely {top[0][0]} ({top[0][1]:.0%}), P(within 2)={self.threat:.0%}",
                         tr.report.explain(), cell=top[0][0])
        self.turn_entries["WUMPUS_BELIEF"] = e.id

    # ================================================================ risk model
    def _blocked(self) -> set[Pos]:
        return self.collapsed | {c for c, s in self.statuses.items() if s.status == "pit"}

    def wumpus_risk(self, c: Pos) -> float:
        if not self.tracker.alive:
            return 0.0
        if manhattan(c, self.pos) <= 2:
            return self.tracker.entry_risk(c, self._blocked(), self.pit_prob)
        return self.tracker.prob(c)

    def cell_risk(self, c: Pos) -> float:
        if c in self.collapsed:
            return 1.0
        p = self.pit_prob.get(c, 0.0 if c in self.visited else 1.0)
        return 1.0 - (1.0 - p) * (1.0 - self.wumpus_risk(c))

    def _pit_known_safe(self, c: Pos) -> bool:
        return c in self.visited or (c in self.statuses and self.statuses[c].status == "safe")

    # ================================================================ decision
    def decide(self) -> ExplorerAction:
        t, L = self.turn, self.log
        self.last_pos = self.pos
        if self.glitter and not self.has_gold:
            return self._commit(GRAB, "glitter perceived -> grab the gold", [self.turn_entries["PERCEPT"]])
        if self.has_gold and self.pos == self.info.start:
            return self._commit(CLIMB, "at the exit with the gold -> climb out", [])

        if self._should_hunt():
            shot = self._best_shot()
            if shot is not None:
                cell, d, p_hit = shot
                if cell == self.pos:
                    return self._commit(ExplorerAction.shoot(d), f"hunt: SHOOT {d.name}, P(hit)={p_hit:.0%}",
                                        [self.turn_entries["TARGET"], self.turn_entries["WUMPUS_BELIEF"]])
                return self._execute(cell, f"hunt: reach {cell} to shoot {d.name}")

        goal, purpose, candidates = self._select_goal()
        if goal is None:
            if self.pos == self.info.start:
                return self._commit(CLIMB, "no acceptable frontier cell and at the exit -> retreat (climb)",
                                    [self.turn_entries.get("TARGET", 0)])
            goal, purpose = self.info.start, "retreat to the exit"
        return self._execute(goal, purpose)

    def _execute(self, goal: Pos, purpose: str) -> ExplorerAction:
        self._ensure_plan(goal, purpose)

        if self.tracker.alive and self.threat >= self.cfg.threat_threshold:
            return self._minimax_action(goal)
        if self.plan and self.plan.actions:
            a = self.plan.actions.pop(0)
            nxt = self.plan.cells.pop(0)
            self.plan.step_risk.pop(0)
            return self._commit(a, f"{a} -> {nxt}  ({purpose}; {len(self.plan.actions)} steps left)",
                                [self.turn_entries[k] for k in ("TARGET", "ASTAR") if k in self.turn_entries])
        return self._commit(WAIT, "no path available -> wait", [self.turn_entries.get("TARGET", 0)])

    def _commit(self, a: ExplorerAction, why: str, refs: list[int]) -> ExplorerAction:
        self.last_action = a
        self.log.add(self.turn, "DECISION", why, [f"action: {a}", f"reason: {why}",
                                                  "depends on entries: " + ", ".join(f"#{r}" for r in refs)],
                     refs=refs)
        return a

    # ---------------------------------------------------------------- progress / hunting
    def stuck(self) -> int:
        return self.turn - self.last_progress_turn

    def risk_cap(self) -> float:
        return self.cfg.relaxed_risk if self.stuck() >= self.cfg.stuck_turns else self.cfg.max_explore_risk

    def _should_hunt(self) -> bool:
        return (self.tracker.alive and self.arrows > 0 and not self.has_gold
                and self.stuck() >= self.cfg.stuck_turns)

    def _best_shot(self):
        """Pick (cell, direction) maximising P(Wumpus on the arrow's ray) - 1% per step to get there."""
        dist, q = {self.pos: 0}, deque([self.pos])
        while q:
            c = q.popleft()
            for n in neighbors(c, self.info.size):
                if n not in dist and n not in self.collapsed and self._pit_known_safe(n):
                    dist[n] = dist[c] + 1
                    q.append(n)
        rows = []
        for c, k in dist.items():
            for d in DIRECTIONS:
                cells = ray(c, d, self.info.size)
                if cells:
                    p = sum(self.tracker.prob(x) for x in cells)
                    rows.append((p - 0.01 * k, p, c, d, k))
        rows.sort(key=lambda r: (-r[0], r[2], r[3].name))
        if not rows or rows[0][1] < self.cfg.min_shot_prob:
            return None
        _, p, c, d, k = rows[0]
        details = [f"No progress for {self.stuck()} turns and an arrow is left -> hunt the Wumpus.",
                   "For every reachable pit-safe cell s and direction d:  P(hit) = Σ belief(ray(s,d))",
                   f"score = P(hit) - 0.01·steps;  shoot only if P(hit) ≥ {self.cfg.min_shot_prob:.0%}",
                   f"{'cell':<7}{'dir':<4}{'steps':>6}{'P(hit)':>9}{'score':>8}"]
        details += [f"{str(r[2]):<7}{r[3].name:<4}{r[4]:>6}{r[1]:>9.1%}{r[0]:>8.3f}" for r in rows[:8]]
        e = self.log.add(self.turn, "TARGET", f"hunt: shoot {d.name} from {c} (P(hit)={p:.0%})", details,
                         refs=[self.turn_entries["WUMPUS_BELIEF"]], cell=c)
        self.turn_entries["TARGET"] = e.id
        return c, d, p

    # ---------------------------------------------------------------- goal selection
    def _frontier(self) -> list[Pos]:
        out = set()
        for v in self.visited:
            for n in neighbors(v, self.info.size):
                if n not in self.visited and n not in self.collapsed and self.pit_prob.get(n, 1.0) < 1.0:
                    out.add(n)
        return sorted(out)

    def _select_goal(self):
        cfg, L = self.cfg, self.log
        if self.has_gold:
            self.retreating = False
            e = L.add(self.turn, "TARGET", "carrying the gold -> head for the exit",
                      ["has_gold = True (from percept)", f"goal = exit {self.info.start}; then CLIMB"],
                      cell=self.info.start)
            self.turn_entries["TARGET"] = e.id
            return self.info.start, "return to the exit with the gold", []
        rows, ok = [], []
        for c in self._frontier():
            pit, wr = self.pit_prob.get(c, 0.0), self.wumpus_risk(c)
            total = 1 - (1 - pit) * (1 - wr)
            rows.append((c, pit, wr, total))
            if total <= self.risk_cap():
                ok.append(c)
        details = ["Frontier = unvisited cells adjacent to visited cells.",
                   f"Acceptable if combined risk ≤ {self.risk_cap():.0%};  risk = 1-(1-P(pit))(1-P(Wumpus))",
                   f"{'cell':<7}{'P(pit)':>8}{'P(W)':>8}{'risk':>8}"]
        details += [f"{str(c):<7}{p:>8.1%}{w:>8.1%}{r:>8.1%}{'  ok' if c in ok else ''}" for c, p, w, r in rows]
        # plan commitment: keep the current goal while it is still acceptable
        if self.plan and self.plan.goal in ok and not self.retreating:
            e = L.add(self.turn, "TARGET", f"keep target {self.plan.goal} ({len(ok)} acceptable)", details,
                      cell=self.plan.goal)
            self.turn_entries["TARGET"] = e.id
            return self.plan.goal, self.plan.purpose, ok
        if self.stuck() >= 3 * cfg.stuck_turns:
            self.retreating = True
            e = L.add(self.turn, "TARGET", f"no progress for {self.stuck()} turns -> retreat to the exit",
                      details, cell=self.info.start)
            self.turn_entries["TARGET"] = e.id
            return None, "", rows
        if not ok:
            self.retreating = True
            e = L.add(self.turn, "TARGET", f"no frontier cell below {self.risk_cap():.0%} risk -> retreat",
                      details, cell=self.info.start)
            self.turn_entries["TARGET"] = e.id
            return None, "", rows
        self.retreating = False
        res = self._astar(set(ok))
        if not res.found:
            e = L.add(self.turn, "TARGET", "acceptable cells unreachable -> retreat", details + res.explain())
            self.turn_entries["TARGET"] = e.id
            self.retreating = True
            return None, "", rows
        details.append(f"A* over all acceptable cells picks the cheapest: {res.goal} (cost {res.cost:.2f})")
        e = L.add(self.turn, "TARGET", f"explore {res.goal} (risk {self.cell_risk(res.goal):.0%}, "
                  f"{len(ok)} acceptable)", details, cell=res.goal)
        self.turn_entries["TARGET"] = e.id
        return res.goal, f"explore {res.goal}", ok

    # ---------------------------------------------------------------- planning
    def _astar(self, goals: set[Pos]) -> PathResult:
        cfg = self.cfg

        def cost(c: Pos):
            if c in self.collapsed:
                return None
            if c in goals or self._pit_known_safe(c):
                r = self.cell_risk(c)
                return None if r > max(self.risk_cap(), 0.5) else cfg.risk_weight * r
            return None
        return astar(self.pos, goals, self.info.size, cost, self.jumps, self.info.pads, cfg.jump_cost)

    def _plan_invalid_reason(self, goal: Pos) -> str | None:
        p = self.plan
        if p is None:
            reason, self.plan_dropped = self.plan_dropped, None
            return reason or "no plan"
        if not p.actions:
            return "previous plan completed"
        if p.goal != goal:
            return f"goal changed {p.goal} -> {goal}"
        for c, r0 in zip(p.cells, p.step_risk):
            if c in self.collapsed:
                return f"cell {c} on the path collapsed"
            r = self.cell_risk(c)
            if r > r0 + 0.05:
                return f"risk of {c} rose {r0:.0%} -> {r:.0%}"
        a = p.actions[0]
        if a.kind is ExplorerActionType.JUMP and self.jumps == 0:
            return "no jumps left"
        return None

    def _ensure_plan(self, goal: Pos, purpose: str) -> None:
        reason = self._plan_invalid_reason(goal)
        if reason is None:
            return
        res = self._astar({goal})
        tag_reason = reason
        if res.found:
            self.plan = Plan(goal, purpose, list(res.actions), list(res.cells), res.cost,
                             [self.cell_risk(c) for c in res.cells])
        else:
            self.plan = None
        e = self.log.add(self.turn, "ASTAR", (f"path to {goal}: {len(res.actions)} steps, cost {res.cost:.1f}"
                                              if res.found else f"no path to {goal}"), res.explain(), cell=goal)
        self.turn_entries["ASTAR"] = e.id
        if reason not in ("no plan", "previous plan completed"):
            self.log.add(self.turn, "REPLAN", tag_reason, [f"old plan invalidated: {tag_reason}",
                                                          f"new plan: see entry #{e.id}"], refs=[e.id])

    # ---------------------------------------------------------------- adversarial
    def _goal_distance(self, goal: Pos) -> dict[Pos, float]:
        """Dijkstra-like BFS distances to the goal over cells the explorer may use."""
        ok = lambda c: c not in self.collapsed and (self._pit_known_safe(c) or c == goal or
                                                    self.pit_prob.get(c, 1.0) <= self.cfg.max_explore_risk)
        dist, q = {goal: 0.0}, deque([goal])
        while q:
            c = q.popleft()
            for n in neighbors(c, self.info.size):
                if n not in dist and ok(n):
                    dist[n] = dist[c] + 1
                    q.append(n)
        return dist

    def _minimax_action(self, goal: Pos) -> ExplorerAction:
        cfg = self.cfg
        game = DuelGame(size=self.info.size, pads=self.info.pads, blocked=frozenset(self._blocked()),
                        pit_risk={c: (0.0 if self._pit_known_safe(c) else p) for c, p in self.pit_prob.items()},
                        goal_dist=self._goal_distance(goal), max_step_risk=cfg.max_explore_risk)
        root = DuelState(self.pos, self.pos, True, self.arrows, self.jumps)
        hyps = self.tracker.top(cfg.minimax_hypotheses)
        res = determinized_decision(game, root, hyps, cfg.minimax_depth)
        e = self.log.add(self.turn, "MINIMAX", f"threat {self.threat:.0%} -> best {res.best_action} "
                         f"(E[V]={res.expected_value:.0f}, {res.stats.nodes} nodes, {res.stats.cutoffs} cutoffs)",
                         res.explain(), refs=[self.turn_entries["WUMPUS_BELIEF"]])
        a = res.best_action
        if self.plan and self.plan.actions and self.plan.actions[0] == a:
            self.plan.actions.pop(0), self.plan.cells.pop(0), self.plan.step_risk.pop(0)
        else:
            self.plan = None           # tactical deviation: plan will be rebuilt next turn
            self.plan_dropped = f"Minimax deviated from the plan ({a})"
        return self._commit(a, f"{a} chosen by Minimax (Wumpus threat {self.threat:.0%})",
                            [e.id, self.turn_entries["WUMPUS_BELIEF"]])
