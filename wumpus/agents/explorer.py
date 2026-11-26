"""The explorer AI. Receives only PublicInfo + Percepts (never the hidden WorldState).

Per turn:  observe(percept)  -> belief update (KB, Bayes pits, Wumpus filter, portal/tile knowledge)
           decide()          -> hunt / goal selection, plan validation / A* replanning, Minimax if threatened
Everything noteworthy is written to `self.log` with the evidence that produced it.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from ..config import AgentConfig
from ..core.actions import CLIMB, FORWARD, GRAB, JUMP, SHOOT, TURN_LEFT, TURN_RIGHT, ExplorerAction
from ..core.geometry import DIRECTIONS, Direction, Pos, all_cells, manhattan, neighbors, ray
from ..core.percepts import Percept, PublicInfo
from .knowledge_base import CellStatus, DungeonKB, WumpusEvidence
from .minimax import DuelGame, DuelState, determinized_decision
from .pathfinding import PathResult, PlanningMode, RiskModel, RoutePlan, plan, plan_routes
from .bayesian import CellRiskReport, HazardEstimate, estimate_hazard, estimate_pits
from .trace import BrainLog
from .wumpus_tracker import WumpusTracker


@dataclass
class Plan:
    goal: Pos
    purpose: str
    actions: list[ExplorerAction]
    cells: list[Pos]                   # cell after each action
    facings: list[Direction]           # facing after each action
    cost: float
    step_risk: list[float] = field(default_factory=list)   # risk of each cell when planned
    expect: tuple[Pos, Direction] | None = None             # where we should be before the next action

    def pop(self) -> tuple[ExplorerAction, Pos]:
        a, c, f = self.actions.pop(0), self.cells.pop(0), self.facings.pop(0)
        self.step_risk.pop(0)
        self.expect = (c, f)
        return a, c


class ExplorerAgent:
    def __init__(self, info: PublicInfo, config: AgentConfig | None = None):
        self.info = info
        self.cfg = config or AgentConfig()
        self.log = BrainLog()
        self.turn = 0
        self.pos = info.start
        self.facing = info.start_facing
        self.has_gold = False
        self.arrows = info.arrows
        self.jump_available = info.jumps > 0
        self.visited: set[Pos] = {info.start}
        self.collapsed: dict[Pos, int] = {}        # cell -> turn from which it is a pit
        self.unstable_here: Pos | None = None      # cell we stand on that will collapse when left
        self.portals: dict[Pos, Pos] = {}          # learned portal pairs (symmetric)
        self.breeze_obs: dict[Pos, tuple[bool, int]] = {}
        self.signal_obs: dict[Pos, tuple[bool, int]] = {}
        self.stench = False
        self.glitter = False
        self.dkb = DungeonKB(info)
        self.statuses: dict[Pos, CellStatus] = {}
        self.portal_status: dict[Pos, CellStatus] = {}
        self.wumpus_status: dict[Pos, CellStatus] = {}   # time-sensitive: valid for self.turn only
        self.safe_status: dict[Pos, CellStatus] = {}     # SAFE_x_y (no pit AND no Wumpus) right now
        self.percepts: dict[int, Percept] = {}           # observed percepts by turn
        self.shuffle_tiles_known: set[Pos] = set()       # cells where a shuffle triggered
        self.inferred: dict[str, tuple[int, str]] = {}   # proposition -> (turn proven, explanation)
        self.risk_reports: dict[Pos, CellRiskReport] = {}
        self.pit_prob: dict[Pos, float] = {}
        self.hazards: dict[Pos, HazardEstimate] = {}     # per-cell pit / Wumpus / death estimates
        self.last_route: RoutePlan | None = None
        self.tracker = WumpusTracker(info, self.cfg.stay_prob, self.cfg.pursuit_prob)
        self.plan: Plan | None = None
        self.retreating = False
        self.last_action: ExplorerAction | None = None
        self.last_pos: Pos = info.start
        self.last_facing: Direction = info.start_facing
        self.threat = 0.0
        self.plan_dropped: str | None = None       # why the plan was discarded outside _ensure_plan
        self.last_progress_turn = 1                # last turn a new cell was visited / gold taken
        self.minimax_streak = 0                    # consecutive Minimax decisions in the same cell
        self.prediction: tuple | None = None       # (turn, predicted W cell, reply, hypothesis) from Minimax
        self.last_minimax = None                   # MinimaxResult of the latest search
        self.reports: list[DecisionReport] = []    # one structured report per decision
        self._turn_minimax = None
        self._turn_route = None
        self.minimax_cell: Pos | None = None
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
        e = L.add(p.turn, "PERCEPT", f"at {p.position} facing {p.facing.label}: {p.flags()}",
                  [f"position {p.position}, facing {p.facing.label}, step {p.step}, score {p.score}",
                   f"breeze={p.breeze} stench={p.stench} glitter={p.glitter} bump={p.bump} scream={p.scream}",
                   f"portal_signal={p.portal_signal} unstable={p.unstable} shuffled={p.shuffled} "
                   f"teleported={p.teleported}",
                   f"has_gold={p.has_gold} arrows={p.arrows} jump_available={p.jump_available}"], cell=p.position)
        self.turn_entries["PERCEPT"] = e.id
        self.percepts[p.turn] = p
        self.tracker.begin(p.turn)
        la, moved = self.last_action, p.position != self.pos

        # --- evidence from our own previous action (before the Wumpus moved)
        if la is SHOOT and self.tracker.alive and self.arrows > p.arrows:
            if p.scream:
                self.tracker.kill()
                L.add(p.turn, "EVENT", "SCREAM: the arrow killed the Wumpus",
                      [f"arrow shot {self.last_facing.label} from {self.last_pos}", "scream percept received"])
            else:
                self.tracker.observe_arrow_missed(ray(self.last_pos, self.last_facing, self.info.size))
        if p.bump:
            L.add(p.turn, "EVENT", f"BUMP: wall ahead of {p.position} ({p.facing.label})",
                  ["bump percept: the last FORWARD/JUMP hit the outer wall; position unchanged"])
            if self.plan:
                self.plan_dropped = "unexpected bump"
                self.plan = None
        if p.teleported and la in (FORWARD, JUMP):
            src = self.last_pos.step(self.last_facing, 1 if la is FORWARD else self.info.jump_distance)
            self.portals[src], self.portals[p.position] = p.position, src
            self.tracker.portals = dict(self.portals)
            L.add(p.turn, "EVENT", f"TELEPORTED: learned portal pair {src} <-> {p.position}",
                  [f"stepped onto {src} with {la}, percept says we are now at {p.position}",
                   "rule: a portal moves the entity to its paired portal -> pair learned",
                   "both cells are portals (O facts in the KB) and are never pits"], cell=p.position)
        if self.tracker.alive and moved:
            self.tracker.observe_survived_at(p.position)

        # --- collapsing floor: the cell we left is a pit from now on
        if self.unstable_here is not None and p.position != self.unstable_here:
            c = self.unstable_here
            self.collapsed[c] = p.turn
            L.add(p.turn, "EVENT", f"tile {c} collapsed behind us -> now a pit",
                  [f"unstable-floor warning was felt on {c}", "rule: unstable tiles collapse when left",
                   "the cell is impassable from now on; breezes next to it are explained by it"], cell=c)
            self.unstable_here = None
            if self.plan:
                self.plan_dropped = self.plan_dropped or f"cell {c} on the path collapsed"
        if p.unstable and self.unstable_here != p.position:
            self.unstable_here = p.position
            L.add(p.turn, "EVENT", f"UNSTABLE floor at {p.position}: it will collapse when we leave",
                  ["unstable-floor warning percept", "planning will never route back through this cell"],
                  cell=p.position)

        if p.shuffled:
            self.shuffle_tiles_known.add(p.position)
            stale = len(self.breeze_obs)
            self.breeze_obs = {}
            L.add(p.turn, "EVENT", f"SHUFFLE: {stale} breeze observations invalidated",
                  ["shuffle indicator: a pit was swapped with an empty unvisited cell",
                   "visited cells stay pit-free; every older breeze observation is dropped"])

        self.pos, self.facing, self.has_gold, self.arrows = p.position, p.facing, p.has_gold, p.arrows
        self.jump_available, self.stench, self.glitter = p.jump_available, p.stench, p.glitter
        if p.position not in self.visited or (p.has_gold and la is GRAB):
            self.last_progress_turn = p.turn
        self.visited.add(p.position)
        self.breeze_obs[p.position] = (p.breeze, p.turn)
        self.signal_obs[p.position] = (p.portal_signal, p.turn)

        self._update_pits()
        self._update_wumpus(p)
        self._check_prediction()
        self._update_hazards()

    def valid_breeze(self) -> dict[Pos, tuple[bool, int]]:
        """Breeze observations about ORIGINAL pits: a breeze sensed next to an already-collapsed
        tile is explained by that tile and carries no information, so it is left out."""
        out = {}
        for c, (b, t) in self.breeze_obs.items():
            if b and any(n in self.collapsed and self.collapsed[n] <= t for n in neighbors(c, self.info.size)):
                continue
            out[c] = (b, t)
        return out

    def _wumpus_evidence(self) -> WumpusEvidence:
        return WumpusEvidence(self.turn, self.pos, self.stench, self.tracker.alive)

    def _rebuild_kb(self, breeze):
        frontier = self._frontier_cells(c for c in all_cells(self.info.size)
                                        if c not in self.visited and c not in self.collapsed)
        return self.dkb.rebuild(self.visited, self.collapsed, breeze, set(self.portals), self.signal_obs,
                                self._wumpus_evidence(), safe_queries=frontier)

    def _revise_until_consistent(self, kb):
        """Belief revision: on a contradiction, log the minimal conflicting set and drop the OLDEST
        observation that takes part in it, until the KB is consistent again."""
        report = kb.consistency()
        while not report.consistent:
            obs = []
            for _, src in report.core:
                m = re.match(r"percept turn (\d+): (?:no )?(breeze|portal signal) at \((\d+),(\d+)\)", src)
                if m:
                    obs.append((int(m.group(1)), m.group(2), Pos(int(m.group(3)), int(m.group(4)))))
            if not obs:
                self.log.add(self.turn, "WARNING", "contradiction without a revisable observation",
                             report.explain())
                return kb
            t, kind, c = min(obs)
            (self.breeze_obs if kind == "breeze" else self.signal_obs).pop(c, None)
            self.log.add(self.turn, "WARNING", f"CONTRADICTION -> dropped {kind} observation at {c} (turn {t})",
                         report.explain() + [f"Revision: the oldest observation in the conflict "
                                             f"({kind} at {c}, turn {t}) is discarded."])
            kb = self._rebuild_kb(self.valid_breeze())
            report = kb.consistency()
        return kb

    def _update_pits(self) -> None:
        L, t = self.log, self.turn
        old_texts = {self.dkb.kb.clause_text(c) for c in self.dkb.kb.clauses}
        expired = [x for x in self.dkb.kb.time_sensitive() if x[2] < t]
        kb = self._revise_until_consistent(self._rebuild_kb(self.valid_breeze()))
        breeze = self.valid_breeze()
        new = [(kb.clause_text(c), s) for c, s in zip(kb.clauses, kb.sources) if kb.clause_text(c) not in old_texts]
        details = [f"  {c}    <- {s}" for c, s in new] or ["no new clauses"]
        if expired:
            details += [f"Belief revision: {len(expired)} time-sensitive Wumpus facts from earlier turns expired "
                        "(the Wumpus moves), e.g.:"] + [f"  {c}    <- {s} [valid turn {v}]" for c, s, v in expired[:4]]
        e = L.add(t, "KB_TELL", f"{len(kb.clauses)} clauses ({len(new)} new, {len(expired)} expired)", details)
        self.turn_entries["KB_TELL"] = e.id

        cells = [c for c in all_cells(self.info.size) if c not in self.visited and c not in self.collapsed]
        prev, prev_portal = self.statuses, self.portal_status
        self.statuses = self.dkb.classify(cells)
        frontier = self._frontier_cells(cells)
        self.portal_status = self.dkb.classify_portals(frontier)
        self.wumpus_status = self.dkb.classify_wumpus(neighbors(self.pos, self.info.size))
        self.safe_status = self.dkb.classify_safe(frontier)
        for c, st in sorted(self.statuses.items()):
            if st.status != "unknown" and (c not in prev or prev[c].status != st.status):
                prop = f"{'¬' if st.status == 'safe' else ''}PIT_{c.x}_{c.y}"
                self.inferred[prop] = (t, st.explanation)
                L.add(t, "INFERENCE", f"Cell {c} proven {'SAFE (no pit)' if st.status == 'safe' else 'PIT'}",
                      [st.explanation] + st.proof.explain(), refs=[self.turn_entries["KB_TELL"]], cell=c)
        for c, st in sorted(self.portal_status.items()):
            if st.status == "portal" and c not in self.portals and (
                    c not in prev_portal or prev_portal[c].status != "portal"):
                self.inferred[f"PORTAL_{c.x}_{c.y}"] = (t, st.explanation)
                L.add(t, "INFERENCE", f"Cell {c} proven PORTAL (destination unknown -> avoid)",
                      [st.explanation] + st.proof.explain(), refs=[self.turn_entries["KB_TELL"]], cell=c)
        for c, st in sorted(self.wumpus_status.items()):
            if st.status == "wumpus":
                L.add(t, "INFERENCE", f"Cell {c}: Wumpus certainly there NOW (valid this turn only)",
                      [st.explanation] + st.proof.explain(), refs=[self.turn_entries["KB_TELL"]], cell=c)
        n_pit = sum(s.status == "pit" for s in self.statuses.values())
        n_unknown = sum(s.status == "unknown" for s in self.statuses.values())
        prior = max(0.0, min(1.0, (self.info.pit_count - n_pit) / n_unknown)) if n_unknown else 0.0
        old_prob = self.pit_prob
        self.risk_reports = estimate_pits(self.info.size, self.statuses, breeze, prior,
                                          self.cfg.max_component_cells)
        self.pit_prob = {c: r.p_pit for c, r in self.risk_reports.items()}
        for c in self.visited:
            self.pit_prob[c] = 0.0
        for c in self.collapsed:
            self.pit_prob[c] = 1.0
        self._old_pit_prob = old_prob

    def _update_wumpus(self, p: Percept) -> None:
        tr = self.tracker
        if not tr.alive:
            return
        if p.turn > 1:
            tr.predict(p.position, self._blocked(), self.pit_prob)
            tr.observe_survived_at(p.position)
        tr.observe_stench(p.position, p.stench)
        nb = set(neighbors(p.position, self.info.size))
        fallback = [c for c in all_cells(self.info.size)
                    if c != p.position and (c in nb) == p.stench and c not in self._blocked()]
        tr.normalize(fallback_cells=fallback)
        self.threat = tr.prob_within(self.pos, 2)
        top = tr.top(1)
        e = self.log.add(p.turn, "WUMPUS_BELIEF",
                         f"most likely {top[0][0]} ({top[0][1]:.0%}), P(within 2)={self.threat:.0%}",
                         tr.report.explain(), cell=top[0][0])
        self.turn_entries["WUMPUS_BELIEF"] = e.id

    def _check_prediction(self) -> None:
        """Compare Minimax's predicted Wumpus reply with what the evidence now says."""
        pred, self.prediction = self.prediction, None
        if pred is None or not self.tracker.alive or self.turn != pred[0] + 1:
            return
        _, cell, reply, hyp = pred
        p = self.tracker.prob(cell)
        if p < 0.05:
            self.log.add(self.turn, "REPLAN", f"Wumpus did not act as predicted ({reply} to {cell}) -> "
                         "prediction discarded, replanning",
                         [f"Minimax (turn {pred[0]}) assumed the Wumpus at {hyp} and predicted {reply} -> {cell}",
                          f"after this turn's evidence P(Wumpus at {cell}) = {p:.1%}: the prediction is refuted",
                          "the plan built on it is dropped; A* and Minimax run again on the new belief"],
                         refs=[self.turn_entries["WUMPUS_BELIEF"]], cell=cell)
            self.plan = None

    def _update_hazards(self) -> None:
        """Combined pit / Wumpus / death estimate for every non-visited cell (bayesian.estimate_hazard)."""
        old = self.hazards
        blocked = self._blocked()
        tr = self.tracker
        w_ev = []
        if tr.alive:
            w_ev = [f"Wumpus filter: {s.label}" for s in tr.report.steps][-4:]
        else:
            w_ev = ["the Wumpus is dead"]
        new: dict[Pos, HazardEstimate] = {}
        for c in all_cells(self.info.size):
            if c in self.visited and c not in self.collapsed and c != self.pos:
                continue
            w_now = tr.prob(c) if tr.alive else 0.0
            w_in = 0.0
            if tr.alive and manhattan(c, self.pos) <= 2:
                w_in = max(0.0, tr.entry_risk(c, blocked, self.pit_prob) - w_now)
            conf = ("collapsed tile" if c in self.collapsed else
                    "proven pit" if c in self.statuses and self.statuses[c].status == "pit" else None)
            ws = self.wumpus_status.get(c)
            new[c] = estimate_hazard(c, self.risk_reports.get(c), self.pit_prob.get(c, 0.0), w_now, w_in,
                                     known_pit_free=self._pit_known_safe(c), confirmed_hazard=conf,
                                     wumpus_status=ws.status if ws else None, wumpus_evidence=w_ev)
        self.hazards = new
        frontier = set(self._frontier_cells(new))
        for c in sorted(frontier):
            h, r = new[c], self.risk_reports.get(c)
            prev = old.get(c)
            if prev is not None and abs(prev.p_death - h.p_death) < 0.005:
                continue
            if h.p_death == 0.0 and (prev is None or prev.p_death == 0.0):
                continue
            details = h.explain() + ([""] + r.explain() if r is not None and r.method in ("enumeration", "approx") else [])
            self.log.add(self.turn, "RISK_CALC", f"Cell {c}: {h.p_death:.0%} death risk "
                         f"(pit {h.p_pit:.0%}, Wumpus {h.p_wumpus + h.p_wumpus_in:.0%})", details, cell=c)

    # ================================================================ read-only knowledge view
    def knowledge(self) -> "KnowledgeSnapshot":
        """Immutable snapshot of everything the explorer knows (for the Brain View)."""
        size = self.info.size
        st = self.statuses
        pit_free = frozenset(self.visited | set(self.portals) | {c for c, x in st.items() if x.status == "safe"}) \
            - frozenset(self.collapsed)
        w_ruled_out = {c for c, x in self.wumpus_status.items() if x.status == "none"} | {self.pos}
        known_safe = frozenset(c for c in pit_free if not self.tracker.alive or c in w_ruled_out
                               or (c in self.safe_status and self.safe_status[c].status == "safe"))
        confirmed = {c: "pit (proven)" for c, x in st.items() if x.status == "pit"}
        confirmed.update({c: "collapsed tile" for c in self.collapsed})
        confirmed.update({c: "Wumpus (now)" for c, x in self.wumpus_status.items() if x.status == "wumpus"})
        suspected_pits = {c: p for c, p in self.pit_prob.items() if 0 < p < 1 and c not in self.visited
                          and self.risk_reports.get(c) is not None and self.risk_reports[c].method != "prior"}
        time_facts = tuple(self.dkb.kb.time_sensitive())
        return KnowledgeSnapshot(
            turn=self.turn, position=self.pos, facing=self.facing,
            visited=frozenset(self.visited), pit_free=pit_free, known_safe=known_safe,
            suspected_pits=MappingProxyType(dict(suspected_pits)),
            suspected_wumpus=MappingProxyType({c: p for c, p in self.tracker.belief.items() if p >= 0.01}),
            confirmed_hazards=MappingProxyType(confirmed),
            percepts=MappingProxyType(dict(self.percepts)),
            special_tiles=MappingProxyType({**{c: "unstable (collapses when left)" for c in
                                               ([self.unstable_here] if self.unstable_here else [])},
                                            **{c: "collapsed" for c in self.collapsed},
                                            **{c: "used shuffle tile" for c in self.shuffle_tiles_known},
                                            **{c: "portal" for c in self.portals}}),
            portal_connections=MappingProxyType(dict(self.portals)),
            time_sensitive_facts=time_facts,
            inferred=MappingProxyType(dict(self.inferred)),
            ambiguous=frozenset(c for c, x in st.items() if x.status == "unknown" and c in suspected_pits),
            pit_probability=MappingProxyType(dict(self.pit_prob)),
            wumpus_alive=self.tracker.alive, size=size)

    # ================================================================ risk model
    def _blocked(self) -> set[Pos]:
        return set(self.collapsed) | {c for c, s in self.statuses.items() if s.status == "pit"}

    def _unknown_portal(self, c: Pos) -> bool:
        st = self.portal_status.get(c)
        return c not in self.portals and st is not None and st.status == "portal"

    def wumpus_risk(self, c: Pos) -> float:
        if not self.tracker.alive:
            return 0.0
        if not self.cfg.use_bayes:                  # logic only: certain Wumpus = 1, else 0
            ws = self.wumpus_status.get(c)
            return 1.0 if ws is not None and ws.status == "wumpus" else 0.0
        if manhattan(c, self.pos) <= 2:
            return self.tracker.entry_risk(c, self._blocked(), self.pit_prob)
        return self.tracker.prob(c)

    def cell_risk(self, c: Pos) -> float:
        if c in self.collapsed or c == self.unstable_here:
            return 1.0
        if self.cfg.use_bayes:
            p = self.pit_prob.get(c, 0.0 if c in self.visited else 1.0)
        else:                                       # logic only: anything unproven counts as fatal
            p = 0.0 if self._pit_known_safe(c) else 1.0
        return 1.0 - (1.0 - p) * (1.0 - self.wumpus_risk(c))

    def _pit_known_safe(self, c: Pos) -> bool:
        return c not in self.collapsed and (
            c in self.visited or c in self.portals or (c in self.statuses and self.statuses[c].status == "safe"))

    # ================================================================ decision
    def decide(self) -> ExplorerAction:
        """Decision pipeline (see DecisionReport): priorities -> plan/Minimax -> hazard guard -> report."""
        self._turn_minimax = self._turn_route = None
        n_before = len(self.log.entries)
        action = self._decide()
        action = self._guard(action)
        self.last_action = action
        self.reports.append(self._report(action, n_before))
        return action

    def _decide(self) -> ExplorerAction:
        self.last_pos, self.last_facing = self.pos, self.facing
        if self.glitter and not self.has_gold:
            return self._commit(GRAB, "glitter perceived -> grab the gold", [self.turn_entries["PERCEPT"]])
        if self.has_gold and self.pos == self.info.start:
            return self._commit(CLIMB, "at the exit with the gold -> climb out", [])

        if self._should_hunt():
            shot = self._best_shot()
            if shot is not None:
                cell, d, p_hit = shot
                if cell == self.pos:
                    refs = [self.turn_entries["TARGET"], self.turn_entries["WUMPUS_BELIEF"]]
                    if self.facing == d:
                        return self._commit(SHOOT, f"hunt: SHOOT {d.label}, P(hit)={p_hit:.0%}", refs)
                    turn = TURN_LEFT if self.facing.left() == d else TURN_RIGHT
                    return self._commit(turn, f"hunt: {turn} to face {d.label} before shooting", refs)
                return self._execute(cell, f"hunt: reach {cell} to shoot {d.label}")

        goal, purpose, _ = self._select_goal()
        if goal is None:
            if self.pos == self.info.start:
                return self._commit(CLIMB, "no acceptable frontier cell and at the exit -> retreat (climb)",
                                    [self.turn_entries.get("TARGET", 0)])
            goal, purpose = self.info.start, "retreat to the exit"
        return self._execute(goal, purpose)

    def _execute(self, goal: Pos, purpose: str) -> ExplorerAction:
        self._ensure_plan(goal, purpose)
        if self.minimax_cell != self.pos:
            self.minimax_streak, self.minimax_cell = 0, self.pos
        if self.cfg.use_minimax and self.tracker.alive and self.threat >= self.cfg.threat_threshold:
            paralysed = self.minimax_streak >= 4 or self.stuck() >= 2 * self.cfg.stuck_turns
            if not paralysed:
                self.minimax_streak += 1
                return self._minimax_action(goal)
            if self.minimax_streak == 4:           # log once; suppression lasts until we leave this cell
                self.minimax_streak = 5
                self.log.add(self.turn, "WARNING", f"anti-paralysis: Minimax kept us in {self.pos} for 4 turns "
                             "-> follow the plan until we leave this cell",
                             [f"threat {self.threat:.0%} would normally trigger Minimax",
                              "Minimax assumes a Wumpus that always attacks; we survived next to it for",
                              "4 turns, so it is not attacking -> the worst-case veto is suspended."])
        if self.plan and self.plan.actions:
            a, nxt = self.plan.pop()
            refs = [self.turn_entries[k] for k in ("TARGET", "A*_PLANNER") if k in self.turn_entries]
            if a in (FORWARD, JUMP) and nxt in self.visited and nxt != goal:
                return self._commit(a, f"{a} -> {nxt}: {self._backtrack_reason(goal)}", refs)
            return self._commit(a, f"{a} -> {nxt}  ({purpose}; {len(self.plan.actions)} actions left)", refs)
        if self.pos == self.info.start:
            return self._commit(CLIMB, "no path anywhere and at the exit -> climb out",
                                [self.turn_entries.get("TARGET", 0)])
        fallback = self._least_risky_frontier(cut_off_from_exit=(goal == self.info.start))
        if fallback is not None and fallback != goal:
            return self._execute(fallback, f"cut off from {goal}: explore {fallback} to find a new route")
        return self._commit(TURN_LEFT, "no path to any cell -> turn in place (waiting)",
                            [self.turn_entries.get("TARGET", 0)])

    def _least_risky_frontier(self, cut_off_from_exit: bool) -> Pos | None:
        """Last resort when the goal is unreachable: the cheapest reachable frontier cell.
        Above the normal risk cap only when cut off from the exit (otherwise a certain loss)."""
        cap = 0.5 if cut_off_from_exit else self.risk_cap()
        cells = [c for c in self._frontier() if self.cell_risk(c) <= cap]
        if cells and cut_off_from_exit:     # equally risky options: prefer the ones towards the exit
            low = min(self.cell_risk(c) for c in cells)
            cells = [c for c in cells if self.cell_risk(c) <= low + 0.02]
            near = min(manhattan(c, self.info.start) for c in cells)
            cells = [c for c in cells if manhattan(c, self.info.start) == near]
        res = self._astar(set(cells), allow_unknown=True) if cells else None
        if res is None or not res.found:
            return None
        self.log.add(self.turn, "TARGET", f"goal unreachable -> explore {res.goal} to open a new route "
                     f"(risk {self.cell_risk(res.goal):.0%})",
                     ["The planned goal cannot be reached even through unexplored cells.",
                      f"Fallback: A* to the cheapest frontier cell with risk ≤ {cap:.0%}"
                      + (" (cut off from the exit: staying means a certain loss)" if cut_off_from_exit else "")]
                     + res.explain(), cell=res.goal)
        return res.goal

    def _backtrack_reason(self, goal: Pos) -> str:
        """Plain-language reason for stepping back onto an already visited cell."""
        warn = []
        p = self.percepts.get(self.turn)
        if p is not None:
            if p.breeze:
                warn.append("breeze here, so a neighbour may be a pit")
            if p.stench:
                warn.append("stench here, so the Wumpus is next to us")
            if p.portal_signal:
                warn.append("a portal is next to us")
            if p.unstable:
                warn.append("the floor here will collapse")
        what = "the exit" if goal == self.info.start else f"{goal}"
        msg = f"backtracking over safe, already visited cells to reach {what}"
        return msg + (f" ({'; '.join(warn)})" if warn else " (no safe unexplored cell is closer)")

    def predicted_state(self) -> tuple[Pos, Direction]:
        """Where the explorer expects to be after its last decided action (its own prediction:
        a bump or a hidden portal can still surprise it; the next percept tells)."""
        a, pos, f = self.last_action, self.pos, self.facing
        if a is TURN_LEFT:
            return pos, f.left()
        if a is TURN_RIGHT:
            return pos, f.right()
        if a in (FORWARD, JUMP):
            c = pos.step(f, 1 if a is FORWARD else self.info.jump_distance)
            if 0 <= c.x < self.info.size and 0 <= c.y < self.info.size:
                return self.portals.get(c, c), f
        return pos, f

    def _commit(self, a: ExplorerAction, why: str, refs: list[int]) -> ExplorerAction:
        self.last_action = a
        self.log.add(self.turn, "DECISION", why, [f"action: {a}", f"reason: {why}",
                                                  "depends on entries: " + ", ".join(f"#{r}" for r in refs)],
                     refs=refs)
        return a

    # ---------------------------------------------------------------- guard and report
    def _landing(self, a: ExplorerAction) -> Pos | None:
        if a not in (FORWARD, JUMP):
            return None
        c = self.pos.step(self.facing, 1 if a is FORWARD else self.info.jump_distance)
        if not (0 <= c.x < self.info.size and 0 <= c.y < self.info.size):
            return None
        return self.portals.get(c, c)

    def _confirmed_fatal(self, c: Pos) -> str | None:
        if c in self.collapsed:
            return "collapsed tile"
        st = self.statuses.get(c)
        if st is not None and st.status == "pit":
            return "proven pit"
        ws = self.wumpus_status.get(c)
        if ws is not None and ws.status == "wumpus":
            return "the Wumpus is certainly there"
        return None

    def _guard(self, a: ExplorerAction) -> ExplorerAction:
        """Last line of defence: never knowingly enter a confirmed fatal hazard."""
        c = self._landing(a)
        why = self._confirmed_fatal(c) if c is not None else None
        if why is None:
            return a
        self.log.add(self.turn, "WARNING", f"guard: refused {a} into {c} ({why}) -> turn instead",
                     [f"selected action {a} would land on {c}", f"{c} is a confirmed hazard: {why}",
                      "rule: never knowingly enter a confirmed fatal hazard"], cell=c)
        self.plan, self.plan_dropped = None, f"guard refused {a} into {c}"
        return TURN_LEFT

    def candidate_actions(self) -> list[tuple[str, str]]:
        """Every legal explorer action with a one-line assessment (for the decision report)."""
        out = []
        for a in (FORWARD, JUMP):
            raw = self.pos.step(self.facing, 1 if a is FORWARD else self.info.jump_distance)
            if a is JUMP and not self.jump_available:
                out.append((str(a), "unavailable: jump already used"))
                continue
            if not (0 <= raw.x < self.info.size and 0 <= raw.y < self.info.size):
                out.append((str(a), "useless: wall ahead (bump)"))
                continue
            c = self.portals.get(raw, raw)
            fatal = self._confirmed_fatal(c)
            if fatal:
                out.append((str(a), f"forbidden: {c} is a {fatal}"))
                continue
            h = self.hazards.get(c)
            risk = h.p_death if h else self.cell_risk(c)
            tag = "proven pit-free" if self._pit_known_safe(c) else "UNPROVEN"
            out.append((str(a), f"-> {c} ({tag}), P(death) {risk:.0%}"))
        out.append((str(TURN_LEFT), f"face {self.facing.left().label}"))
        out.append((str(TURN_RIGHT), f"face {self.facing.right().label}"))
        out.append((str(GRAB), "gold here (glitter)" if self.glitter and not self.has_gold else "nothing to grab"))
        if self.arrows > 0 and self.tracker.alive:
            p_hit = sum(self.tracker.prob(x) for x in ray(self.pos, self.facing, self.info.size))
            out.append((str(SHOOT), f"P(hit) = {p_hit:.0%} along {self.facing.label}"))
        else:
            out.append((str(SHOOT), "no arrow" if self.arrows == 0 else "the Wumpus is dead"))
        out.append((str(CLIMB), ("exit: " + ("escape with the gold" if self.has_gold else "retreat (draw)"))
                    if self.pos == self.info.start else "not at the exit"))
        return out

    def _report(self, action: ExplorerAction, first_entry: int) -> "DecisionReport":
        turn_entries = self.log.for_turn(self.turn)
        new_facts = tuple(e.summary for e in turn_entries if e.tag == "INFERENCE")
        decision = [e for e in self.log.entries[first_entry:] if e.tag == "DECISION"]
        why = decision[-1].summary if decision else str(action)
        frontier = self._frontier_cells(self.hazards)
        hz = {str(c): self.hazards[c].to_dict() for c in frontier if c in self.hazards}
        p = self.percepts.get(self.turn)
        return DecisionReport(
            turn=self.turn, observations={"position": tuple(self.pos), "facing": self.facing.label,
                                          "percepts": p.flags() if p else "", "has_gold": self.has_gold,
                                          "arrows": self.arrows, "jump": self.jump_available},
            new_facts=new_facts, hazards=MappingProxyType(hz), candidates=tuple(self.candidate_actions()),
            astar=self._turn_route.to_dict() if self._turn_route else (
                {"plan": [str(a) for a in self.plan.actions], "cells": [tuple(c) for c in self.plan.cells],
                 "note": "continuing the existing plan"} if self.plan else None),
            minimax=self._turn_minimax.to_dict() if self._turn_minimax else None,
            selected=str(action), explanation=why)

    # ---------------------------------------------------------------- progress / hunting
    def stuck(self) -> int:
        return self.turn - self.last_progress_turn

    def risk_cap(self) -> float:
        return self.cfg.relaxed_risk if self.stuck() >= self.cfg.stuck_turns else self.cfg.max_explore_risk

    def _should_hunt(self) -> bool:
        return (self.cfg.use_bayes and self.tracker.alive and self.arrows > 0 and not self.has_gold
                and self.stuck() >= self.cfg.stuck_turns)

    def _best_shot(self):
        """Pick (cell, direction) maximising P(Wumpus on the arrow's ray) - 1% per step to get there."""
        dist, q = {self.pos: 0}, deque([self.pos])
        while q:
            c = q.popleft()
            for n in neighbors(c, self.info.size):
                if n not in dist and self._pit_known_safe(n) and n not in self.portals:
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
                   f"{'cell':<7}{'dir':<6}{'steps':>6}{'P(hit)':>9}{'score':>8}"]
        details += [f"{str(r[2]):<7}{r[3].label:<6}{r[4]:>6}{r[1]:>9.1%}{r[0]:>8.3f}" for r in rows[:8]]
        e = self.log.add(self.turn, "TARGET", f"hunt: shoot {d.label} from {c} (P(hit)={p:.0%})", details,
                         refs=[self.turn_entries["WUMPUS_BELIEF"]], cell=c)
        self.turn_entries["TARGET"] = e.id
        return c, d, p

    # ---------------------------------------------------------------- goal selection
    def _frontier_cells(self, candidates) -> list[Pos]:
        cand = set(candidates)
        out = {n for v in self.visited for n in neighbors(v, self.info.size)
               if n in cand and n not in self.visited and n not in self.collapsed}
        return sorted(out)

    def _frontier(self) -> list[Pos]:
        stuck = self.stuck() >= self.cfg.stuck_turns
        return [c for c in self._frontier_cells(all_cells(self.info.size))
                if self.pit_prob.get(c, 1.0) < 1.0 and (stuck or not self._unknown_portal(c))]

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
            pit = self.pit_prob.get(c, 0.0) if self.cfg.use_bayes else (0.0 if self._pit_known_safe(c) else 1.0)
            wr = self.wumpus_risk(c)
            total = 1 - (1 - pit) * (1 - wr)
            rows.append((c, pit, wr, total))
            if total <= self.risk_cap():
                ok.append(c)
        details = ["Frontier = unvisited cells adjacent to visited cells (proven portals with an",
                   "unknown destination are skipped unless stuck).",
                   f"Acceptable if combined risk ≤ {self.risk_cap():.0%};  risk = 1-(1-P(pit))(1-P(Wumpus))",
                   f"{'cell':<7}{'P(pit)':>8}{'P(W)':>8}{'risk':>8}"]
        details += [f"{str(c):<7}{p:>8.1%}{w:>8.1%}{r:>8.1%}{'  ok' if c in ok else ''}" for c, p, w, r in rows]
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
        escape = self._exit_connected()
        walk_escape = self._exit_connected(use_jump=False)
        with_exit = [c for c in ok if c in escape]
        if with_exit and len(with_exit) < len(ok):
            details.append(f"escape route: {len(ok) - len(with_exit)} candidates would cut us off from the "
                           f"exit -> only {len(with_exit)} with a known way back are kept")
            ok = with_exit
        proven = [c for c, _, _, r in rows if c in ok and r <= cfg.safe_risk and self._pit_known_safe(c)]
        safe = [c for c, _, _, r in rows if c in ok and r <= cfg.safe_risk]
        goals = proven or safe or ok
        details.append(f"tier 1: {len(proven)} proven pit-free (and low Wumpus risk); tier 2: {len(safe)} with "
                       f"risk ≤ {cfg.safe_risk:.0%}; tier 3: {len(ok)} within the cap -> using tier "
                       + ("1 (proven safe)" if proven else "2 (low risk, unproven)" if safe
                          else "3 (no safe option: calculated risk)"))
        res = self._astar(set(goals))
        while res.found and JUMP in res.actions and res.goal not in walk_escape:
            # the only way back from this target is the jump, and getting there would spend it
            details.append(f"escape route: reaching {res.goal} uses the jump, and the jump is its only way "
                           "back -> rejected")
            goals = [c for c in goals if c != res.goal]
            res = self._astar(set(goals)) if goals else PathResult(False, self.pos)
        if not res.found:
            e = L.add(self.turn, "TARGET", "acceptable cells unreachable -> retreat", details + res.explain())
            self.turn_entries["TARGET"] = e.id
            self.retreating = True
            return None, "", rows
        details.append(f"A* over all acceptable cells picks the cheapest: {res.goal} (cost {res.cost:.2f})")
        e = L.add(self.turn, "TARGET", f"explore {res.goal} (risk {self.cell_risk(res.goal):.0%}, "
                  f"{len(goals)} candidates)", details, cell=res.goal)
        self.turn_entries["TARGET"] = e.id
        return res.goal, f"explore {res.goal}", ok

    def _exit_connected(self, use_jump: bool = True) -> set[Pos]:
        """Cells from which the entrance is reachable over proven pit-free cells, treating the tile
        we stand on as collapsed if it is unstable (it will be a pit once we leave)."""
        bad = set(self.collapsed) | ({self.unstable_here} if self.unstable_here else set())
        ok = lambda c: c not in bad and self._pit_known_safe(c)
        seen, q = {self.info.start}, deque([self.info.start])
        while q:
            c = q.popleft()
            for n in neighbors(c, self.info.size):
                if n not in seen and ok(n):
                    seen.add(n)
                    q.append(n)
        # a frontier cell keeps an escape route if it touches that network, or if the unused jump
        # (3 cells ahead in some direction, crossing anything) lands in it
        out = seen | {c for c in all_cells(self.info.size) if c not in bad
                      and any(n in seen for n in neighbors(c, self.info.size))}
        if use_jump and self.jump_available:
            out |= {c for c in all_cells(self.info.size) if c not in bad
                    and any(c.step(d, self.info.jump_distance) in seen for d in DIRECTIONS)}
        return out

    # ---------------------------------------------------------------- planning
    def risk_model(self) -> RiskModel:
        """The planner's view of the world, built only from the explorer's knowledge."""
        blocked = {c: "collapsed tile" for c in self.collapsed}
        if self.unstable_here is not None:
            blocked[self.unstable_here] = "collapses when we leave it"
        blocked.update({c: "proven pit" for c, s in self.statuses.items() if s.status == "pit"})
        avoid = {c: "portal with unknown destination" for c in self.portal_status if self._unknown_portal(c)}
        proven = frozenset(c for c in all_cells(self.info.size) if self._pit_known_safe(c))
        risk = {c: self.cell_risk(c) for c in all_cells(self.info.size) if c not in blocked}
        return RiskModel(self.info.size, proven, MappingProxyType(risk), MappingProxyType(blocked),
                         MappingProxyType(avoid), MappingProxyType(dict(self.portals)))

    def _plan(self, goals: set[Pos], mode: PlanningMode = PlanningMode.STRICT) -> PathResult:
        return plan(self.pos, self.facing, goals, self.risk_model(), mode, self.risk_cap(), self.jump_available,
                    self.cfg.risk_weight, self.cfg.jump_cost, self.info.jump_distance)

    def _astar(self, goals: set[Pos], allow_unknown: bool = False) -> PathResult:
        """Strict-safe planning, or risk-tolerant when allow_unknown (cut off from the exit)."""
        return self._plan(goals, PlanningMode.TOLERANT if allow_unknown else PlanningMode.STRICT)

    def _plan_invalid_reason(self, goal: Pos) -> str | None:
        p = self.plan
        if p is None:
            reason, self.plan_dropped = self.plan_dropped, None
            return reason or "no plan"
        if not p.actions:
            return "previous plan completed"
        if p.goal != goal:
            return f"goal changed {p.goal} -> {goal}"
        if p.expect is not None and p.expect != (self.pos, self.facing):
            return f"unexpected state: expected {p.expect[0]} {p.expect[1].label}, at {self.pos} {self.facing.label}"
        for c, r0 in zip(p.cells, p.step_risk):
            if c in self.collapsed:
                return f"cell {c} on the path collapsed"
            r = self.cell_risk(c)
            if r > r0 + 0.05:
                return f"risk of {c} rose {r0:.0%} -> {r:.0%}"
        if p.actions[0] is JUMP and not self.jump_available:
            return "jump already used"
        return None

    def _ensure_plan(self, goal: Pos, purpose: str) -> None:
        reason = self._plan_invalid_reason(goal)
        if reason is None:
            return
        model, mode = self.risk_model(), PlanningMode.STRICT
        route = plan_routes(self.pos, self.facing, {goal}, model, mode, self.risk_cap(), self.jump_available,
                            risk_weight=self.cfg.risk_weight, jump_cost=self.cfg.jump_cost,
                            jump_distance=self.info.jump_distance)
        if not route.best.found and goal == self.info.start:      # cut off from the exit
            mode = PlanningMode.TOLERANT
            route = plan_routes(self.pos, self.facing, {goal}, model, mode, self.risk_cap(), self.jump_available,
                                risk_weight=self.cfg.risk_weight, jump_cost=self.cfg.jump_cost,
                                jump_distance=self.info.jump_distance)
            if route.best.found:
                reason = f"{reason}; safe cells do not reach {goal} -> route through unexplored cells"
        self.last_route = route
        self._turn_route = route
        res = route.best
        if res.found:
            self.plan = Plan(goal, purpose, list(res.actions), list(res.cells), list(res.facings), res.cost,
                             [self.cell_risk(c) for c in res.cells])
        else:
            self.plan = None
        e = self.log.add(self.turn, "A*_PLANNER", (f"path to {goal}: {len(res.actions)} actions, cost {res.cost:.1f} "
                                                   f"[{route.mode.value}]"
                                                   if res.found else f"no path to {goal}"), route.explain(), cell=goal)
        self.turn_entries["A*_PLANNER"] = e.id
        if reason not in ("no plan", "previous plan completed"):
            self.log.add(self.turn, "REPLAN", reason, [f"old plan invalidated: {reason}",
                                                      f"new plan: see entry #{e.id}"], refs=[e.id])

    # ---------------------------------------------------------------- adversarial
    def _goal_distance(self, goal: Pos) -> dict[Pos, float]:
        """BFS distances to the goal over cells the explorer may use (for the Minimax leaf value)."""
        ok = lambda c: c not in self.collapsed and c != self.unstable_here and (
            self._pit_known_safe(c) or c == goal or self.pit_prob.get(c, 1.0) <= self.cfg.max_explore_risk)
        dist, q = {goal: 0.0}, deque([goal])
        while q:
            c = q.popleft()
            for n in neighbors(c, self.info.size):
                if n not in dist and ok(n):
                    dist[n] = dist[c] + 1
                    q.append(n)
        return dist

    def _walk_in_check(self, res, a: ExplorerAction, ref: int) -> ExplorerAction:
        """Minimax assumes a Wumpus that always attacks. When every action loses under the most likely
        hypothesis, the choice is decided by the unlikely ones and can walk straight into the Wumpus'
        most likely cell. Under our own motion model, entering that cell can be deadlier than staying:
        then take Minimax's best action whose Wumpus risk is not above the risk of staying put."""
        c = self._landing(a)
        if c is None or not self.tracker.alive:
            return a
        blocked = self._blocked()
        risk = lambda cell: self.tracker.entry_risk(cell, blocked, self.pit_prob)
        enter, stay = risk(c), risk(self.pos)
        if enter <= stay:
            return a
        options = [b for b in res.root_values if b is not a and (self._landing(b) is None or risk(self._landing(b)) <= stay)]
        if not options:
            return a
        alt = max(options, key=res.root_values.get)
        self.log.add(self.turn, "WARNING", f"walk-in check: Minimax chose {a} into {c}, but entering is riskier "
                     f"({enter:.0%}) than staying ({stay:.0%}) -> {alt}",
                     ["Minimax assumes the Wumpus always attacks, so under the most likely hypothesis every",
                      "action scored as a loss and the unlikely hypotheses decided.",
                      f"own motion model: P(eaten | enter {c}) = {enter:.1%} (Wumpus there now or moving in)",
                      f"own motion model: P(eaten | stay at {self.pos}) = {stay:.1%} (Wumpus must choose to attack)",
                      f"-> {alt}: best Minimax value among actions no riskier than staying "
                      f"(E[V]={res.root_values[alt]:.0f} vs {res.root_values[a]:.0f})"], refs=[ref], cell=c)
        return alt

    def _minimax_action(self, goal: Pos) -> ExplorerAction:
        cfg = self.cfg
        blocked = self._blocked() | ({self.unstable_here} if self.unstable_here else set())
        blocked |= {c for c in self.portal_status if self._unknown_portal(c)}
        game = DuelGame(size=self.info.size, portals=dict(self.portals), blocked=frozenset(blocked),
                        pit_risk={c: (0.0 if self._pit_known_safe(c) else p) for c, p in self.pit_prob.items()},
                        goal_dist=self._goal_distance(goal), exit_dist=self._goal_distance(self.info.start),
                        max_step_risk=cfg.max_explore_risk, jump_distance=self.info.jump_distance)
        root = DuelState(self.pos, self.facing, self.pos, True, self.arrows, self.jump_available)
        hyps = self.tracker.top(cfg.minimax_hypotheses)
        res = determinized_decision(game, root, hyps, cfg.minimax_depth)
        e = self.log.add(self.turn, "MINIMAX", f"threat {self.threat:.0%} -> best {res.best_action} "
                         f"(E[V]={res.expected_value:.0f}, {res.stats.nodes} nodes, {res.stats.cutoffs} cutoffs)",
                         res.explain(), refs=[self.turn_entries["WUMPUS_BELIEF"]])
        a = self._walk_in_check(res, res.best_action, e.id)
        self.last_minimax = res
        self._turn_minimax = res
        hyp, reply = res.predicted_reply(a)
        if reply is not None:
            after = game.result(replace(root, wumpus=hyp), a)
            cell = game._land(after.wumpus.step(reply.direction)) if reply.direction else after.wumpus
            self.prediction = (self.turn, cell, reply, hyp)
        if self.plan and self.plan.actions and self.plan.actions[0] == a:
            self.plan.pop()
        else:
            self.plan = None           # tactical deviation: plan will be rebuilt next turn
            self.plan_dropped = f"Minimax deviated from the plan ({a})"
        return self._commit(a, f"{a} chosen by Minimax (Wumpus threat {self.threat:.0%})",
                            [e.id, self.turn_entries["WUMPUS_BELIEF"]])


@dataclass(frozen=True)
class KnowledgeSnapshot:
    """Read-only view of the explorer's knowledge. All collections are immutable."""
    turn: int
    position: Pos
    facing: Direction
    visited: frozenset
    pit_free: frozenset              # proven to contain no pit (permanent unless shuffled)
    known_safe: frozenset            # no pit AND Wumpus ruled out right now
    suspected_pits: MappingProxyType     # cell -> P(pit), only cells constrained by evidence
    suspected_wumpus: MappingProxyType   # cell -> P(Wumpus), belief >= 1%
    confirmed_hazards: MappingProxyType  # cell -> reason
    percepts: MappingProxyType           # turn -> Percept
    special_tiles: MappingProxyType      # cell -> kind
    portal_connections: MappingProxyType
    time_sensitive_facts: tuple          # (clause, source, valid_turn)
    inferred: MappingProxyType           # proposition -> (turn, explanation)
    ambiguous: frozenset
    pit_probability: MappingProxyType
    wumpus_alive: bool
    size: int


@dataclass(frozen=True)
class DecisionReport:
    """Structured record of one decision (Brain View / tests)."""
    turn: int
    observations: dict
    new_facts: tuple
    hazards: MappingProxyType          # "(x,y)" -> HazardEstimate.to_dict() for frontier cells
    candidates: tuple                  # (action, assessment)
    astar: dict | None                 # RoutePlan.to_dict() if a route was (re)planned this turn
    minimax: dict | None               # MinimaxResult.to_dict() if Minimax ran this turn
    selected: str
    explanation: str
