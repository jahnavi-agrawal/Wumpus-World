"""Terminal version: the human plays the Wumpus, the AI plays the explorer.

The human is omniscient (sees the true dungeon). The explorer only gets Percepts.
Every Brain Activity entry has an id; `i <id>` prints the evidence behind it.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from .agents.explorer import ExplorerAgent
from .config import GameConfig
from .control.wumpus_controllers import make_controller
from .core.actions import STAY, WumpusAction
from .core.geometry import Direction, Pos
from .environment.engine import GameEngine, Phase

FACING_GLYPH = {Direction.N: "^", Direction.E: ">", Direction.S: "v", Direction.W: "<"}
SUMMARY_TAGS = ("EVENT", "TARGET", "REPLAN", "MINIMAX", "WARNING", "DECISION")

HELP = """Wumpus commands (your turn):
  w / a / s / d   move up / left / down / right      x   stay
  i <id>          inspect a Brain Activity entry (evidence + calculation)
  b               all brain entries of the explorer's last turn
  k               the explorer's knowledge map (pit probabilities)
  m               the explorer's belief about where YOU are
  l               your legal moves          h   this help          q   quit"""


# ------------------------------------------------------------------ rendering (pure)
def _portal_labels(world) -> dict[Pos, str]:
    labels, n = {}, 0
    for a in sorted(world.portals):
        if a not in labels:
            letter = "AB"[n] if n < 2 else str(n)
            labels[a] = labels[world.portals[a]] = letter
            n += 1
    return labels


def render_board(world) -> str:
    """God view of the hidden dungeon (the human Wumpus may see everything)."""
    portals = _portal_labels(world)
    rows = []
    for y in range(world.size - 1, -1, -1):
        cells = []
        for x in range(world.size):
            p = Pos(x, y)
            if p == world.explorer and world.explorer_alive:
                t = "E" + FACING_GLYPH[world.facing]
            elif p == world.wumpus and world.wumpus_alive:
                t = "W"
            elif p in world.collapsed:
                t = "XX"
            elif p in world.pits:
                t = "P"
            elif p == world.gold:
                t = "G"
            elif p in portals:
                t = "o" + portals[p]
            elif p in world.unstable:
                t = "u"
            elif p in world.shuffle_tiles:
                t = "s"
            else:
                t = "." if p in world.visited else ""
            cells.append(f"{t:^3}")
        rows.append(f"{y} |" + "|".join(cells) + "|")
    rows.append("   " + "".join(f"{x:^4}" for x in range(world.size)))
    return "\n".join(rows)


LEGEND = ("E>=explorer(facing)  W=Wumpus(you)  P=pit  XX=collapsed  G=gold  oA/oB=portal pairs  "
          "u=unstable  s=shuffle  .=visited")


def render_knowledge(agent: ExplorerAgent, explorer_now: tuple[Pos, Direction] | None = None) -> str:
    """What the explorer believes about pits (never the true layout). Beliefs are as of the
    explorer's last observation; `explorer_now` marks where it stands after its last action."""
    pos, facing = explorer_now or (agent.pos, agent.facing)
    rows = []
    for y in range(agent.info.size - 1, -1, -1):
        cells = []
        for x in range(agent.info.size):
            p = Pos(x, y)
            st = agent.statuses.get(p)
            if p == pos:
                t = "E" + FACING_GLYPH[facing]
            elif p in agent.collapsed:
                t = "XX"
            elif p in agent.visited:
                t = "."
            elif st is not None and st.status == "pit":
                t = "PIT"
            elif st is not None and st.status == "safe":
                t = "ok"
            else:
                t = f"{agent.pit_prob.get(p, 0):.0%}"
            cells.append(f"{t:^4}")
        rows.append(f"{y} |" + "|".join(cells) + "|")
    rows.append("   " + "".join(f"{x:^5}" for x in range(agent.info.size)))
    return "Explorer knowledge: P(pit) per cell  (ok = proven safe by the KB, PIT = proven pit)\n" + "\n".join(rows)


def render_belief(agent: ExplorerAgent, explorer_now: tuple[Pos, Direction] | None = None) -> str:
    pos, facing = explorer_now or (agent.pos, agent.facing)
    if not agent.tracker.alive:
        return "The explorer knows you are dead."
    rows = []
    for y in range(agent.info.size - 1, -1, -1):
        cells = []
        for x in range(agent.info.size):
            p = Pos(x, y)
            v = agent.tracker.prob(p)
            t = "E" + FACING_GLYPH[facing] if p == pos else (f"{v:.0%}" if v >= 0.005 else "")
            cells.append(f"{t:^4}")
        rows.append(f"{y} |" + "|".join(cells) + "|")
    rows.append("   " + "".join(f"{x:^5}" for x in range(agent.info.size)))
    return "Explorer belief: P(Wumpus is here)\n" + "\n".join(rows)


def render_status(engine: GameEngine) -> str:
    s = engine.status()
    inv = s.inventory
    return (f"Turn {s.turn}  Step {s.step_count}  Score {s.score}  |  gold: {'yes' if inv['gold'] else 'no'}  "
            f"arrows: {inv['arrows']}  jump: {'ready' if inv['jump'] else 'used'}  |  to move: {s.current_player}")


def render_entry(agent: ExplorerAgent, entry_id: int) -> str:
    if not 0 <= entry_id < len(agent.log.entries):
        return f"No brain entry #{entry_id} (valid: 0..{len(agent.log.entries) - 1})"
    e = agent.log.get(entry_id)
    lines = [f"#{e.id}  turn {e.turn}  {e.line()}" + (f"   [cell {e.cell}]" if e.cell else "")]
    lines += ["    " + d for d in e.details]
    if e.refs:
        lines.append("  depends on: " + ", ".join(f"#{r} {agent.log.get(r).line()[:60]}" for r in e.refs))
    return "\n".join(lines)


def render_turn_summary(agent: ExplorerAgent, turn: int, full: bool = False) -> str:
    entries = agent.log.for_turn(turn)
    shown = [e for e in entries if full or e.tag in SUMMARY_TAGS]
    lines = [f"  #{e.id:<4} {e.line()}" for e in shown]
    hidden = len(entries) - len(shown)
    if hidden:
        lines.append(f"  ... {hidden} more (KB/RISK_CALC/belief) - type 'b' to list, 'i <id>' to inspect")
    return "Brain Activity:\n" + "\n".join(lines)


# ------------------------------------------------------------------ commands (pure)
@dataclass(frozen=True)
class Command:
    kind: str                      # move | inspect | brain | knowledge | belief | legal | help | quit | invalid
    action: WumpusAction | None = None
    arg: int | None = None
    error: str = ""


KEYS = {"w": Direction.N, "up": Direction.N, "d": Direction.E, "right": Direction.E,
        "s": Direction.S, "down": Direction.S, "a": Direction.W, "left": Direction.W}


def parse_command(text: str) -> Command:
    t = text.strip().lower()
    if t in KEYS:
        return Command("move", WumpusAction.move(KEYS[t]))
    if t in ("x", "stay", ""):
        return Command("move", STAY) if t else Command("invalid", error="type h for help")
    if t.startswith("i"):
        rest = t[1:].strip().lstrip("#")
        return Command("inspect", arg=int(rest)) if rest.isdigit() else Command("invalid", error="usage: i <id>")
    simple = {"b": "brain", "k": "knowledge", "m": "belief", "l": "legal", "h": "help", "?": "help",
              "q": "quit", "quit": "quit", "exit": "quit"}
    if t in simple:
        return Command(simple[t])
    return Command("invalid", error=f"unknown command '{text.strip()}' - type h for help")


# ------------------------------------------------------------------ game loop
def run(config: GameConfig, wumpus: str = "human", inp=input, out=print, show_board: bool = True):
    """Play one game. `wumpus` is 'human' or a scripted mode ('idle', 'random', 'hunter')."""
    engine = GameEngine(config)
    agent = ExplorerAgent(engine.public_info())
    bot = None if wumpus == "human" else make_controller(wumpus, config.seed)
    out(f"WUMPUS: THE MIND GAMES - terminal edition (seed {config.seed}, Wumpus: {wumpus})")
    out(LEGEND)
    if bot is None:
        out(HELP)
    while engine.phase is not Phase.OVER:
        if engine.phase is Phase.EXPLORER:
            turn = engine.turn
            action = agent.act(engine.percept())
            events = engine.apply_explorer_action(action)
            out("")
            out(f"=== Turn {turn}: explorer -> {action}")
            for ev in events:
                out(f"  {ev}")
            out(render_turn_summary(agent, turn))
            continue
        if show_board:
            out(render_board(engine.world))
            out(render_status(engine))
        legal = engine.legal_wumpus_actions()
        if bot is not None:
            a = bot.choose(engine.world, legal)
        else:
            a = _ask_human(engine, agent, legal, inp, out)
            if a is None:
                out("Game abandoned.")
                return engine, agent
        for ev in engine.apply_wumpus_action(a):
            out(f"  {ev}")
    out("")
    out(render_board(engine.world))
    out(f"GAME OVER: {engine.result.outcome.value} ({engine.result.reason}).  Final score {engine.score}")
    return engine, agent


def _ask_human(engine, agent, legal, inp, out) -> WumpusAction | None:
    while True:
        try:
            text = inp("Wumpus> ")
        except EOFError:
            return None
        cmd = parse_command(text)
        if cmd.kind == "move":
            if cmd.action in legal:
                return cmd.action
            out(f"Illegal move {cmd.action} (wall or pit). Legal: {', '.join(map(str, legal))}")
        elif cmd.kind == "inspect":
            out(render_entry(agent, cmd.arg))
        elif cmd.kind == "brain":
            out(render_turn_summary(agent, agent.turn, full=True))
        elif cmd.kind == "knowledge":
            out(render_knowledge(agent, (engine.world.explorer, engine.world.facing)))
        elif cmd.kind == "belief":
            out(render_belief(agent, (engine.world.explorer, engine.world.facing)))
        elif cmd.kind == "legal":
            out("Legal: " + ", ".join(map(str, legal)))
        elif cmd.kind == "help":
            out(HELP)
        elif cmd.kind == "quit":
            return None
        else:
            out(cmd.error)


def configure_console() -> None:
    """Windows consoles default to cp1252; explanations use symbols such as Σ, δ, ∨."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
