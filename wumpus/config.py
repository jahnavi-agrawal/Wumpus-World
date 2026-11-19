"""All tunable parameters in one place. Every random decision is driven by `seed`."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    seed: int = 7
    size: int = 8
    pit_count: int = 7
    arrows: int = 1
    jumps: int = 2
    max_turns: int = 150
    wumpus_min_start_distance: int = 3
    use_pads: bool = True
    collapse_charges: int = 3
    collapse_cooldown: int = 3
    collapse_range: int = 2
    shuffle_charges: int = 1


@dataclass(frozen=True)
class AgentConfig:
    """Explorer AI parameters (documented in DECISIONS.md)."""
    max_explore_risk: float = 0.30     # never step into a frontier cell riskier than this
    risk_weight: float = 25.0          # A* step cost = 1 + risk_weight * risk
    jump_cost: float = 2.5             # base A* cost of a jump (>= 2 keeps heuristic admissible)
    threat_threshold: float = 0.30     # P(Wumpus within 2) that triggers Minimax
    minimax_depth: int = 4             # plies (explorer + wumpus)
    minimax_hypotheses: int = 4        # top-k Wumpus positions used for determinization
    pursuit_prob: float = 0.5          # Wumpus motion model: P(move greedily towards explorer)
    stay_prob: float = 0.15            # Wumpus motion model: P(stay)
    max_component_cells: int = 18      # enumeration cap per frontier component
    stuck_turns: int = 12              # turns without visiting a new cell = "no progress"
    relaxed_risk: float = 0.40         # risk cap once stuck (and no good shot)
    min_shot_prob: float = 0.35        # hunt only if P(arrow hits) is at least this
