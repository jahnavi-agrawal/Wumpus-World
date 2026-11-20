"""All tunable parameters in one place. Every random decision is driven by `seed`."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    seed: int = 7
    size: int = 8
    pit_density: float = 0.135         # fraction of eligible cells holding a pit (12-15%)
    portal_pairs: int = 2
    unstable_tiles: int = 3
    shuffle_tiles: int = 2
    arrows: int = 1
    jumps: int = 1
    jump_distance: int = 3             # lands 3 cells ahead, crossing 2
    max_turns: int = 150
    wumpus_min_start_distance: int = 3
    score_step: int = -1
    score_shoot: int = -10
    score_escape_with_gold: int = 1000
    score_death: int = -1000

    def special_count(self) -> int:
        return 2 * self.portal_pairs + self.unstable_tiles + self.shuffle_tiles

    def eligible_count(self) -> int:
        """Cells outside the 3-cell safe start zone that are not special."""
        return self.size * self.size - 3 - self.special_count()

    def pit_count(self) -> int:
        return round(self.pit_density * self.eligible_count())


@dataclass(frozen=True)
class AgentConfig:
    """Explorer AI parameters (documented in DECISIONS.md)."""
    max_explore_risk: float = 0.30     # never step into a frontier cell riskier than this
    safe_risk: float = 0.05            # if any frontier cell is at most this risky, explore only those
    risk_weight: float = 25.0          # A* step cost = 1 + risk_weight * risk
    jump_cost: float = 3.0             # base A* cost of a jump (>= 3 cells keeps heuristic admissible)
    threat_threshold: float = 0.30     # P(Wumpus within 2) that triggers Minimax
    minimax_depth: int = 4             # plies (explorer + wumpus)
    minimax_hypotheses: int = 4        # top-k Wumpus positions used for determinization
    pursuit_prob: float = 0.5          # Wumpus motion model: P(move greedily towards explorer)
    stay_prob: float = 0.15            # Wumpus motion model: P(stay)
    max_component_cells: int = 18      # enumeration cap per frontier component
    stuck_turns: int = 12              # turns without visiting a new cell = "no progress"
    relaxed_risk: float = 0.40         # risk cap once stuck (and no good shot)
    min_shot_prob: float = 0.35        # hunt only if P(arrow hits) is at least this
