import itertools
import random

import pytest

from wumpus.ai.logic import DungeonKB
from wumpus.ai.probability import estimate_pits
from wumpus.config import GameConfig
from wumpus.core.geometry import Pos, all_cells, neighbors
from wumpus.env.engine import GameEngine

INFO = GameEngine(GameConfig(seed=1)).public_info()


def run(visited, breezy, prior):
    obs = {v: (v in breezy, 1) for v in visited}
    dkb = DungeonKB(INFO)
    dkb.rebuild(set(visited), set(), obs)
    cells = [c for c in all_cells(8) if c not in visited]
    return estimate_pits(8, dkb.classify(cells), obs, prior)


def test_aima_textbook_numbers():
    """AIMA 12.7: breeze in (1,0) and (0,1), prior 0.2 -> P(1,1)=0.86, P(2,0)=P(0,2)=0.31."""
    r = run([Pos(0, 0), Pos(1, 0), Pos(0, 1)], {Pos(1, 0), Pos(0, 1)}, 0.2)
    assert r[Pos(1, 1)].p_pit == pytest.approx(0.2 / 0.232, abs=1e-9)
    assert r[Pos(2, 0)].p_pit == pytest.approx(0.072 / 0.232, abs=1e-9)
    assert r[Pos(0, 2)].p_pit == pytest.approx(0.072 / 0.232, abs=1e-9)
    assert r[Pos(1, 1)].worlds_consistent == 5
    assert r[Pos(5, 5)].method == "prior" and r[Pos(5, 5)].p_pit == 0.2
    text = "\n".join(r[Pos(1, 1)].explain())
    assert "Consistent models enumerated: 5" in text and "B(1,0)" in text


def test_kb_proven_cells_fixed():
    r = run([Pos(0, 0), Pos(1, 0), Pos(0, 1)], {Pos(1, 0)}, 0.2)
    assert r[Pos(2, 0)].p_pit == 1.0 and r[Pos(2, 0)].method == "kb-pit"
    assert r[Pos(1, 1)].p_pit == 0.0 and r[Pos(1, 1)].method == "kb-safe"


def brute_force(visited, breezy, prior, query):
    """Full joint enumeration over all unknown cells near the evidence (ground truth)."""
    unknown = sorted({n for v in visited for n in neighbors(v, 8)} - set(visited) - INFO.safe_start_zone - set(INFO.pads))
    num = den = 0.0
    for bits in itertools.product([0, 1], repeat=len(unknown)):
        pits = {c for c, b in zip(unknown, bits) if b}
        if all(any(n in pits for n in neighbors(v, 8)) == (v in breezy) for v in visited):
            w = 1.0
            for b in bits:
                w *= prior if b else 1 - prior
            den += w
            num += w * (query in pits)
    return num / den


@pytest.mark.parametrize("seed", range(15))
def test_matches_brute_force_on_random_evidence(seed):
    rng = random.Random(seed)
    truth = set(rng.sample([c for c in all_cells(8) if c not in INFO.safe_start_zone and c not in INFO.pads], 9))
    visited, frontier = [Pos(0, 0)], [Pos(0, 0)]
    while frontier and len(visited) < 7:   # random safe walk
        p = frontier.pop(rng.randrange(len(frontier)))
        for n in neighbors(p, 8):
            if n not in truth and n not in visited and rng.random() < 0.6 and len(visited) < 7:
                visited.append(n)
                frontier.append(n)
    breezy = {v for v in visited if any(n in truth for n in neighbors(v, 8))}
    r = run(visited, breezy, 0.15)
    for c in sorted({n for v in visited for n in neighbors(v, 8)} - set(visited) - INFO.safe_start_zone - set(INFO.pads)):
        assert r[c].p_pit == pytest.approx(brute_force(visited, breezy, 0.15, c), abs=1e-9)
