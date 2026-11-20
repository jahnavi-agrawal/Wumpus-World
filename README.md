# WUMPUS: THE MIND GAMES

You play the **Wumpus**. An AI **explorer** that cannot see the dungeon tries to find the gold
and escape. It uses propositional logic (DPLL), Bayesian probability, an HMM filter over your
position, A* and Minimax with alpha-beta pruning. Every decision it makes can be inspected.

## Setup
```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Play (terminal version)
```
.venv\Scripts\python main.py                    # you are the Wumpus, seed 7
.venv\Scripts\python main.py --seed 12          # another reproducible dungeon
.venv\Scripts\python main.py --wumpus hunter    # watch the AI vs a scripted Wumpus (idle/random/hunter)
```
Your turn: `w a s d` move, `x` stay, `i <id>` inspect a Brain Activity entry (its evidence and
calculation), `b` all entries of the explorer's last turn, `k` explorer's pit knowledge,
`m` explorer's belief about where you are, `l` legal moves, `h` help, `q` quit.

## Tests
```
.venv\Scripts\python -m pytest
```

Docs: [RULES.md](RULES.md) (all state-transition rules), [ARCHITECTURE.md](ARCHITECTURE.md),
[DECISIONS.md](DECISIONS.md) (why each algorithm, how the probabilities work, Minimax assumptions,
simplifications).
