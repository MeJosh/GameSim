# GameSim

A **game-agnostic simulation and deep reinforcement learning (DRL) framework** for
training and analyzing game-playing agents.

You write a small, data-driven **engine** for a game (its rules, state, and legal
actions). GameSim gives you everything around it: a uniform agent interface,
deterministic logging and replay, optional game-specific visualization, and an
adapter into standard DRL tooling so you can train agents by self-play.

## Why this exists

This is run as an **experiment**: a place to learn DRL and good simulation design by
iterating. The near-term proving ground is **Connect Four** (small, perfect
information, easy to test end-to-end). The long-term target is **simplified Magic:
The Gathering decks** — a deliberately hard case with hidden information,
stochasticity, huge variable action spaces, and rules-as-data. Designing for MTG's
hard parts from the start keeps the framework honest.

## The four subsystems

1. **Engine** — the authoritative, in-memory simulator. Enforces the rules,
   validates every action, and is the single source of truth for game state.
2. **Agent** — anything that chooses actions: a DRL policy, a scripted bot, or a
   human. Agents are fully decoupled from the engine and hot-swappable.
3. **Visualization** — optional, game-specific renderers that can attach to a live
   simulation or step through a recorded log.
4. **Logging** — toggleable, deterministic event recording that supports exact
   replay of any game from a seed plus its action log.

## Status

Phase 0 (planning + scaffold) — see [`plans/roadmap.md`](plans/roadmap.md).

## Layout

```
docs/       Architecture, glossary, and decision records (ADRs)
plans/      Roadmap and detailed per-phase implementation plans
progress/   Dated write-ups on what changed and what was learned
src/gamesim Framework source
tests/      Test suite (TDD, red -> green)
```

## Getting started (dev)

With `make` (macOS, Linux, WSL, or Git Bash on Windows):

```bash
make install   # create .venv and install the package + dev tools
make test      # run the test suite
make check     # lint + format check + type-check + test
```

`make install` also installs the repository's pre-commit hook. It automatically
applies Ruff fixes and formatting to staged Python files; stage the resulting
changes and commit again. Use `make hooks` to run the same hook suite across the
whole repository.

On Windows, the Makefile uses the Python launcher (`py -3`) by default because
`python3` is often only a Microsoft Store alias. Install Python 3.10+ from
python.org, or override the interpreter if needed:

```bash
make install-rl PYTHON=python
```

Run `make help` to list every target (`format`, `typecheck`, `test-cov`,
`install-rl`, `clean`, ...).

### Without make (e.g. native Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -m pre_commit install
pytest
```

The DRL stack is a separate extra so you don't pull in torch until you start
training: `make install-rl` (or `pip install -e ".[dev,rl]"`).

### Local browser play

The optional web adapter lets you play a locally adjudicated Connect Four game
against a random baseline or a trained policy. It is deliberately outside the engine
and training packages: the browser only submits columns, while the engine validates
and applies every move.

```bash
make install-web
make serve
```

Then open `http://127.0.0.1:8000`. Pick an opponent: **Random** or **Minimax** need no
RL dependencies; **Trained policy** uses the default checkpoint at
`checkpoints/connect_four_maskable_ppo.zip` (requires the RL extras). The service is
local and in-memory, so active games are not persisted when the server stops.

### Record a match and view it

`record-matches` plays any two agents and writes a versioned ZIP match archive
(`manifest.json` index + one replayable JSON per game). Each side is `random`,
`minimax[:depth]`, or `trained:<checkpoint>` — so `minimax`-vs-`random` records with no
RL dependency at all, while trained matchups need the RL extras:

```bash
make record-matches GAMES=100                        # trained vs random (default)
make record-matches AGENT_A=minimax:4 AGENT_B=random # no torch needed
```

Turn any match archive into a **single self-contained HTML report** (summary stats +
click-through of every game, no server, no dependencies to open):

```bash
make report REPORT_LOG=logs/connect_four_trained_vs_random.zip REPORT_OUT=reports/match.html
```

Or explore it in the browser: open the **Replay** tab, upload the archive, pick a game,
and step through each engine-reconstructed turn (the panel also shows a summary).

### Incremental training smoke run

For a bounded overnight-style check of learning progress, run the exploratory
experiment script directly rather than extending the normal training CLI:

```bash
python scripts/run_incremental_smoke.py --run-dir runs/incremental-smoke-001
```

It evaluates an untrained baseline, then continues one PPO model through 2,048,
4,096, and 8,192 additional timesteps. Each stage saves a checkpoint, evaluates the
policy against **random and minimax** baselines, records a representative match log per
stage under `matches/`, and plays **head-to-head vs earlier checkpoints**. A versioned
`progress.json` captures the metrics (win-rate, game length, opening-move distribution,
head-to-head). The script refuses to reuse an existing directory, so it cannot overwrite
prior runs.

Turn a completed run into a **self-contained HTML progress report** (win-rate and
game-length trends, opening-strategy shift, and a head-to-head matrix — torch-free):

```bash
make progress-report PROGRESS_JSON=runs/incremental-smoke-001/progress.json PROGRESS_REPORT_OUT=reports/progress.html
```

After training, evaluate the saved policy against random and minimax baselines:

```bash
make train TIMESTEPS=100000 SEED=0
make evaluate GAMES=100
```

Both CLIs show Rich progress bars with elapsed/estimated remaining time and live
stats. Use `--no-progress` if you want plain output for logs or scripts.

The default checkpoint is `checkpoints/connect_four_maskable_ppo.zip`. You can
evaluate just one baseline or point at another checkpoint:

```bash
make evaluate OPPONENT=random CHECKPOINT=checkpoints/my_run.zip
make evaluate OPPONENT=minimax GAMES=50
```

See [`docs/architecture.md`](docs/architecture.md) for the design and
[`plans/roadmap.md`](plans/roadmap.md) for what's being built next.
