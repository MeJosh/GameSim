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

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

See [`docs/architecture.md`](docs/architecture.md) for the design and
[`plans/roadmap.md`](plans/roadmap.md) for what's being built next.
