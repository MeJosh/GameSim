# Architecture

GameSim separates a game into four cleanly-decoupled subsystems. The guiding
principle is that the **Engine is authoritative and self-contained** — it never
imports agents, renderers, or DRL code. Everything else depends on the engine
through narrow interfaces, never the other way around.

```
                +-----------------------------------------------+
                |                    Engine                     |
                |  (authoritative rules + in-memory game state) |
                |                                               |
   Agent -----> |  step(agent, action)  --> validate --> apply |----> Events
   (action)     |  observation(agent)   --> per-agent view     |        |
                |  legal_actions(agent) --> action mask        |        |
                +-----------------------------------------------+        |
                       ^                    |                            |
                       |                    v                            v
                   observation +     +-------------+              +-------------+
                   action mask       | Runner /    |              |  Recorder   |
                       |             | game loop   |              | (logging)   |
                +-------------+      +-------------+              +-------------+
                |   Agent(s)  |                                         |
                | policy /    |                                         v
                | scripted /  |                                   +-------------+
                | human       |                                   |  Replay +   |
                +-------------+                                   | Visualizer  |
                                                                  +-------------+
```

## Core design decisions (summary)

These are recorded in full as ADRs under [`docs/adr/`](adr/):

- **Python-first.** Fastest iteration and the best DRL ecosystem. The engine
  boundary is kept clean enough to port a hot core to Rust/C++ later if needed.
- **N-agent interface from day one.** The core addresses players by `AgentId`, never
  by hardcoded `player_1/player_2`. Two-player is just the N=2 case; adding more
  players later is naming discipline, not a rewrite.
- **Action masking is first-class.** The engine always reports which actions are
  legal for the agent on turn. This is what makes DRL tractable on games with large,
  mostly-illegal action spaces (and it's essential for MTG).
- **Data-driven rules.** Game definitions and actions are data plus small effect
  primitives, not bespoke hardcoded classes, so games are easy to modify and extend.
- **Deterministic, event-sourced logging.** Any game can be replayed exactly from a
  seed plus its action log.

---

## 1. Engine

The engine is the **single source of truth**. It owns game state in memory, enforces
the rules, and validates every action before applying it. Agents propose actions; the
engine decides what actually happens.

### The turn cycle (AEC model)

Turn-based games follow an **Agent–Environment Cycle** (the model used by PettingZoo):
at any moment exactly one agent is "to move," and the engine drives the loop. This
generalizes cleanly from Connect Four's strict alternation to MTG's priority passing.

The core engine interface (see `src/gamesim/core/engine.py`):

```python
class Engine(Protocol[StateT, ActionT]):
    def reset(self, *, seed: int | None = None) -> None: ...
    def current_agent(self) -> AgentId: ...          # whose turn is it?
    def legal_actions(self, agent: AgentId) -> ActionMask: ...
    def step(self, agent: AgentId, action: ActionT) -> StepResult: ...
    def observation(self, agent: AgentId) -> Observation: ...
    def rewards(self) -> Mapping[AgentId, float]: ...
    def is_terminal(self) -> bool: ...
    def agents(self) -> Sequence[AgentId]: ...        # all players
```

Key rules the engine enforces:

- **Validation.** `step` rejects illegal actions (raises, rather than silently
  ignoring) so bugs surface loudly. `legal_actions` is the contract agents must
  respect; DRL agents get it as a mask.
- **Determinism.** The engine owns a single seeded RNG. All shuffles, draws, and
  random outcomes go through it. Same seed + same actions => identical game. This is
  what makes replay and reproducible experiments possible.
- **Observation boundary.** `observation(agent)` returns only what that agent is
  allowed to see. For Connect Four that's the full board; for MTG it's your hand and
  the shared board but *not* the opponent's hand or library order. Building this
  boundary now means hidden-information games are a natural extension, not a retrofit.

### State, and the (optional) ECS layer

State is separated from rules: a `GameState` holds the mutable data; the engine holds
the logic that reads and transforms it. How state is stored internally is an engine
implementation detail.

The **Entity-Component-System (ECS)** approach is a supported and encouraged pattern
for richer games, but it is deliberately *optional* (a secondary priority). Simple
games like Connect Four don't need it — a small typed state object is clearer. The
plan is to prove the framework with a plain state first, then introduce a lightweight
ECS (`entities` = ids, `components` = plain data records, `systems` = functions over
component queries) when a game's complexity justifies it — MTG being the obvious
driver. The engine interface above is agnostic to whether ECS is used underneath.

### Data-driven rules and actions

Rather than hardcoding each action as a method, actions are **data** interpreted by
the engine:

- An **action** is an identifiable, serializable value (e.g. Connect Four:
  `DropDisc(column=3)`).
- Effects are composed from small reusable **primitives** (move/place, draw, modify
  a counter, end phase, ...). A card or rule is then a *composition of primitives over
  components* — data, not code.

Connect Four barely exercises this (one action type), but establishing the pattern now
is exactly what lets MTG cards become data-with-effects instead of thousands of
bespoke classes.

---

## 2. Agent

An **agent** chooses an action given what it can see. It is completely decoupled from
the engine — it receives an observation and an action mask, and returns an action. It
never touches game state directly.

```python
class Agent(Protocol):
    def act(self, observation: Observation, mask: ActionMask) -> ActionT: ...
```

Because the interface is this narrow, agents are **hot-swappable**. The same engine
run can pit any mix of:

- **PolicyAgent** — wraps a trained DRL policy (Phase 2).
- **ScriptedAgent** — heuristics/search (e.g. random, minimax) for baselines and
  opponents.
- **HumanAgent** — reads input from a UI or CLI (the "play against it" use case).

The engine doesn't know or care which is which. Swapping opponents, running
self-play (the same policy on both sides), or dropping a human in are all just
different agent wiring in the **Runner**.

### The Runner (game loop)

A small `Runner` orchestrates a game: ask the engine whose turn it is, get that
agent's observation + mask, ask the agent for an action, `step` the engine, emit
events, repeat until terminal. The Runner is where agents, logging, and (optionally)
visualization are wired together. Training loops are a specialized runner.

---

## 3. Visualization

Visualization is **optional and game-specific**. During training, games run far too
fast to watch, so nothing renders by default. When you *do* want to see a game, a
renderer for that game hooks in.

```python
class Renderer(Protocol):
    def render(self, observation: Observation) -> None: ...
```

Two modes, one interface:

- **Live** — attach a renderer to a running engine (via the Runner's event stream) to
  watch a game as it plays.
- **Replay** — load a recorded log and step forward/backward through it. Because logs
  are deterministic, replay reconstructs exact states. This is the primary debugging
  and analysis tool.

Renderers subscribe to engine **events**; they are pure consumers and can never affect
the game. Each game ships its own renderer (Connect Four: a grid; MTG: a board view).

---

## 4. Logging

Logging is **toggleable** (off by default during bulk training, on when debugging or
assessing progress) and **event-sourced**.

- The engine emits a stream of **events** (game started with seed X, agent A took
  action Y, phase changed, game ended). A `Recorder` consumes them.
- **NullRecorder** does nothing (zero overhead). **JsonlRecorder** appends events to a
  `.jsonl` file.
- A recorded log = `{seed, agent list, ordered actions/events}`. Feeding it back
  through the engine **replays the game exactly**, because the engine is deterministic.

This gives us three things cheaply: reproducible experiments, a debugging time
machine, and the data source for replay visualization.

```python
class Recorder(Protocol):
    def record(self, event: Event) -> None: ...
    def close(self) -> None: ...
```

---

## How the DRL side connects

The native engine stays independent of any DRL library. A thin **adapter** exposes it
through standard interfaces:

- A **PettingZoo AEC** wrapper presents the engine as a standard multi-agent env.
- Per-game **encoders** convert engine observations to tensors and back, and expose the
  engine's legal-action mask to the learner.
- Training uses **`sb3-contrib` MaskablePPO** initially (PPO + action masking — the
  workhorse for this problem class), driven by **self-play** for two-player games. The
  `Agent` interface is kept clean so a hand-rolled learner or a different library
  (e.g. CleanRL) can be swapped in later for learning purposes.

```
Engine  <--adapter-->  PettingZoo AEC env  <--encoder-->  MaskablePPO (self-play)
```

Keeping the encoder and adapter as separate layers means the engine never has tensors
or neural-net concerns leaking into it, and a game is playable by scripted/human agents
with no DRL dependency installed at all.

---

## Dependency direction (the rule that keeps this clean)

```
games ---------> core <--------- agents
  |               ^                 ^
  |               |                 |
  +--> encoders --+--- rl adapter --+
       (rl)            (rl)
  viz ---> core        logging ---> core
```

`core` depends on nothing in the project. Everything points *inward* toward `core`.
If you ever find `core` importing an agent, a renderer, or torch, something has gone
wrong.
