# Glossary

Shared vocabulary for the project. Keep terms here consistent across code and docs.

**Agent** — Anything that selects an action from an observation and a legal-action
mask. May be a DRL policy, a scripted heuristic, or a human. Decoupled from the engine.

**AgentId** — Stable identifier for a player/seat in a game. The core never hardcodes
player counts; it addresses everyone by `AgentId`.

**Action** — An identifiable, serializable choice an agent can make (e.g.
`DropDisc(column=3)`). Actions are data the engine interprets, not code.

**Action mask** — A boolean vector marking which actions are currently legal for the
agent on turn. The contract agents must respect; the mechanism that makes DRL work on
large action spaces.

**AEC (Agent–Environment Cycle)** — Turn-based interaction model where exactly one
agent acts at a time and the environment advances between actions. The model GameSim
(and PettingZoo) use.

**Component** — In the optional ECS layer, a plain data record attached to an entity
(e.g. `Position`, `Owner`, `Counter`). No behavior.

**ECS (Entity-Component-System)** — Optional data-oriented state design: `entities`
are ids, `components` are data, `systems` are functions over component queries. Used
for complex games; skipped for simple ones.

**Effect primitive** — A small reusable state transformation (place, draw, modify
counter, change phase). Rules and cards are compositions of primitives.

**Engine** — The authoritative in-memory simulator: owns state, enforces rules,
validates actions. Depends on nothing else in the project.

**Entity** — In ECS, an opaque id that components attach to.

**Event** — An immutable record of something that happened in a game (started, action
taken, phase changed, ended). The unit of logging and the signal renderers subscribe
to.

**Observation** — The subset of game state a specific agent is allowed to see. Full
board in Connect Four; hand + shared board (not opponent's hand) in MTG.

**Recorder** — Consumes the engine's event stream and persists it (or discards it, in
the null case). Enables replay.

**Replay** — Reconstructing a game exactly by feeding a recorded seed + action log back
through the deterministic engine.

**Runner** — The game loop that wires engine + agents (+ optional logging and
visualization) together and drives a game to completion.

**Self-play** — Training setup where the same policy controls all sides, learning by
playing itself. The default for two-player games.

**State** — The mutable data of a game in progress, separate from the rules that
transform it.
