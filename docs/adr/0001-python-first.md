# 0001 — Python-first stack

**Status:** Accepted — 2026-07-23

## Context

The framework has two halves with different pressures. The engine wants performance and
would benefit from a systems language (Rust/C++, ECS libraries like `bevy_ecs`). The DRL
half lives most naturally in Python (PyTorch, Gymnasium, PettingZoo, Stable-Baselines3).
The project is an experiment focused on learning and fast iteration, and the author is
newer to DRL.

## Decision

Build everything in **Python first**. Use NumPy for state/observation math and the
standard Python DRL ecosystem. Keep the `Engine` boundary (see
[0002](0002-n-agent-interface.md)) clean and free of framework leakage so a hot core
could be ported to Rust/C++ (via PyO3/pybind11) later without touching agents or the
DRL layer.

## Consequences

- **+** Fastest path to a working end-to-end pipeline; least new tooling to learn.
- **+** Direct access to the best DRL libraries; no FFI friction during the learning phase.
- **+** A clean engine interface leaves the door open to a native core later.
- **−** Pure-Python simulation is slower; large-scale MTG training may eventually need
  optimization (vectorization, or a ported core). Accepted for now — correctness and
  iteration speed matter more than raw throughput at this stage.
