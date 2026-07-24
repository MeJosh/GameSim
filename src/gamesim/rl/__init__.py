"""DRL adapter layer: bridges the native engine to standard RL tooling.

Keeps tensors and neural-net concerns *out* of the engine. Per-game encoders convert
observations to tensors and expose the legal-action mask; a PettingZoo adapter presents
the engine as a standard multi-agent env. Concrete code arrives in Phase 2
(see docs/architecture.md "How the DRL side connects").
"""

from .encoder import Encoder

__all__ = ["Encoder"]
