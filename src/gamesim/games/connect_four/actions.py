"""Connect Four actions.

Kept deliberately simple (see plans/phase-01-engine-core.md): the only action kind
is "drop a disc into a column," so an action is just the column's integer index.
No wrapper type is needed for a single action kind.
"""

from __future__ import annotations

# An action is the integer column index to drop a disc into: 0..NUM_COLUMNS-1
# (see gamesim.games.connect_four.state for board dimensions). This is also the
# encoder's contract for Phase 2 (mask index i <-> column i).
Action = int
