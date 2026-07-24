const boardElement = document.querySelector("#board");
const opponentElement = document.querySelector("#opponent");
const newGameElement = document.querySelector("#new-game");
const statusElement = document.querySelector("#status");

let game = null;
let pending = false;

function statusText(snapshot) {
  const messages = {
    in_progress: "Your turn",
    human_won: "You won",
    opponent_won: "Opponent won",
    draw: "Draw",
  };
  return messages[snapshot.outcome];
}

function render() {
  if (!game) return;
  boardElement.replaceChildren();
  const playable = !pending && game.outcome === "in_progress";
  for (const row of [...game.board].reverse()) {
    row.forEach((token, column) => {
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = `cell token-${token}`;
      cell.disabled = !playable || !game.legal_columns.includes(column);
      cell.setAttribute("aria-label", `Drop disc in column ${column + 1}`);
      cell.addEventListener("click", () => playMove(column));
      boardElement.append(cell);
    });
  }
  statusElement.textContent = pending ? "Opponent is thinking..." : statusText(game);
}

async function newGame() {
  pending = true;
  statusElement.textContent = "Starting game...";
  try {
    const response = await fetch("/api/games", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ opponent: opponentElement.value }),
    });
    game = await response.json();
    if (!response.ok) throw new Error(game.detail);
  } catch (error) {
    statusElement.textContent = error.message;
  } finally {
    pending = false;
    render();
  }
}

async function playMove(column) {
  if (!game || pending) return;
  pending = true;
  render();
  try {
    const response = await fetch(`/api/games/${game.game_id}/moves`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ column }),
    });
    game = await response.json();
    if (!response.ok) throw new Error(game.detail);
  } catch (error) {
    statusElement.textContent = error.message;
  } finally {
    pending = false;
    render();
  }
}

newGameElement.addEventListener("click", newGame);
newGame();
