const boardElement = document.querySelector("#board");
const opponentElement = document.querySelector("#opponent");
const newGameElement = document.querySelector("#new-game");
const statusElement = document.querySelector("#status");
const playPanel = document.querySelector("#play-panel");
const replayPanel = document.querySelector("#replay-panel");
const playTab = document.querySelector("#play-tab");
const replayTab = document.querySelector("#replay-tab");
const logFileElement = document.querySelector("#log-file");
const replayStatusElement = document.querySelector("#replay-status");
const dropZoneElement = document.querySelector("#drop-zone");
const replayInsightsElement = document.querySelector("#replay-insights");
const winnerChartElement = document.querySelector("#winner-chart");
const turnChartElement = document.querySelector("#turn-chart");
const gameListElement = document.querySelector("#game-list");
const replayBoardElement = document.querySelector("#replay-board");
const replayTurnElement = document.querySelector("#replay-turn");
const previousMoveElement = document.querySelector("#previous-move");
const nextMoveElement = document.querySelector("#next-move");
const moveCounterElement = document.querySelector("#move-counter");

let game = null;
let pending = false;
let replay = null;
let replayState = null;
let selectedGameIndex = 0;
let selectedMove = 0;

function statusText(snapshot) {
  const messages = {
    in_progress: "Your turn",
    human_won: "You won",
    opponent_won: "Opponent won",
    draw: "Draw",
  };
  return messages[snapshot.outcome];
}

function renderBoard(element, board) {
  element.replaceChildren();
  if (!board) return;
  for (const row of [...board].reverse()) {
    row.forEach((token) => {
      const cell = document.createElement("span");
      cell.className = `cell token-${token}`;
      element.append(cell);
    });
  }
}

function renderPlay() {
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

function renderReplay() {
  gameListElement.replaceChildren();
  if (!replay) return;
  renderReplayInsights();
  replay.games.forEach((item) => {
    const entry = document.createElement("button");
    const title = document.createElement("span");
    const details = document.createElement("span");
    entry.type = "button";
    entry.className = item.index === selectedGameIndex ? "game-entry selected" : "game-entry";
    title.className = "game-entry-title";
    details.className = "game-entry-details";
    title.textContent = `Game ${item.index + 1}`;
    details.textContent = item.outcome === "draw"
      ? `Draw - ${item.total_moves} turns`
      : `${item.outcome} won - ${item.total_moves} turns`;
    entry.append(title, details);
    entry.addEventListener("click", () => selectGame(item.index));
    gameListElement.append(entry);
  });
  if (!replayState) return;
  renderBoard(replayBoardElement, replayState.board);
  moveCounterElement.textContent = `${replayState.move} / ${replayState.total_moves}`;
  previousMoveElement.disabled = replayState.move === 0;
  nextMoveElement.disabled = replayState.move === replayState.total_moves;
  replayTurnElement.textContent = replayState.current_player
    ? `${replayState.current_player} to move`
    : replayState.outcome === "draw"
      ? "Draw"
      : `${replayState.outcome} won`;
}

function renderReplayInsights() {
  const outcomes = new Map();
  const turnCounts = new Map();
  replay.games.forEach((item) => {
    outcomes.set(item.outcome, (outcomes.get(item.outcome) || 0) + 1);
    turnCounts.set(item.total_moves, (turnCounts.get(item.total_moves) || 0) + 1);
  });
  replayInsightsElement.hidden = false;
  renderWinnerChart([...outcomes.entries()]);
  renderTurnChart([...turnCounts.entries()].sort(([left], [right]) => left - right));
}

function renderWinnerChart(outcomes) {
  winnerChartElement.replaceChildren();
  const total = replay.games.length;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "donut");
  svg.setAttribute("viewBox", "0 0 120 120");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `Match outcomes across ${total} games`);
  const background = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  background.setAttribute("class", "donut-background");
  background.setAttribute("cx", "60");
  background.setAttribute("cy", "60");
  background.setAttribute("r", "42");
  svg.append(background);
  let offset = 0;
  outcomes.forEach(([label, count], index) => {
    const slice = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    slice.setAttribute("class", `donut-slice donut-slice-${index % 3}`);
    slice.setAttribute("cx", "60");
    slice.setAttribute("cy", "60");
    slice.setAttribute("r", "42");
    slice.setAttribute("pathLength", "100");
    slice.setAttribute("stroke-dasharray", `${(count / total) * 100} ${100 - (count / total) * 100}`);
    slice.setAttribute("stroke-dashoffset", String(-offset));
    slice.setAttribute("aria-label", `${label}: ${count} games`);
    offset += (count / total) * 100;
    svg.append(slice);
  });
  const value = document.createElement("strong");
  value.className = "donut-total";
  value.textContent = total;
  const label = document.createElement("span");
  label.className = "donut-label";
  label.textContent = "games";
  const legend = document.createElement("div");
  legend.className = "chart-legend";
  outcomes.forEach(([outcome, count], index) => {
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("i");
    swatch.className = `legend-swatch donut-slice-${index % 3}`;
    item.append(swatch, `${outcome}: ${count}`);
    legend.append(item);
  });
  const center = document.createElement("div");
  center.className = "donut-center";
  center.append(value, label);
  const wrapper = document.createElement("div");
  wrapper.className = "donut-wrap";
  wrapper.append(svg, center);
  winnerChartElement.append(wrapper, legend);
}

function renderTurnChart(turnCounts) {
  turnChartElement.replaceChildren();
  const maximum = Math.max(...turnCounts.map(([, count]) => count));
  turnCounts.forEach(([turns, count]) => {
    const group = document.createElement("div");
    group.className = "turn-bar-group";
    group.setAttribute("aria-label", `${turns} turns: ${count} games`);
    const value = document.createElement("span");
    value.className = "turn-bar-value";
    value.textContent = count;
    const bar = document.createElement("div");
    bar.className = "turn-bar";
    bar.style.height = `${Math.max((count / maximum) * 100, 5)}%`;
    const label = document.createElement("span");
    label.className = "turn-bar-label";
    label.textContent = turns;
    group.append(value, bar, label);
    turnChartElement.append(group);
  });
}

function setMode(mode) {
  const replayMode = mode === "replay";
  playPanel.hidden = replayMode;
  replayPanel.hidden = !replayMode;
  playTab.classList.toggle("active", !replayMode);
  replayTab.classList.toggle("active", replayMode);
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
    renderPlay();
  }
}

async function playMove(column) {
  if (!game || pending) return;
  pending = true;
  renderPlay();
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
    renderPlay();
  }
}

async function loadReplay(file) {
  replayStatusElement.textContent = "Loading match log...";
  try {
    const archive = file.name.toLowerCase().endsWith(".zip");
    const request = archive
      ? {
          url: "/api/replays/archive",
          body: { archive_base64: bytesToBase64(new Uint8Array(await file.arrayBuffer())) },
        }
      : { url: "/api/replays", body: { log: JSON.parse(await file.text()) } };
    const response = await fetch(request.url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request.body),
    });
    replay = await response.json();
    if (!response.ok) throw new Error(replay.detail);
    replayStatusElement.textContent = `${replay.games.length} games loaded`;
    await selectGame(0);
  } catch (error) {
    replayStatusElement.textContent = error.message;
  }
}

function bytesToBase64(bytes) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

async function selectGame(gameIndex) {
  selectedGameIndex = gameIndex;
  selectedMove = 0;
  await loadReplayMove();
}

async function loadReplayMove() {
  if (!replay) return;
  const response = await fetch(
    `/api/replays/${replay.match_id}/games/${selectedGameIndex}?move=${selectedMove}`,
  );
  replayState = await response.json();
  if (!response.ok) {
    replayStatusElement.textContent = replayState.detail;
    return;
  }
  renderReplay();
}

playTab.addEventListener("click", () => setMode("play"));
replayTab.addEventListener("click", () => setMode("replay"));
newGameElement.addEventListener("click", newGame);
logFileElement.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file) loadReplay(file);
});
dropZoneElement.addEventListener("dragover", (event) => {
  event.preventDefault();
  event.stopPropagation();
  dropZoneElement.classList.add("dragging");
});
dropZoneElement.addEventListener("dragleave", () => dropZoneElement.classList.remove("dragging"));
dropZoneElement.addEventListener("drop", (event) => {
  event.preventDefault();
  event.stopPropagation();
  dropZoneElement.classList.remove("dragging");
  const [file] = event.dataTransfer.files;
  if (file) loadReplay(file);
});
document.addEventListener("dragover", (event) => {
  if (event.dataTransfer.types.includes("Files")) event.preventDefault();
});
document.addEventListener("drop", (event) => {
  if (!event.dataTransfer.types.includes("Files")) return;
  event.preventDefault();
  const [file] = event.dataTransfer.files;
  if (file) {
    setMode("replay");
    loadReplay(file);
  }
});
previousMoveElement.addEventListener("click", () => {
  if (selectedMove > 0) {
    selectedMove -= 1;
    loadReplayMove();
  }
});
nextMoveElement.addEventListener("click", () => {
  if (replayState && selectedMove < replayState.total_moves) {
    selectedMove += 1;
    loadReplayMove();
  }
});
document.addEventListener("keydown", (event) => {
  if (replayPanel.hidden || !replayState) return;
  if (event.key === "ArrowLeft" && selectedMove > 0) {
    selectedMove -= 1;
    loadReplayMove();
  }
  if (event.key === "ArrowRight" && selectedMove < replayState.total_moves) {
    selectedMove += 1;
    loadReplayMove();
  }
});

newGame();
