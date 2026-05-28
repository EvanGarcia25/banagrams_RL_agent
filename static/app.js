const GRID_SIZE = 20;
let lastStateHash = null;
let isDraggingSlider = false;

const replayControls = () => document.getElementById("replay-controls");
const replaySlider = () => document.getElementById("replay-slider");
const replayToggleButton = () => document.getElementById("replay-toggle");
const replayPrevButton = () => document.getElementById("replay-prev");
const replayNextButton = () => document.getElementById("replay-next");

async function postJson(url, body = {}) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

function applyReplayUi(state) {
  const controls = replayControls();
  const slider = replaySlider();
  const toggleButton = replayToggleButton();
  const prevButton = replayPrevButton();
  const nextButton = replayNextButton();
  const isReplay = state.viewer_mode === "replay";

  controls.classList.toggle("hidden", !isReplay);

  if (!isReplay) {
    return;
  }

  const maxFrame = state.replay_max_frame_index ?? state.replay_step_count ?? 0;
  slider.max = String(maxFrame);
  slider.value = String(state.replay_step_index ?? 0);
  slider.disabled = false;

  const paused = !!state.replay_paused;
  toggleButton.textContent = paused ? "Play" : "Pause";
  prevButton.disabled = (state.replay_step_index ?? 0) <= 0;
  nextButton.disabled = (state.replay_step_index ?? 0) >= maxFrame;
}

async function replaySeek(frameIndex) {
  await postJson("/api/replay/seek", { frame_index: frameIndex });
  await fetchState(true);
}

async function replayStep(delta) {
  await postJson("/api/replay/step", { delta });
  await fetchState(true);
}

async function replayToggle() {
  await postJson("/api/replay/toggle", {});
  await fetchState(true);
}

async function fetchState(forceRender = false) {
  try {
    const res = await fetch("/api/state");
    const state = await res.json();
    const hash = JSON.stringify(state);
    if (forceRender || hash !== lastStateHash) {
      lastStateHash = hash;
      render(state);
    }
  } catch (_) {}
}

function render(state) {
  renderGrid(state.grid);
  renderHand(state.hand);
  renderWords(state.words, state.invalid_words);

  document.getElementById("bag-count").textContent  = state.bag_count;
  document.getElementById("hand-count").textContent = state.hand.length;
  document.getElementById("replay-step").textContent =
    state.replay_step_index !== undefined ? `${state.replay_step_index}/${state.replay_step_count}` : "—";
  document.getElementById("training-step").textContent =
    state.episode_run && state.episode_run.training_step !== undefined && state.episode_run.training_step !== null
      ? state.episode_run.training_step
      : "—";
  document.getElementById("last-action").textContent = state.last_action || "—";

  document.getElementById("win-banner").classList.toggle("hidden", !state.won);
  applyReplayUi(state);
}

function renderGrid(grid) {
  const container = document.getElementById("grid");
  container.innerHTML = "";
  container.style.gridTemplateColumns = `40px repeat(${GRID_SIZE}, var(--cell))`;
  container.style.gridTemplateRows = `40px repeat(${GRID_SIZE}, var(--cell))`;

  const corner = document.createElement("div");
  corner.className = "axis-corner";
  container.appendChild(corner);

  for (let c = 0; c < GRID_SIZE; c++) {
    const axis = document.createElement("div");
    axis.className = "axis-label axis-top";
    axis.textContent = c;
    axis.title = `Column ${c}`;
    container.appendChild(axis);
  }

  for (let r = 0; r < GRID_SIZE; r++) {
    const rowAxis = document.createElement("div");
    rowAxis.className = "axis-label axis-left";
    rowAxis.textContent = r;
    rowAxis.title = `Row ${r}`;
    container.appendChild(rowAxis);

    for (let c = 0; c < GRID_SIZE; c++) {
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.title = `(${r}, ${c})`;
      const letter = grid[r][c];
      if (letter) {
        cell.classList.add("filled");
        cell.textContent = letter;
      }
      container.appendChild(cell);
    }
  }
}

function renderHand(hand) {
  const container = document.getElementById("hand-bar");
  container.innerHTML = "";
  hand.forEach(letter => {
    const tile = document.createElement("div");
    tile.className = "tile";
    tile.textContent = letter;
    container.appendChild(tile);
  });
}

function renderWords(words, invalidWords) {
  const area = document.getElementById("words-area");
  if (!words.length) {
    area.innerHTML = `<span style="color:var(--text-dim)">None yet</span>`;
    return;
  }
  const invalidSet = new Set(invalidWords);
  area.innerHTML = words.map(w => {
    const cls = invalidSet.has(w) ? "word-invalid" : "word-valid";
    return `<span class="${cls}">${w}</span>`;
  }).join("  ");
}

document.getElementById("replay-slider").addEventListener("pointerdown", () => {
  isDraggingSlider = true;
});

document.getElementById("replay-slider").addEventListener("input", async event => {
  isDraggingSlider = true;
  await replaySeek(Number(event.target.value));
});

document.getElementById("replay-slider").addEventListener("pointerup", async event => {
  isDraggingSlider = false;
  await replaySeek(Number(event.target.value));
});

document.getElementById("replay-slider").addEventListener("change", async event => {
  if (!isDraggingSlider) {
    await replaySeek(Number(event.target.value));
  }
});

document.getElementById("replay-prev").addEventListener("click", async () => {
  await replayStep(-1);
});

document.getElementById("replay-next").addEventListener("click", async () => {
  await replayStep(1);
});

document.getElementById("replay-toggle").addEventListener("click", async () => {
  await replayToggle();
});

window.addEventListener("keydown", async event => {
  const isReplay = document.getElementById("replay-controls") && !document.getElementById("replay-controls").classList.contains("hidden");
  if (!isReplay) {
    return;
  }
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    await replayStep(-1);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    await replayStep(1);
  }
});

fetchState();
setInterval(fetchState, 500);
