const GRID_SIZE = 20;
let lastStateHash = null;

async function fetchState() {
  try {
    const res = await fetch("/api/state");
    const state = await res.json();
    const hash = JSON.stringify(state);
    if (hash !== lastStateHash) {
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
  document.getElementById("last-action").textContent = state.last_action || "—";

  document.getElementById("win-banner").classList.toggle("hidden", !state.won);
}

function renderGrid(grid) {
  const container = document.getElementById("grid");
  container.innerHTML = "";
  for (let r = 0; r < GRID_SIZE; r++) {
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

fetchState();
setInterval(fetchState, 500);
