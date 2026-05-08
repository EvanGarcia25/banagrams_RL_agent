const boardEl = document.getElementById("board");
const handEl = document.getElementById("hand");
const bagCountEl = document.getElementById("bagCount");
const validWordsEl = document.getElementById("validWords");
const lastActionEl = document.getElementById("lastAction");
const messageEl = document.getElementById("message");

const placeForm = document.getElementById("placeForm");
const removeForm = document.getElementById("removeForm");
const dumpForm = document.getElementById("dumpForm");
const resetBtn = document.getElementById("resetBtn");
const validateBtn = document.getElementById("validateBtn");

function setMessage(payload, isError = false) {
  messageEl.textContent = JSON.stringify(payload, null, 2);
  messageEl.classList.toggle("error", isError);
}

function renderBoard(grid) {
  boardEl.innerHTML = "";
  for (let r = 0; r < grid.length; r += 1) {
    for (let c = 0; c < grid[r].length; c += 1) {
      const cell = document.createElement("div");
      cell.className = "cell";
      if (grid[r][c]) {
        cell.classList.add("filled");
        cell.textContent = grid[r][c];
      }
      cell.title = `(${r}, ${c})`;
      boardEl.appendChild(cell);
    }
  }
}

function renderState(state) {
  renderBoard(state.grid);
  handEl.textContent = state.hand.join(" ");
  bagCountEl.textContent = state.tile_bag_count;
  validWordsEl.textContent = state.valid_words.length ? state.valid_words.join(", ") : "none";
  lastActionEl.textContent = state.last_action || "none";
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  const body = await response.json();
  if (!response.ok) {
    throw body;
  }
  return body;
}

async function loadState() {
  try {
    const state = await fetchJSON("/api/state");
    renderState(state);
    setMessage({ info: "State loaded" });
  } catch (err) {
    setMessage(err, true);
  }
}

async function dispatchAction(action) {
  try {
    const result = await fetchJSON("/api/action", {
      method: "POST",
      body: JSON.stringify(action),
    });
    renderState(result.state);
    setMessage(result);
  } catch (err) {
    const fallbackState = await fetchJSON("/api/state");
    renderState(fallbackState);
    setMessage(err, true);
  }
}

placeForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const action = {
    type: "place",
    letter: document.getElementById("placeLetter").value,
    row: Number(document.getElementById("placeRow").value),
    col: Number(document.getElementById("placeCol").value),
  };
  await dispatchAction(action);
});

removeForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const action = {
    type: "remove",
    row: Number(document.getElementById("removeRow").value),
    col: Number(document.getElementById("removeCol").value),
  };
  await dispatchAction(action);
});

dumpForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const action = {
    type: "dump",
    letter: document.getElementById("dumpLetter").value,
  };
  await dispatchAction(action);
});

resetBtn.addEventListener("click", async () => {
  try {
    const state = await fetchJSON("/api/reset", { method: "POST" });
    renderState(state);
    setMessage({ info: "Game reset" });
  } catch (err) {
    setMessage(err, true);
  }
});

validateBtn.addEventListener("click", async () => {
  try {
    const result = await fetchJSON("/api/validate", { method: "POST" });
    setMessage(result, result.invalid_words && result.invalid_words.length > 0);
  } catch (err) {
    setMessage(err, true);
  }
});

loadState();
