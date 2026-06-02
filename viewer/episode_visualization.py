from __future__ import annotations

import argparse
import threading
import time
from copy import deepcopy

from flask import Flask, jsonify, render_template, request

from .episode_log import episode_frames, load_episode
from game import BananagramsGame

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))
game = BananagramsGame()


class ViewerController:
    def __init__(self):
        self._lock = threading.RLock()
        self._mode = "live"
        self._episode: dict | None = None
        self._frames: list[dict] = []
        self._frame_index = 0
        self._step_delay = 0.75
        self._replay_paused = True
        self._thread: threading.Thread | None = None

    def set_live(self) -> None:
        with self._lock:
            self._mode = "live"
            self._episode = None
            self._frames = []
            self._frame_index = 0
            self._replay_paused = True

    def set_replay(self, episode: dict, *, step_delay: float = 0.75) -> None:
        with self._lock:
            self._mode = "replay"
            self._episode = episode
            self._frames = episode_frames(episode)
            self._frame_index = 0
            self._step_delay = step_delay
            self._replay_paused = True

    def start_replay(self) -> None:
        with self._lock:
            if self._mode != "replay":
                return
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._auto_advance, daemon=True)
            self._thread.start()

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._replay_paused = paused

    def toggle_paused(self) -> bool:
        with self._lock:
            self._replay_paused = not self._replay_paused
            return self._replay_paused

    def seek(self, frame_index: int) -> int:
        with self._lock:
            if self._mode != "replay" or not self._frames:
                return 0
            self._frame_index = max(0, min(frame_index, len(self._frames) - 1))
            return self._frame_index

    def step(self, delta: int) -> int:
        with self._lock:
            target_index = self._frame_index + delta
            return self.seek(target_index)

    def max_frame_index(self) -> int:
        with self._lock:
            return max(0, len(self._frames) - 1)

    def _auto_advance(self) -> None:
        while True:
            time.sleep(self._step_delay)
            with self._lock:
                if (
                    self._mode != "replay"
                    or self._replay_paused
                    or self._frame_index >= len(self._frames) - 1
                ):
                    break
                self._frame_index += 1

    def is_live(self) -> bool:
        with self._lock:
            return self._mode == "live"

    def get_state(self) -> dict:
        with self._lock:
            if self._mode == "live":
                return game.get_state()

            if not self._frames:
                return game.get_state()

            state = deepcopy(self._frames[self._frame_index])
            state["viewer_mode"] = "replay"
            state["replay_step_index"] = self._frame_index
            state["replay_step_count"] = max(0, len(self._frames) - 1)
            state["replay_total_frames"] = len(self._frames)
            state["replay_paused"] = self._replay_paused
            state["replay_max_frame_index"] = max(0, len(self._frames) - 1)
            state["episode_run"] = deepcopy((self._episode or {}).get("run", {}))
            state["episode_summary"] = deepcopy((self._episode or {}).get("summary", {}))
            return state


viewer = ViewerController()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def state():
    return jsonify(viewer.get_state())


def _live_only_response(message: str):
    return jsonify({"success": False, "message": message, "state": viewer.get_state()}), 409


@app.route("/api/place", methods=["POST"])
def place():
    if not viewer.is_live():
        return _live_only_response("Replay mode is read-only.")
    body = request.get_json(force=True)
    return jsonify(game.place(body["letter"], int(body["row"]), int(body["col"])))


@app.route("/api/remove", methods=["POST"])
def remove():
    if not viewer.is_live():
        return _live_only_response("Replay mode is read-only.")
    body = request.get_json(force=True)
    return jsonify(game.remove(int(body["row"]), int(body["col"])))


@app.route("/api/dump", methods=["POST"])
def dump():
    if not viewer.is_live():
        return _live_only_response("Replay mode is read-only.")
    body = request.get_json(force=True)
    return jsonify(game.dump(body["letter"]))


@app.route("/api/reset", methods=["POST"])
def reset():
    if not viewer.is_live():
        return _live_only_response("Replay mode is read-only.")
    body = request.get_json(silent=True) or {}
    seed = body.get("seed")
    game.reset(seed=seed)
    return jsonify(game.get_state())


@app.route("/api/replay/seek", methods=["POST"])
def replay_seek():
    if viewer.is_live():
        return _live_only_response("Replay controls are only available in replay mode.")
    body = request.get_json(force=True)
    frame_index = viewer.seek(int(body["frame_index"]))
    viewer.set_paused(True)
    return jsonify(viewer.get_state() | {"success": True, "message": "seek", "frame_index": frame_index})


@app.route("/api/replay/step", methods=["POST"])
def replay_step():
    if viewer.is_live():
        return _live_only_response("Replay controls are only available in replay mode.")
    body = request.get_json(force=True)
    delta = int(body.get("delta", 1))
    frame_index = viewer.step(delta)
    viewer.set_paused(True)
    return jsonify(viewer.get_state() | {"success": True, "message": "step", "frame_index": frame_index})


@app.route("/api/replay/toggle", methods=["POST"])
def replay_toggle():
    if viewer.is_live():
        return _live_only_response("Replay controls are only available in replay mode.")
    paused = viewer.toggle_paused()
    if not paused:
        viewer.start_replay()
    return jsonify(viewer.get_state() | {"success": True, "message": "toggle", "paused": paused})


@app.route("/api/replay/state")
def replay_state():
    return jsonify(viewer.get_state())


def configure_replay(episode_file: str, *, step_delay: float = 0.75) -> None:
    try:
        episode_data = load_episode(episode_file)
        viewer.set_replay(episode_data, step_delay=step_delay)
    except Exception as e:
        print(f"Error loading episode file '{episode_file}': {e}")
        print("Falling back to live mode.")


def run(*, episode_file: str | None = None, step_delay: float = 0.75) -> None:
    viewer.set_live()
    if episode_file:
        configure_replay(episode_file, step_delay=step_delay)
    app.run(debug=False, use_reloader=False, port=8080)


def start_background(*, episode_file: str | None = None, step_delay: float = 0.75) -> None:
    if episode_file:
        configure_replay(episode_file, step_delay=step_delay)
    t = threading.Thread(target=lambda: app.run(debug=False, use_reloader=False, port=8080), daemon=True)
    t.start()
    print("Display server running at http://localhost:8080")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Bananagrams visual server")
    parser.add_argument("--episode-file", type=str, default=None, help="Replay a saved episode log")
    parser.add_argument("--step-delay", type=float, default=0.75, help="Seconds between replay frames")
    args = parser.parse_args(argv)

    run(episode_file=args.episode_file, step_delay=args.step_delay)


if __name__ == "__main__":
    main()
