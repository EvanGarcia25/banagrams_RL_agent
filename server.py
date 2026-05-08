from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from bananagrams_rl import BananagramsEnv, BananagramsGame, Dictionary, GameConfig, InvalidAction


def create_app() -> Flask:
    app = Flask(__name__)

    custom_dict_file = Path("words.txt")
    if custom_dict_file.exists():
        dictionary = Dictionary.from_file(custom_dict_file)
    else:
        dictionary = Dictionary.from_system_or_fallback()

    game = BananagramsGame(
        dictionary=dictionary,
        config=GameConfig(grid_size=15, initial_hand_size=12),
        seed=42,
    )
    env = BananagramsEnv(game)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/state")
    def state():
        return jsonify(game.state())

    @app.post("/api/reset")
    def reset():
        return jsonify(env.reset())

    @app.post("/api/action")
    def action():
        payload = request.get_json(silent=True) or {}
        state, reward, done, info = env.step(payload)
        status = 200 if not info.get("error") else 400
        return jsonify({"state": state, "reward": reward, "done": done, "info": info}), status

    @app.post("/api/validate")
    def validate():
        valid_words, invalid_words = game.validate_board_words()
        return jsonify({"valid_words": valid_words, "invalid_words": invalid_words})

    @app.errorhandler(InvalidAction)
    def invalid_action(err: InvalidAction):
        return jsonify({"error": str(err)}), 400

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
