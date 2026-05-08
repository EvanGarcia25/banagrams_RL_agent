import threading
from flask import Flask, jsonify, render_template, request
from game import BananagramsGame

app = Flask(__name__)
game = BananagramsGame()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def state():
    return jsonify(game.get_state())


@app.route("/api/place", methods=["POST"])
def place():
    body = request.get_json(force=True)
    return jsonify(game.place(body["letter"], int(body["row"]), int(body["col"])))


@app.route("/api/remove", methods=["POST"])
def remove():
    body = request.get_json(force=True)
    return jsonify(game.remove(int(body["row"]), int(body["col"])))


@app.route("/api/dump", methods=["POST"])
def dump():
    body = request.get_json(force=True)
    return jsonify(game.dump(body["letter"]))


@app.route("/api/reset", methods=["POST"])
def reset():
    game.reset()
    return jsonify(game.get_state())


def run():
    app.run(debug=False, use_reloader=False, port=8080)


def start_background():
    t = threading.Thread(target=run, daemon=True)
    t.start()
    print("Display server running at http://localhost:8080")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=8080)
