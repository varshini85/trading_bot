import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from bot import BasicBot

load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
FUTURES_URL = os.getenv("BINANCE_FUTURES_URL")

if not API_KEY or not API_SECRET:
    raise RuntimeError("Set BINANCE_API_KEY and BINANCE_API_SECRET in .env")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["JSON_SORT_KEYS"] = False

bot = BasicBot(API_KEY, API_SECRET, testnet=True, futures_url=FUTURES_URL)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/place_order", methods=["POST"])
def api_place_order():
    data = request.get_json(force=True)
    symbol = data.get("symbol")
    side = data.get("side")
    order_type = data.get("type")
    qty = data.get("qty")
    price = data.get("price", None)

    if not symbol or not side or not order_type or qty is None:
        return jsonify({"error": "symbol, side, type and qty are required"}), 400

    try:
        resp = bot.place_futures_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=float(qty),
            price=float(price) if price is not None else None
        )
        return jsonify({"ok": True, "result": resp})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/open_orders", methods=["GET"])
def api_open_orders():
    symbol = request.args.get("symbol", "BTCUSDT")
    try:
        orders = bot.client.futures_get_open_orders(symbol=symbol)
        return jsonify({"ok": True, "open_orders": orders})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/position", methods=["GET"])
def api_position():
    symbol = request.args.get("symbol", "BTCUSDT")
    try:
        pos = bot.client.futures_position_information(symbol=symbol)
        return jsonify({"ok": True, "position": pos})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
