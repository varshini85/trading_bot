#!/usr/bin/env python3
"""
bot.py - Fixed Binance Futures Testnet Bot (MARKET / LIMIT / TWAP)
This version strictly aligns LIMIT prices to tickSize and bumps min_allowed by one tick
to avoid -4014: "Price not increased by tick size".

Requirements:
    pip install python-binance python-dotenv

.env:
    BINANCE_API_KEY
    BINANCE_API_SECRET
    (optional) BINANCE_FUTURES_URL
"""
import os
import argparse
import logging
import time
from logging.handlers import RotatingFileHandler
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
from dotenv import load_dotenv

from binance import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException

# increase precision
getcontext().prec = 28

load_dotenv()

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("basicbot")
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
logger.addHandler(ch)

fh = RotatingFileHandler(os.path.join(LOG_DIR, "basicbot.log"), maxBytes=5_000_000, backupCount=3)
fh.setLevel(logging.DEBUG)
fh.setFormatter(fmt)
logger.addHandler(fh)


class BasicBot:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True, futures_url: str | None = None):
        if not api_key or not api_secret:
            raise ValueError("API key/secret must be provided")
        self.client = Client(api_key, api_secret, testnet=testnet)
        env_url = futures_url or os.getenv("BINANCE_FUTURES_URL")
        if env_url:
            self.client.FUTURES_URL = env_url.rstrip("/")
            logger.info("Overrode FUTURES_URL -> %s", self.client.FUTURES_URL)
        logger.info("Binance client initialized (testnet=%s)", testnet)

    # ---------- helpers ----------
    def _get_exchange_info_for_symbol(self, symbol: str) -> dict:
        info = self.client.futures_exchange_info()
        for s in info.get("symbols", []):
            if s["symbol"].upper() == symbol.upper():
                fdict = {}
                for f in s.get("filters", []):
                    ftype = f.get("filterType") or f.get("type")
                    if ftype:
                        fdict[ftype] = f
                return fdict
        raise ValueError(f"Symbol {symbol} not found in exchangeInfo")

    def _get_mark_price(self, symbol: str) -> Decimal:
        mp = self.client.futures_mark_price(symbol=symbol)
        p = mp.get("markPrice") or mp.get("price")
        if p is None:
            raise RuntimeError("Could not fetch mark price for " + symbol)
        return Decimal(str(p))

    def _tick_quant_and_places(self, tick_size: str):
        """
        Returns (quant, places, tick_decimal)
        quant = Decimal quant used for Decimal.quantize (e.g. Decimal('0.01'))
        places = number of decimal places (int)
        tick_decimal = Decimal(tick_size)
        """
        tick = Decimal(str(tick_size))
        exp = tick.as_tuple().exponent
        if exp >= 0:
            quant = Decimal(1)
            places = 0
        else:
            quant = Decimal(1).scaleb(exp)
            places = -exp
        return quant, places, tick

    def _quantize_qty(self, qty: Decimal, step_size: str, rounding=ROUND_DOWN) -> Decimal:
        step = Decimal(str(step_size))
        exp = step.as_tuple().exponent
        if exp >= 0:
            quant = Decimal(1)
        else:
            quant = Decimal(1).scaleb(exp)
        return qty.quantize(quant, rounding=rounding)

    def _ceil_to_tick(self, price: Decimal, tick_size: str) -> Decimal:
        quant, _, _ = self._tick_quant_and_places(tick_size)
        n = (price / quant).to_integral_value(rounding=ROUND_UP)
        return (n * quant).quantize(quant, rounding=ROUND_DOWN)

    def _format_price_str(self, price: Decimal, tick_size: str) -> str:
        _, places, _ = self._tick_quant_and_places(tick_size)
        if places == 0:
            return format(price.quantize(Decimal(1)), 'f')
        fmt = "{0:." + str(places) + "f}"
        return fmt.format(price)

    # ---------- notional helper ----------
    def compute_min_qty_for_notional(self, symbol: str, target_notional: Decimal = Decimal("100")) -> Decimal:
        filters = self._get_exchange_info_for_symbol(symbol)
        price = self._get_mark_price(symbol)
        lot = filters.get("LOT_SIZE") or filters.get("MARKET_LOT_SIZE")
        if not lot:
            for f in filters.values():
                if "stepSize" in f:
                    lot = f
                    break
        if not lot or "stepSize" not in lot:
            raise RuntimeError("Could not find stepSize for symbol")
        step_size = lot["stepSize"]
        raw_qty = (target_notional / price)
        qty = self._quantize_qty(raw_qty, step_size, rounding=ROUND_DOWN)
        if qty == Decimal("0"):
            qty = Decimal(str(step_size))
        return qty

    # ---------- validate & adjust ----------
    def validate_and_adjust_price_qty(self, symbol: str, side: str, qty: Decimal | None, price: Decimal | None,
                                      target_notional: Decimal = Decimal("100")) -> (Decimal, Decimal | None):
        symbol = symbol.upper()
        side = side.upper()
        filters = self._get_exchange_info_for_symbol(symbol)

        pf = filters.get("PRICE_FILTER", {})
        tick_size = pf.get("tickSize") or pf.get("tick_size") or pf.get("tick")
        min_price_filter = pf.get("minPrice")
        max_price_filter = pf.get("maxPrice")

        lot = filters.get("LOT_SIZE") or filters.get("MARKET_LOT_SIZE")
        if not lot:
            for f in filters.values():
                if "stepSize" in f:
                    lot = f
                    break
        step_size = lot.get("stepSize") if lot else None
        if step_size is None:
            raise RuntimeError("Could not find stepSize for symbol")

        mark_price = self._get_mark_price(symbol)
        logger.debug("mark_price for %s = %s", symbol, mark_price)

        final_price = None

        # Price adjustment for LIMIT orders
        if price is not None:
            if not tick_size:
                raise RuntimeError("tickSize missing from PRICE_FILTER; cannot quantize price")

            # start by aligning price down to tick (safe baseline)
            quant, places, tick = self._tick_quant_and_places(tick_size)
            price_q_down = price.quantize(quant, rounding=ROUND_DOWN)

            # determine min_allowed: prefer minPrice filter, then for SELL ensure >= mark_price
            min_allowed = Decimal("-Infinity")
            if min_price_filter is not None:
                try:
                    min_allowed = Decimal(str(min_price_filter))
                except Exception:
                    min_allowed = Decimal("-Infinity")
            if side == "SELL" and mark_price > min_allowed:
                min_allowed = mark_price

            # If the quantized price is below min_allowed, we bump min_allowed by 1 tick and ceil
            if price_q_down < min_allowed:
                bumped = (min_allowed + tick)  # add one tick to guarantee strictly increased
                adjusted_price = self._ceil_to_tick(bumped, tick_size)
                logger.info("Adjusted LIMIT price from %s to %s (bumped min_allowed %s by tick %s)",
                            price, adjusted_price, min_allowed, tick)
                price_q = adjusted_price
            else:
                # ensure price is exactly on a tick and not smaller than original price
                price_q = self._ceil_to_tick(price_q_down, tick_size)

            # respect maxPrice
            if max_price_filter is not None:
                try:
                    maxp = Decimal(str(max_price_filter))
                    if price_q > maxp:
                        price_q = price_q.quantize(quant, rounding=ROUND_DOWN)
                        logger.info("Adjusted LIMIT price down to maxPrice %s", price_q)
                except Exception:
                    pass

            final_price = price_q

        # Qty handling (ensure step size, and notional)
        if qty is not None:
            qty_q = self._quantize_qty(qty, step_size, rounding=ROUND_DOWN)
        else:
            qty_q = None

        effective_price = final_price if final_price is not None else mark_price

        if qty_q is None:
            raw_qty = (target_notional / effective_price)
            qty_q = self._quantize_qty(raw_qty, step_size, rounding=ROUND_DOWN)
            if qty_q == Decimal("0"):
                qty_q = Decimal(str(step_size))
            logger.info("Computed qty=%s to meet notional %s with price %s", qty_q, target_notional, effective_price)
        else:
            notional = (effective_price * qty_q)
            if notional < target_notional:
                needed = (target_notional / effective_price)
                new_qty = self._quantize_qty(needed, step_size, rounding=ROUND_UP)
                if new_qty == Decimal("0"):
                    new_qty = Decimal(str(step_size))
                logger.info("Qty %s produces notional %s < %s; increasing to %s", qty_q, notional, target_notional, new_qty)
                qty_q = new_qty

        return qty_q, final_price

    # ---------- place order (with test validation) ----------
    def place_futures_order(self, symbol: str, side: str, order_type: str,
                            quantity: float | None = None, price: float | None = None,
                            auto_fix: bool = True, target_notional: Decimal = Decimal("100"),
                            time_in_force: str = "GTC", **kwargs):
        symbol = symbol.upper()
        side = side.upper()
        order_type = order_type.upper()

        if side not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")
        if order_type not in ("MARKET", "LIMIT"):
            raise ValueError("order_type must be MARKET or LIMIT")

        qty_dec = Decimal(str(quantity)) if quantity is not None else None
        price_dec = Decimal(str(price)) if price is not None else None

        if order_type == "LIMIT" and price_dec is None:
            raise ValueError("LIMIT orders require a price.")

        if auto_fix:
            try:
                qty_dec, price_dec = self.validate_and_adjust_price_qty(
                    symbol, side, qty_dec, price_dec, target_notional=target_notional)
            except Exception as e:
                logger.error("Failed to auto-validate/adjust price & qty: %s", e)
                raise

        if qty_dec is None:
            raise ValueError("Quantity must be provided (or computable).")

        payload = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": str(qty_dec)
        }

        if order_type == "LIMIT":
            pf = self._get_exchange_info_for_symbol(symbol).get("PRICE_FILTER", {})
            tick_size = pf.get("tickSize") or pf.get("tick_size") or pf.get("tick")
            if not tick_size:
                price_str = format(price_dec, 'f')
            else:
                price_str = self._format_price_str(price_dec, tick_size)
            payload.update({"price": price_str, "timeInForce": time_in_force})

        logger.info("Final order payload (will test): %s", payload)

        # Try test order first (does not place) if available - safer
        try:
            if hasattr(self.client, "futures_create_test_order"):
                # note: some python-binance builds may not include it, so check first
                self.client.futures_create_test_order(**payload)
                logger.debug("futures_create_test_order succeeded for payload.")
            else:
                logger.debug("No futures_create_test_order available in python-binance; skipping test order.")
        except Exception as e:
            logger.error("Test order failed: %s", e)
            # bubble the error so user sees why payload would be rejected
            raise

        # Now send real order
        try:
            resp = self.client.futures_create_order(**payload)
            logger.info("Order placed: %s %s %s qty=%s price=%s", side, order_type, symbol, payload["quantity"], payload.get("price"))
            logger.debug("Order response: %s", resp)
            return resp
        except BinanceAPIException as e:
            logger.error("BinanceAPIException: %s", e)
            raise
        except BinanceOrderException as e:
            logger.error("BinanceOrderException: %s", e)
            raise
        except Exception as e:
            logger.exception("Unexpected error placing order")
            raise

    # ---------- TWAP ----------
    def submit_twap(self, symbol: str, side: str, total_qty: float, duration: int, slices: int,
                    order_kind: str = "MARKET", price: float | None = None, auto_fix: bool = True):
        if slices <= 0:
            raise ValueError("slices must be > 0")
        if duration < 0:
            raise ValueError("duration must be >= 0")
        order_kind = order_kind.upper()
        if order_kind not in ("MARKET", "LIMIT"):
            raise ValueError("order_kind must be MARKET or LIMIT")

        total_qty_dec = Decimal(str(total_qty))
        slice_raw = (total_qty_dec / Decimal(slices))
        delay = duration / max(1, slices)

        logger.info("TWAP: total=%s over %ss in %s slices (~%s per slice, delay=%ss)",
                    total_qty_dec, duration, slices, slice_raw, delay)

        results = []
        for idx in range(1, slices + 1):
            slice_qty = float(slice_raw)
            logger.info("[TWAP] placing slice %s/%s qty=%s kind=%s", idx, slices, slice_qty, order_kind)
            try:
                resp = self.place_futures_order(
                    symbol=symbol, side=side, order_type=order_kind, quantity=slice_qty,
                    price=price, auto_fix=auto_fix
                )
                logger.info("[TWAP] slice %s placed orderId=%s", idx, resp.get("orderId"))
                results.append(resp)
            except Exception as e:
                logger.exception("[TWAP] slice %s failed: %s", idx, e)
                results.append({"error": str(e)})
            if idx < slices:
                time.sleep(delay)
        return results


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Basic Binance Futures Testnet Bot (fixed)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--side", required=True, choices=["BUY", "SELL"])
    p.add_argument("--type", dest="order_type", required=True, choices=["MARKET", "LIMIT", "TWAP"])
    p.add_argument("--qty", type=float, required=True)
    p.add_argument("--price", type=float)
    p.add_argument("--duration", type=int, default=60)
    p.add_argument("--slices", type=int, default=5)
    p.add_argument("--no-auto-fix", dest="auto_fix", action="store_false")
    p.add_argument("--testnet", action="store_true", default=True)
    p.add_argument("--futures-url", type=str)
    return p.parse_args()


def main():
    args = parse_args()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        logger.error("Missing API keys")
        print("Set BINANCE_API_KEY and BINANCE_API_SECRET in .env")
        return

    bot = BasicBot(api_key, api_secret, testnet=args.testnet, futures_url=args.futures_url)

    try:
        if args.order_type == "TWAP":
            results = bot.submit_twap(
                symbol=args.symbol, side=args.side,
                total_qty=args.qty, duration=args.duration, slices=args.slices,
                order_kind="MARKET" if args.price is None else "LIMIT",
                price=args.price, auto_fix=args.auto_fix
            )
            print("TWAP results:")
            for r in results:
                print(r)
        else:
            resp = bot.place_futures_order(
                symbol=args.symbol, side=args.side,
                order_type=args.order_type, quantity=args.qty,
                price=args.price, auto_fix=args.auto_fix
            )
            print("Order Result JSON:")
            print(resp)
    except Exception as e:
        logger.error("Operation failed: %s", e)
        print("Operation failed. See logs for details.")


if __name__ == "__main__":
    main()
