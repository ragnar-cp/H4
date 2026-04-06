"""
Grid Strategy — Live Engine (Vantage MT5 Direct)
=================================================
Uses the official MetaTrader5 Python library — FREE, no subscription.

SETUP:
1. Make sure Vantage MT5 is installed and logged in on your PC
2. Install dependencies:
       pip install MetaTrader5 pandas numpy
3. Fill in your MT5 login details in CONFIG below
4. Enable Algo Trading in MT5 (green triangle button)
5. Run:
       python 005_4hr_v3.py

# ===========================================================
# VERSION 3 — MAJOR CHANGES
# ===========================================================
#
#   4H TREND FILTER
#   ───────────────
#   • H4 bias is read from the CURRENTLY FORMING H4 candle,
#     updated every 300ms in real-time.
#   • H4 currently GREEN (close > open) → BUY only on M15.
#   • H4 currently RED   (close < open) → SELL only on M15.
#   • H4 currently Doji  (close == open) → skip all entries.
#
#   GRID LOGIC
#   ──────────
#   • Grids fire ONLY on price distance (every 15 pts from entry).
#   • No grid orders on candle close — purely price-based.
#   • No new entry while a basket is active.
#
#   COOLDOWN LOGIC
#   ──────────────
#   • After any trade exits → 0 bars cooldown.
#   • When a NEW H4 candle opens → 1 M15 bar cooldown.
#   • Daily SL hit → trading halted for remainder of day.
#
# ===========================================================
# CURRENT PARAMETERS
# ===========================================================
#
#   FIXED_LOT            = 0.05
#   GRID_LOT_MULTIPLIERS = [2, 4]
#       Entry  = 0.05
#       Grid 1 = 0.05 × 2 = 0.10
#       Grid 2 = 0.05 × 4 = 0.20
#       Total max exposure = 0.35 lots
#
#   SINGLE_TRADE_TP_PTS  = 10   (no-grid exit at 10 pts)
#   TARGET_PROFIT        = 50   (Grid 1 TP = $50)
#   TARGET_PROFIT_G2     = 20   (Grid 2 TP = $20)
#   DAILY_SL_PCT         = 0.30 (30% daily stop loss)
#   GRID_STEP            = 15   (grid spacing in pts)
#   COOLDOWN_BARS        = 0    (no cooldown after normal exit)
#   H4_COOLDOWN_BARS     = 1    (1 M15 bar cooldown on new H4 open)
#
# ===========================================================
# EXIT LOGIC PER GRID LEVEL
# ===========================================================
#
#   Entry only  → 10 pts profit     [TP (no grid)]
#   Grid 1 hit  → $50 profit        [TARGET PROFIT G1]
#   Grid 2 hit  → $20 profit        [TARGET PROFIT G2]
#   Grid 3 hit  → $0.01 profit      [BREAK-EVEN RECOVERY]
#   Any level   → Daily SL breach   [STOP LOSS]
#
# ===========================================================
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import pandas as pd
import numpy as np


# ============================================================
# CONFIG — FILL THESE IN
# ============================================================

MT5_LOGIN    = 24521770
MT5_PASSWORD = "IQz#*O8p"
MT5_SERVER   = "VantageInternational-Demo"
SYMBOL       = "XAUUSD"
PAPER_TRADE  = True

# ============================================================


# ---------------------------
# STRATEGY PARAMETERS
# ---------------------------

TIMEFRAME            = mt5.TIMEFRAME_M15
TIMEFRAME_H4         = mt5.TIMEFRAME_H4

GRID_STEP            = 15
SINGLE_TRADE_TP_PTS  = 10
TARGET_PROFIT        = 50
TARGET_PROFIT_G2     = 20
DAILY_SL_PCT         = 0.30

COOLDOWN_BARS        = 0
H4_COOLDOWN_BARS     = 1

EMA_PERIOD           = 200
ATR_PERIOD           = 14
GRID_LOT_MULTIPLIERS = [2, 4]
FIXED_LOT            = 0.05
INITIAL_CAPITAL      = 10000
STATE_FILE           = "grid_state.json"
WARMUP_BARS          = 250
MAGIC_NUMBER         = 20250101
ENTRY_GUARD_SECS     = 3


# ---------------------------
# LOGGING
# ---------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("grid_live.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ---------------------------
# HELPER — MT5 SERVER TIME
# ---------------------------

def get_server_time():
    try:
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick:
            utc_time = datetime.utcfromtimestamp(tick.time)
            return utc_time + timedelta(hours=3)
    except Exception:
        pass
    return datetime.now()


# ---------------------------
# INCREMENTAL INDICATORS
# ---------------------------

class IncrementalEMA:
    def __init__(self, period):
        self.period = period
        self.k      = 2 / (period + 1)
        self.value  = None
        self.buffer = []

    def update(self, price):
        if self.value is None:
            self.buffer.append(price)
            if len(self.buffer) >= self.period:
                self.value = sum(self.buffer) / len(self.buffer)
        else:
            self.value = price * self.k + self.value * (1 - self.k)
        return self.value

    def is_ready(self):
        return self.value is not None


class IncrementalATR:
    def __init__(self, period):
        self.period     = period
        self.value      = None
        self.prev_close = None
        self.tr_buffer  = []

    def update(self, high, low, close):
        if self.prev_close is None:
            self.prev_close = close
            return None
        tr = max(high - low,
                 abs(high - self.prev_close),
                 abs(low  - self.prev_close))
        self.prev_close = close
        if self.value is None:
            self.tr_buffer.append(tr)
            if len(self.tr_buffer) >= self.period:
                self.value = sum(self.tr_buffer) / self.period
        else:
            self.value = (self.value * (self.period - 1) + tr) / self.period
        return self.value

    def is_ready(self):
        return self.value is not None


# ---------------------------
# STATE PERSISTENCE
# ---------------------------

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        log.info(f"Resumed state  Capital:{state['capital']:.2f}")
        return state
    log.info("Fresh start — no previous state found")
    return {
        "capital"             : INITIAL_CAPITAL,
        "basket_active"       : False,
        "direction"           : None,
        "entry_price"         : None,
        "entry_time"          : None,
        "entry_candle_time"   : None,
        "trades"              : [],
        "triggered_grids"     : [],
        "cooldown_counter"    : 0,
        "daily_sl_hit"        : False,
        "daily_sl_date"       : None,
        "capital_at_day_open" : INITIAL_CAPITAL,
        "current_day"         : None,
        "total_profit"        : 0.0,
        "total_loss"          : 0.0,
        "win_trades"          : 0,
        "loss_trades"         : 0,
        "trade_log"           : [],
        "open_tickets"        : [],
        "effective_grid_step" : GRID_STEP,
        "trade_lot"           : FIXED_LOT,
        "trend_tag"           : "",
        "grid_lots"           : [],
        "session_trades"      : 0,
        "session_profit"      : 0.0,
        "last_exit_reason"    : "",
        "last_profit"         : 0.0,
        "float_pnl"           : 0.0,
        "grids_hit"           : 0,
        "h4_bias"             : None,
        "h4_candle_time"      : None,
    }


# ---------------------------
# MT5 BROKER CLASS
# ---------------------------

class MT5Broker:

    def connect(self):
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        authorized = mt5.login(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
        if not authorized:
            raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
        info = mt5.account_info()
        log.info("Connected to MT5")
        log.info(f"  Account : {info.login}")
        log.info(f"  Name    : {info.name}")
        log.info(f"  Server  : {info.server}")
        log.info(f"  Balance : {info.balance}")
        log.info(f"  Leverage: 1:{info.leverage}")
        return True

    def disconnect(self):
        mt5.shutdown()
        log.info("MT5 disconnected")

    def get_balance(self):
        info = mt5.account_info()
        if info is None:
            raise RuntimeError(f"Cannot get balance: {mt5.last_error()}")
        return info.balance

    def get_latest_candles(self, symbol, timeframe, count):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count + 1)
        if rates is None:
            raise RuntimeError(f"Cannot get candles: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return [{"open": r["open"], "high": r["high"],
                 "low": r["low"], "close": r["close"], "time": r["time"]}
                for _, r in df.iterrows()]

    def get_symbol_info(self, symbol):
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Symbol {symbol} not found: {mt5.last_error()}")
        return info

    def place_order(self, direction, symbol, lot, comment=""):
        order_type  = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        symbol_info = self.get_symbol_info(symbol)
        price       = symbol_info.ask if direction == "BUY" else symbol_info.bid
        request = {
            "action"      : mt5.TRADE_ACTION_DEAL,
            "symbol"      : symbol,
            "volume"      : float(lot),
            "type"        : order_type,
            "price"       : price,
            "deviation"   : 20,
            "magic"       : MAGIC_NUMBER,
            "comment"     : comment,
            "type_time"   : mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            raise RuntimeError(f"Order failed  retcode:{retcode}  error:{mt5.last_error()}")
        log.info(f"  {direction} placed  lot:{lot}  ticket:{result.order}  fill:{result.price}")
        return result.order, result.price

    def close_position(self, ticket):
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            log.warning(f"  Position {ticket} not found — may already be closed")
            return
        position    = positions[0]
        close_type  = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY \
                      else mt5.ORDER_TYPE_BUY
        symbol_info = self.get_symbol_info(position.symbol)
        price       = symbol_info.bid if position.type == mt5.ORDER_TYPE_BUY \
                      else symbol_info.ask
        request = {
            "action"      : mt5.TRADE_ACTION_DEAL,
            "symbol"      : position.symbol,
            "volume"      : position.volume,
            "type"        : close_type,
            "position"    : ticket,
            "price"       : price,
            "deviation"   : 20,
            "magic"       : MAGIC_NUMBER,
            "comment"     : "close",
            "type_time"   : mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(f"  Close failed ticket:{ticket}  retcode:{result.retcode if result else 'None'}")
        else:
            log.info(f"  Closed ticket:{ticket}")

    def close_all_positions(self, tickets):
        for ticket in tickets:
            try:
                self.close_position(int(ticket))
            except Exception as e:
                log.error(f"  Failed to close {ticket}: {e}")


# ---------------------------
# GLOBAL OBJECTS
# ---------------------------

ema    = IncrementalEMA(EMA_PERIOD)
atr    = IncrementalATR(ATR_PERIOD)
state  = load_state()
broker = MT5Broker()

bot_start_time = datetime.now()

# Candle watcher timestamps
last_processed_candle    = None
last_processed_h4_candle = None

# Flag set by update_h4_bias() when a new H4 bar opens,
# read and cleared by on_new_candle() to apply cooldown
_h4_changed_flag = False

mt5_status = {
    "connected"      : False,
    "last_ping"      : None,
    "account_name"   : "",
    "account_number" : "",
    "server"         : "",
    "balance"        : 0.0,
    "equity"         : 0.0,
    "margin_free"    : 0.0,
    "ping_ms"        : None,
    "algo_trading"   : False,
    "error"          : "",
    "open_positions" : 0,
}


# ---------------------------
# MT5 CONNECTION CHECKER
# ---------------------------

def check_mt5_connection():
    global mt5_status
    try:
        t0   = time.time()
        tick = mt5.symbol_info_tick(SYMBOL)
        ping = round((time.time() - t0) * 1000, 1)

        if tick is None:
            mt5_status["connected"] = False
            mt5_status["error"]     = str(mt5.last_error())
            return

        info = mt5.account_info()
        if info is None:
            mt5_status["connected"] = False
            mt5_status["error"]     = str(mt5.last_error())
            return

        terminal = mt5.terminal_info()

        mt5_status["connected"]      = True
        mt5_status["last_ping"]      = datetime.now().strftime("%H:%M:%S")
        mt5_status["account_name"]   = info.name
        mt5_status["account_number"] = str(info.login)
        mt5_status["server"]         = info.server
        mt5_status["balance"]        = info.balance
        mt5_status["equity"]         = info.equity
        mt5_status["margin_free"]    = info.margin_free
        mt5_status["ping_ms"]        = ping
        mt5_status["algo_trading"]   = terminal.trade_allowed if terminal else False
        mt5_status["error"]          = ""

        positions = mt5.positions_get()
        mt5_status["open_positions"] = len(positions) if positions else 0

    except Exception as e:
        mt5_status["connected"] = False
        mt5_status["error"]     = str(e)


# ===========================================================
# H4 BIAS ENGINE — updates every 300ms in real-time
# ===========================================================

def update_h4_bias():
    """
    Reads the CURRENTLY FORMING H4 candle (candles[-1]) every 300ms.
    Updates state["h4_bias"] in real-time as the candle develops.

    H4 currently GREEN (close > open) → bias = "BUY"
    H4 currently RED   (close < open) → bias = "SELL"
    H4 currently Doji  (close == open) → bias = None

    Sets _h4_changed_flag = True when a new H4 candle timestamp is seen.
    on_new_candle() reads and clears this flag to apply the 1-bar cooldown.
    """
    global last_processed_h4_candle, _h4_changed_flag

    try:
        candles = broker.get_latest_candles(SYMBOL, TIMEFRAME_H4, 2)

        # candles[-1] = currently forming H4 candle (real-time)
        h4_live = candles[-1]
        h4_time = str(h4_live["time"])
        h4_o    = float(h4_live["open"])
        h4_c    = float(h4_live["close"])

        # Detect new H4 bar (timestamp changed)
        if h4_time != last_processed_h4_candle:
            last_processed_h4_candle = h4_time
            _h4_changed_flag         = True
            log.info(f"New H4 candle opened  time:{h4_time}  O:{h4_o}")

        # Real-time bias from forming candle
        if h4_c > h4_o:
            new_bias = "BUY"
        elif h4_c < h4_o:
            new_bias = "SELL"
        else:
            new_bias = None

        # Log only when direction flips
        if new_bias != state.get("h4_bias"):
            log.info(
                f"H4 bias updated: {state.get('h4_bias')} → {new_bias}  "
                f"O:{h4_o}  C:{h4_c}  [{h4_time}]"
            )

        state["h4_bias"]        = new_bias
        state["h4_candle_time"] = h4_time

    except Exception as e:
        log.error(f"H4 bias update error: {e}")


# ---------------------------
# EXIT LOGIC HELPER
# ---------------------------

def get_exit_decision(profit, trades, direction, entry_price, h, l, live=False, live_price=None):
    """
    Unified exit logic.

    Exit per grid level:
        len == 1  → Entry only  → TP at 10 pts         [TP (no grid)]
        len == 2  → Grid 1 hit  → TP at $50            [TARGET PROFIT G1]
        len == 3  → Grid 2 hit  → TP at $20            [TARGET PROFIT G2]
        len >= 4  → Grid 3 hit  → break-even ($0.01)   [BREAK-EVEN RECOVERY]
    """
    exit_trade  = False
    exit_reason = ""
    basket_sl   = -(state.get("capital_at_day_open", INITIAL_CAPITAL) * DAILY_SL_PCT)

    if profit <= basket_sl:
        exit_trade  = True
        exit_reason = "STOP LOSS"

    elif len(trades) == 1:
        if live:
            if direction == "BUY"  and live_price >= entry_price + SINGLE_TRADE_TP_PTS:
                exit_trade  = True
                exit_reason = "TP (no grid)"
                profit      = SINGLE_TRADE_TP_PTS * trades[0][1] * 100
            elif direction == "SELL" and live_price <= entry_price - SINGLE_TRADE_TP_PTS:
                exit_trade  = True
                exit_reason = "TP (no grid)"
                profit      = SINGLE_TRADE_TP_PTS * trades[0][1] * 100
        else:
            if direction == "BUY"  and h >= entry_price + SINGLE_TRADE_TP_PTS:
                exit_trade  = True
                exit_reason = "TP (no grid)"
                profit      = SINGLE_TRADE_TP_PTS * trades[0][1] * 100
            elif direction == "SELL" and l <= entry_price - SINGLE_TRADE_TP_PTS:
                exit_trade  = True
                exit_reason = "TP (no grid)"
                profit      = SINGLE_TRADE_TP_PTS * trades[0][1] * 100

    elif len(trades) == 2 and profit >= TARGET_PROFIT:
        exit_trade  = True
        exit_reason = "TARGET PROFIT G1"

    elif len(trades) == 3 and profit >= TARGET_PROFIT_G2:
        exit_trade  = True
        exit_reason = "TARGET PROFIT G2"

    elif len(trades) >= 4 and profit > 0:
        exit_trade  = True
        exit_reason = "BREAK-EVEN RECOVERY"

    return exit_trade, exit_reason, profit


# ---------------------------
# STRATEGY LOGIC
# ---------------------------

def on_new_candle(candle):
    """
    Called every M15 candle close.

    H4 bias is already up to date — updated every 300ms in main loop.
    Reads _h4_changed_flag to detect new H4 bar for 1-bar cooldown.

    Grid triggering is NOT done here.
    Grids fire purely on price in check_grids_realtime() every 300ms.
    No new entry is placed while a basket is active.
    """
    global state, _h4_changed_flag

    # Read and immediately clear the H4 changed flag
    h4_just_changed  = _h4_changed_flag
    _h4_changed_flag = False

    o = float(candle["open"])
    h = float(candle["high"])
    l = float(candle["low"])
    c = float(candle["close"])
    t = candle["time"]

    if isinstance(t, str):
        t = datetime.fromisoformat(t)

    bar_date_str = str(t.date())

    log.info(
        f"M15 Candle {t}  O:{o}  H:{h}  L:{l}  C:{c}  "
        f"H4_bias:{state.get('h4_bias', '?')}"
    )

    ema.update(c)
    atr.update(h, l, c)

    if not ema.is_ready():
        log.info("EMA warming up — skip")
        return

    # ---- Daily reset ----
    if bar_date_str != state["current_day"]:
        state["current_day"]         = bar_date_str
        state["capital_at_day_open"] = state["capital"]
        state["daily_sl_hit"]        = False
        state["session_trades"]      = 0
        state["session_profit"]      = 0.0
        log.info(
            f"NEW DAY {bar_date_str}  "
            f"Capital:{state['capital']:.2f}  "
            f"DailySL:{state['capital'] * DAILY_SL_PCT:.2f}"
        )

    # ---- Daily kill-switch ----
    if not state["daily_sl_hit"]:
        daily_loss = state["capital"] - state["capital_at_day_open"]
        if daily_loss <= -(state["capital_at_day_open"] * DAILY_SL_PCT):
            state["daily_sl_hit"]  = True
            state["daily_sl_date"] = bar_date_str
            log.warning(f"DAILY STOP LOSS HIT  Loss:{daily_loss:.2f}")
            if state["basket_active"] and state["open_tickets"]:
                broker.close_all_positions(state["open_tickets"])
                state["basket_active"]     = False
                state["open_tickets"]      = []
                state["float_pnl"]         = 0.0
                state["grids_hit"]         = 0
                state["entry_candle_time"] = None
                state["trades"]            = []
                state["triggered_grids"]   = []

    if state["daily_sl_hit"]:
        save_state(state)
        return

    # ================================================================
    # BASKET ACTIVE — only check SL at candle close, then return.
    # Grids and TP are handled every 300ms in check_grids_realtime().
    # Never open a new entry while basket is open.
    # ================================================================
    if state["basket_active"]:
        direction   = state["direction"]
        entry_price = state["entry_price"]
        trades      = [tuple(x) for x in state["trades"]]
        triggered   = set(state["triggered_grids"])

        if direction == "BUY":
            profit = sum((c - p) * lot * 100 for p, lot in trades)
        else:
            profit = sum((p - c) * lot * 100 for p, lot in trades)

        state["float_pnl"] = round(profit, 2)
        log.info(
            f"Basket active — Float P&L:{profit:.2f}  "
            f"Trades:{len(trades)}  Grids hit:{len(triggered)}"
        )

        # Only SL check at candle close
        basket_sl = -(state.get("capital_at_day_open", INITIAL_CAPITAL) * DAILY_SL_PCT)
        if profit <= basket_sl:
            broker.close_all_positions(state["open_tickets"])
            state["capital"]          += profit
            state["session_trades"]   += 1
            state["session_profit"]   += profit
            state["last_exit_reason"]  = "STOP LOSS"
            state["last_profit"]       = round(profit, 2)
            state["float_pnl"]         = 0.0
            state["grids_hit"]         = 0
            state["total_loss"]       += abs(profit)
            state["loss_trades"]      += 1
            log.warning(
                f"EXIT [STOP LOSS]  P&L:{profit:.2f}  "
                f"Capital:{state['capital']:.2f}"
            )
            state["trade_log"].append({
                "time"       : str(t),
                "direction"  : direction,
                "profit"     : round(profit, 2),
                "capital"    : round(state["capital"], 2),
                "exit_reason": "STOP LOSS",
                "grids_hit"  : len(triggered),
                "h4_bias"    : state.get("h4_bias"),
            })
            state["basket_active"]     = False
            state["trades"]            = []
            state["triggered_grids"]   = []
            state["open_tickets"]      = []
            state["entry_candle_time"] = None
            state["cooldown_counter"]  = COOLDOWN_BARS

        save_state(state)
        return   # always return — never fall through to entry logic

    # ================================================================
    # NO ACTIVE BASKET — apply cooldowns then check for new entry
    # ================================================================

    # H4 cooldown: 1 bar pause when new H4 candle just opened
    if h4_just_changed:
        state["cooldown_counter"] = H4_COOLDOWN_BARS
        log.info(f"New H4 candle — applying {H4_COOLDOWN_BARS}-bar cooldown")
        save_state(state)
        return

    # Normal cooldown
    if state["cooldown_counter"] > 0:
        state["cooldown_counter"] -= 1
        log.info(f"Cooldown: {state['cooldown_counter']} bars remaining")
        if state["cooldown_counter"] > 0:
            save_state(state)
            return

    # H4 bias check
    h4_bias = state.get("h4_bias")
    if h4_bias is None:
        log.info("H4 doji — no bias, skip entry")
        save_state(state)
        return

    # M15 direction must match H4 bias
    is_bull = c > o
    is_bear = c < o

    if not is_bull and not is_bear:
        log.info("M15 Doji — skip")
        save_state(state)
        return

    m15_direction = "BUY" if is_bull else "SELL"

    if m15_direction != h4_bias:
        log.info(
            f"M15 signal {m15_direction} ≠ H4 bias {h4_bias} — skip"
        )
        save_state(state)
        return

    # ---- Place entry ----
    try:
        state["capital"] = broker.get_balance()
    except Exception as e:
        log.warning(f"Balance sync failed: {e}")

    trade_lot = FIXED_LOT
    grid_lots = [trade_lot * m for m in GRID_LOT_MULTIPLIERS]
    direction = h4_bias
    trend_tag = f"H4-{h4_bias}"

    log.info(
        f"{direction} [{trend_tag}] entry  "
        f"candle close:{c}  EMA:{ema.value:.2f}"
    )

    ticket, fill_price = broker.place_order(
        direction, SYMBOL, trade_lot, comment=f"Entry {trend_tag}"
    )

    state["basket_active"]       = True
    state["direction"]           = direction
    state["entry_price"]         = fill_price
    state["entry_time"]          = str(datetime.now())
    state["entry_candle_time"]   = str(t)
    state["trades"]              = [[fill_price, trade_lot]]
    state["triggered_grids"]     = []
    state["open_tickets"]        = [ticket]
    state["effective_grid_step"] = GRID_STEP
    state["trade_lot"]           = trade_lot
    state["trend_tag"]           = trend_tag
    state["grid_lots"]           = grid_lots
    state["float_pnl"]           = 0.0
    state["grids_hit"]           = 0

    save_state(state)


# ---------------------------
# WARMUP
# ---------------------------

def warmup():
    log.info(f"Warming up on {WARMUP_BARS} historical M15 bars...")
    candles = broker.get_latest_candles(SYMBOL, TIMEFRAME, WARMUP_BARS)
    for c in candles[:-1]:
        ema.update(c["close"])
        atr.update(c["high"], c["low"], c["close"])
    log.info(f"Warmup done  EMA:{ema.value:.2f}  ATR:{atr.value:.4f}")

    # Seed H4 bias from current forming candle
    log.info("Seeding H4 bias from current forming H4 candle...")
    update_h4_bias()
    log.info(f"H4 bias on startup: {state.get('h4_bias', '?')}")


# ---------------------------
# CANDLE WATCHER (M15)
# ---------------------------

def check_new_candle():
    """
    Called every 300ms. Detects when a new M15 candle closes by comparing
    candles[-2].time to the last processed timestamp.
    """
    global last_processed_candle

    try:
        candles   = broker.get_latest_candles(SYMBOL, TIMEFRAME, 3)
        closed    = candles[-2]
        cand_time = str(closed["time"])

        if cand_time == last_processed_candle:
            return

        last_processed_candle = cand_time
        log.info(f"New M15 candle closed: {cand_time} — processing signal")
        on_new_candle(closed)

    except Exception as e:
        log.error(f"Candle watch error: {e}", exc_info=True)


# ---------------------------
# REAL-TIME GRID CHECK
# ---------------------------

def check_grids_realtime():
    """
    Called every 300ms.

    Responsibilities:
      1. Detect manual closes.
      2. Trigger grid orders when live price crosses entry ± (N × GRID_STEP).
      3. Exit when TP conditions are met on live price.

    Grid orders fire purely on price distance — never on candle close.
    """
    global state

    if not state.get("basket_active"):
        return

    try:
        # ---- Manual close detector ----
        open_tickets = state.get("open_tickets", [])
        if open_tickets:
            still_open = []
            for ticket in open_tickets:
                pos = mt5.positions_get(ticket=int(ticket))
                if pos:
                    still_open.append(ticket)

            if len(still_open) == 0:
                log.warning(
                    f"MANUAL CLOSE DETECTED — "
                    f"all {len(open_tickets)} position(s) closed outside bot"
                )
                state["basket_active"]     = False
                state["trades"]            = []
                state["triggered_grids"]   = []
                state["open_tickets"]      = []
                state["entry_candle_time"] = None
                state["float_pnl"]         = 0.0
                state["grids_hit"]         = 0
                state["cooldown_counter"]  = COOLDOWN_BARS
                state["last_exit_reason"]  = "MANUAL CLOSE"
                state["last_profit"]       = 0.0
                save_state(state)
                log.warning("State reset — bot ready for next candle")
                return

            elif len(still_open) < len(open_tickets):
                log.warning(
                    f"PARTIAL MANUAL CLOSE — "
                    f"{len(open_tickets) - len(still_open)} position(s) closed outside bot"
                )
                state["open_tickets"] = still_open
                save_state(state)

        # ---- Entry guard ----
        entry_time = state.get("entry_time")
        if entry_time:
            entry_dt         = datetime.fromisoformat(str(entry_time))
            secs_since_entry = (datetime.now() - entry_dt).total_seconds()
            if secs_since_entry < ENTRY_GUARD_SECS:
                return

        symbol_info = broker.get_symbol_info(SYMBOL)
        direction   = state["direction"]
        live_price  = symbol_info.bid if direction == "BUY" else symbol_info.ask

        entry_price  = state["entry_price"]
        trades       = [tuple(x) for x in state["trades"]]
        triggered    = set(state["triggered_grids"])
        grid_lots    = state["grid_lots"]
        eff_step     = state["effective_grid_step"]
        new_grid_hit = False

        # ---- Grid trigger: price distance only ----
        for idx, lot in enumerate(grid_lots):
            if idx in triggered:
                continue
            level      = (idx + 1) * eff_step
            grid_price = entry_price - level if direction == "BUY" \
                         else entry_price + level
            hit = (direction == "BUY"  and live_price <= grid_price) or \
                  (direction == "SELL" and live_price >= grid_price)
            if hit:
                ticket, fill_price = broker.place_order(
                    direction, SYMBOL, lot, comment=f"Grid#{idx+1}"
                )
                trades.append((fill_price, lot))
                triggered.add(idx)
                state["open_tickets"].append(ticket)
                state["grids_hit"] = len(triggered)
                new_grid_hit       = True
                log.info(
                    f"GRID #{idx+1} triggered  "
                    f"live:{live_price}  fill:{fill_price}  "
                    f"grid_price:{grid_price}  Lot:{lot}"
                )

        # ---- Live P&L ----
        if direction == "BUY":
            profit = sum((live_price - p) * l * 100 for p, l in trades)
        else:
            profit = sum((p - live_price) * l * 100 for p, l in trades)

        state["float_pnl"] = round(profit, 2)

        if new_grid_hit:
            state["trades"]          = [list(x) for x in trades]
            state["triggered_grids"] = list(triggered)
            save_state(state)

        # ---- TP / SL exit check ----
        exit_trade, exit_reason, profit = get_exit_decision(
            profit, trades, direction, entry_price,
            h=None, l=None, live=True, live_price=live_price
        )

        if exit_trade:
            broker.close_all_positions(state["open_tickets"])
            state["capital"]          += profit
            state["session_trades"]   += 1
            state["session_profit"]   += profit
            state["last_exit_reason"]  = exit_reason
            state["last_profit"]       = round(profit, 2)
            state["float_pnl"]         = 0.0
            state["grids_hit"]         = 0
            state["basket_active"]     = False
            state["trades"]            = []
            state["triggered_grids"]   = []
            state["open_tickets"]      = []
            state["entry_candle_time"] = None
            state["cooldown_counter"]  = COOLDOWN_BARS

            if profit > 0:
                state["total_profit"] += profit
                state["win_trades"]   += 1
            else:
                state["total_loss"]   += abs(profit)
                state["loss_trades"]  += 1

            state["trade_log"].append({
                "time"       : str(datetime.now()),
                "direction"  : direction,
                "profit"     : round(profit, 2),
                "capital"    : round(state["capital"], 2),
                "exit_reason": exit_reason,
                "grids_hit"  : len(triggered),
                "h4_bias"    : state.get("h4_bias"),
            })

            log.info(
                f"REALTIME EXIT [{exit_reason}]  "
                f"P&L:{profit:.2f}  Capital:{state['capital']:.2f}"
            )
            save_state(state)

    except Exception as e:
        log.error(f"Realtime grid check error: {e}")


# ---------------------------
# DASHBOARD HELPERS
# ---------------------------

def get_market_session(server_time):
    h    = server_time.hour + server_time.minute / 60.0
    wday = server_time.weekday()

    asian    = 2.0  <= h < 10.0
    london   = 10.0 <= h < 18.0
    new_york = 15.0 <= h < 23.0

    weekend = (wday == 4 and h >= 23.0) or (wday == 5) or (wday == 6 and h < 2.0)

    if weekend:
        days_left = 2 if wday == 4 else (1 if wday == 5 else 0)
        target    = server_time.replace(hour=2, minute=0, second=0, microsecond=0)
        if days_left > 0:
            target += timedelta(days=days_left)
        elif server_time >= target:
            target += timedelta(days=7)
        diff     = target - server_time
        tot      = int(diff.total_seconds())
        opens_in = f"{tot // 3600}h {(tot % 3600) // 60:02d}m"
        return "CLOSED", "WEEKEND", opens_in

    if not (asian or london or new_york):
        target = server_time.replace(hour=2, minute=0, second=0, microsecond=0)
        if server_time >= target:
            target += timedelta(days=1)
        diff     = target - server_time
        tot      = int(diff.total_seconds())
        opens_in = f"{tot // 3600}h {(tot % 3600) // 60:02d}m"
        return "CLOSED", "DAILY CLOSE", opens_in

    if london and new_york:
        session = "LONDON + NEW YORK [OVERLAP]"
    elif london:
        session = "LONDON"
    elif new_york:
        session = "NEW YORK"
    elif asian:
        session = "ASIAN"
    else:
        session = "UNKNOWN"

    return "OPEN", session, ""


def get_uptime():
    delta = datetime.now() - bot_start_time
    total = int(delta.total_seconds())
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def get_time_in_trade():
    entry_time = state.get("entry_time")
    if not entry_time or not state.get("basket_active"):
        return None
    try:
        delta = datetime.now() - datetime.fromisoformat(str(entry_time))
        total = int(delta.total_seconds())
        h, m, s = total // 3600, (total % 3600) // 60, total % 60
        return f"{h}h {m:02d}m {s:02d}s" if h > 0 else f"{m}m {s:02d}s"
    except Exception:
        return None


def get_daily_sl_bar(daily_loss, daily_sl_limit, width=20):
    if daily_sl_limit <= 0:
        return ""
    pct    = min(abs(daily_loss) / daily_sl_limit, 1.0)
    filled = int(pct * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}]  {pct*100:.0f}%  (${abs(daily_loss):.0f} of ${daily_sl_limit:.0f} limit)"


# ---------------------------
# DASHBOARD
# ---------------------------

def print_dashboard():
    os.system('cls' if os.name == 'nt' else 'clear')

    now = get_server_time()

    # Next M15 candle countdown
    total_secs = now.minute * 60 + now.second
    next_secs  = None
    for t in [0, 15, 30, 45]:
        if t * 60 > total_secs:
            next_secs = t * 60 - total_secs
            break
    if next_secs is None:
        next_secs = (60 - now.minute) * 60 - now.second
    next_secs = max(0, next_secs)
    next_mins = next_secs // 60
    next_secs = next_secs % 60

    total     = state.get("win_trades", 0) + state.get("loss_trades", 0)
    win_rate  = (state.get("win_trades", 0) / total * 100) if total > 0 else 0.0
    net_pnl   = state.get("capital", INITIAL_CAPITAL) - INITIAL_CAPITAL
    daily_pnl = state.get("capital", INITIAL_CAPITAL) - state.get("capital_at_day_open", INITIAL_CAPITAL)
    daily_sl  = state.get("capital_at_day_open", INITIAL_CAPITAL) * DAILY_SL_PCT

    # H4 bias display
    h4_bias     = state.get("h4_bias")
    h4_time     = state.get("h4_candle_time", "?")
    h4_bias_str = h4_bias if h4_bias else "DOJI/NONE"
    h4_arrow    = "▲ BUY ONLY" if h4_bias == "BUY" else ("▼ SELL ONLY" if h4_bias == "SELL" else "── SKIP")

    if state.get("basket_active"):
        grids         = state.get("grids_hit", 0)
        max_grids     = len(GRID_LOT_MULTIPLIERS)
        basket_str    = f"{state.get('direction')} | Grids: {grids}/{max_grids} | Float: ${state.get('float_pnl', 0.0):+.2f}"
        entry_str     = f"Entry @ {state.get('entry_price', '?')}"
        time_in_trade = get_time_in_trade()
        tit_str       = f"Time in trade  : {time_in_trade}" if time_in_trade else ""
    else:
        last_reason = state.get("last_exit_reason", "")
        last_profit = state.get("last_profit", 0.0)
        last        = f"${last_profit:+.2f} [{last_reason}]" if last_reason else "None yet"
        basket_str  = f"WAITING  |  Last trade: {last}"
        cooldown    = state.get("cooldown_counter", 0)
        entry_str   = f"Cooldown: {cooldown} M15 bar(s)" if cooldown > 0 else "Ready to trade"
        tit_str     = ""

    daily_sl_status = "HIT" if state.get("daily_sl_hit") else ""

    if mt5_status["connected"]:
        conn_str  = "CONNECTED"
        algo_str  = "ON" if mt5_status["algo_trading"] else "OFF  <-- ENABLE IN MT5!"
        ping_str  = f"{mt5_status['ping_ms']} ms" if mt5_status["ping_ms"] else "?"
        acct_str  = f"{mt5_status['account_name']}  ({mt5_status['account_number']})"
        bal_str   = (f"${mt5_status['balance']:.2f}  |  "
                     f"Equity: ${mt5_status['equity']:.2f}  |  "
                     f"Free Margin: ${mt5_status['margin_free']:.2f}")
        last_ping = mt5_status["last_ping"] or "?"
        pos_str   = str(mt5_status["open_positions"])
    else:
        conn_str  = "DISCONNECTED  <-- MT5 NOT RUNNING OR NOT LOGGED IN"
        algo_str  = "UNKNOWN"
        ping_str  = "N/A"
        acct_str  = "N/A"
        bal_str   = "N/A"
        last_ping = "Never"
        pos_str   = "N/A"

    ema_str  = f"{ema.value:.2f}"  if ema.is_ready() else "Warming up..."
    atr_str  = f"{atr.value:.4f}" if atr.is_ready() else "Warming up..."
    mkt_status, mkt_session, mkt_opens_in = get_market_session(now)
    mkt_str  = (f"OPEN  |  Session: {mkt_session}" if mkt_status == "OPEN"
                else f"CLOSED  ({mkt_session})  |  Opens in: {mkt_opens_in}")
    uptime_str      = get_uptime()
    sl_bar          = (get_daily_sl_bar(daily_pnl, daily_sl) if daily_pnl < 0
                       else f"[{'░'*20}]  0%  ($0 of ${daily_sl:.0f} limit)")
    last_candle_str = last_processed_candle if last_processed_candle else "None yet"

    print("=" * 66)
    print(f"  GRID BOT LIVE v3  |  {now.strftime('%Y-%m-%d  %H:%M:%S')}  [Server Time]")
    print("=" * 66)
    print(f"  MT5 Status     : {conn_str}")
    print(f"  Account        : {acct_str}")
    print(f"  Server         : {mt5_status['server'] or MT5_SERVER}")
    print(f"  Algo Trading   : {algo_str}")
    print(f"  Ping           : {ping_str}  |  Last check: {last_ping}")
    if not mt5_status["connected"] and mt5_status["error"]:
        print(f"  Error          : {mt5_status['error']}")
    print(f"  Broker Balance : {bal_str}")
    print(f"  Open Positions : {pos_str}")
    print("-" * 66)
    print(f"  Market Status  : {mkt_str}")
    print(f"  Bot Uptime     : {uptime_str}")
    print("-" * 66)
    print(f"  ── H4 TREND FILTER (LIVE / REAL-TIME) ──────────────")
    print(f"  H4 Bias        : {h4_bias_str}  {h4_arrow}")
    print(f"  H4 Candle Time : {h4_time}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Next M15 in    : {next_mins:02d}:{next_secs:02d}")
    print(f"  Last M15       : {last_candle_str}")
    print(f"  EMA(200)       : {ema_str}  |  ATR(14): {atr_str}")
    print("-" * 66)
    print(f"  Bot Capital    : ${state.get('capital', INITIAL_CAPITAL):.2f}")
    print(f"  Net P&L        : ${net_pnl:+.2f}")
    print(f"  Today P&L      : ${daily_pnl:+.2f}  {'  DAILY SL HIT' if daily_sl_status else ''}")
    print(f"  Daily SL       : {sl_bar}")
    print(f"  Session trades : {state.get('session_trades', 0)}  |  Profit: ${state.get('session_profit', 0.0):+.2f}")
    print("-" * 66)
    print(f"  Total trades   : {total}  |  Win rate: {win_rate:.1f}%")
    print(f"  Total profit   : ${state.get('total_profit', 0.0):.2f}")
    print(f"  Total loss     : ${state.get('total_loss', 0.0):.2f}")
    print("-" * 66)
    print(f"  TP levels      : No grid:10pts | G1:$50 | G2:$20 | G3:B/E")
    print(f"  Basket         : {basket_str}")
    print(f"                   {entry_str}")
    if tit_str:
        print(f"  {tit_str}")
    print("=" * 66)


# ---------------------------
# MAIN
# ---------------------------

async def main():
    log.info("=" * 60)
    log.info("Grid Strategy Live v3 — Vantage MT5 Direct")
    log.info(f"Account    : {MT5_LOGIN}")
    log.info(f"Server     : {MT5_SERVER}")
    log.info(f"Symbol     : {SYMBOL}")
    log.info(f"Lot Size   : {FIXED_LOT}")
    log.info(f"Grid Step  : {GRID_STEP} pts")
    log.info(f"Daily SL   : {DAILY_SL_PCT*100:.0f}%")
    log.info(f"EMA Period : {EMA_PERIOD}")
    log.info(f"H4 Filter  : LIVE real-time — BUY when H4 green, SELL when H4 red")
    log.info(f"Cooldown   : 0 bars after trade, 1 bar on new H4 candle")
    log.info(f"TP Levels  : No grid:10pts | G1:$50 | G2:$20 | G3:B/E")
    log.info("=" * 60)

    broker.connect()
    warmup()

    while True:
        update_h4_bias()          # H4 bias updated every 300ms in real-time
        check_new_candle()        # fires on_new_candle() when M15 closes
        check_mt5_connection()    # update connection status panel
        check_grids_realtime()    # realtime TP/grid price checks every 300ms

        print_dashboard()

        await asyncio.sleep(0.3)


# ---------------------------
# ENTRY POINT
# ---------------------------

if __name__ == "__main__":
    asyncio.run(main())
