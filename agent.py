import requests
import time
import os
import numpy as np
from collections import deque

# =============================
# API SETTINGS
# =============================
API_URL = os.getenv("API_URL")
API_KEY = os.getenv("TEAM_API_KEY")
if not API_URL or not API_KEY:
    raise Exception("API_URL or TEAM_API_KEY not set")

HEADERS = {"X-API-Key": API_KEY}

# =============================
# DATA STORAGE (capped deques — no manual slicing)
# =============================
MAX_HISTORY  = 1000
closes       = deque(maxlen=MAX_HISTORY)
volumes      = deque(maxlen=MAX_HISTORY)
rolling_vols = deque(maxlen=MAX_HISTORY)

# =============================
# POSITION STATE
# =============================
in_position     = False
entry_price     = 0.0
entry_qty       = 0
peak_unrealized = 0.0
hold_ticks      = 0
partial_sold    = False

# =============================
# STRATEGY CONSTANTS
# (only things that never need to change)
# =============================
VOL_PERIOD        = 5
Z_PERIOD          = 20
VOL_RATIO_PERIOD  = 10
MAX_HOLD_TICKS    = 20
MAX_RISK_PCT      = 0.25        # 25% of net worth per trade
CASH_LIMIT_PCT    = 0.30        # max 30% of cash per trade
MIN_PEAK_TO_TRAIL = 0.030       # trail only after +0.030% gain
TRAIL_ATR_MULT    = 3.5         # wide enough to survive tick noise
EMERGENCY_Z       = 1.8         # bypass all filters at extreme z
VOL_RATIO_MIN     = 0.25        # only block on extremely thin volume

# =============================
# PERFORMANCE TRACKER
# =============================
trade_log = []   # list of dicts: {entry, exit, pct, reason}

def log_trade(entry_p, exit_p, reason):
    pct = (exit_p - entry_p) / entry_p * 100
    trade_log.append({"entry": entry_p, "exit": exit_p, "pct": pct, "reason": reason})
    wins  = sum(1 for t in trade_log if t["pct"] > 0)
    total = len(trade_log)
    avg   = np.mean([t["pct"] for t in trade_log]) if trade_log else 0
    print(f"  ── TRADE #{total} | {'WIN' if pct > 0 else 'LOSS'} {pct:+.4f}% | "
          f"Win rate: {wins}/{total} | Avg: {avg:+.4f}% | Reason: {reason}")


# =============================
# DYNAMIC PARAMETERS
# (recalculated every tick from real market data)
# =============================
def compute_dynamic_params():
    """
    TP and SL scaled to actual recent price range.
    In tight chop (range=0.15%), TP~0.07%, SL~0.08%.
    In wider moves (range=0.40%), TP~0.20%, SL~0.22%.
    Always reachable, always proportional.
    """
    if len(closes) < 22:
        return 0.055, -0.065
    arr          = list(closes)
    recent_range = max(arr[-20:]) - min(arr[-20:])
    range_pct    = (recent_range / arr[-1]) * 100
    profit_target = max(0.040, round(range_pct * 0.50, 4))
    stop_loss     = -max(0.045, round(range_pct * 0.55, 4))
    return profit_target, stop_loss


def compute_atr(n=10):
    """
    ATR over n bars as % of price.
    n=10 for stability (5-bar ATR is too noisy on 10s ticks).
    """
    arr = list(closes)
    if len(arr) < n + 1:
        return 0.005
    moves = [abs(arr[-i] - arr[-i - 1]) for i in range(1, n + 1)]
    return (np.mean(moves) / arr[-1]) * 100


def detect_regime():
    """
    CHOP  = fast MA ≈ slow MA  (price going nowhere)
    TREND = fast MA meaningfully above/below slow MA
    """
    arr = list(closes)
    if len(arr) < 25:
        return "chop"
    ma_fast = np.mean(arr[-5:])
    ma_slow = np.mean(arr[-20:])
    sep = abs(ma_fast - ma_slow) / ma_slow
    return "trend" if sep > 0.0008 else "chop"


# =============================
# API FUNCTIONS
# =============================
def get_price():
    try:
        r = requests.get(f"{API_URL}/api/price", headers=HEADERS, timeout=6)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [price error] {e}")
        return None

def get_portfolio():
    try:
        r = requests.get(f"{API_URL}/api/portfolio", headers=HEADERS, timeout=6)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [portfolio error] {e}")
        return None

def get_history():
    try:
        r = requests.get(f"{API_URL}/api/history", headers=HEADERS, timeout=10)
        return r.json()
    except:
        return []

def buy(qty):
    try:
        r = requests.post(f"{API_URL}/api/buy", json={"quantity": qty},
                          headers=HEADERS, timeout=6)
        r.raise_for_status()
        print(f"  >>> BUY  {qty} shares ✓")
        return True
    except Exception as e:
        print(f"  [buy error] {e}")
        return False

def sell(qty):
    try:
        r = requests.post(f"{API_URL}/api/sell", json={"quantity": qty},
                          headers=HEADERS, timeout=6)
        r.raise_for_status()
        print(f"  >>> SELL {qty} shares ✓")
        return True
    except Exception as e:
        print(f"  [sell error] {e}")
        return False


# =============================
# ENTRY SIGNAL
# =============================
def get_signal():
    """
    Returns: (signal, z, vol_ratio, cur_vol, debug_dict)

    Signal logic:
    - EMERGENCY: |z| > 1.8  →  enter immediately, no vol filter
    - NORMAL:    |z| > threshold  AND  vol not suspiciously thin
                 Threshold is lower in CHOP (catch small dips)
                 Vol ratio only needs to be > 0.25 (not a spike, just not empty)
    """
    arr = list(closes)
    vol = list(volumes)

    if len(arr) < Z_PERIOD + 5:
        return "hold", 0, 0, 0, {}

    cur_vol   = np.std(arr[-VOL_PERIOD:])
    vol_ratio = (
        vol[-1] / np.mean(vol[-VOL_RATIO_PERIOD:])
        if len(vol) >= VOL_RATIO_PERIOD else 1.0
    )

    mean_z = np.mean(arr[-Z_PERIOD:])
    std_z  = np.std(arr[-Z_PERIOD:])
    z      = (arr[-1] - mean_z) / std_z if std_z > 0 else 0

    # Emergency bypass
    if abs(z) > EMERGENCY_Z:
        return ("buy" if z < 0 else "sell"), z, vol_ratio, cur_vol, {"trigger": "EMERGENCY"}

    regime = detect_regime()

    # Adaptive vol threshold from real history
    rv = list(rolling_vols)
    if len(rv) > 50:
        low_vol_thresh = np.percentile(rv[-300:], 40)
    elif len(rv) > 5:
        low_vol_thresh = np.percentile(rv, 60)
    else:
        low_vol_thresh = cur_vol * 2.0   # bootstrap: always passable

    z_threshold = 0.50 if regime == "chop" else 0.65

    low_vol          = cur_vol < low_vol_thresh
    extreme_z        = abs(z) > z_threshold
    reasonable_vol   = vol_ratio >= VOL_RATIO_MIN

    dbg = {
        "regime": regime,
        "z_thresh": z_threshold,
        "low_vol": low_vol,
        "extreme_z": extreme_z,
        "reasonable_vol": reasonable_vol,
        "low_vol_thresh": low_vol_thresh,
    }

    if low_vol and extreme_z and reasonable_vol:
        return ("buy" if z < 0 else "sell"), z, vol_ratio, cur_vol, dbg

    return "hold", z, vol_ratio, cur_vol, dbg


# =============================
# EXIT LOGIC
# =============================
def check_exit(unrealized, peak_unr, ticks, atr, profit_target, stop_loss):
    """
    Priority order:
    1. Take profit  (full TP)
    2. Stop loss    (hard floor)
    3. Trailing stop (only after MIN_PEAK_TO_TRAIL, wide ATR buffer)
    4. Max hold timer (always fires — no conditions)
    """
    if unrealized >= profit_target:
        return True, "TAKE_PROFIT"

    if unrealized <= stop_loss:
        return True, "STOP_LOSS"

    if peak_unr >= MIN_PEAK_TO_TRAIL:
        trail_level = peak_unr - (atr * TRAIL_ATR_MULT)
        if unrealized < trail_level:
            return True, f"TRAILING_STOP(trail={trail_level:+.4f}%)"

    if ticks >= MAX_HOLD_TICKS:
        return True, "MAX_HOLD_TIMER"

    return False, ""


# =============================
# POSITION SIZING
# =============================
def compute_qty(price, net_worth, cash, stop_loss_pct):
    """
    Risk-based sizing: lose at most MAX_RISK_PCT of net worth if SL hits.
    Also capped at CASH_LIMIT_PCT of available cash.
    Minimum 1 share.
    """
    risk_per_share = price * (abs(stop_loss_pct) / 100)
    if risk_per_share <= 0:
        return 1
    qty = int((net_worth * MAX_RISK_PCT) / risk_per_share)
    qty = min(qty, int(cash * CASH_LIMIT_PCT / price))
    return max(1, qty)


# =============================
# BOOTSTRAP
# =============================
print("=" * 70)
print("  AGENT v8 — FINAL | ADAPTIVE EXITS | CHOP-OPTIMISED | BUG-FREE")
print(f"  Trail: >{MIN_PEAK_TO_TRAIL}% peak | ATR×{TRAIL_ATR_MULT} | "
      f"Vol min: {VOL_RATIO_MIN} | MaxHold: {MAX_HOLD_TICKS} ticks")
print("=" * 70)

hist = get_history()
if hist:
    for t in hist:
        c  = float(t.get("close",  0))
        vl = float(t.get("volume", 1000))
        closes.append(c)
        volumes.append(vl)
    # Pre-build rolling_vols so threshold is real from tick 1
    tmp = list(closes)
    for i in range(len(tmp)):
        start = max(0, i - VOL_PERIOD + 1)
        rv    = np.std(tmp[start:i + 1]) if i >= VOL_PERIOD - 1 else 0.0
        rolling_vols.append(rv)
    non_zero = sum(1 for v in rolling_vols if v > 0)
    print(f"  Bootstrap: {len(closes)} ticks | rolling_vols ready: {non_zero}\n")


# =============================
# MAIN LOOP
# =============================
try:
    while True:
        # ── 1. Fetch price ──────────────────────────────────────────────
        tick = get_price()
        if not tick:
            time.sleep(10)
            continue

        price = float(tick["close"])
        vol   = float(tick.get("volume", 1000))

        closes.append(price)
        volumes.append(vol)
        rolling_vols.append(
            np.std(list(closes)[-VOL_PERIOD:]) if len(closes) >= VOL_PERIOD else 0
        )

        # ── 2. Fetch portfolio ──────────────────────────────────────────
        port = get_portfolio()
        if not port:
            time.sleep(10)
            continue

        cash      = float(port["cash"])
        net_worth = float(port["net_worth"])
        pnl_pct   = float(port.get("pnl_pct", 0))

        # ── 3. Compute indicators ───────────────────────────────────────
        PROFIT_TARGET, STOP_LOSS = compute_dynamic_params()
        atr    = compute_atr(n=10)
        regime = detect_regime()

        # ── 4. Status line ──────────────────────────────────────────────
        pnl_arrow = "▲" if pnl_pct >= 0 else "▼"
        print(f"{'─' * 70}")
        print(
            f"  {price:.4f} | Net={net_worth:.2f} | "
            f"P/L={pnl_arrow}{abs(pnl_pct):.3f}% | "
            f"{regime.upper()} | ATR={atr:.5f}% | "
            f"TP={PROFIT_TARGET:.4f}% SL={STOP_LOSS:.4f}%"
        )

        # ── 5. POSITION MANAGEMENT ──────────────────────────────────────
        if in_position and entry_qty > 0:
            unrealized = (price - entry_price) / entry_price * 100
            hold_ticks += 1

            if unrealized > peak_unrealized:
                peak_unrealized = unrealized

            trail_display = (
                f"{peak_unrealized - atr * TRAIL_ATR_MULT:+.4f}%"
                if peak_unrealized >= MIN_PEAK_TO_TRAIL else "–"
            )

            print(
                f"  POSITION qty={entry_qty} | entry={entry_price:.4f} | "
                f"unr={unrealized:+.4f}% | peak={peak_unrealized:+.4f}% | "
                f"trail={trail_display} | tick {hold_ticks}/{MAX_HOLD_TICKS}"
            )

            # Partial exit at 50% of TP — lock half, let rest run
            if (not partial_sold
                    and unrealized >= PROFIT_TARGET * 0.50
                    and entry_qty >= 2):
                half = entry_qty // 2
                print(f"  ◑ PARTIAL PROFIT {unrealized:+.4f}% → selling {half} shares")
                if sell(half):
                    entry_qty    -= half
                    partial_sold  = True

            else:
                do_exit, reason = check_exit(
                    unrealized, peak_unrealized, hold_ticks,
                    atr, PROFIT_TARGET, STOP_LOSS
                )
                if do_exit:
                    print(f"  ✕ EXIT [{reason}] at {unrealized:+.4f}%")
                    if sell(entry_qty):
                        log_trade(entry_price, price, reason)
                        in_position     = False
                        entry_qty       = 0
                        peak_unrealized = 0.0
                        hold_ticks      = 0
                        partial_sold    = False
                else:
                    print(f"  ✓ HOLD")

            time.sleep(10)
            continue

        # ── 6. ENTRY ────────────────────────────────────────────────────
        signal, z, vol_ratio, cur_vol, dbg = get_signal()

        if signal == "buy" and not in_position and cash > price * 5:
            qty = compute_qty(price, net_worth, cash, STOP_LOSS)
            print(
                f"  ▶ ENTER LONG {qty} | z={z:+.3f} "
                f"vol_ratio={vol_ratio:.2f} | {dbg.get('trigger', 'signal')}"
            )
            if buy(qty):
                in_position     = True
                entry_price     = price
                entry_qty       = qty
                peak_unrealized = 0.0
                hold_ticks      = 0
                partial_sold    = False

        elif signal == "sell" and in_position and entry_qty > 0:
            print(f"  ▶ SIGNAL SELL | z={z:+.3f}")
            if sell(entry_qty):
                log_trade(entry_price, price, "SIGNAL_SELL")
                in_position     = False
                entry_qty       = 0
                peak_unrealized = 0.0
                hold_ticks      = 0
                partial_sold    = False

        else:
            # Transparent WAIT — shows exactly which condition blocked entry
            blockers = []
            if not dbg.get("extreme_z"):
                blockers.append(f"z={z:+.3f} need>{dbg.get('z_thresh', 0.5):.2f}")
            if not dbg.get("low_vol"):
                blockers.append(f"vol={cur_vol:.5f} not<thresh={dbg.get('low_vol_thresh', 0):.5f}")
            if not dbg.get("reasonable_vol"):
                blockers.append(f"vol_ratio={vol_ratio:.2f} need>{VOL_RATIO_MIN}")
            reason_str = " | ".join(blockers) if blockers else "no signal"
            print(f"  – WAIT [{reason_str}]")

        time.sleep(10)

except KeyboardInterrupt:
    print("\n\nAgent stopped.")
    if trade_log:
        wins = sum(1 for t in trade_log if t["pct"] > 0)
        avg  = np.mean([t["pct"] for t in trade_log])
        print(f"\n  Final: {len(trade_log)} trades | "
              f"{wins} wins | avg {avg:+.4f}%")
except Exception as e:
    print(f"Fatal: {e}")
    raise
