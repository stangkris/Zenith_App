import numpy as np
import pandas as pd


def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["ema50"] = data["close"].ewm(span=50, adjust=False).mean()
    data["ema200"] = data["close"].ewm(span=200, adjust=False).mean()

    delta = data["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    data["rsi14"] = 100 - (100 / (1 + rs))
    data["rsi14"] = data["rsi14"].fillna(50)

    data["vol_sma20"] = data["volume"].rolling(20, min_periods=1).mean()
    data["atr14"] = _atr(data, 14)
    return data


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def detect_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> tuple[list[int], list[int]]:
    highs = []
    lows = []
    for i in range(left, len(df) - right):
        h = df["high"].iloc[i]
        l = df["low"].iloc[i]
        if h >= df["high"].iloc[i - left : i + right + 1].max():
            highs.append(i)
        if l <= df["low"].iloc[i - left : i + right + 1].min():
            lows.append(i)
    return highs, lows


def detect_bos(df: pd.DataFrame) -> list[dict]:
    swing_highs, _ = detect_swings(df)
    events: list[dict] = []
    for idx in range(1, len(df)):
        prior_highs = [h for h in swing_highs if h < idx]
        if not prior_highs:
            continue
        last_swing = prior_highs[-1]
        level = float(df["high"].iloc[last_swing])
        if df["close"].iloc[idx] > level and df["close"].iloc[idx - 1] <= level:
            events.append(
                {
                    "index": idx,
                    "swing_index": last_swing,
                    "level": level,
                    "direction": "bullish",
                }
            )
    return events


def detect_fvg(df: pd.DataFrame) -> list[dict]:
    zones: list[dict] = []
    for i in range(2, len(df)):
        high_2 = float(df["high"].iloc[i - 2])
        low_2 = float(df["low"].iloc[i - 2])
        high_0 = float(df["high"].iloc[i])
        low_0 = float(df["low"].iloc[i])

        # Bullish imbalance: current low is above the high 2 candles ago
        if low_0 > high_2:
            zones.append(
                {
                    "type": "bullish",
                    "start_index": i - 2,
                    "end_index": i,
                    "low": high_2,
                    "high": low_0,
                }
            )

        # Bearish imbalance
        if high_0 < low_2:
            zones.append(
                {
                    "type": "bearish",
                    "start_index": i - 2,
                    "end_index": i,
                    "low": high_0,
                    "high": low_2,
                }
            )
    return zones


def detect_order_blocks(df: pd.DataFrame, bos_events: list[dict], lookback: int = 20) -> list[dict]:
    zones: list[dict] = []
    for bos in bos_events:
        if bos["direction"] != "bullish":
            continue
        idx = bos["index"]
        start = max(1, idx - lookback)
        bearish_candidates = [i for i in range(start, idx) if df["close"].iloc[i] < df["open"].iloc[i]]
        if not bearish_candidates:
            continue
        ob_idx = bearish_candidates[-1]
        candle_open = float(df["open"].iloc[ob_idx])
        candle_close = float(df["close"].iloc[ob_idx])
        zone_high = max(candle_open, candle_close)
        zone_low = float(df["low"].iloc[ob_idx])
        zones.append(
            {
                "type": "bullish",
                "index": ob_idx,
                "trigger_index": idx,
                "low": zone_low,
                "high": zone_high,
            }
        )
    return zones


def bullish_candle_signal(df: pd.DataFrame, idx: int | None = None) -> bool:
    if len(df) < 2:
        return False
    i = len(df) - 1 if idx is None else idx
    if i < 1:
        return False

    o = float(df["open"].iloc[i])
    c = float(df["close"].iloc[i])
    h = float(df["high"].iloc[i])
    l = float(df["low"].iloc[i])

    prev_o = float(df["open"].iloc[i - 1])
    prev_c = float(df["close"].iloc[i - 1])

    body = abs(c - o)
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)

    bullish_engulfing = c > o and prev_c < prev_o and o <= prev_c and c >= prev_o
    bullish_pinbar = c > o and lower_wick >= 2 * max(body, 1e-9) and upper_wick <= max(body, 1e-9)

    return bullish_engulfing or bullish_pinbar


def price_in_zone(price: float, zone_low: float, zone_high: float, tolerance: float) -> bool:
    return (zone_low - tolerance) <= price <= (zone_high + tolerance)


def bounced_from_zone(df: pd.DataFrame, zone_low: float, zone_high: float, tolerance: float) -> bool:
    if len(df) < 2:
        return False
    prev_close = float(df["close"].iloc[-2])
    last_close = float(df["close"].iloc[-1])
    touched = (zone_low - tolerance) <= prev_close <= (zone_high + tolerance)
    bounced = last_close > prev_close and last_close > zone_high
    return touched and bounced


def analyze_smc(df: pd.DataFrame) -> dict:
    bos_events = detect_bos(df)
    fvg_zones = [z for z in detect_fvg(df) if z["type"] == "bullish"]
    ob_zones = detect_order_blocks(df, bos_events)

    last_price = float(df["close"].iloc[-1])
    atr = float(df["atr14"].iloc[-1]) if not np.isnan(df["atr14"].iloc[-1]) else last_price * 0.01
    tol = max(atr * 0.2, last_price * 0.002)

    candidate_zones = []
    if fvg_zones:
        candidate_zones.append(("FVG", fvg_zones[-1]))
    if ob_zones:
        candidate_zones.append(("OB", ob_zones[-1]))

    zone_pass = False
    active_zone = None
    zone_type = None
    for z_type, zone in candidate_zones:
        z_low = float(zone["low"])
        z_high = float(zone["high"])
        inside = price_in_zone(last_price, z_low, z_high, tol)
        bounce = bounced_from_zone(df, z_low, z_high, tol)
        if inside or bounce:
            zone_pass = True
            active_zone = zone
            zone_type = z_type
            break

    # Relaxed volume spike from 2x to 1.5x
    volume_spike = float(df["volume"].iloc[-1]) > 1.5 * float(df["vol_sma20"].iloc[-1])
    candle_reversal_in_zone = zone_pass and bullish_candle_signal(df)
    # Allow trigger if EITHER candle signal OR volume spike (was previously combined or implicit)
    trigger_pass = candle_reversal_in_zone or (zone_pass and volume_spike)

    trend_pass = len(bos_events) > 0
    checklist = {
        "trend": trend_pass,
        "zone": zone_pass,
        "trigger": trigger_pass,
    }

    # --- TP Calculation (Structure Based) ---
    # Try to find a recent major swing high to act as liquidity target
    # If no clear swing, fallback to 2R
    fib = fibonacci_levels(df)
    structure_target = float(fib["high_price"])
    
    if active_zone is not None:
        entry = float(active_zone["high"])
        sl = float(active_zone["low"] - atr * 0.2)
    else:
        entry = last_price
        sl = float(last_price - atr)

    risk = max(entry - sl, last_price * 0.005)
    
    # Use structure target if it offers > 1R reward, else fallback or extend
    if structure_target > entry + risk:
        tp = structure_target
    else:
        # Fallback to Fib 1.618 extension
        tp = float(fib["161_8"])

    return {
        "strategy": "SMC",
        "checklist": checklist,
        "status": "BUY" if all(checklist.values()) else "WAIT",
        "entry": round(entry, 4),
        "sl": round(sl, 4),
        "tp": round(tp, 4),
        "order_type": "limit",
        "fvg_zones": fvg_zones,
        "ob_zones": ob_zones,
        "bos_events": bos_events,
        "active_zone": active_zone,
        "active_zone_type": zone_type,
        "volume_spike": volume_spike,
    }


def _recent_resistance(df: pd.DataFrame, length: int = 20, include_current: bool = False) -> float:
    if df.empty:
        return 0.0

    scope = df if include_current else df.iloc[:-1]
    if scope.empty:
        scope = df
    window = scope.tail(length)
    return float(window["high"].max())


def _is_tight_consolidation(df: pd.DataFrame, length: int = 10) -> bool:
    window = df.tail(length)
    if len(window) < length:
        return False
    high = float(window["high"].max())
    low = float(window["low"].min())
    close_mean = float(window["close"].mean())
    width = (high - low) / max(close_mean, 1e-9)

    swings = [
        float(window["high"].iloc[-10:-7].max() - window["low"].iloc[-10:-7].min()),
        float(window["high"].iloc[-7:-4].max() - window["low"].iloc[-7:-4].min()),
        float(window["high"].iloc[-4:].max() - window["low"].iloc[-4:].min()),
    ]
    contracting = swings[2] <= swings[1] <= swings[0]
    # Relaxed from 0.06 to 0.15 (15%) to catch more setups
    return width < 0.15 and contracting


def analyze_momentum_breakout(df: pd.DataFrame) -> dict:
    resistance = _recent_resistance(df, 20, include_current=False)
    last_close = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else last_close
    last_ema50 = float(df["ema50"].iloc[-1])
    avg_volume = float(df["vol_sma20"].iloc[-1])
    last_volume = float(df["volume"].iloc[-1])

    base_window = df.iloc[:-1] if len(df) > 11 else df
    trend_pass = last_close > last_ema50
    zone_pass = _is_tight_consolidation(base_window, 10) and abs(last_close - resistance) / max(resistance, 1e-9) < 0.03
    trigger_pass = prev_close <= resistance and last_close > resistance and last_volume > 2 * avg_volume

    breakout_idx = len(df) - 1 if trigger_pass else None

    sl = float(min(df["low"].tail(5).min(), last_ema50))
    entry = max(last_close, resistance)
    risk = max(entry - sl, last_close * 0.005)
    
    # --- TP Calculation (Measured Move) ---
    # Height of consolidation = Resistance - Lowest Low in consolidation
    consolidation_low = float(df["low"].tail(10).min())
    pattern_height = max(resistance - consolidation_low, risk)
    measured_move = entry + pattern_height
    
    # If measured move is too small (< 1.5R), aim for 2R
    if measured_move < entry + 1.5 * risk:
        tp = float(entry + 2 * risk)
    else:
        tp = float(measured_move)

    return {
        "strategy": "Momentum",
        "checklist": {
            "trend": trend_pass,
            "zone": zone_pass,
            "trigger": trigger_pass,
        },
        "status": "BUY" if trend_pass and zone_pass and trigger_pass else "WAIT",
        "entry": round(entry, 4),
        "sl": round(sl, 4),
        "tp": round(tp, 4),
        "order_type": "stop",
        "resistance": resistance,
        "breakout_idx": breakout_idx,
        "volume_spike": last_volume > 2 * avg_volume,
    }


def _major_swing_low_high(df: pd.DataFrame, lookback: int = 90) -> tuple[int, int]:
    window = df.tail(lookback)
    start_idx = len(df) - len(window)
    if window.empty:
        return 0, 0

    low_pos = int(np.argmin(window["low"].to_numpy()))
    local_low_idx = start_idx + low_pos
    after_low = df.iloc[local_low_idx:]
    if after_low.empty:
        return max(local_low_idx, 0), max(local_low_idx, 0)

    high_pos = int(np.argmax(after_low["high"].to_numpy()))
    local_high_idx = local_low_idx + high_pos

    if local_high_idx <= local_low_idx:
        local_low_idx = start_idx
        local_high_idx = len(df) - 1

    return local_low_idx, local_high_idx


def fibonacci_levels(df: pd.DataFrame) -> dict:
    low_idx, high_idx = _major_swing_low_high(df, 90)
    low_price = float(df["low"].iloc[low_idx])
    high_price = float(df["high"].iloc[high_idx])
    diff = high_price - low_price

    return {
        "low_idx": low_idx,
        "high_idx": high_idx,
        "low_price": low_price,
        "high_price": high_price,
        "50": high_price - 0.5 * diff,
        "61_8": high_price - 0.618 * diff,
        "161_8": high_price + 0.618 * diff,
    }


def _bullish_divergence(df: pd.DataFrame) -> bool:
    _, lows = detect_swings(df, 2, 2)
    if len(lows) < 2:
        return False
    i1, i2 = lows[-2], lows[-1]
    price_lower_low = float(df["low"].iloc[i2]) < float(df["low"].iloc[i1])
    rsi_higher_low = float(df["rsi14"].iloc[i2]) > float(df["rsi14"].iloc[i1])
    return price_lower_low and rsi_higher_low


def analyze_pullback_reversal(df: pd.DataFrame) -> dict:
    fib = fibonacci_levels(df)
    last_close = float(df["close"].iloc[-1])
    ema200 = float(df["ema200"].iloc[-1])
    rsi = float(df["rsi14"].iloc[-1])

    fib_top = max(fib["50"], fib["61_8"])
    fib_bottom = min(fib["50"], fib["61_8"])

    trend_pass = last_close > ema200
    zone_pass = fib_bottom <= last_close <= fib_top
    # Relaxed RSI from < 30 to < 40 for strong uptrend pullbacks
    trigger_pass = (rsi < 40 or _bullish_divergence(df)) and bullish_candle_signal(df)

    sl = float(fib_bottom - df["atr14"].iloc[-1] * 0.5)
    entry = float(last_close)
    risk = max(entry - sl, last_close * 0.005)
    
    # --- TP Calculation (Swing High / Fib 0%) ---
    # Target the recent swing high (Fib 0% level)
    swing_high = float(fib["high_price"])
    
    if swing_high > entry + 1.0 * risk:
        tp = swing_high
    else:
        # Fallback to 2R if swing high is too close (already near top)
        tp = float(entry + 2 * risk)

    return {
        "strategy": "Pullback",
        "checklist": {
            "trend": trend_pass,
            "zone": zone_pass,
            "trigger": trigger_pass,
        },
        "status": "BUY" if trend_pass and zone_pass and trigger_pass else "WAIT",
        "entry": round(entry, 4),
        "sl": round(sl, 4),
        "tp": round(tp, 4),
        "order_type": "market",
        "fib": fib,
        "rsi": rsi,
        "bullish_divergence": _bullish_divergence(df),
    }


def volume_profile_poc(df: pd.DataFrame, bins: int = 24) -> dict:
    price_min = float(df["low"].min())
    price_max = float(df["high"].max())
    edges = np.linspace(price_min, price_max, bins + 1)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    hist, edges = np.histogram(typical_price, bins=edges, weights=df["volume"])

    # Basic POC
    max_idx = int(np.argmax(hist))
    poc = float((edges[max_idx] + edges[max_idx + 1]) / 2)

    # Calculate Value Area (70%) - Start from POC and expand
    total_vol = hist.sum()
    target_vol = total_vol * 0.7
    current_vol = hist[max_idx]
    
    left = max_idx
    right = max_idx
    
    # Expand until 70% reached
    while current_vol < target_vol:
        can_go_left = left > 0
        can_go_right = right < len(hist) - 1
        
        vol_left = hist[left - 1] if can_go_left else 0
        vol_right = hist[right + 1] if can_go_right else 0
        
        if not can_go_left and not can_go_right:
            break
            
        if vol_left > vol_right:
            current_vol += vol_left
            left -= 1
        else:
            current_vol += vol_right
            right += 1
            
    val = float(edges[left])
    vah = float(edges[right + 1])

    return {
        "poc": poc,
        "val": val,
        "vah": vah,
        "edges": edges,
        "hist": hist,
    }


def analyze_volume_profile(df: pd.DataFrame) -> dict:
    vp = volume_profile_poc(df)
    poc = vp["poc"]
    last_close = float(df["close"].iloc[-1])
    atr = float(df["atr14"].iloc[-1])
    tol = max(atr * 0.3, last_close * 0.003)

    tested_zone = abs(last_close - poc) <= tol or abs(float(df["low"].iloc[-1]) - poc) <= tol
    strong_volume = float(df["volume"].iloc[-1]) > 1.5 * float(df["vol_sma20"].iloc[-1])
    bounce = bullish_candle_signal(df) and last_close > poc

    # Trend Filter: Price > EM50 (added in V1.1.0)
    ema50 = float(df["ema50"].iloc[-1])
    trend_pass = last_close > ema50

    checklist = {
        "trend": trend_pass,
        "zone": tested_zone,
        "trigger": bounce and strong_volume,
    }

    sl = float(poc - atr * 0.5)
    entry = max(last_close, poc)
    risk = max(entry - sl, last_close * 0.005)
    
    # --- TP Calculation (Value Area High) ---
    vah = float(vp.get("vah", poc + 2 * risk))
    
    if vah > entry + 1.0 * risk:
        tp = vah
    else:
        tp = float(entry + 2 * risk)

    return {
        "strategy": "Volume Profile",
        "checklist": checklist,
        "status": "BUY" if all(checklist.values()) else "WAIT",
        "entry": round(entry, 4),
        "sl": round(sl, 4),
        "tp": round(tp, 4),
        "order_type": "limit",
        "poc": round(poc, 4),
        "volume_profile": vp,
    }


def run_strategy(df: pd.DataFrame, strategy_name: str) -> dict:
    if strategy_name == "SMC (BOS/OB/FVG)":
        return analyze_smc(df)
    if strategy_name == "Momentum Breakout (EMA50 + VCP)":
        return analyze_momentum_breakout(df)
    if strategy_name == "Pullback Reversal (EMA200 + Fib + RSI)":
        return analyze_pullback_reversal(df)
    if strategy_name == "Volume Profile (POC)":
        return analyze_volume_profile(df)
    raise ValueError(f"Unknown strategy: {strategy_name}")
