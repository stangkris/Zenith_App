import os
import re
import html
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from strategies import enrich_indicators, run_strategy


def _load_local_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    try:
        lines = env_path.read_text(encoding="utf-8-sig").splitlines()
    except Exception:
        return

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


_load_local_env(Path(__file__).with_name(".env"))

DEFAULT_TICKERS = ["IREN", "EOSE", "TSM", "MSFT", "IONQ", "NVDA", "TSLA", "AMD", "META", "COIN"]
STRATEGY_OPTIONS = [
    "SMC (BOS/OB/FVG)",
    "Momentum Breakout (EMA50 + VCP)",
    "Pullback Reversal (EMA200 + Fib + RSI)",
    "Volume Profile (POC)",
]
INTERVAL_OPTIONS = ["15min", "1h", "4h", "1day"]
FIGURE_SCHEMA_VERSION = 7
BACKTEST_CACHE_FILE = Path(__file__).with_name("backtest_trades.csv")
BACKTEST_SCHEMA_VERSION = 3
BACKTEST_DEFAULT_REFRESH_DAYS = 1
BACKTEST_MAX_BARS = {
    "15min": 5000,
    "1h": 4000,
    "4h": 2000,
    "1day": 1200,
}
BACKTEST_HORIZON_BARS = {
    "15min": 32,
    "1h": 18,
    "4h": 12,
    "1day": 8,
}
BACKTEST_MIN_WARMUP = 220
BACKTEST_STRATEGY_LOOKBACK = {
    "15min": 1400,
    "1h": 900,
    "4h": 620,
    "1day": 360,
}


def _backtest_signal_bounds(total_bars: int, interval: str) -> tuple[int | None, int | None, int]:
    if total_bars <= 0 or total_bars < BACKTEST_MIN_WARMUP:
        return None, None, 0

    horizon = int(BACKTEST_HORIZON_BARS.get(interval, 12))
    max_bars = int(BACKTEST_MAX_BARS.get(interval, 600))
    analysis_start = max(0, total_bars - max_bars)
    work_start = max(0, analysis_start - BACKTEST_MIN_WARMUP)

    local_loop_start = analysis_start - work_start
    local_latest_signal = (total_bars - work_start) - horizon - 1
    if local_latest_signal < local_loop_start:
        return None, None, work_start

    global_start = work_start + local_loop_start
    global_end = work_start + local_latest_signal
    return global_start, global_end, work_start


def _compute_source_digest(df: pd.DataFrame) -> str:
    if df.empty:
        return ""

    needed = ["timestamp", "open", "high", "low", "close", "volume"]
    cols = [c for c in needed if c in df.columns]
    if not cols:
        return ""

    payload = df[cols].copy()
    if "timestamp" in payload.columns:
        ts = pd.to_datetime(payload["timestamp"], errors="coerce")
        payload["timestamp"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")

    for col in ["open", "high", "low", "close", "volume"]:
        if col in payload.columns:
            payload[col] = pd.to_numeric(payload[col], errors="coerce").round(8)

    if "timestamp" in payload.columns:
        payload = payload.sort_values("timestamp")
    raw = payload.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()



def _strategy_signal_series(
    df: pd.DataFrame,
    strategy_name: str,
    interval: str,
    progress_callback=None,
) -> pd.Series:
    n = len(df)
    signals = pd.Series(False, index=df.index)
    loop_start_global, loop_end_global, work_start = _backtest_signal_bounds(n, interval)
    if loop_start_global is None or loop_end_global is None:
        return signals

    work = df.iloc[work_start:].reset_index(drop=True)
    loop_start = int(loop_start_global - work_start)
    loop_end = int(loop_end_global - work_start)
    total = int(loop_end - loop_start + 1)
    lookback = int(BACKTEST_STRATEGY_LOOKBACK.get(interval, 800))

    for step, i in enumerate(range(loop_start, loop_end + 1), start=1):
        sub_start = max(0, i + 1 - lookback)
        sub = work.iloc[sub_start : i + 1]
        try:
            res = run_strategy(sub, strategy_name)
            global_i = work_start + i
            signals.iloc[global_i] = bool(res.get("status") == "BUY")
        except Exception:
            pass

        if progress_callback and (step == total or step % 20 == 0):
            if progress_callback("signals", step, total) is False:
                break

    return signals


def _load_backtest_cache() -> pd.DataFrame:
    if not BACKTEST_CACHE_FILE.exists():
        return pd.DataFrame()
    try:
        cached = pd.read_csv(BACKTEST_CACHE_FILE)
    except Exception:
        return pd.DataFrame()
    if cached.empty:
        return cached
    if "schema_version" not in cached.columns:
        return pd.DataFrame()
    try:
        versions = pd.to_numeric(cached["schema_version"], errors="coerce").dropna().astype(int).unique().tolist()
        if not versions or any(v != BACKTEST_SCHEMA_VERSION for v in versions):
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    for col in [
        "ticker",
        "interval",
        "strategy",
        "market_mode",
        "entry_time",
        "exit_time",
        "entry",
        "exit",
        "sl",
        "tp",
        "r_result",
        "order_type",
        "exit_reason",
        "data_start",
        "data_end",
        "data_digest",
    ]:
        if col not in cached.columns:
            cached[col] = np.nan

    cached["entry_time"] = pd.to_datetime(cached["entry_time"], errors="coerce")
    cached["exit_time"] = pd.to_datetime(cached["exit_time"], errors="coerce")
    cached["data_start"] = pd.to_datetime(cached["data_start"], errors="coerce")
    cached["data_end"] = pd.to_datetime(cached["data_end"], errors="coerce")
    return cached


def _should_refresh_backtest_cache(last_refresh_utc: pd.Timestamp | None) -> bool:
    if last_refresh_utc is None or pd.isna(last_refresh_utc):
        return True
    ts = pd.to_datetime(last_refresh_utc, errors="coerce")
    if pd.isna(ts):
        return True
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    age_days = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 86400.0
    return age_days >= float(BACKTEST_DEFAULT_REFRESH_DAYS)


def _calculate_equity_curve_and_mdd(r_results: list[float]) -> tuple[list[float], float]:
    equity: list[float] = []
    value = 0.0
    peak = 0.0
    mdd = 0.0
    for r in r_results:
        value += float(r)
        equity.append(value)
        if value > peak:
            peak = value
        dd = peak - value
        if dd > mdd:
            mdd = dd
    return equity, float(mdd)


def _compute_advanced_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "breakeven_count": 0,
            "avg_r": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "max_drawdown_r": 0.0,
            "trades_per_month": 0.0,
        }

    # For metrics: exclude canceled
    filled = trades[trades["exit_reason"].ne("CANCELED")].copy()
    if filled.empty:
        return {
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "breakeven_count": 0,
            "avg_r": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "max_drawdown_r": 0.0,
            "trades_per_month": 0.0,
        }

    filled["r_result"] = pd.to_numeric(filled["r_result"], errors="coerce")
    filled = filled.dropna(subset=["r_result"])
    if filled.empty:
        return {
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "breakeven_count": 0,
            "avg_r": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "max_drawdown_r": 0.0,
            "trades_per_month": 0.0,
        }

    sort_cols: list[str] = []
    if "exit_time" in filled.columns:
        sort_cols.append("exit_time")
    if "entry_time" in filled.columns:
        sort_cols.append("entry_time")
    if sort_cols:
        filled = filled.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    r = filled["r_result"]
    wins = r[r > 0]
    losses = r[r < 0]
    breakeven = r[r == 0]

    trade_count = int(len(r))
    win_count = int(len(wins))
    loss_count = int(len(losses))
    breakeven_count = int(len(breakeven))

    avg_r = float(r.mean())
    avg_win_r = float(wins.mean()) if win_count else 0.0
    avg_loss_r = float(losses.abs().mean()) if loss_count else 0.0

    denom = max((win_count + loss_count + breakeven_count), 1)
    win_rate = float(win_count) / float(denom) * 100.0

    win_rate_dec = float(win_count) / max(trade_count, 1)
    loss_rate_dec = float(loss_count) / max(trade_count, 1)
    expectancy = (win_rate_dec * avg_win_r) - (loss_rate_dec * avg_loss_r)

    _, mdd = _calculate_equity_curve_and_mdd(r.tolist())

    # Trade frequency per month based on exit_time
    filled = filled.dropna(subset=["exit_time"])
    if filled.empty:
        trades_per_month = 0.0
    else:
        start = filled["exit_time"].min()
        end = filled["exit_time"].max()
        months = max(((end.year - start.year) * 12 + (end.month - start.month) + 1), 1)
        trades_per_month = float(len(filled)) / float(months)

    return {
        "trade_count": trade_count,
        "win_count": win_count,
        "loss_count": loss_count,
        "breakeven_count": breakeven_count,
        "avg_r": avg_r,
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "win_rate": win_rate,
        "expectancy": float(expectancy),
        "max_drawdown_r": float(mdd),
        "trades_per_month": float(trades_per_month),
    }


def _simulate_trades_from_signals(
    df: pd.DataFrame,
    signals: pd.Series,
    interval: str,
    strategy_name: str,
    progress_callback=None,
) -> pd.DataFrame:
    horizon = int(BACKTEST_HORIZON_BARS.get(interval, 12))
    rows: list[dict] = []
    lookback = int(BACKTEST_STRATEGY_LOOKBACK.get(interval, 800))
    signal_indices = signals[signals].index.to_numpy()
    total = int(len(signal_indices))

    def emit_progress(step_idx: int) -> bool:
        if not progress_callback or total <= 0:
            return True
        if step_idx == total or step_idx % 10 == 0:
            return progress_callback("trades", step_idx, total) is not False
        return True

    last_exit_idx = -1  # Track last trade exit to prevent overlapping trades
    SLIPPAGE_PCT = 0.0005  # 0.05% slippage on entry/exit
    COMMISSION_PCT = 0.001  # 0.1% commission per round trip (of total trade value)

    for step, idx in enumerate(signal_indices, start=1):
        si = int(idx)
        # Skip if previous trade is still open
        if si <= last_exit_idx:
            if not emit_progress(step):
                break
            continue

        if si >= len(df) - 2:
            if not emit_progress(step):
                break
            continue

        try:
            sub_start = max(0, si + 1 - lookback)
            levels = run_strategy(df.iloc[sub_start : si + 1], strategy_name)
        except Exception:
            if not emit_progress(step):
                break
            continue

        raw_entry = float(levels.get("entry", np.nan))
        raw_sl = float(levels.get("sl", np.nan))
        raw_tp = float(levels.get("tp", np.nan))
        signal_close = float(df["close"].iloc[si])
        order_type = str(levels.get("order_type", "")).strip().lower()
        if order_type not in {"limit", "stop", "market"}:
            tol = max(abs(signal_close) * 0.0005, 1e-6)
            if abs(raw_entry - signal_close) <= tol:
                order_type = "market"
            elif raw_entry > signal_close:
                order_type = "stop"
            else:
                order_type = "limit"

        if not (np.isfinite(raw_entry) and np.isfinite(raw_sl) and np.isfinite(raw_tp)):
            if not emit_progress(step):
                break
            continue
        if raw_tp <= raw_entry or raw_sl >= raw_entry:
            if not emit_progress(step):
                break
            continue

        end_idx = min(len(df) - 1, si + horizon)
        future = df.iloc[si + 1 : end_idx + 1]
        fill_i: int | None = None
        fill_price_base: float | None = None

        if order_type == "market":
            if si + 1 <= end_idx:
                fill_i = si + 1
                fill_price_base = float(df["open"].iloc[fill_i])
        elif not future.empty:
            if order_type == "limit":
                fill_mask = future["low"] <= raw_entry
                if fill_mask.any():
                    fill_pos = int(np.argmax(fill_mask.to_numpy()))
                    fill_i = int(future.index[fill_pos])
                    fill_price_base = raw_entry
            else:  # stop
                fill_mask = future["high"] >= raw_entry
                if fill_mask.any():
                    fill_pos = int(np.argmax(fill_mask.to_numpy()))
                    fill_i = int(future.index[fill_pos])
                    fill_price_base = max(raw_entry, float(df["open"].iloc[fill_i]))

        if fill_i is None or fill_price_base is None:
            rows.append(
                {
                    "entry_idx": si,
                    "exit_idx": np.nan,
                    "entry_time": df["timestamp"].iloc[si],
                    "exit_time": pd.NaT,
                    "entry": raw_entry,
                    "exit": np.nan,
                    "sl": raw_sl,
                    "tp": raw_tp,
                    "r_result": np.nan,
                    "exit_reason": "CANCELED",
                    "order_type": order_type,
                }
            )
            if not emit_progress(step):
                break
            continue

        # Apply slippage after order is considered filled.
        entry = float(fill_price_base) * (1 + SLIPPAGE_PCT)
        sl = raw_sl
        tp = raw_tp
        if tp <= entry or sl >= entry:
            rows.append(
                {
                    "entry_idx": fill_i,
                    "exit_idx": np.nan,
                    "entry_time": df["timestamp"].iloc[fill_i],
                    "exit_time": pd.NaT,
                    "entry": entry,
                    "exit": np.nan,
                    "sl": sl,
                    "tp": tp,
                    "r_result": np.nan,
                    "exit_reason": "CANCELED",
                    "order_type": order_type,
                }
            )
            if not emit_progress(step):
                break
            continue

        exit_idx: int | None = None
        exit_price: float | None = None
        exit_reason: str | None = None

        for j in range(fill_i, end_idx + 1):
            hi = float(df["high"].iloc[j])
            lo = float(df["low"].iloc[j])

            hit_tp = hi >= tp
            hit_sl = lo <= sl

            if hit_tp and hit_sl:
                exit_idx = j
                exit_price = sl  # Pessimistic intrabar
                exit_reason = "SL_INTRABAR"
                break
            if hit_sl:
                exit_idx = j
                exit_price = sl
                exit_reason = "SL"
                break
            if hit_tp:
                exit_idx = j
                exit_price = tp
                exit_reason = "TP"
                break

        if exit_idx is None:
            exit_idx = end_idx
            exit_price = float(df["close"].iloc[end_idx])
            exit_reason = "TIME_STOP"

        if exit_price is None or exit_reason is None:
            continue
            
        last_exit_idx = exit_idx

        # Apply Slippage to Exit (Sell lower)
        # If SL_INTRABAR, we already assumed worst case (SL), apply slippage to that
        final_exit_price = exit_price * (1 - SLIPPAGE_PCT)

        risk = max(entry - sl, 1e-9)
        gross_pnl = final_exit_price - entry
        commission_cost = (entry + final_exit_price) * (COMMISSION_PCT / 2.0)
        net_pnl = gross_pnl - commission_cost
        r_result = net_pnl / risk

        rows.append(
            {
                "entry_idx": fill_i,
                "exit_idx": exit_idx,
                "entry_time": df["timestamp"].iloc[fill_i],
                "exit_time": df["timestamp"].iloc[exit_idx],
                "entry": float(entry),
                "exit": float(final_exit_price),
                "sl": float(sl),
                "tp": float(tp),
                "r_result": float(r_result),
                "exit_reason": str(exit_reason),
                "order_type": order_type,
            }
        )

        if not emit_progress(step):
            break

    return pd.DataFrame(rows)


def _canonicalize_trade_log(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    entry_time_col = out["entry_time"] if "entry_time" in out.columns else pd.Series(pd.NaT, index=out.index)
    exit_time_col = out["exit_time"] if "exit_time" in out.columns else pd.Series(pd.NaT, index=out.index)
    out["entry_time"] = pd.to_datetime(entry_time_col, errors="coerce")
    out["exit_time"] = pd.to_datetime(exit_time_col, errors="coerce")
    for col in ["entry", "exit", "sl", "tp", "r_result"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    exit_reason_col = out["exit_reason"] if "exit_reason" in out.columns else pd.Series("", index=out.index)
    out["exit_reason"] = exit_reason_col.astype(str)
    order_type_col = out["order_type"] if "order_type" in out.columns else pd.Series("limit", index=out.index)
    out["order_type"] = order_type_col.astype(str).str.lower().replace({"": "limit", "nan": "limit"})
    return out


def _append_and_dedup_trades(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        combined = new.copy()
    else:
        combined = pd.concat([existing, new], ignore_index=True)

    combined = _canonicalize_trade_log(combined)

    # Dedup: a trade is uniquely identified by (ticker, interval, strategy, market_mode, entry_time, exit_time, exit_reason)
    subset = [
        "ticker",
        "interval",
        "strategy",
        "market_mode",
        "entry_time",
        "exit_time",
        "exit_reason",
        "entry",
        "sl",
        "tp",
        "order_type",
    ]
    subset = [c for c in subset if c in combined.columns]
    if not subset:
        return combined.reset_index(drop=True)
    combined = combined.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)
    return combined


def _get_last_exit_time(trades: pd.DataFrame) -> pd.Timestamp | None:
    if trades.empty or "exit_time" not in trades.columns:
        return None
    usable = trades[(trades["exit_reason"] != "__META__") & (trades["exit_reason"] != "CANCELED")].copy()
    if usable.empty:
        return None
    ts = pd.to_datetime(usable["exit_time"], errors="coerce")
    ts = ts.dropna()
    if ts.empty:
        return None
    return ts.max()


def _get_backtest_summary(
    df: pd.DataFrame,
    ticker: str,
    interval: str,
    strategy_name: str,
    market_mode: str,
    progress_callback=None,
) -> tuple[dict, bool]:
    cached = _load_backtest_cache()
    total_bars = int(len(df))
    source_start = pd.to_datetime(df["timestamp"].iloc[0], errors="coerce") if total_bars else pd.NaT
    source_end = pd.to_datetime(df["timestamp"].iloc[-1], errors="coerce") if total_bars else pd.NaT
    source_digest = _compute_source_digest(df)

    if cached.empty:
        key_mask = pd.Series(False, index=cached.index, dtype=bool)
    elif {"ticker", "interval", "strategy", "market_mode"}.issubset(cached.columns):
        key_mask = (
            (cached["ticker"] == ticker)
            & (cached["interval"] == interval)
            & (cached["strategy"] == strategy_name)
            & (cached["market_mode"] == market_mode)
        )
    else:
        key_mask = pd.Series(False, index=cached.index, dtype=bool)

    key_rows = cached[key_mask].copy() if not cached.empty else pd.DataFrame()

    if not cached.empty and "exit_reason" in cached.columns:
        meta_mask = key_mask & (cached["exit_reason"] == "__META__")
    else:
        meta_mask = pd.Series(False, index=cached.index, dtype=bool)

    meta_rows = cached[meta_mask].copy() if not cached.empty else pd.DataFrame()
    last_refresh: pd.Timestamp | None = None
    cached_data_end: pd.Timestamp | None = None
    cached_data_digest = ""
    if not meta_rows.empty:
        try:
            last_refresh = pd.to_datetime(meta_rows["exit_time"].iloc[-1], errors="coerce")
        except Exception:
            last_refresh = None
        if "data_end" in meta_rows.columns:
            try:
                cached_data_end = pd.to_datetime(meta_rows["data_end"].iloc[-1], errors="coerce")
            except Exception:
                cached_data_end = None
        if "data_digest" in meta_rows.columns:
            try:
                cached_data_digest = str(meta_rows["data_digest"].iloc[-1] or "").strip()
            except Exception:
                cached_data_digest = ""

    # Split meta and trades
    if not key_rows.empty and "exit_reason" in key_rows.columns:
        trades = key_rows[key_rows["exit_reason"] != "__META__"].copy()
    else:
        trades = key_rows.copy() if not key_rows.empty else pd.DataFrame()
    trades = _canonicalize_trade_log(trades)

    # Reuse cache if source range didn't move forward AND didn't expand backward.
    cached_data_start: pd.Timestamp | None = None
    if not meta_rows.empty and "data_start" in meta_rows.columns:
        try:
            cached_data_start = pd.to_datetime(meta_rows["data_start"].iloc[-1], errors="coerce")
        except Exception:
            cached_data_start = None

    force_full_recalc = False
    if cached_data_end is not None and not pd.isna(cached_data_end) and not pd.isna(source_end):
        end_ok = bool(source_end <= cached_data_end)
        # Also invalidate if the new source starts earlier than what was cached
        # (e.g. fetch window expanded from 183 → 540 days).
        if cached_data_start is not None and not pd.isna(cached_data_start) and not pd.isna(source_start):
            start_ok = bool(source_start >= cached_data_start)
        else:
            start_ok = True
        cache_hit = end_ok and start_ok
    else:
        cache_hit = bool((not trades.empty) and (not _should_refresh_backtest_cache(last_refresh)))

    if not cached_data_digest:
        cache_hit = False
        force_full_recalc = True
    elif source_digest and cached_data_digest != source_digest:
        cache_hit = False
        force_full_recalc = True

    def run_backtest_window(df_window: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        sig = _strategy_signal_series(
            df_window,
            strategy_name,
            interval,
            progress_callback=progress_callback,
        )
        trs = _simulate_trades_from_signals(
            df_window,
            sig,
            interval,
            strategy_name,
            progress_callback=progress_callback,
        )
        return sig, trs

    # Incremental update: compute only newly appended range, with continuity lookback.
    signals = None
    if not cache_hit:
        if progress_callback:
            progress_callback("prepare", 1, 1)
        lookback = max(int(BACKTEST_STRATEGY_LOOKBACK.get(interval, 800)), BACKTEST_MIN_WARMUP)
        horizon = int(BACKTEST_HORIZON_BARS.get(interval, 12))
        continuity_bars = lookback + horizon + 8

        anchor_ts = cached_data_end
        if anchor_ts is None or pd.isna(anchor_ts):
            anchor_ts = _get_last_exit_time(trades)

        if force_full_recalc:
            trades = pd.DataFrame()
            signals, new_trades = run_backtest_window(df)
        elif trades.empty or anchor_ts is None or pd.isna(anchor_ts):
            signals, new_trades = run_backtest_window(df)
        else:
            ts_all = pd.to_datetime(df["timestamp"], errors="coerce")
            candidate_idx = ts_all[ts_all >= anchor_ts].index
            anchor_idx = int(candidate_idx[0]) if len(candidate_idx) else max(total_bars - 1, 0)
            start_idx = max(0, anchor_idx - continuity_bars)
            window_start = pd.to_datetime(df["timestamp"].iloc[start_idx], errors="coerce")
            df_recent = df.iloc[start_idx:].copy().reset_index(drop=True)

            if df_recent.empty or len(df_recent) < 60:
                signals, new_trades = run_backtest_window(df)
            else:
                signals, new_trades = run_backtest_window(df_recent)
                if not trades.empty and "entry_time" in trades.columns and not pd.isna(window_start):
                    trades = trades[(trades["entry_time"].isna()) | (trades["entry_time"] < window_start)].copy()

        # Attach key columns
        new_trades["schema_version"] = BACKTEST_SCHEMA_VERSION
        new_trades["ticker"] = ticker
        new_trades["interval"] = interval
        new_trades["strategy"] = strategy_name
        new_trades["market_mode"] = market_mode

        # Append + dedup
        trades = _append_and_dedup_trades(trades, new_trades)

        # Write back: keep other keys, replace this key's trades + meta
        remaining = cached[~key_mask] if not cached.empty else pd.DataFrame()
        meta_row = pd.DataFrame(
            [
                {
                    "schema_version": BACKTEST_SCHEMA_VERSION,
                    "ticker": ticker,
                    "interval": interval,
                    "strategy": strategy_name,
                    "market_mode": market_mode,
                    "entry_time": pd.NaT,
                    "exit_time": pd.Timestamp.now(tz="UTC"),
                    "entry": np.nan,
                    "exit": np.nan,
                    "sl": np.nan,
                    "tp": np.nan,
                    "r_result": np.nan,
                    "order_type": np.nan,
                    "exit_reason": "__META__",
                    "data_start": source_start,
                    "data_end": source_end,
                    "data_digest": source_digest,
                }
            ]
        )
        updated = pd.concat([remaining, trades, meta_row], ignore_index=True)
        try:
            updated.to_csv(BACKTEST_CACHE_FILE, index=False)
        except Exception:
            pass

        if progress_callback:
            progress_callback("finalize", 1, 1)

    # Metrics
    metrics = _compute_advanced_metrics(trades)

    signal_start_idx, signal_end_idx, _ = _backtest_signal_bounds(total_bars, interval)
    if signal_start_idx is not None and signal_end_idx is not None:
        start = pd.to_datetime(df["timestamp"].iloc[signal_start_idx], errors="coerce")
        end = pd.to_datetime(df["timestamp"].iloc[signal_end_idx], errors="coerce")
    else:
        start = df["timestamp"].iloc[0] if total_bars else None
        end = df["timestamp"].iloc[-1] if total_bars else None

    canceled_count = int((trades["exit_reason"] == "CANCELED").sum()) if not trades.empty else 0
    signal_count = int(trades.shape[0])
    filled_count = int(metrics["trade_count"])
    fill_rate = (float(filled_count) / float(signal_count) * 100.0) if signal_count else 0.0

    summary = {
        "total_bars": total_bars,
        "signal_count": signal_count,
        "filled_count": filled_count,
        "fill_rate": fill_rate,
        "win_count": metrics["win_count"],
        "loss_count": metrics["loss_count"],
        "breakeven_count": metrics["breakeven_count"],
        "canceled_count": canceled_count,
        "win_rate": metrics["win_rate"],
        "avg_r": metrics["avg_r"],
        "expectancy": metrics["expectancy"],
        "max_drawdown_r": metrics["max_drawdown_r"],
        "trades_per_month": metrics["trades_per_month"],
        "start": start,
        "end": end,
        "source_start": source_start,
        "source_end": source_end,
        "last_refresh_utc": last_refresh if cache_hit and isinstance(last_refresh, pd.Timestamp) and not pd.isna(last_refresh) else pd.Timestamp.now(tz="UTC"),
    }

    if progress_callback:
        progress_callback("complete", 1, 1)

    return summary, cache_hit


st.set_page_config(
    page_title="Zenith | Elite Stock Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css(mini_mode: bool = False) -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg-main: #edf1f6;
                --bg-soft: #f4f6f9;
                --panel: #ffffff;
                --panel-border: #dde4ee;
                --text: #1f2937;
                --muted: #61738b;
                --accent: #0f172a;
                --ok: #0f766e;
                --warn: #b91c1c;
                --section-gap: 0.56rem;
            }

            #MainMenu, footer {visibility: hidden;}

            header[data-testid="stHeader"] {
                background: rgba(0, 0, 0, 0);
            }

            header[data-testid="stHeader"] button {
                color: #0f172a !important;
                border: 1px solid #cbd5e1 !important;
                background: #ffffff !important;
                border-radius: 10px !important;
                cursor: pointer !important;
            }

            [data-testid="stToolbar"] {
                display: none !important;
            }

            [data-testid="stSidebarCollapseButton"],
            [data-testid="stSidebarCollapsedControl"] {
                display: none !important;
            }

            [data-testid="stSidebarContent"] {
                display: flex;
                flex-direction: column;
                height: calc(100vh - 0.6rem);
                padding-top: 0.08rem;
                padding-bottom: 0.45rem;
                overflow: hidden !important;
            }

            [data-testid="stSidebarUserContent"] {
                display: flex;
                flex-direction: column;
                min-height: 100%;
                height: 100%;
            }

            [data-testid="stSidebarUserContent"] > div {
                display: flex;
                flex-direction: column;
                min-height: 100%;
                height: 100%;
            }

            .sidebar-toggle-wrap {
                margin-bottom: 0.38rem;
            }

            .sidebar-toggle-wrap button {
                border-radius: 12px !important;
                font-size: 0.92rem !important;
                font-weight: 700 !important;
                min-height: 2.35rem !important;
                cursor: pointer !important;
            }

            .stApp {
                background: linear-gradient(180deg, var(--bg-main) 0%, var(--bg-soft) 100%);
                color: var(--text);
                min-height: 100vh;
                overflow: hidden !important;
            }

            html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
                height: 100vh;
                overflow: hidden !important;
            }

            .block-container {
                padding-top: 0.32rem;
                padding-bottom: 0.12rem;
                padding-left: 1rem;
                padding-right: 1rem;
                max-width: 1860px;
                height: calc(100vh - 0.35rem);
                overflow: hidden !important;
            }

            .block-container > div[data-testid="stVerticalBlock"] {
                gap: var(--section-gap);
                height: 100%;
                overflow: hidden !important;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #020617 0%, #0b1220 100%);
                border-right: 1px solid #1e293b;
                overflow: hidden !important;
            }

            [data-testid="stSidebar"] > div {
                overflow: hidden !important;
            }

            [data-testid="stSidebar"] * {
                color: #e2e8f0 !important;
            }

            [data-testid="stSidebar"] .controls-title {
                color: #94a3b8 !important;
            }

            [data-testid="stSidebar"] div[data-baseweb="select"] > div,
            [data-testid="stSidebar"] div[data-baseweb="input"] > div,
            [data-testid="stSidebar"] .stTextInput input {
                background: #0f172a !important;
                border: 1px solid #334155 !important;
                border-radius: 12px !important;
                color: #f8fafc !important;
                box-shadow: none !important;
            }

            [data-testid="stSidebar"] div[data-baseweb="select"] * {
                color: #f8fafc !important;
            }

            [data-testid="stSidebar"] .stButton > button {
                background: #0f172a !important;
                border: 1px solid #334155 !important;
                border-radius: 14px !important;
                color: #ffffff !important;
                font-weight: 700 !important;
                cursor: pointer !important;
            }

            [data-testid="stSidebar"] .stButton > button:hover {
                background: #111827 !important;
                border-color: #475569 !important;
            }

            /* Header card + top actions */
            .topbar-card-marker {
                display: none;
            }
            div[data-testid="stVerticalBlock"]:has(.topbar-card-marker) {
                background: #ffffff;
                border: 1px solid #dde4ee;
                border-radius: 16px;
                box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
                padding: 0.58rem 0.9rem;
                margin-bottom: var(--section-gap);
            }
            div[data-testid="stVerticalBlock"]:has(.topbar-card-marker) > div[data-testid="stHorizontalBlock"] {
                align-items: center !important;
            }
            div[data-testid="stVerticalBlock"]:has(.topbar-card-marker) .stButton > button {
                background: #ffffff !important;
                color: #334155 !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 12px !important;
                font-weight: 600 !important;
                min-height: 2.35rem !important;
                font-size: 0.9rem !important;
                letter-spacing: 0.01em !important;
                box-shadow: none !important;
            }
            div[data-testid="stVerticalBlock"]:has(.topbar-card-marker) .stButton > button:hover {
                background: #f8fafc !important;
                border-color: #cbd5e1 !important;
                color: #334155 !important;
            }
            div[data-testid="stVerticalBlock"]:has(.topbar-card-marker) .stButton > button * {
                color: #334155 !important;
            }
            div[data-testid="stVerticalBlock"]:has(.topbar-card-marker) [data-testid="stPopover"] > button,
            div[data-testid="stVerticalBlock"]:has(.topbar-card-marker) [data-testid="stPopover"] button,
            div[data-testid="stVerticalBlock"]:has(.topbar-card-marker) button[data-testid="stPopoverButton"] {
                background: #ffffff !important;
                color: #334155 !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 12px !important;
                font-weight: 600 !important;
                min-height: 2.35rem !important;
                font-size: 0.9rem !important;
                letter-spacing: 0.01em !important;
                box-shadow: none !important;
                cursor: pointer !important;
            }
            div[data-testid="stVerticalBlock"]:has(.topbar-card-marker) [data-testid="stPopover"] > button:hover,
            div[data-testid="stVerticalBlock"]:has(.topbar-card-marker) [data-testid="stPopover"] button:hover,
            div[data-testid="stVerticalBlock"]:has(.topbar-card-marker) button[data-testid="stPopoverButton"]:hover,
            div[data-testid="stVerticalBlock"]:has(.topbar-card-marker) button[data-testid="stPopoverButton"][aria-expanded="true"] {
                background: #f8fafc !important;
                border-color: #cbd5e1 !important;
                color: #334155 !important;
            }
            div[data-testid="stVerticalBlock"]:has(.topbar-card-marker) [data-testid="stPopover"] button *,
            div[data-testid="stVerticalBlock"]:has(.topbar-card-marker) button[data-testid="stPopoverButton"] * {
                color: #334155 !important;
            }
            div[data-testid="stVerticalBlock"]:has(.topbar-card-marker) .stButton > button:disabled {
                background: #f8fafc !important;
                border-color: #e2e8f0 !important;
                color: #94a3b8 !important;
                box-shadow: none !important;
            }
            div[data-baseweb="popover"] {
                z-index: 1100 !important;
            }
            div[data-testid="stPopoverBody"] {
                background: #ffffff !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 14px !important;
                box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12) !important;
                max-width: min(620px, 94vw) !important;
                max-height: min(72vh, 640px) !important;
                overflow-y: auto !important;
                overflow-x: hidden !important;
                padding: 0.35rem 0.5rem !important;
                color: #1f2937 !important;
            }
            div[data-testid="stPopoverBody"],
            div[data-testid="stPopoverBody"] p,
            div[data-testid="stPopoverBody"] li,
            div[data-testid="stPopoverBody"] span,
            div[data-testid="stPopoverBody"] div {
                color: #1f2937 !important;
            }
            div[data-testid="stPopoverBody"] h4,
            div[data-testid="stPopoverBody"] .m-head {
                color: #0f172a !important;
                font-weight: 800 !important;
            }
            div[data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"],
            div[data-testid="stPopoverBody"] .stMarkdown {
                background: #ffffff !important;
                color: #1f2937 !important;
                border: 0 !important;
                box-shadow: none !important;
                padding: 0 !important;
                margin: 0 !important;
            }
            div[data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] > *,
            div[data-testid="stPopoverBody"] [data-testid="stElementContainer"],
            div[data-testid="stPopoverBody"] [data-testid="element-container"],
            div[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] > div {
                background: #ffffff !important;
                color: #1f2937 !important;
            }
            div[data-baseweb="popover"] [data-testid="stMarkdownContainer"],
            div[data-baseweb="popover"] .stMarkdown,
            div[data-baseweb="popover"] [data-testid="stMarkdownContainer"] > * {
                background: #ffffff !important;
                color: #1f2937 !important;
            }
            .zenith-popover-content {
                display: block;
                width: 100%;
                max-width: 100%;
                overflow-wrap: anywhere;
                word-break: break-word;
                line-height: 1.56;
                color: #1f2937 !important;
                padding-bottom: 2px;
                background: #ffffff !important;
            }
            .zenith-popover-content * {
                background: #ffffff !important;
                color: #1f2937 !important;
            }
            .zenith-popover-content ul,
            .zenith-popover-content ol {
                list-style-position: outside;
                padding-left: 1.15rem;
            }
            .zenith-popover-content h4 {
                margin: 0.1rem 0 0.5rem 0;
                color: #0f172a !important;
                font-size: 1.02rem;
                line-height: 1.38;
            }
            .zenith-popover-content .m-head {
                margin: 0.55rem 0 0.24rem 0;
                color: #334155 !important;
                font-weight: 700 !important;
            }
            .zenith-popover-content ul {
                margin: 0.15rem 0 0.5rem 0;
                padding-left: 1.15rem;
                list-style-position: outside;
            }
            .zenith-popover-content li {
                margin-bottom: 0.26rem;
                line-height: 1.5;
            }
            .zenith-popover-content .bt-headline {
                margin: 0 0 0.48rem 0;
                color: #0f172a !important;
                font-weight: 700;
                line-height: 1.4;
                font-size: 0.96rem;
            }
            .zenith-popover-content .bt-note {
                margin-top: 0.35rem;
                padding: 0.5rem 0.62rem;
                border-radius: 10px;
                border: 1px solid #e2e8f0;
                background: #f8fafc !important;
                color: #334155 !important;
                font-size: 0.9rem;
                line-height: 1.45;
            }
            .zenith-popover-content .bt-note.error {
                border-color: #fecaca;
                background: #fef2f2 !important;
                color: #7f1d1d !important;
            }
            .zenith-popover-content .bt-note.hint {
                border-color: #dbeafe;
                background: #eff6ff !important;
                color: #1e3a8a !important;
            }

            .section-gap {
                height: var(--section-gap);
            }

            .brand-row {
                display: flex;
                align-items: center;
                gap: 12px;
            }

            .brand-mark {
                font-size: 1.35rem;
                color: #111827;
                line-height: 1;
            }

            .brand-title {
                font-size: 2rem;
                font-weight: 700;
                color: #111827;
                letter-spacing: 0.01em;
                margin-right: 4px;
            }

            .brand-divider {
                color: #9ca3af;
                font-size: 1.75rem;
                line-height: 1;
            }

            .brand-sub {
                font-size: 1.45rem;
                font-weight: 500;
                color: #1f2937;
            }

            .manual-details {
                margin-left: auto;
                position: relative;
            }

            .manual-details summary {
                list-style: none;
                cursor: pointer;
                user-select: none;
                background: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 0.45rem 0.72rem;
                font-weight: 700;
                font-size: 0.88rem;
            }

            .manual-details summary::-webkit-details-marker { display: none; }
            .manual-details summary::marker { content: ""; }

            .manual-details[open] summary {
                background: #0f172a;
            }

            .manual-content {
                position: absolute;
                right: 0;
                top: calc(100% + 10px);
                width: min(560px, 72vw);
                background: #ffffff;
                color: #0f172a;
                border: 1px solid #cbd5e1;
                border-radius: 14px;
                box-shadow: 0 18px 38px rgba(15, 23, 42, 0.16);
                padding: 12px 14px;
                line-height: 1.64;
                z-index: 9999;
                font-size: 0.98rem;
            }

            .manual-content h4 {
                margin: 0.2rem 0 0.5rem 0;
                color: #0f172a;
                font-size: 1.02rem;
            }

            .manual-content .m-head {
                color: #334155;
                font-weight: 700;
                margin-top: 0.5rem;
                margin-bottom: 0.2rem;
            }

            .manual-content ul {
                margin: 0.1rem 0 0.45rem 1rem;
                padding: 0;
            }

            .manual-content li {
                color: #1f2937;
                margin-bottom: 0.18rem;
            }

            .glass, .action-card, .check-card, .kpi-card, .chart-shell, .summary-card, .stat-card {
                background: var(--panel);
                border: 1px solid var(--panel-border);
                border-radius: 20px;
                box-shadow: 0 5px 14px rgba(15, 23, 42, 0.035);
            }

            .glass { padding: 14px 16px; }
            .stat-card {
                padding: 7px 12px;
                min-height: 58px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .chart-shell { padding: 10px; }
            .action-card, .check-card, .kpi-card, .summary-card { padding: 12px 14px; }
            .check-card { margin-bottom: 8px; }

            .tiny-label {
                color: var(--muted);
                font-size: 0.73rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 600;
            }

            .tiny-value {
                color: var(--text);
                font-weight: 700;
                margin-top: 2px;
            }

            .kpi-num {
                color: #111827;
                font-size: 1.14rem;
                font-weight: 800;
            }

            .stat-value {
                color: #111827;
                font-size: 1.35rem;
                font-weight: 800;
                letter-spacing: 0.01em;
                margin-top: 1px;
            }

            .change-context {
                margin-top: 2px;
                display: flex;
                align-items: center;
                gap: 7px;
                flex-wrap: nowrap;
                font-size: 0.75rem;
                font-weight: 700;
                line-height: 1.25;
            }

            .phase-pill {
                font-size: 0.66rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                border: 1px solid currentColor;
                border-radius: 999px;
                padding: 2px 7px;
                line-height: 1.2;
                font-weight: 800;
            }

            .status-buy { color: var(--ok); font-weight: 800; }
            .status-wait { color: var(--warn); font-weight: 800; }
            .check-pass { color: var(--ok); font-weight: 700; }
            .check-fail { color: var(--warn); font-weight: 700; }

            .summary-card {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }

            .summary-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 5px 8px;
                border-radius: 10px;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
            }

            .summary-key {
                color: #64748b;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                display: inline-flex;
                align-items: center;
                gap: 6px;
            }

            .summary-val {
                color: #111827;
                font-size: 0.92rem;
                font-weight: 700;
            }

            .status-pill {
                border-radius: 12px;
                padding: 8px 10px;
                font-weight: 800;
                font-size: 1rem;
                border: 1px solid #e2e8f0;
                background: #ffffff;
            }

            .info-tip {
                display: inline-flex;
                width: 15px;
                height: 15px;
                border-radius: 50%;
                align-items: center;
                justify-content: center;
                border: 1px solid #94a3b8;
                color: #475569;
                font-size: 10px;
                font-weight: 700;
                cursor: help;
            }

            .stPlotlyChart {
                border-radius: 18px;
                overflow: hidden;
                border: 1px solid #dbe2ea;
                background: #ffffff;
                position: relative;
            }

            .stSelectbox label, .stTextInput label, .stRadio label, .stMarkdown, .stCaption {
                color: var(--text) !important;
            }

            /* BaseWeb controls (select/input) */
            .stApp div[data-baseweb="select"] > div,
            .stApp div[data-baseweb="input"] > div,
            .stApp .stTextInput > div > div > input {
                background: #ffffff !important;
                border: 1px solid #d1d5db !important;
                color: #111827 !important;
                border-radius: 10px !important;
                box-shadow: none !important;
            }

            .stApp div[data-baseweb="select"] > div:hover,
            .stApp div[data-baseweb="input"] > div:hover {
                border-color: #9ca3af !important;
            }

            .stApp div[data-baseweb="select"] * {
                color: #111827 !important;
            }

            div[data-baseweb="select"],
            div[data-baseweb="select"] *[role="button"],
            button[role="combobox"],
            [role="option"] {
                cursor: pointer !important;
            }

            /* Force hand cursor for all interactive controls (buttons/selects/popovers/sidebar controls) */
            .stApp button,
            .stApp [role="button"],
            .stApp [data-testid="stPopover"] > button,
            .stApp [data-testid="stPopover"] button,
            .stApp button[data-testid="stPopoverButton"],
            .stApp div[data-baseweb="select"],
            .stApp div[data-baseweb="select"] *,
            .stApp div[data-baseweb="input"] > div,
            .stApp [data-testid="stSelectbox"] *,
            .stApp [data-testid="stTextInput"] *,
            .stApp [data-testid="stSidebar"] .stCaption,
            .stApp [data-testid="stSidebar"] label {
                cursor: pointer !important;
            }

            /* Alerts readability */
            div[data-testid="stAlert"] {
                border-radius: 12px;
                border: 1px solid #dbeafe;
                color: #1f2937;
            }

            div[data-testid="stInfo"] {
                background: #eff6ff;
            }

            div[data-testid="stWarning"] {
                background: #fffbeb;
                border-color: #fde68a;
            }

            div[data-testid="stError"] {
                background: #fef2f2;
                border-color: #fecaca;
            }

            .controls-title {
                color: #9ca3af;
                letter-spacing: 0.07em;
                text-transform: uppercase;
                font-size: 1rem;
                font-weight: 700;
                margin: 0 0 0.5rem 0;
            }

            .side-footer {
                margin-top: auto !important;
                border-top: 1px solid #334155;
                padding-top: 0.7rem;
                color: #94a3b8;
                font-size: 0.84rem;
                line-height: 1.4;
                padding-bottom: 0.12rem;
            }

            .side-spacer {
                flex: 1 1 auto;
                min-height: 2px;
            }

            .panel-title {
                color: #9ca3af;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 0.8rem;
                font-weight: 700;
                margin-bottom: 0.28rem;
            }

            .hint-card {
                background: #f8fafc;
                border: 1px dashed #cbd5e1;
                color: #475569;
                border-radius: 12px;
                padding: 10px 12px;
                font-size: 0.96rem;
            }

            .advice-card {
                margin-top: 8px;
                background: #fffbeb;
                border: 1px solid #fde68a;
                border-radius: 14px;
                padding: 9px 11px;
                color: #78350f;
            }

            .advice-title {
                font-size: 0.86rem;
                font-weight: 800;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                margin-bottom: 4px;
            }

            .advice-text {
                font-size: 0.92rem;
                line-height: 1.4;
            }

            @media (max-width: 900px) {
                .block-container { padding-top: 0.5rem; }
                .brand-title { font-size: 1.35rem; }
                .brand-sub { font-size: 1.1rem; }
                .brand-divider { font-size: 1.35rem; }
                .brand-row { gap: 8px; }
                div[data-testid="stVerticalBlock"]:has(.topbar-card-marker) .stButton > button {
                    min-height: 2.1rem !important;
                    font-size: 0.84rem !important;
                }
                div[data-testid="stVerticalBlock"]:has(.topbar-card-marker) [data-testid="stPopover"] > button,
                div[data-testid="stVerticalBlock"]:has(.topbar-card-marker) [data-testid="stPopover"] button,
                div[data-testid="stVerticalBlock"]:has(.topbar-card-marker) button[data-testid="stPopoverButton"] {
                    min-height: 2.1rem !important;
                    font-size: 0.84rem !important;
                }
                .manual-content {
                    right: -16px;
                    width: min(92vw, 560px);
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if mini_mode:
        st.markdown(
            """
            <style>
                [data-testid="stSidebar"] {
                    min-width: 84px !important;
                    max-width: 84px !important;
                    width: 84px !important;
                }

                [data-testid="stSidebarContent"] {
                    padding-left: 10px !important;
                    padding-right: 10px !important;
                }

                [data-testid="stSidebar"] .sidebar-toggle-wrap button {
                    padding: 0.35rem 0.2rem !important;
                    font-size: 1rem !important;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
                [data-testid="stSidebar"] {
                    min-width: 250px !important;
                    max-width: 250px !important;
                    width: 250px !important;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame | None:
    if df.empty:
        st.warning("ข้อมูลว่างเปล่า")
        return None

    data = df.copy()
    parsed_ts = pd.to_datetime(data["timestamp"], errors="coerce")
    if getattr(parsed_ts.dt, "tz", None) is None:
        parsed_ts = parsed_ts.dt.tz_localize("Asia/Bangkok")
    else:
        parsed_ts = parsed_ts.dt.tz_convert("Asia/Bangkok")
    data["timestamp"] = parsed_ts.dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
    data["volume"] = data["volume"].fillna(0)
    data = data.reset_index(drop=True)

    if data.empty:
        st.warning("ข้อมูลหลังจัดรูปแบบว่างเปล่า")
        return None

    return enrich_indicators(data)


@st.cache_data(show_spinner=False, ttl=300)
def fetch_ohlcv_twelvedata(symbol: str, api_key: str, interval: str) -> pd.DataFrame | None:
    # Scale the lookback window by timeframe so that larger intervals
    # have enough bars for the 220-bar warmup + scanning range.
    _fetch_days = {"15min": 300, "1h": 600, "4h": 1200, "1day": 2000}.get(interval, 365)
    end = datetime.utcnow()
    start = end - timedelta(days=_fetch_days)
    
    # We remove 'end_date' to let the API default to "now" (latest available).
    # This prevents timezone conflicts where 'now' in UTC might be interpreted
    # as 'past' in Bangkok time, cutting off recent data.
    params = {
        "symbol": symbol,
        "interval": interval,
        "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
        "outputsize": 5000,
        "format": "JSON",
        "timezone": "Asia/Bangkok",
        "apikey": api_key,
    }
    url = f"https://api.twelvedata.com/time_series?{urlencode(params)}"
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=20) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            detail = ""
        if exc.code == 429:
            st.warning("Twelve Data เกิน Rate Limit (429) กรุณารอสักครู่แล้วลองใหม่")
            return None
        if detail:
            st.error(f"Twelve Data HTTP {exc.code}: {detail[:240]}")
        else:
            st.error(f"ไม่สามารถดึงข้อมูล Twelve Data ได้: HTTP Error {exc.code}")
        return None
    except Exception as exc:
        message = str(exc)
        if "429" in message:
            st.warning("Twelve Data เกิน Rate Limit (429) กรุณารอสักครู่แล้วลองใหม่")
            return None
        st.error(f"ไม่สามารถดึงข้อมูล Twelve Data ได้: {message}")
        return None

    try:
        obj = json.loads(payload)
    except Exception as exc:
        st.error(f"อ่านข้อมูล Twelve Data ไม่สำเร็จ: {exc}")
        return None

    if not isinstance(obj, dict):
        st.error("รูปแบบข้อมูลจาก Twelve Data ไม่ถูกต้อง")
        return None

    status = str(obj.get("status", ""))
    if status.lower() == "error":
        code = obj.get("code", "")
        message = obj.get("message", "Unknown error")
        if str(code) == "429":
            st.warning(f"Twelve Data Rate Limit: {message}")
        else:
            st.error(f"Twelve Data error ({code}): {message}")
        return None

    values = obj.get("values", [])
    if not values:
        st.warning(f"Twelve Data ไม่พบข้อมูลของ {symbol} (TF: {interval})")
        return None

    df = pd.DataFrame(values).rename(columns={"datetime": "timestamp"})
    if "volume" not in df.columns:
        df["volume"] = 0
    needed = ["timestamp", "open", "high", "low", "close", "volume"]
    return _normalize_ohlcv(df[needed])


@st.cache_data(show_spinner=False, ttl=600)
def search_symbols_twelvedata(query: str, api_key: str) -> list[str]:
    keyword = query.strip().upper()
    if not keyword:
        return []

    params = {
        "symbol": keyword,
        "outputsize": 20,
        "apikey": api_key,
    }
    url = f"https://api.twelvedata.com/symbol_search?{urlencode(params)}"
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=15) as response:
            payload = response.read().decode("utf-8")
    except Exception:
        return []

    try:
        obj = json.loads(payload)
    except Exception:
        return []

    if not isinstance(obj, dict):
        return []

    candidates = obj.get("data", []) or []
    out: list[str] = []
    for item in candidates:
        symbol = str(item.get("symbol", "")).upper().strip()
        if not symbol:
            continue

        instrument_type = str(item.get("instrument_type", "")).lower()
        is_crypto = "crypto" in instrument_type
        if not is_crypto:
            country = str(item.get("country", "")).upper()
            if country and country not in {"US", "USA", "UNITED STATES"}:
                continue
            if instrument_type and ("stock" not in instrument_type and "etf" not in instrument_type and "adr" not in instrument_type):
                continue

        if symbol not in out:
            out.append(symbol)
        if len(out) >= 12:
            break

    return out


def _heuristic_market_mode(symbol: str) -> str:
    raw = symbol.strip().upper()
    normalized = re.sub(r"[^A-Z0-9]", "", raw)
    crypto_roots = {
        "BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "AVAX", "LTC", "DOT",
        "LINK", "UNI", "ATOM", "MATIC", "TRX", "BCH", "ETC", "XLM", "NEAR", "APT",
    }
    if "/" in raw:
        return "crypto_24x7"
    if normalized.endswith("USDT"):
        return "crypto_24x7"
    if len(normalized) > 3 and normalized.endswith("USD") and normalized[:-3] in crypto_roots:
        return "crypto_24x7"
    return "us_equity"


@st.cache_data(show_spinner=False, ttl=3600)
def resolve_market_mode_twelvedata(symbol: str, api_key: str) -> str:
    keyword = symbol.strip().upper()
    if not keyword:
        return "us_equity"
    heuristic_mode = _heuristic_market_mode(keyword)
    if heuristic_mode == "crypto_24x7":
        return heuristic_mode
    if keyword.isalpha() and len(keyword) <= 6:
        return "us_equity"

    params = {
        "symbol": keyword,
        "outputsize": 30,
        "apikey": api_key,
    }
    url = f"https://api.twelvedata.com/symbol_search?{urlencode(params)}"
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=15) as response:
            payload = response.read().decode("utf-8")
        obj = json.loads(payload)
    except Exception:
        return heuristic_mode

    if not isinstance(obj, dict):
        return heuristic_mode

    candidates = obj.get("data", []) or []
    normalized_keyword = re.sub(r"[^A-Z0-9]", "", keyword)
    best_score = -1
    best_item: dict | None = None

    for item in candidates:
        sym = str(item.get("symbol", "")).upper().strip()
        if not sym:
            continue

        normalized_sym = re.sub(r"[^A-Z0-9]", "", sym)
        if normalized_sym != normalized_keyword and sym != keyword:
            continue

        instrument_type = str(item.get("instrument_type", "")).lower()
        country = str(item.get("country", "")).upper()
        exchange = str(item.get("exchange", "")).upper()

        score = 100 if sym == keyword else 85
        if "crypto" in instrument_type:
            score += 40
        if country in {"US", "USA", "UNITED STATES"}:
            score += 15
        if any(tag in exchange for tag in ["NASDAQ", "NYSE", "AMEX"]):
            score += 10

        if score > best_score:
            best_score = score
            best_item = item

    if best_item is None:
        return heuristic_mode

    instrument_type = str(best_item.get("instrument_type", "")).lower()
    if "crypto" in instrument_type:
        return "crypto_24x7"
    if any(tag in instrument_type for tag in ["stock", "etf", "adr"]):
        return "us_equity"
    return heuristic_mode


def to_plot_timestamps(ts: pd.Series, market_mode: str) -> pd.Series:
    parsed = pd.to_datetime(ts, errors="coerce")
    if getattr(parsed.dt, "tz", None) is None:
        parsed = parsed.dt.tz_localize("Asia/Bangkok")
    else:
        parsed = parsed.dt.tz_convert("Asia/Bangkok")
    if market_mode == "us_equity":
        parsed = parsed.dt.tz_convert("America/New_York")
    elif market_mode == "crypto_24x7":
        parsed = parsed.dt.tz_convert("UTC")
    return parsed.dt.tz_localize(None)


def _missing_us_equity_session_days(plot_time: pd.Series | None) -> list[str]:
    if plot_time is None:
        return []

    ts = pd.to_datetime(plot_time, errors="coerce")
    ts = ts.dropna()
    if ts.empty:
        return []

    seen_days = pd.Index(ts.dt.normalize().unique()).sort_values()
    if seen_days.empty:
        return []

    start_day = pd.Timestamp(seen_days.min()).normalize()
    end_day = pd.Timestamp(seen_days.max()).normalize()
    expected_weekdays = pd.date_range(start_day, end_day, freq="B")
    missing_days = expected_weekdays.difference(seen_days)
    if missing_days.empty:
        return []
    return [pd.Timestamp(day).strftime("%Y-%m-%d") for day in missing_days]


def build_xaxis_rangebreaks(interval: str, market_mode: str, plot_time: pd.Series | None = None) -> list[dict[str, Any]]:
    if market_mode == "crypto_24x7":
        return []

    breaks: list[dict[str, Any]] = [dict(bounds=["sat", "mon"])]
    if market_mode == "us_equity":
        missing_session_days = _missing_us_equity_session_days(plot_time)
        if missing_session_days:
            breaks.append(dict(values=missing_session_days))
    if market_mode == "us_equity" and interval in {"15min", "1h", "4h"}:
        breaks.append(dict(bounds=[16, 9.5], pattern="hour"))
    return breaks


@st.cache_data(show_spinner=False, ttl=180)
def run_strategy_cached(df: pd.DataFrame, selected_strategy: str) -> dict:
    return run_strategy(df.copy(), selected_strategy)


def strategy_guide(selected: str) -> str:
    _tools = {
        "SMC (BOS/OB/FVG)": [
            ("📐", "BOS", "การเบรกโครงสร้างราคาเพื่อบอกทิศหลัก"),
            ("📦", "OB", "โซนคำสั่งสะสมที่ราคามักมีแรงตอบสนอง"),
            ("⚡", "FVG", "ช่องว่างราคา (Imbalance) ที่ราคามักกลับมาเติม"),
            ("📊", "Volume", "ปริมาณซื้อขายเพื่อยืนยันแรงเข้า"),
        ],
        "Momentum Breakout (EMA50 + VCP)": [
            ("📈", "EMA50", "เส้นวัดแนวโน้มระยะกลาง"),
            ("🔺", "20D High", "แนวต้านสำคัญรอบล่าสุด"),
            ("🔻", "VCP", "รูปแบบบีบตัวก่อนเบรก"),
            ("📊", "Volume", "ยืนยันแรงเบรกด้วยวอลุ่มสูง"),
        ],
        "Pullback Reversal (EMA200 + Fib + RSI)": [
            ("📈", "EMA200", "เทรนด์หลักระยะยาว"),
            ("🎯", "Fib 50-61.8%", "โซนย่อที่มักมีแรงรับ"),
            ("📉", "RSI(14)", "วัดแรงซื้อ/ขายและภาวะ oversold"),
            ("🕯️", "Candle Signal", "แท่งกลับตัวเพื่อยืนยันจุดเข้า"),
        ],
        "Volume Profile (POC)": [
            ("📊", "POC", "ระดับราคาที่มีการซื้อขายสูงสุด"),
            ("📈", "Volume", "ใช้ยืนยันแรงปฏิเสธที่แนว POC"),
            ("🕯️", "Price Action", "พฤติกรรมเด้ง/หลุดของราคา"),
        ],
    }
    _steps = {
        "SMC (BOS/OB/FVG)": [
            ("Trend", "เช็ก Trend ก่อน: มี BOS ขาขึ้นหรือไม่"),
            ("Zone", "ราคาอยู่/เด้งจาก OB หรือ FVG"),
            ("Trigger", "แท่งกลับตัวหรือวอลุ่มพุ่ง"),
            ("สรุป", "สรุปสถานะ + Entry / SL / TP (Target: Fib 1.618 / Zone)"),
        ],
        "Momentum Breakout (EMA50 + VCP)": [
            ("Trend", "ราคาอยู่เหนือ EMA50 หรือไม่"),
            ("Zone", "ราคาใกล้แนวต้านและมีลักษณะบีบตัว"),
            ("Trigger", "เบรกแนวต้านพร้อมวอลุ่ม > 2x ค่าเฉลี่ย"),
            ("สรุป", "สรุปสถานะ + Entry / SL / TP (Target: Measured Move)"),
        ],
        "Pullback Reversal (EMA200 + Fib + RSI)": [
            ("Trend", "ราคาอยู่เหนือ EMA200 (เทรนด์ยังเป็นบวก)"),
            ("Zone", "ราคา Pullback เข้าช่วง Fib 50%-61.8%"),
            ("Trigger", "RSI อ่อนแรง/bullish divergence + แท่งกลับตัว"),
            ("สรุป", "สรุปสถานะ + Entry / SL / TP (Target: Swing High)"),
        ],
        "Volume Profile (POC)": [
            ("Trend", "ราคา > EMA50 และหา POC จาก Volume Profile"),
            ("Zone", "เช็กว่าราคากำลังเทสโซน POC หรือไม่"),
            ("Trigger", "ดูแรงเด้งกลับ + วอลุ่มยืนยัน"),
            ("สรุป", "สรุปสถานะ + Entry / SL / TP (Target: VAH)"),
        ],
    }
    tools = _tools[selected]
    steps = _steps[selected]

    tools_html = "".join(
        f"<div style='display:flex;align-items:flex-start;gap:8px;padding:6px 0;'>"
        f"<span style='font-size:18px;line-height:1;'>{icon}</span>"
        f"<div><span style='font-weight:700;color:#0f172a;font-size:13px;'>{name}</span>"
        f"<div style='font-size:12px;color:#64748b;margin-top:1px;'>{desc}</div></div></div>"
        for icon, name, desc in tools
    )
    steps_html = "".join(
        f"<div style='display:flex;align-items:flex-start;gap:8px;padding:5px 0;'>"
        f"<span style='display:inline-flex;align-items:center;justify-content:center;"
        f"width:22px;height:22px;border-radius:50%;background:#e0f2fe;color:#0369a1;"
        f"font-size:11px;font-weight:700;flex-shrink:0;'>{i}</span>"
        f"<div style='font-size:12.5px;color:#334155;line-height:1.5;'>"
        f"<b style='color:#0f172a;'>{label}</b>: {desc}</div></div>"
        for i, (label, desc) in enumerate(steps, 1)
    )

    return f"""
    <div style='font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;max-width:380px;'>
        <div style='background:linear-gradient(135deg,#0f766e,#0e7490);color:#fff;padding:12px 16px;
             border-radius:10px;margin-bottom:12px;display:flex;align-items:center;gap:8px;'>
            <span style='font-size:20px;'>📘</span>
            <span style='font-size:14px;font-weight:700;'>คู่มือ: {html.escape(selected)}</span>
        </div>
        <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;margin-bottom:10px;'>
            <div style='font-size:12px;font-weight:700;color:#475569;text-transform:uppercase;
                 letter-spacing:0.5px;margin-bottom:6px;'>🔧 เครื่องมือที่ใช้</div>
            {tools_html}
        </div>
        <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;'>
            <div style='font-size:12px;font-weight:700;color:#475569;text-transform:uppercase;
                 letter-spacing:0.5px;margin-bottom:6px;'>🔍 ขั้นตอนตรวจสัญญาณ</div>
            {steps_html}
        </div>
    </div>
    """.strip()


def render_backtest_results_content(
    current_bt_params: tuple[str, str, str, str],
    ticker: str,
    interval: str,
    strategy: str,
) -> None:
    ticker_safe = html.escape(ticker)
    interval_safe = html.escape(interval)
    strategy_safe = html.escape(strategy)

    # ── Inline style tokens ──
    S = {
        "wrap": "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:320px;",
        "header": (
            "background:linear-gradient(135deg,#0f766e,#0e7490);color:#fff;padding:14px 16px;"
            "border-radius:10px;margin-bottom:12px;display:flex;align-items:center;gap:10px;"
        ),
        "card": (
            "background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;"
            "padding:12px 14px;margin-bottom:10px;white-space:normal;overflow-wrap:break-word;"
        ),
        "metric_grid": "display:grid;grid-template-columns:1fr 1fr;gap:8px;",
        "metric_box": (
            "background:#fff;border:1px solid #e2e8f0;border-radius:8px;"
            "padding:10px 12px;text-align:center;"
        ),
        "metric_val": "font-size:22px;font-weight:800;line-height:1.2;",
        "metric_lbl": "font-size:12px;color:#64748b;margin-top:2px;",
        "detail_row": (
            "display:flex;justify-content:space-between;padding:6px 0;"
            "font-size:13px;color:#334155;border-bottom:1px solid #f1f5f9;"
        ),
        "footer": "font-size:11px;color:#94a3b8;text-align:right;margin-top:6px;",
        "hint": (
            "background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;"
            "padding:14px;text-align:center;color:#0369a1;font-size:13px;"
        ),
        "error": (
            "background:#fef2f2;border:1px solid #fecaca;border-radius:8px;"
            "padding:14px;text-align:center;color:#b91c1c;font-size:13px;"
        ),
    }

    html_parts = [f"<div style='{S['wrap']}'>"]

    # Header badge
    html_parts.append(
        f"<div style='{S['header']}'>"
        f"<span style='font-size:18px;'>📊</span>"
        f"<div><div style='font-size:15px;font-weight:700;letter-spacing:0.3px;'>Backtest Results</div>"
        f"<div style='font-size:12px;opacity:0.85;margin-top:2px;'>"
        f"{ticker_safe} · {interval_safe} · {strategy_safe}</div></div></div>"
    )

    # Methodology notes (collapsible)
    html_parts.append(
        f"<details style='{S['card']}cursor:pointer;'>"
        "<summary style='font-size:13px;font-weight:700;color:#475569;"
        "letter-spacing:0.3px;list-style:none;'>"
        "\u2139\ufe0f \u0e2b\u0e21\u0e32\u0e22\u0e40\u0e2b\u0e15\u0e38\u0e27\u0e34\u0e18\u0e35\u0e04\u0e33\u0e19\u0e27\u0e13 \u25b8</summary>"
        "<div style='margin-top:8px;font-size:12px;color:#64748b;line-height:1.6;'>"
        "<div style='padding:3px 0;'>\u2022 \u0e43\u0e0a\u0e49\u0e2a\u0e31\u0e0d\u0e0d\u0e32\u0e13 \u201cBUY\u201d \u0e02\u0e2d\u0e07\u0e01\u0e25\u0e22\u0e38\u0e17\u0e18\u0e4c \u0e08\u0e33\u0e25\u0e2d\u0e07\u0e04\u0e33\u0e2a\u0e31\u0e48\u0e07 Entry/SL/TP (Structure Based)</div>"
        "<div style='padding:3px 0;'>\u2022 <b>Realism</b>: Slippage 0.05% + Comm 0.1% + No Overlap</div>"
        "<div style='padding:3px 0;'>\u2022 <b>Entry Fill</b>: รองรับ Market / Limit / Stop ตามกลยุทธ์</div>"
        "<div style='padding:3px 0;'>\u2022 <b>Intrabar</b>: \u0e16\u0e49\u0e32\u0e41\u0e17\u0e48\u0e07\u0e40\u0e14\u0e35\u0e22\u0e27\u0e0a\u0e19 TP+SL \u2192 \u0e16\u0e37\u0e2d\u0e27\u0e48\u0e32\u0e41\u0e1e\u0e49</div>"
        "<div style='padding:3px 0;'>\u2022 <b>Time Stop</b>: \u0e04\u0e23\u0e1a Horizon \u2192 \u0e1b\u0e34\u0e14\u0e17\u0e35\u0e48 Close</div>"
        "<div style='padding:3px 0;'>\u2022 \u0e21\u0e35 cache \u0e40\u0e1e\u0e37\u0e48\u0e2d\u0e04\u0e33\u0e19\u0e27\u0e13\u0e40\u0e09\u0e1e\u0e32\u0e30\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e43\u0e2b\u0e21\u0e48</div>"
        "</div></details>"
    )

    if st.session_state.get("backtest_params") == current_bt_params and isinstance(st.session_state.get("backtest_summary"), dict):
        bt_summary = st.session_state["backtest_summary"]
        bt_cache_hit = bool(st.session_state.get("backtest_cache_hit"))
        start_ts = bt_summary.get("start")
        end_ts = bt_summary.get("end")
        updated_ts = bt_summary.get("last_refresh_utc")
        start_txt = start_ts.strftime("%Y-%m-%d") if isinstance(start_ts, pd.Timestamp) and not pd.isna(start_ts) else "-"
        end_txt = end_ts.strftime("%Y-%m-%d") if isinstance(end_ts, pd.Timestamp) and not pd.isna(end_ts) else "-"
        updated_txt = updated_ts.strftime("%Y-%m-%d %H:%M UTC") if isinstance(updated_ts, pd.Timestamp) and not pd.isna(updated_ts) else "-"
        cache_badge = "cache" if bt_cache_hit else "incremental"

        sig_c = int(bt_summary.get("signal_count", 0))
        filled_c = int(bt_summary.get("filled_count", max(sig_c - int(bt_summary.get("canceled_count", 0)), 0)))
        fill_rate = float(bt_summary.get("fill_rate", (filled_c / sig_c * 100.0) if sig_c else 0.0))
        win_c = int(bt_summary.get("win_count", 0))
        loss_c = int(bt_summary.get("loss_count", 0))
        be_c = int(bt_summary.get("breakeven_count", 0))
        cancel_c = int(bt_summary.get("canceled_count", 0))
        win_rate = float(bt_summary.get("win_rate", 0.0))
        avg_r = float(bt_summary.get("avg_r", 0.0))
        expectancy = float(bt_summary.get("expectancy", 0.0))
        max_dd = float(bt_summary.get("max_drawdown_r", 0.0))
        tpm = float(bt_summary.get("trades_per_month", 0.0))

        # Determine win rate color
        if win_rate >= 50:
            wr_color = "#059669"
        elif win_rate >= 35:
            wr_color = "#d97706"
        else:
            wr_color = "#dc2626"

        # Date range badge
        html_parts.append(
            f"<div style='text-align:center;margin-bottom:10px;font-size:13px;color:#475569;'>"
            f"\U0001f4c5 {start_txt}  \u2192  {end_txt}</div>"
        )

        # Key metrics grid (2x2)
        html_parts.append(f"<div style='{S['metric_grid']}margin-bottom:10px;'>")
        # Signals
        html_parts.append(
            f"<div style='{S['metric_box']}'>"
            f"<div style='{S['metric_val']}color:#0f766e;'>{sig_c:,}</div>"
            f"<div style='{S['metric_lbl']}'>\u0e2a\u0e31\u0e0d\u0e0d\u0e32\u0e13 BUY</div></div>"
        )
        # Win Rate
        html_parts.append(
            f"<div style='{S['metric_box']}'>"
            f"<div style='{S['metric_val']}color:{wr_color};'>{win_rate:.1f}%</div>"
            f"<div style='{S['metric_lbl']}'>Win Rate (Filled)</div>"
            f"<div style='height:3px;background:#e2e8f0;border-radius:2px;margin-top:4px;'>"
            f"<div style='height:100%;width:{min(win_rate, 100):.0f}%;background:{wr_color};"
            f"border-radius:2px;'></div></div></div>"
        )
        # Win / Loss
        html_parts.append(
            f"<div style='{S['metric_box']}'>"
            f"<div style='{S['metric_val']}'>"
            f"<span style='color:#059669;'>{win_c}</span>"
            f"<span style='color:#94a3b8;font-size:14px;'> / </span>"
            f"<span style='color:#dc2626;'>{loss_c}</span></div>"
            f"<div style='{S['metric_lbl']}'>\u0e0a\u0e19\u0e30 / \u0e41\u0e1e\u0e49</div></div>"
        )
        # Avg R
        r_color = "#059669" if avg_r > 0 else "#dc2626" if avg_r < 0 else "#64748b"
        html_parts.append(
            f"<div style='{S['metric_box']}'>"
            f"<div style='{S['metric_val']}color:{r_color};'>{avg_r:+.3f}</div>"
            f"<div style='{S['metric_lbl']}'>Avg R</div></div>"
        )
        html_parts.append("</div>")  # close grid

        # Detail rows card
        html_parts.append(f"<div style='{S['card']}'>")
        details = [
            ("Filled / Canceled", f"{filled_c:,} / {cancel_c:,}"),
            ("Fill Rate", f"{fill_rate:.1f}%"),
            ("Breakeven", f"{be_c:,}"),
            ("Expectancy", f"{expectancy:.3f}"),
            ("Max Drawdown (R)", f"{max_dd:.3f}"),
            ("Trades / Month", f"{tpm:.2f}"),
        ]
        for lbl, val in details:
            html_parts.append(
                f"<div style='{S['detail_row']}'>"
                f"<span style='color:#64748b;'>{lbl}</span>"
                f"<span style='font-weight:600;'>{val}</span></div>"
            )
        html_parts.append("</div>")

        # Footer
        html_parts.append(
            f"<div style='{S['footer']}'>\U0001f550 {updated_txt} "
            f"<span style='background:#e2e8f0;padding:1px 6px;border-radius:4px;"
            f"font-size:10px;'>{cache_badge}</span></div>"
        )
    elif st.session_state.get("backtest_params") == current_bt_params and st.session_state.get("backtest_error"):
        err = html.escape(str(st.session_state["backtest_error"]))
        html_parts.append(
            f"<div style='{S['error']}'>"
            f"\u26a0\ufe0f Backtest \u0e17\u0e33\u0e07\u0e32\u0e19\u0e44\u0e21\u0e48\u0e2a\u0e33\u0e40\u0e23\u0e47\u0e08<br>"
            f"<span style='font-size:12px;'>{err}</span></div>"
        )
    else:
        html_parts.append(
            f"<div style='{S['hint']}'>"
            f"\U0001f680 \u0e01\u0e14 <b>Run Backtest</b> \u0e40\u0e1e\u0e37\u0e48\u0e2d\u0e04\u0e33\u0e19\u0e27\u0e13\u0e1c\u0e25\u0e2a\u0e33\u0e2b\u0e23\u0e31\u0e1a "
            f"{ticker_safe} ({interval_safe})</div>"
        )

    html_parts.append("</div>")
    st.markdown("\n".join(html_parts), unsafe_allow_html=True)





_DEAD_CODE = r"""
        "<div class='m-head'>หมายเหตุ</div>",
        "<ul>",
        "<li>ใช้สัญญาณ “สถานะ BUY” ของกลยุทธ์ในแต่ละแท่ง แล้วจำลองคำสั่งตาม Entry/SL/TP ของกลยุทธ์นั้น</li>",
        "<li><b>Entry Fill</b>: ต้องให้ราคาแตะระดับ Entry ก่อน (จำลอง Limit/Stop)</li>",
        "<li><b>Intrabar</b>: ถ้าแท่งเดียวชน TP และ SL ให้ถือว่าแพ้ (Pessimistic Execution)</li>",
        "<li><b>Time Stop</b>: ครบจำนวนแท่ง Horizon แล้วยังไม่จบเกม จะปิดที่ Close และตัดสินจากกำไร/ขาดทุน</li>",
        "<li>มีการ cache ลงไฟล์ เพื่อไม่ต้องรันใหม่ทั้งชุด และจะคำนวณเฉพาะช่วงข้อมูลใหม่เพิ่มเข้ามา</li>",









                "<div class='m-head'>สรุปผล</div>",
                "<ul>",
                f"<li><b>ช่วงข้อมูล</b>: {start_txt} ถึง {end_txt}</li>",
                f"<li><b>สินทรัพย์</b>: {ticker_safe} ({interval_safe})</li>",
                f"<li><b>กลยุทธ์</b>: {strategy_safe}</li>",
จำนวนสัญญาณ BUY</b>: {int(bt_summary.get('signal_count', 0)):,}</li>",
ชนะ / แพ้ / เสมอ</b>: "






อัปเดตล่าสุด</b>: {updated_txt} {cache_badge}</li>",
                "</ul>",

 ทำงานไม่สำเร็จ: {err}</div>")
กด Run Backtest เพื่อคำนวณผลสำหรับ {ticker_safe} ({interval_safe})</div>"







"""
del _DEAD_CODE


def _add_hline(fig: go.Figure, **kwargs: Any) -> None:
    # Plotly's runtime accepts int row/col, but current stubs can mis-type these kwargs.
    fig.add_hline(**kwargs)


def make_figure(
    df: pd.DataFrame,
    selected_strategy: str,
    analysis: dict,
    interval: str,
    market_mode: str,
    chart_height: int,
    chart_revision: int,
) -> go.Figure:
    plot_time = to_plot_timestamps(df["timestamp"], market_mode)

    candle_hover = [
        (
            f"Date: {ts:%d %b %Y %H:%M}<br>"
            f"Open: {o:.2f}<br>"
            f"High: {h:.2f}<br>"
            f"Low: {l:.2f}<br>"
            f"Close: {c:.2f}"
        )
        for ts, o, h, l, c in zip(
            plot_time,
            df["open"],
            df["high"],
            df["low"],
            df["close"],
        )
    ]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02, # Reduced gap
        row_heights=[0.7, 0.15, 0.15], # Taller Price Chart (70%)
        subplot_titles=(
            "MAIN CHART (OHLC + STRATEGY ZONES)",
            "VOLUME",
            "RSI (14)",
        ),
    )

    fig.add_trace(
        go.Candlestick(
            x=plot_time,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color="#10b981",
            decreasing_line_color="#f87171",
            increasing_fillcolor="#86efac",
            decreasing_fillcolor="#fecaca",
            hovertext=candle_hover,
            hoverinfo="text",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    vol_colors = np.where(df["close"] >= df["open"], "rgba(22, 163, 74, 0.82)", "rgba(220, 38, 38, 0.82)").tolist()
    if selected_strategy == "Momentum Breakout (EMA50 + VCP)":
        spike_mask = df["volume"] > (2 * df["vol_sma20"])
        for i, is_spike in enumerate(spike_mask.tolist()):
            if is_spike:
                vol_colors[i] = "rgba(59, 130, 246, 0.62)"

    fig.add_trace(
        go.Bar(
            x=plot_time,
            y=df["volume"],
            marker_color=vol_colors,
            name="Volume",
            showlegend=False,
            hovertemplate="Date: %{x|%d %b %Y %H:%M}<br>Volume: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    if interval == "15min":
        rsi_colors = np.where(
            df["rsi14"].diff().fillna(0) >= 0,
            "rgba(22, 163, 74, 0.38)",
            "rgba(220, 38, 38, 0.38)",
        ).tolist()
        fig.add_trace(
            go.Bar(
                x=plot_time,
                y=df["rsi14"],
                marker_color=rsi_colors,
                name="RSI Momentum",
                showlegend=False,
                opacity=0.95,
                hovertemplate="Date: %{x|%d %b %Y %H:%M}<br>RSI: %{y:.2f}<extra></extra>",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_time,
                y=df["rsi14"],
                name="RSI(14)",
                line=dict(color="#0f172a", width=1.0),
                showlegend=False,
                hovertemplate="Date: %{x|%d %b %Y %H:%M}<br>RSI(14): %{y:.2f}<extra></extra>",
            ),
            row=3,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=plot_time,
                y=df["rsi14"],
                name="RSI(14)",
                line=dict(color="#334155", width=1.9),
                showlegend=False,
                hovertemplate="Date: %{x|%d %b %Y %H:%M}<br>RSI(14): %{y:.2f}<extra></extra>",
            ),
            row=3,
            col=1,
        )

    _add_hline(fig, y=70, line_dash="dash", line_color="#d1d5db", row=3, col=1)
    _add_hline(fig, y=30, line_dash="dash", line_color="#d1d5db", row=3, col=1)
    
    if selected_strategy == "SMC (BOS/OB/FVG)":
        for zone in analysis.get("fvg_zones", [])[-4:]:
            x0_idx = zone["start_index"]
            if x0_idx < len(plot_time):
                x0 = plot_time.iloc[x0_idx]
                fig.add_shape(
                    type="rect", xref="x", yref="y", x0=x0, x1=plot_time.iloc[-1], y0=zone["low"], y1=zone["high"],
                    fillcolor="rgba(147, 197, 253, 0.25)", line=dict(color="rgba(59,130,246,0.38)", width=1), row=1, col=1,
                )
        for zone in analysis.get("ob_zones", [])[-4:]:
            x0_idx = zone["index"]
            if x0_idx < len(plot_time):
                x0 = plot_time.iloc[x0_idx]
                fig.add_shape(
                    type="rect", xref="x", yref="y", x0=x0, x1=plot_time.iloc[-1], y0=zone["low"], y1=zone["high"],
                    fillcolor="rgba(110, 231, 183, 0.28)", line=dict(color="rgba(16,185,129,0.42)", width=1), row=1, col=1,
                )
        for bos in analysis.get("bos_events", [])[-8:]:
            idx = bos["index"]
            if idx < len(plot_time):
                fig.add_annotation(
                    x=plot_time.iloc[idx], y=df["high"].iloc[idx], text="BOS", showarrow=True, arrowhead=2,
                    font=dict(color="#1e3a8a", size=10), bgcolor="rgba(255,255,255,0.75)", bordercolor="rgba(147,197,253,0.7)",
                    row=1, col=1,
                )

    if selected_strategy == "Momentum Breakout (EMA50 + VCP)":
        fig.add_trace(
            go.Scatter(x=plot_time, y=df["ema50"], mode="lines", line=dict(color="#2563eb", width=1.9), name="EMA 50"),
            row=1,
            col=1,
        )
        _add_hline(
            fig,
            y=analysis["resistance"],
            line_color="#9ca3af",
            line_width=1.8,
            annotation_text="20D Resistance",
            annotation_font_color="#6b7280",
            row=1,
            col=1,
        )

    if selected_strategy == "Pullback Reversal (EMA200 + Fib + RSI)":
        fig.add_trace(
            go.Scatter(x=plot_time, y=df["ema200"], mode="lines", line=dict(color="#0f766e", width=2.0), name="EMA 200"),
        )
        fib = analysis["fib"]
        fib50, fib618 = fib["50"], fib["61_8"]
        _add_hline(fig, y=fib50, line_color="#f59e0b", line_dash="dot", annotation_text="Fib 50%", row=1, col=1)
        _add_hline(fig, y=fib618, line_color="#a78bfa", line_dash="dot", annotation_text="Fib 61.8%", row=1, col=1)
        
        # Fib Zone
        low_idx = fib.get("low_idx", 0)
        if 0 <= low_idx < len(plot_time):
            x0 = plot_time.iloc[low_idx]
            fig.add_shape(
                type="rect",
                xref="x",
                yref="y",
                x0=x0,
                x1=plot_time.iloc[-1],
                y0=min(fib50, fib618),
                y1=max(fib50, fib618),
                fillcolor="rgba(254, 243, 199, 0.45)",
                line=dict(color="rgba(245,158,11,0.35)", width=1),
                row=1,
                col=1,
            )

    if selected_strategy == "Volume Profile (POC)":
        _add_hline(
            fig,
            y=analysis["poc"], line_color="#2563eb", line_width=1.9, annotation_text=f"POC {analysis['poc']}", row=1, col=1
        )

    fig.update_layout(
        template="plotly_white",
        height=chart_height,
        margin=dict(l=12, r=12, t=34, b=6),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=10, color="#6b7280"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e293b", size=12),
        hoverlabel=dict(bgcolor="#ffffff", font_color="#111827", bordercolor="#d1d5db"),
        dragmode="pan",
        hovermode="x unified",
        hoverdistance=8,
        spikedistance=1000,
        uirevision=f"zenith-chart-{chart_revision}",
        showlegend=False,
    )
    # Remove spikedistance=-1 (infinite) to reduce hover lag

    fig.update_annotations(font=dict(size=11, color="#475569"))
    fig.update_yaxes(
        title_text="Price",
        row=1,
        col=1,
        showgrid=True,
        gridcolor="#dbe3ec",
        tickfont=dict(color="#1e293b", size=12),
        title_font=dict(color="#334155", size=13),
        tickformat=".2f",
        zeroline=False,
        spikesnap="cursor",
        spikemode="toaxis+across",
        spikecolor="#334155",
        spikethickness=1,
        hoverformat=".2f",
        # Optimize Y-axis spikes
        showspikes=False,
    )
    fig.update_yaxes(
        title_text="Volume",
        row=2,
        col=1,
        showgrid=True,
        gridcolor="#e8edf3",
        tickfont=dict(color="#334155", size=11),
        title_font=dict(color="#475569", size=12),
        tickformat="~s",
        zeroline=False,
        showspikes=True,
        spikesnap="cursor",
        spikemode="toaxis+across",
    )
    fig.update_yaxes(
        title_text="RSI",
        range=[0, 100],
        row=3,
        col=1,
        showgrid=True,
        gridcolor="#e8edf3",
        tickfont=dict(color="#334155", size=11),
        title_font=dict(color="#475569", size=12),
        dtick=20,
        tickformat=".0f",
        zeroline=False,
        showspikes=True,
        spikesnap="cursor",
        spikemode="toaxis+across",
    )
    fig.update_xaxes(
        type="date",
        showgrid=True,
        gridcolor="#f1f5f9",
        fixedrange=False,
        spikecolor="#334155",
        spikethickness=1,
        rangebreaks=build_xaxis_rangebreaks(interval, market_mode, plot_time),
        tickformat="%d %b\n%H:%M" if interval in ("15min", "1h", "4h") else "%d %b\n%Y",
    )

    vol_ceiling = float(df["volume"].quantile(0.97))
    if vol_ceiling > 0:
        fig.update_yaxes(range=[0, vol_ceiling * 1.25], row=2, col=1)

    return fig


def checklist_value(value: bool) -> str:
    if value:
        return "<span class='check-pass'>🟢 ผ่าน</span>"
    return "<span class='check-fail'>🔴 ไม่ผ่าน</span>"


def fmt2(value: float) -> str:
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return str(value)


def market_phase_context(last_ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(last_ts)
    if pd.isna(ts):
        return "REGULAR"

    if ts.tzinfo is None:
        ts_bkk = ts.tz_localize("Asia/Bangkok")
    else:
        ts_bkk = ts.tz_convert("Asia/Bangkok")

    ts_ny = ts_bkk.tz_convert("America/New_York")
    minutes = (ts_ny.hour * 60) + ts_ny.minute
    weekday = ts_ny.weekday()

    if weekday < 5 and 570 <= minutes < 960:   # 09:30-16:00 NY
        phase = "REGULAR"
    elif weekday < 5 and 240 <= minutes < 570:   # 04:00-09:30 NY
        phase = "PRE"
    else:
        phase = "POST"

    return phase


def _daily_change_reference(df: pd.DataFrame, market_mode: str) -> tuple[float, float]:
    last_close = float(df["close"].iloc[-1])
    if len(df) < 2:
        return last_close, last_close

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("Asia/Bangkok")
    else:
        ts = ts.dt.tz_convert("Asia/Bangkok")

    if market_mode == "us_equity":
        ts = ts.dt.tz_convert("America/New_York")
    elif market_mode == "crypto_24x7":
        ts = ts.dt.tz_convert("UTC")

    close_num = pd.to_numeric(df["close"], errors="coerce")
    day_df = pd.DataFrame({"session_day": ts.dt.date, "close": close_num}).dropna()
    if day_df.empty:
        prev_close = float(df["close"].iloc[-2])
        return last_close, prev_close

    session_close = day_df.groupby("session_day", sort=True)["close"].last()
    if len(session_close) >= 2:
        prev_close = float(session_close.iloc[-2])
    else:
        prev_close = float(df["close"].iloc[-2])
    return last_close, prev_close


def render_top_stats(df: pd.DataFrame, ticker: str, interval: str, market_mode: str) -> None:
    last_close, prev_close = _daily_change_reference(df, market_mode)
    chg = last_close - prev_close
    chg_pct = (chg / prev_close * 100) if prev_close else 0.0
    color = "#0f766e" if chg >= 0 else "#b91c1c"
    phase = market_phase_context(df["timestamp"].iloc[-1])

    c1, c2, c3, c4 = st.columns(4, gap="small")
    c1.markdown(f"<div class='stat-card'><div class='tiny-label'>ชื่อหุ้น (Symbol)</div><div class='stat-value'>{ticker}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='stat-card'><div class='tiny-label'>Timeframe</div><div class='stat-value'>{interval}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='stat-card'><div class='tiny-label'>ราคาล่าสุด (USD)</div><div class='stat-value'>{fmt2(last_close)}</div></div>", unsafe_allow_html=True)
    c4.markdown(
        (
            "<div class='stat-card'>"
            f"<div class='tiny-label' style='white-space: nowrap;'>เปลี่ยนแปลง (Daily Change vs Prev Session) • <span class='phase-pill' style='color:{color}; border-color:{color};'>{phase}</span></div>"
            f"<div class='stat-value' style='color:{color};'>{chg:+.2f} ({chg_pct:+.2f}%)</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def main() -> None:
    if "sidebar_mini" not in st.session_state:
        st.session_state["sidebar_mini"] = False
    
    if "interval_sel" not in st.session_state:
        st.session_state["interval_sel"] = INTERVAL_OPTIONS[0]
    if "ticker_sel" not in st.session_state:
        st.session_state["ticker_sel"] = DEFAULT_TICKERS[0]
    if "ticker_options" not in st.session_state:
        st.session_state["ticker_options"] = DEFAULT_TICKERS.copy()
    if "ticker_hint" not in st.session_state:
        st.session_state["ticker_hint"] = "Tip: คลิกเลือกจากรายการ หรือพิมพ์ชื่อหุ้นแล้วกด Enter"
    if "strategy_sel" not in st.session_state:
        st.session_state["strategy_sel"] = STRATEGY_OPTIONS[0]
    if "run_analysis" not in st.session_state:
        st.session_state["run_analysis"] = True
    if "active_params" not in st.session_state:
        st.session_state["active_params"] = (
            st.session_state["ticker_sel"],
            st.session_state["interval_sel"],
            st.session_state["strategy_sel"],
        )
    if "cached_params" not in st.session_state:
        st.session_state["cached_params"] = None
    if "cached_df" not in st.session_state:
        st.session_state["cached_df"] = None
    if "cached_analysis" not in st.session_state:
        st.session_state["cached_analysis"] = None
    if "cached_fig_key" not in st.session_state:
        st.session_state["cached_fig_key"] = None
    if "cached_fig" not in st.session_state:
        st.session_state["cached_fig"] = None
    if "backtest_params" not in st.session_state:
        st.session_state["backtest_params"] = None
    if "backtest_summary" not in st.session_state:
        st.session_state["backtest_summary"] = None
    if "backtest_cache_hit" not in st.session_state:
        st.session_state["backtest_cache_hit"] = False
    if "backtest_error" not in st.session_state:
        st.session_state["backtest_error"] = None
    if "request_run_backtest" not in st.session_state:
        st.session_state["request_run_backtest"] = False
    if "backtest_is_running" not in st.session_state:
        st.session_state["backtest_is_running"] = False
    if "backtest_progress_pct" not in st.session_state:
        st.session_state["backtest_progress_pct"] = 0
    if "backtest_progress_text" not in st.session_state:
        st.session_state["backtest_progress_text"] = ""

    inject_css(st.session_state["sidebar_mini"])

    td_key = os.getenv("TWELVEDATA_API_KEY", "").strip() or os.getenv("TWELVE_DATA_API_KEY", "").strip()
    if not td_key:
        st.error("กรุณาตั้งค่า TWELVEDATA_API_KEY ในไฟล์ .env")
        st.stop()

    def merge_unique_symbols(*groups: list[str]) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for group in groups:
            for item in group:
                sym = str(item).strip().upper()
                if not sym or sym in seen:
                    continue
                seen.add(sym)
                merged.append(sym)
        return merged

    def sync_ticker_options() -> None:
        raw = str(st.session_state.get("ticker_sel", "")).strip().upper()
        base_hint = "Tip: คลิกเลือกจากรายการ หรือพิมพ์ชื่อหุ้นแล้วกด Enter"
        if not raw:
            st.session_state["ticker_options"] = DEFAULT_TICKERS.copy()
            st.session_state["ticker_sel"] = DEFAULT_TICKERS[0]
            st.session_state["ticker_hint"] = base_hint
            return

        if raw in DEFAULT_TICKERS:
            st.session_state["ticker_options"] = DEFAULT_TICKERS.copy()
            st.session_state["ticker_sel"] = raw
            st.session_state["ticker_hint"] = base_hint
            return

        matched = search_symbols_twelvedata(raw, td_key)
        if matched:
            selected = raw if raw in matched else matched[0]
            st.session_state["ticker_options"] = merge_unique_symbols(
                [selected],
                matched,
                DEFAULT_TICKERS,
            )
            st.session_state["ticker_sel"] = selected
            st.session_state["ticker_hint"] = f"ผลค้นหาใกล้เคียง: {', '.join(matched[:6])}"
        else:
            st.session_state["ticker_options"] = merge_unique_symbols([raw], DEFAULT_TICKERS)
            st.session_state["ticker_sel"] = raw
            st.session_state["ticker_hint"] = "ไม่พบในลิสต์อัตโนมัติ แต่จะลองดึงข้อมูลด้วยสัญลักษณ์ที่พิมพ์"

    refresh = False
    with st.sidebar:
        if not st.session_state["sidebar_mini"]:
            _, toggle_col = st.columns([0.72, 0.28], gap="small")
            with toggle_col:
                if st.button("«", key="sidebar_mini_toggle", use_container_width=True, help="ย่อเมนู"):
                    st.session_state["sidebar_mini"] = True
                    st.rerun()
            st.markdown("<div class='controls-title'>Zenith Controls</div>", unsafe_allow_html=True)
            st.selectbox("🕒 Timeframe", INTERVAL_OPTIONS, key="interval_sel")
            st.session_state["ticker_options"] = merge_unique_symbols(
                [st.session_state["ticker_sel"]],
                st.session_state["ticker_options"],
                DEFAULT_TICKERS,
            )
            st.selectbox(
                "💵 ชื่อหุ้น (Ticker Symbol - USD)",
                st.session_state["ticker_options"],
                key="ticker_sel",
                accept_new_options=True,
                on_change=sync_ticker_options,
                placeholder="เลือกจากลิสต์ หรือพิมพ์สัญลักษณ์แล้วกด Enter",
            )
            st.caption(st.session_state["ticker_hint"])
            st.selectbox("🧠 กลยุทธ์ (Strategy)", STRATEGY_OPTIONS, key="strategy_sel")
            refresh = st.button("📊 Analyze", use_container_width=True)
            st.markdown("<div class='side-spacer'></div>", unsafe_allow_html=True)
            _patch_note_content_120 = (
                "<h3 style='margin:0 0 6px 0;color:#0f766e;'>🚀 Patch Note — V1.2.0</h3>"
                "<p style='font-size:12px;color:#64748b;margin:0 0 10px 0;'>Release Date: 2026-02-18</p>"
                "<div style='background:#f0fdfa;padding:10px 12px;border-radius:8px;border:1px solid #ccfbf1;margin-bottom:10px;'>"
                "<div style='font-size:12px;font-weight:700;color:#115e59;margin-bottom:4px;'>Backtest + UI Quality Upgrade</div>"
                "<ul style='margin:0;padding-left:16px;font-size:12px;color:#334155;line-height:1.55;'>"
                "<li>Backtest รองรับ Order Type แยกตามกลยุทธ์: Market / Limit / Stop</li>"
                "<li>แก้ Momentum breakout trigger ให้ทำงานตามโครงสร้าง breakout จริง</li>"
                "<li>เพิ่ม data digest ใน cache เพื่อกันผล backtest ค้างเมื่อข้อมูลย้อนหลังถูกแก้</li>"
                "<li>แก้การคำนวณช่วงสแกนสัญญาณ + ช่วงวันที่ใน summary ให้ตรงกับช่วงที่คำนวณจริง</li>"
                "<li>เพิ่ม Filled Trades / Fill Rate แยกจาก Signal Count ใน Backtest Results</li>"
                "<li>ปรับ Max Drawdown ให้คำนวณตามลำดับเวลาอย่างถูกต้อง</li>"
                "<li>ปรับ UX เพิ่มเติม: cursor แบบมือใน controls ที่กดได้, กราฟขยายเต็มพื้นที่มากขึ้น</li>"
                "<li>แก้ Daily Change ให้เทียบกับ Previous Session Close แทน bar ก่อนหน้า</li>"
                "<li>ปรับความต่อเนื่องกราฟ intraday (ซ่อนช่วงนอกเวลาเทรดของ US equity)</li>"
                "</ul>"
                "</div>"
                "<div style='font-size:11px;color:#94a3b8;'>"
                "🧪 เพิ่ม regression tests ครอบคลุม core logic ของ backtest และ strategy หลัก"
                "</div>"
            )
            _patch_note_content_111 = (
                "<h3 style='margin:0 0 6px 0;color:#0f766e;'>🔬 Patch Note — V1.1.1</h3>"
                "<p style='font-size:12px;color:#64748b;margin:0 0 10px 0;'>Release Date: 2026-02-17</p>"
                "<div style='font-size:12px;color:#475569;line-height:1.6;'>"
                "• Smart Take Profit (Structure-Based) สำหรับทุกกลยุทธ์<br>"
                "• ปรับ Realism Engine: Slippage / Commission / No Overlap<br>"
                "• ปรับปรุง UI consistency และรายละเอียดใน popover ต่างๆ"
                "</div>"
                "<hr style='border:none;border-top:1px solid #e2e8f0;margin:8px 0;'>"
                "<div style='font-size:11px;color:#94a3b8;'>"
                "📝 รายละเอียดเต็ม: patch_notes/V1.1.0.md และ patch_notes/V1.1.1.md</div>"
            )
            st.markdown(
                "<div class='side-footer'>"
                "Zenith Analysis v1.2.0 | © 2026"
                "<br>Developed by: Krisanu Kinkhuntod"
                "</div>",
                unsafe_allow_html=True,
            )
            _pn_id = "zenith-patch-note-modal"
            # Escape content for JS string injection
            _safe_content_120 = _patch_note_content_120.replace("'", "\\'").replace("\n", "").replace('"', '\\"')
            _safe_content_111 = _patch_note_content_111.replace("'", "\\'").replace("\n", "").replace('"', '\\"')

            # Inject modal logic via components.html to bypass Streamlit markdown limits
            components.html(
                f"""
                <style>
                    body {{ margin: 0; font-family: sans-serif; overflow: hidden; }}
                    /* Button Style matching sidebar theme */
                    .patch-note-btn {{
                        display: inline-block;
                        width: 100%;
                        padding: 6px 12px;
                        font-size: 13px;
                        background: linear-gradient(135deg, #0f766e, #0e7490);
                        color: #ffffff;
                        border: none;
                        border-radius: 6px;
                        cursor: pointer;
                        text-align: center;
                        font-weight: 600;
                        letter-spacing: 0.3px;
                        transition: opacity 0.2s;
                        box-sizing: border-box;
                    }}
                    .patch-note-btn:hover {{ opacity: 0.9; }}
                </style>
                <button class="patch-note-btn" onclick="openPatchNote()">📋 Patch Note</button>
                <script>
                    function openPatchNote() {{
                        const parentDoc = window.parent.document;
                        if (!parentDoc) return;

                        let backdrop = parentDoc.getElementById('{_pn_id}-backdrop');
                        
                        // Create modal if not exists
                        if (!backdrop) {{
                            const overlay = parentDoc.createElement('div');
                            overlay.id = '{_pn_id}-backdrop';
                            overlay.onclick = (e) => {{
                                if (e.target.id === '{_pn_id}-backdrop') {{
                                    overlay.style.display = 'none';
                                }}
                            }};
                            Object.assign(overlay.style, {{
                                display: 'none',
                                position: 'fixed',
                                top: '0', left: '0', width: '100%', height: '100%',
                                background: 'rgba(0,0,0,0.5)',
                                zIndex: '999999',
                                justifyContent: 'center',
                                alignItems: 'center'
                            }});

                            // Content Box
                            const box = parentDoc.createElement('div');
                            Object.assign(box.style, {{
                                background: '#fff',
                                borderRadius: '14px',
                                padding: '24px 28px',
                                maxWidth: '460px',
                                width: '90%',
                                maxHeight: '90vh',
                                overflowY: 'auto',
                                position: 'relative',
                                boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
                                animation: 'pnSlideIn 0.25s ease-out'
                            }});
                            
                            // Close Button
                            const closeBtn = parentDoc.createElement('button');
                            closeBtn.innerHTML = '✕';
                            closeBtn.onclick = () => {{ overlay.style.display = 'none'; }};
                            Object.assign(closeBtn.style, {{
                                position: 'absolute', top: '10px', right: '14px',
                                background: 'none', border: 'none',
                                fontSize: '20px', cursor: 'pointer', color: '#94a3b8', lineHeight: '1'
                            }});
                            
                            // Inject Content
                            const contentDiv = parentDoc.createElement('div');
                            contentDiv.innerHTML = `
                                <div style="display:flex;gap:8px;margin-bottom:10px;">
                                    <button type="button" data-tab="v120" style="flex:1;padding:6px 8px;border:1px solid #99f6e4;background:#ecfeff;border-radius:8px;font-size:12px;font-weight:700;color:#115e59;cursor:pointer;">V1.2.0 (Latest)</button>
                                    <button type="button" data-tab="v111" style="flex:1;padding:6px 8px;border:1px solid #cbd5e1;background:#f8fafc;border-radius:8px;font-size:12px;font-weight:700;color:#475569;cursor:pointer;">V1.1.1</button>
                                </div>
                                <div data-pane="v120">{_safe_content_120}</div>
                                <div data-pane="v111" style="display:none;">{_safe_content_111}</div>
                            `;

                            const tabButtons = contentDiv.querySelectorAll('[data-tab]');
                            const panes = contentDiv.querySelectorAll('[data-pane]');
                            function activateTab(tabId) {{
                                panes.forEach((p) => {{
                                    p.style.display = (p.getAttribute('data-pane') === tabId) ? 'block' : 'none';
                                }});
                                tabButtons.forEach((b) => {{
                                    const active = b.getAttribute('data-tab') === tabId;
                                    b.style.background = active ? '#ecfeff' : '#f8fafc';
                                    b.style.borderColor = active ? '#99f6e4' : '#cbd5e1';
                                    b.style.color = active ? '#115e59' : '#475569';
                                }});
                            }}
                            tabButtons.forEach((btn) => {{
                                btn.onclick = (ev) => {{
                                    ev.preventDefault();
                                    ev.stopPropagation();
                                    activateTab(btn.getAttribute('data-tab'));
                                }};
                            }});
                            activateTab('v120');
                            
                            box.appendChild(closeBtn);
                            box.appendChild(contentDiv);
                            overlay.appendChild(box);
                            parentDoc.body.appendChild(overlay);

                            // Add animation style if not present
                            if (!parentDoc.getElementById('pn-anim-style')) {{
                                const style = parentDoc.createElement('style');
                                style.id = 'pn-anim-style';
                                style.textContent = `@keyframes pnSlideIn {{ from {{ transform:translateY(30px);opacity:0; }} to {{ transform:translateY(0);opacity:1; }} }}`;
                                parentDoc.head.appendChild(style);
                            }}
                            
                            backdrop = overlay;
                        }}
                        
                        backdrop.style.display = 'flex';
                    }}
                </script>
                """,
                height=45,
                scrolling=False
            )
        else:
            if st.button("»", key="sidebar_mini_toggle", use_container_width=True, help="เปิดเมนู"):
                st.session_state["sidebar_mini"] = False
                st.rerun()

    if refresh:
        st.session_state["run_analysis"] = True
        st.session_state["active_params"] = (
            st.session_state["ticker_sel"],
            st.session_state["interval_sel"],
            st.session_state["strategy_sel"],
        )
        # Ensure the UI updates immediately (fixes cases where the chart never renders
        # because Streamlit doesn't rerun after setting session_state from a button).
        st.rerun()

    selected_interval = st.session_state["interval_sel"]
    selected_ticker = st.session_state["ticker_sel"]
    selected_strategy = st.session_state["strategy_sel"]
    active_params = st.session_state["active_params"]
    display_strategy = active_params[2] if active_params else selected_strategy

    manual_html = strategy_guide(display_strategy)

    if st.session_state["active_params"] is None:
        st.session_state["active_params"] = (selected_ticker, selected_interval, selected_strategy)

    ticker, interval, strategy = st.session_state["active_params"]
    market_mode = resolve_market_mode_twelvedata(ticker, td_key)
    needs_refresh = (
        st.session_state["cached_params"] != st.session_state["active_params"]
        or st.session_state["cached_df"] is None
        or st.session_state["cached_analysis"] is None
    )

    if needs_refresh:
        with st.spinner("กำลังดึงข้อมูลและวิเคราะห์..."):
            df = fetch_ohlcv_twelvedata(ticker, td_key, interval)

        if df is None or len(df) < 60:
            st.warning("ข้อมูลไม่เพียงพอสำหรับการวิเคราะห์ (ต้องการอย่างน้อย 60 แท่ง)")
            st.stop()

        analysis = run_strategy_cached(df, strategy)
        st.session_state["cached_df"] = df
        st.session_state["cached_analysis"] = analysis
        st.session_state["cached_params"] = st.session_state["active_params"]
    else:
        df = st.session_state["cached_df"]
        analysis = st.session_state["cached_analysis"]

    current_bt_params = (ticker, interval, strategy, market_mode)
    if (
        st.session_state.get("backtest_params") != current_bt_params
        and not st.session_state.get("backtest_is_running")
        and not st.session_state.get("request_run_backtest")
    ):
        st.session_state["backtest_progress_pct"] = 0
        st.session_state["backtest_progress_text"] = ""

    top_container = st.container()
    with top_container:
        st.markdown("<div class='topbar-card-marker'></div>", unsafe_allow_html=True)
        cols = st.columns([2.6, 1.1, 1.1, 1.1], gap="small")
        with cols[0]:
            st.markdown(
                """
                <div class='brand-row'>
                    <div class='brand-mark'>↗</div>
                    <div class='brand-title'>Zenith</div>
                    <div class='brand-divider'>|</div>
                    <div class='brand-sub'>Smart Stock Analysis Terminal</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with cols[1]:
            with st.popover("📖 Strategy Manual", use_container_width=True):
                st.markdown(
                    f"<div class='zenith-popover-content zenith-manual-content'>{manual_html}</div>",
                    unsafe_allow_html=True,
                )

        with cols[2]:
            run_bt_top = st.button(
                "🚀 Run Backtest",
                key="run_backtest_top",
                use_container_width=True,
                disabled=bool(
                    st.session_state.get("request_run_backtest")
                    or st.session_state.get("backtest_is_running")
                ),
            )

        with cols[3]:
            with st.popover("📊 Backtest Results", use_container_width=True):
                render_backtest_results_content(current_bt_params, ticker, interval, strategy)

    if run_bt_top:
        st.session_state["request_run_backtest"] = True
        st.session_state["backtest_is_running"] = True
        st.session_state["backtest_error"] = None
        st.session_state["backtest_progress_pct"] = 0
        st.session_state["backtest_progress_text"] = f"Backtest {ticker}: เตรียมข้อมูล 0%"
        st.rerun()

    bt_progress_text = str(st.session_state.get("backtest_progress_text", "")).strip()
    bt_progress_pct = int(st.session_state.get("backtest_progress_pct", 0) or 0)
    top_progress_slot = st.empty()
    if bt_progress_text and st.session_state.get("backtest_is_running"):
        top_progress_slot.progress(max(0, min(bt_progress_pct, 100)), text=bt_progress_text)

    if st.session_state.get("request_run_backtest"):
        st.session_state["backtest_is_running"] = True
        progress = top_progress_slot.progress(
            max(0, min(bt_progress_pct, 100)),
            text=(bt_progress_text or f"Backtest {ticker}: เตรียมข้อมูล 0%"),
        )
        state = {"last_pct": max(int(bt_progress_pct), -1)}

        def on_backtest_progress(stage: str, done: int, total: int) -> None:
            total_safe = max(int(total), 1)
            done_safe = max(0, min(int(done), total_safe))
            ratio = float(done_safe) / float(total_safe)

            if stage == "signals":
                pct = int(ratio * 70.0)
                text = f"Backtest {ticker}: สแกนสัญญาณ {pct}%"
            elif stage == "trades":
                pct = 70 + int(ratio * 28.0)
                text = f"Backtest {ticker}: จำลองออเดอร์ {pct}%"
            elif stage in {"finalize", "complete"}:
                pct = 99 if stage == "finalize" else 100
                text = f"Backtest {ticker}: สรุปผล {pct}%"
            else:
                pct = max(state["last_pct"], 5)
                text = f"Backtest {ticker}: เตรียมข้อมูล {pct}%"

            pct = max(0, min(pct, 100))
            if pct > state["last_pct"]:
                progress.progress(pct, text=text)
                st.session_state["backtest_progress_pct"] = pct
                st.session_state["backtest_progress_text"] = text
                state["last_pct"] = pct

        _bt_ok = False
        try:
            bt_summary, bt_cache_hit = _get_backtest_summary(
                df, ticker, interval, strategy, market_mode, progress_callback=on_backtest_progress
            )
            st.session_state["backtest_params"] = current_bt_params
            st.session_state["backtest_summary"] = bt_summary
            st.session_state["backtest_cache_hit"] = bt_cache_hit
            st.session_state["backtest_error"] = None
            st.session_state["backtest_progress_pct"] = 100
            st.session_state["backtest_progress_text"] = f"Backtest {ticker}: สรุปผล 100%"
            progress.progress(100, text=st.session_state["backtest_progress_text"])
            _bt_ok = True
        except Exception as exc:
            st.session_state["backtest_params"] = current_bt_params
            st.session_state["backtest_summary"] = None
            st.session_state["backtest_cache_hit"] = False
            st.session_state["backtest_error"] = str(exc)

        # Always clean up running flags
        st.session_state["request_run_backtest"] = False
        st.session_state["backtest_is_running"] = False
        st.session_state["backtest_progress_pct"] = 0
        st.session_state["backtest_progress_text"] = ""
        top_progress_slot.empty()

        if _bt_ok:
            win_r = bt_summary.get("win_rate", 0)
            sig_c = bt_summary.get("signal_count", 0)
            win_c = bt_summary.get("win_count", 0)
            loss_c = bt_summary.get("loss_count", 0)
            st.session_state["_bt_popup"] = {
                "type": "success",
                "title": f"✅ Backtest {ticker} เสร็จสิ้น",
                "detail": (
                    f"สัญญาณ {sig_c} รายการ  •  ชนะ {win_c} / แพ้ {loss_c}\n"
                    f"Win Rate {win_r:.1f}%"
                ),
            }
        else:
            err_msg = st.session_state.get("backtest_error", "Unknown error")
            st.session_state["_bt_popup"] = {
                "type": "error",
                "title": f"❌ Backtest {ticker} ล้มเหลว",
                "detail": str(err_msg)[:200],
            }

        # Rerun to update UI (popover, etc.)
        st.rerun()

    # Show persistent error banner if backtest failed
    bt_err = st.session_state.get("backtest_error")
    if bt_err and st.session_state.get("backtest_params") == current_bt_params:
        st.error(f"Backtest ทำงานไม่สำเร็จ: {bt_err}")

    # ---------- Backtest completion popup ----------
    _bt_popup = st.session_state.pop("_bt_popup", None)
    if isinstance(_bt_popup, dict):
        _pop_type = html.escape(str(_bt_popup.get("type", "success")))
        _pop_title = html.escape(str(_bt_popup.get("title", "")))
        _pop_detail = html.escape(str(_bt_popup.get("detail", ""))).replace("\n", "<br>")
        _pop_bg = "#0f766e" if _pop_type == "success" else "#b91c1c"
        _pop_icon = "🎯" if _pop_type == "success" else "⚠️"
        components.html(
            f"""
            <script>
            (function() {{
                var w = window.parent;
                if (!w) return;
                var doc = w.document;

                // Remove any previous popup
                var old = doc.getElementById('btPopupOverlay');
                if (old) old.remove();

                // Inject CSS once
                if (!doc.getElementById('btPopupStyles')) {{
                    var style = doc.createElement('style');
                    style.id = 'btPopupStyles';
                    style.textContent = `
                        .bt-popup-overlay {{
                            position: fixed; inset: 0; z-index: 99999;
                            display: flex; align-items: center; justify-content: center;
                            background: rgba(15, 23, 42, 0.45);
                            animation: btFadeIn 0.2s ease;
                        }}
                        .bt-popup-card {{
                            background: #ffffff; border-radius: 16px;
                            box-shadow: 0 20px 60px rgba(15, 23, 42, 0.25);
                            padding: 28px 36px; max-width: 420px; width: 90%;
                            text-align: center; position: relative;
                            animation: btSlideUp 0.3s ease;
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        }}
                        .bt-popup-icon {{ font-size: 38px; margin-bottom: 8px; }}
                        .bt-popup-title {{
                            font-size: 17px; font-weight: 700;
                            margin-bottom: 8px;
                        }}
                        .bt-popup-detail {{
                            font-size: 14px; color: #475569; line-height: 1.6;
                        }}
                        .bt-popup-bar {{
                            height: 4px; border-radius: 2px;
                            margin-top: 16px; animation: btShrink 4s linear forwards;
                        }}
                        .bt-popup-hint {{
                            font-size: 11px; color: #94a3b8; margin-top: 10px;
                        }}
                        @keyframes btFadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
                        @keyframes btSlideUp {{ from {{ transform: translateY(20px); opacity: 0; }} to {{ transform: translateY(0); opacity: 1; }} }}
                        @keyframes btShrink {{ from {{ width: 100%; }} to {{ width: 0%; }} }}
                    `;
                    doc.head.appendChild(style);
                }}

                var popColor = '{_pop_bg}';

                var overlay = doc.createElement('div');
                overlay.id = 'btPopupOverlay';
                overlay.className = 'bt-popup-overlay';
                overlay.onclick = function() {{ overlay.remove(); }};

                overlay.innerHTML = `
                    <div class="bt-popup-card" onclick="event.stopPropagation()">
                        <div class="bt-popup-icon">{_pop_icon}</div>
                        <div class="bt-popup-title" style="color:${{popColor}}">{_pop_title}</div>
                        <div class="bt-popup-detail">{_pop_detail}</div>
                        <div class="bt-popup-bar" style="background:${{popColor}}"></div>
                        <div class="bt-popup-hint">คลิกที่ใดก็ได้เพื่อปิด</div>
                    </div>
                `;
                doc.body.appendChild(overlay);

                setTimeout(function() {{
                    var el = doc.getElementById('btPopupOverlay');
                    if (el) el.remove();
                }}, 4000);
            }})();
            </script>
            """,
            height=0,
            width=0,
        )

    render_top_stats(df, ticker, interval, market_mode)
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

    left, right = st.columns([3.0, 1.0], gap="small")

    with left:
        chart_height = 680
        fig_key = (
            ticker,
            interval,
            strategy,
            market_mode,
            chart_height,
            FIGURE_SCHEMA_VERSION,
        )
        if st.session_state["cached_fig_key"] != fig_key or st.session_state["cached_fig"] is None:
            fig = make_figure(
                df,
                strategy,
                analysis,
                interval,
                market_mode,
                chart_height=chart_height,
                chart_revision=abs(hash(fig_key)) % 1_000_000,
            )
            st.session_state["cached_fig"] = fig
            st.session_state["cached_fig_key"] = fig_key
        else:
            fig = st.session_state["cached_fig"]
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "displayModeBar": False,
                "scrollZoom": True,
                "doubleClick": "reset+autosize",
                "responsive": True,
            },
        )
        components.html(
            """
            <script>
            (function() {
              const w = window.parent;
              if (!w) return;

              function formatPrice(value) {
                if (!Number.isFinite(value)) return '';
                const abs = Math.abs(value);
                const digits = abs >= 1 ? 2 : 4;
                return value.toLocaleString(undefined, {
                  minimumFractionDigits: digits,
                  maximumFractionDigits: digits
                });
              }

              function ensureYAxisCursorWidgets(host) {
                let label = host.querySelector('.zenith-y-cursor-label');
                let guide = host.querySelector('.zenith-y-cursor-guide');

                if (!label) {
                  label = w.document.createElement('div');
                  label.className = 'zenith-y-cursor-label';
                  Object.assign(label.style, {
                    position: 'absolute',
                    zIndex: '13',
                    left: '6px',
                    top: '6px',
                    padding: '2px 6px',
                    borderRadius: '6px',
                    border: '1px solid #cbd5e1',
                    background: 'rgba(248, 250, 252, 0.95)',
                    color: '#334155',
                    fontSize: '11px',
                    fontWeight: '600',
                    lineHeight: '1.2',
                    pointerEvents: 'none',
                    display: 'none',
                    userSelect: 'none',
                  });
                  host.appendChild(label);
                }

                if (!guide) {
                  guide = w.document.createElement('div');
                  guide.className = 'zenith-y-cursor-guide';
                  Object.assign(guide.style, {
                    position: 'absolute',
                    zIndex: '11',
                    left: '0px',
                    top: '0px',
                    width: '0px',
                    borderTop: '1px dashed rgba(71, 85, 105, 0.75)',
                    pointerEvents: 'none',
                    display: 'none',
                    userSelect: 'none',
                  });
                  host.appendChild(guide);
                }

                return { label, guide };
              }

              function bindYAxisCursor(host, plot) {
                const widgets = ensureYAxisCursorWidgets(host);
                const label = widgets.label;
                const guide = widgets.guide;

                if (typeof host.__zenithYCursorCleanup === 'function') {
                  host.__zenithYCursorCleanup();
                }

                const hideLabel = () => {
                  label.style.display = 'none';
                  guide.style.display = 'none';
                };
                const onMove = (ev) => {
                  const full = plot && plot._fullLayout;
                  if (!full || !full._size || !full.yaxis || !full.yaxis.range || !full.yaxis.domain) {
                    hideLabel();
                    return;
                  }

                  const size = full._size;
                  const axis = full.yaxis;
                  const rect = host.getBoundingClientRect();
                  const x = ev.clientX - rect.left;
                  const y = ev.clientY - rect.top;

                  const xMin = size.l;
                  const xMax = size.l + size.w;
                  const yTop = size.t + (1 - axis.domain[1]) * size.h;
                  const yBottom = size.t + (1 - axis.domain[0]) * size.h;
                  if (x < xMin || x > xMax || y < yTop || y > yBottom) {
                    hideLabel();
                    return;
                  }

                  const ratio = (y - yTop) / Math.max(1, (yBottom - yTop));
                  const r0 = Number(axis.range[0]);
                  const r1 = Number(axis.range[1]);
                  if (!Number.isFinite(r0) || !Number.isFinite(r1)) {
                    hideLabel();
                    return;
                  }

                  const topVal = Math.max(r0, r1);
                  const bottomVal = Math.min(r0, r1);
                  const value = topVal + ratio * (bottomVal - topVal);

                  label.textContent = formatPrice(value);
                  label.style.left = `${Math.max(4, size.l - 64)}px`;
                  label.style.top = `${Math.max(6, Math.min(rect.height - 22, y - 10))}px`;
                  label.style.display = 'block';

                  guide.style.left = `${xMin}px`;
                  guide.style.width = `${Math.max(0, xMax - xMin)}px`;
                  guide.style.top = `${Math.max(0, y)}px`;
                  guide.style.display = 'block';
                };

                host.addEventListener('mousemove', onMove);
                host.addEventListener('mouseleave', hideLabel);
                host.addEventListener('mousedown', hideLabel);
                if (plot && typeof plot.on === 'function') {
                  plot.on('plotly_relayout', hideLabel);
                  plot.on('plotly_doubleclick', hideLabel);
                }

                host.__zenithYCursorCleanup = () => {
                  host.removeEventListener('mousemove', onMove);
                  host.removeEventListener('mouseleave', hideLabel);
                  host.removeEventListener('mousedown', hideLabel);
                  if (plot && typeof plot.removeListener === 'function') {
                    plot.removeListener('plotly_relayout', hideLabel);
                    plot.removeListener('plotly_doubleclick', hideLabel);
                  }
                };
              }

              function getPlotlyApi(plot) {
                if (w.Plotly && typeof w.Plotly.relayout === 'function') return w.Plotly;
                if (plot && plot._context && plot._context._plotlyjs && typeof plot._context._plotlyjs.relayout === 'function') {
                  return plot._context._plotlyjs;
                }
                if (plot && plot.__plotly && plot.__plotly.Plotly && typeof plot.__plotly.Plotly.relayout === 'function') {
                  return plot.__plotly.Plotly;
                }
                return null;
              }

              function fitPlotHeight(host, plot) {
                if (!host || !plot) return;
                const api = getPlotlyApi(plot);
                if (!api) return;
                const doc = w.document;
                const viewportH = Math.max(
                  Number(w.innerHeight) || 0,
                  Number(doc.documentElement && doc.documentElement.clientHeight) || 0
                );
                if (!Number.isFinite(viewportH) || viewportH <= 0) return;

                const rect = host.getBoundingClientRect();
                if (!rect || !Number.isFinite(rect.top)) return;

                const bottomPadding = 14;
                const available = Math.floor(viewportH - rect.top - bottomPadding);
                const target = Math.max(620, Math.min(980, available));
                if (!Number.isFinite(target) || target <= 0) return;

                const prev = Number(host.getAttribute('data-zenith-fit-h') || '0');
                if (Math.abs(prev - target) < 6) return;

                host.setAttribute('data-zenith-fit-h', String(target));
                host.style.minHeight = `${target}px`;
                host.style.height = `${target}px`;
                api.relayout(plot, { height: target });
              }

              function resetPlotView(host, plot) {
                const api = getPlotlyApi(plot);
                if (api) {
                  api.relayout(plot, {
                    'xaxis.autorange': true,
                    'xaxis2.autorange': true,
                    'xaxis3.autorange': true,
                    'yaxis.autorange': true,
                    'yaxis2.autorange': true,
                    'yaxis3.autorange': true,
                    'xaxis.range': null,
                    'xaxis2.range': null,
                    'xaxis3.range': null,
                    'yaxis.range': null,
                    'yaxis2.range': null,
                    'yaxis3.range': null
                  }).then(() => fitPlotHeight(host, plot)).catch(() => {});
                  return;
                }

                const dragLayer = host.querySelector('.draglayer');
                if (dragLayer) {
                  dragLayer.dispatchEvent(new MouseEvent('dblclick', { bubbles: true, cancelable: true, view: w }));
                }
              }

              function setupToolbar() {
                const chartHosts = w.document.querySelectorAll('div[data-testid="stPlotlyChart"]');
                if (!chartHosts.length) return false;
                const host = chartHosts[chartHosts.length - 1];
                const plot = host.querySelector('.js-plotly-plot');
                if (!plot) return false;
                const modebar = host.querySelector('.modebar');
                if (modebar) modebar.style.display = 'none';

                bindYAxisCursor(host, plot);
                fitPlotHeight(host, plot);

                let tools = host.querySelector('.zenith-plot-tools');
                if (!tools) {
                  host.style.position = 'relative';
                  tools = w.document.createElement('div');
                  tools.className = 'zenith-plot-tools';
                  Object.assign(tools.style, {
                    position: 'absolute',
                    top: '10px',
                    right: '10px',
                    zIndex: '12',
                    display: 'flex',
                    gap: '6px',
                    padding: '4px',
                    borderRadius: '999px',
                    border: '1px solid rgba(148, 163, 184, 0.55)',
                    background: 'rgba(255, 255, 255, 0.94)',
                    backdropFilter: 'blur(4px)',
                    boxShadow: '0 8px 20px rgba(15, 23, 42, 0.12)'
                  });
                  host.appendChild(tools);
                }
                let resetBtn = tools.querySelector('.zenith-reset-btn');
                if (!resetBtn) {
                  const resetIcon = `
                    <svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24'
                         fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
                      <path d='M3 2v6h6'></path>
                      <path d='M3 8a9 9 0 1 0 3-6.7'></path>
                    </svg>
                  `;
                  resetBtn = w.document.createElement('button');
                  resetBtn.type = 'button';
                  resetBtn.className = 'zenith-reset-btn';
                  resetBtn.title = 'Reset view';
                  resetBtn.setAttribute('aria-label', 'Reset view');
                  resetBtn.innerHTML = `${resetIcon}<span>Reset View</span>`;
                  Object.assign(resetBtn.style, {
                    height: '32px',
                    padding: '0 12px',
                    borderRadius: '999px',
                    border: '1px solid #99f6e4',
                    background: 'linear-gradient(135deg, #ecfeff 0%, #f0fdfa 100%)',
                    color: '#0f766e',
                    cursor: 'pointer',
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '6px',
                    fontSize: '12px',
                    fontWeight: '700',
                    letterSpacing: '0.02em',
                    boxShadow: '0 2px 8px rgba(15, 23, 42, 0.10)',
                    transition: 'all 140ms ease'
                  });
                  resetBtn.onmouseenter = () => {
                    resetBtn.style.transform = 'translateY(-1px)';
                    resetBtn.style.boxShadow = '0 4px 12px rgba(15, 23, 42, 0.14)';
                    resetBtn.style.borderColor = '#5eead4';
                    resetBtn.style.color = '#0f766e';
                  };
                  resetBtn.onmouseleave = () => {
                    resetBtn.style.transform = 'translateY(0)';
                    resetBtn.style.boxShadow = '0 2px 8px rgba(15, 23, 42, 0.10)';
                    resetBtn.style.borderColor = '#99f6e4';
                    resetBtn.style.color = '#0f766e';
                  };
                  tools.appendChild(resetBtn);
                }
                resetBtn.onclick = (ev) => {
                  ev.preventDefault();
                  ev.stopPropagation();
                  resetPlotView(host, plot);
                };
                setTimeout(() => fitPlotHeight(host, plot), 80);
                setTimeout(() => fitPlotHeight(host, plot), 220);
                return true;
              }

              function setupToolbarWithRetry(attempt = 0) {
                const ok = setupToolbar();
                if (ok) return;
                if (attempt >= 30) return;
                const delay = Math.min(120 + (attempt * 25), 600);
                setTimeout(() => setupToolbarWithRetry(attempt + 1), delay);
              }

              if (!w.__zenithToolbarObserverBound) {
                w.__zenithToolbarObserverBound = true;
                const observer = new MutationObserver(() => setupToolbarWithRetry(0));
                observer.observe(w.document.body, { childList: true, subtree: true });
              }
              if (!w.__zenithPlotResizeBound) {
                w.__zenithPlotResizeBound = true;
                let resizeTimer = null;
                w.addEventListener('resize', () => {
                  if (resizeTimer) {
                    w.clearTimeout(resizeTimer);
                  }
                  resizeTimer = w.setTimeout(() => setupToolbarWithRetry(0), 120);
                }, { passive: true });
              }

              setupToolbarWithRetry(0);
              setTimeout(() => setupToolbarWithRetry(0), 60);
              setTimeout(() => setupToolbarWithRetry(0), 220);
            })();
            </script>
            """,
            height=0,
            width=0,
        )

    with right:
        checklist = analysis["checklist"]
        status_buy = analysis["status"] == "BUY"
        status_text = "🟢 เข้าซื้อได้" if status_buy else "⏳ รอจังหวะ"
        status_class = "status-buy" if status_buy else "status-wait"
        entry_txt = fmt2(analysis["entry"])
        sl_txt = fmt2(analysis["sl"])
        tp_txt = fmt2(analysis["tp"])
        st.markdown(
            f"""
            <div class='summary-card'>
                <div class='panel-title'>📋 สรุปสัญญาณ (Signal Summary)</div>
                <div class='summary-row'>
                    <span class='summary-key'>แนวโน้ม (Trend)<span class='info-tip' title='แนวโน้มหลักของราคาตามเงื่อนไขกลยุทธ์'>i</span></span>
                    <span class='summary-val'>{checklist_value(checklist['trend'])}</span>
                </div>
                <div class='summary-row'>
                    <span class='summary-key'>โซน (Zone)<span class='info-tip' title='ตำแหน่งราคาที่ควรจับตาตามโซนเทคนิค'>i</span></span>
                    <span class='summary-val'>{checklist_value(checklist['zone'])}</span>
                </div>
                <div class='summary-row'>
                    <span class='summary-key'>ทริกเกอร์ (Trigger)<span class='info-tip' title='สัญญาณยืนยันการเข้าเทรด เช่นแท่งกลับตัวหรือวอลุ่ม'>i</span></span>
                    <span class='summary-val'>{checklist_value(checklist['trigger'])}</span>
                </div>
                <div class='status-pill {status_class}'>🎯 สถานะ: {status_text}</div>
                <div class='summary-row'>
                    <span class='summary-key'>📌 Entry<span class='info-tip' title='ราคาที่แนะนำสำหรับจุดเข้า'>i</span></span>
                    <span class='summary-val'>{entry_txt}</span>
                </div>
                <div class='summary-row'>
                    <span class='summary-key'>🛑 Stop Loss<span class='info-tip' title='ระดับตัดขาดทุนเมื่อราคาวิ่งผิดทาง'>i</span></span>
                    <span class='summary-val'>{sl_txt}</span>
                </div>
                <div class='summary-row'>
                    <span class='summary-key'>🎯 Take Profit<span class='info-tip' title='ระดับเป้าหมายทำกำไร'>i</span></span>
                    <span class='summary-val'>{tp_txt}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        missing = [name for name, passed in checklist.items() if not passed]
        if status_buy:
            advice_text = (
                "สัญญาณหลักผ่านครบแล้ว แต่ยังควรรอแท่งปิดยืนยันก่อนเข้าจริง "
                "และควบคุมขนาดไม้ไม่เกินความเสี่ยงที่รับได้."
            )
        else:
            missing_txt = ", ".join(missing).upper()
            advice_text = (
                "ยังไม่ควรวาง Limit Order ทันทีที่ราคา Entry. "
                f"รอให้เงื่อนไขที่ยังไม่ผ่าน ({missing_txt}) กลับมาเป็นบวกก่อน "
                "จากนั้นค่อยพิจารณาเข้าเพื่อเพิ่มความแม่นยำ."
            )
        st.markdown(
            f"""
            <div class='advice-card'>
                <div class='advice-title'>⚠ คำแนะนำการเข้าเทรด</div>
                <div class='advice-text'>{advice_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
