# ================================================================
# PRABU ULTIMATE TREND PYTHON SCANNER
# ================================================================
# 5 MINUTE HEIKIN ASHI
# NIFTY 50
# RSI + VWAP + SUPERTREND
# SL = 0.75%
# RR = 1:2
# TELEGRAM ALERTS
# ================================================================

import os
import time
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
import requests


# ================================================================
# CONFIGURATION
# ================================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")

CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

TEST_MODE = os.getenv("TEST_MODE", "0") == "1"

TIMEZONE = ZoneInfo("Asia/Kolkata")

TIMEFRAME = "5m"

# Higher-timeframe confirmation
HTF_15M = "15m"
HTF_1H = "1h"

# Require both higher timeframes to agree with the 5M direction
USE_15M_FILTER = True
USE_1H_FILTER = True

# Number of candles downloaded.
# Keep reasonably high for indicator calculations.
PERIOD = "5d"

# ------------------------------------------------
# Risk Management
# ------------------------------------------------

SL_PERCENT = 0.75
RR = 2.0
TP_PERCENT = SL_PERCENT * RR

# ------------------------------------------------
# RSI
# ------------------------------------------------

RSI_LENGTH = 14

# ------------------------------------------------
# VWAP
# ------------------------------------------------

USE_VWAP_FILTER = True

# ------------------------------------------------
# Supertrend
# ------------------------------------------------

ST_FACTOR = 12.0
ST_ATR_PERIOD = 90

# ------------------------------------------------
# Trend baseline
# ------------------------------------------------

WMA_LENGTH = 40
EMA_LENGTH = 14

# ------------------------------------------------
# Scanner
# ------------------------------------------------

SCAN_INTERVAL_SECONDS = 300

# ------------------------------------------------
# Signal alerts
# ------------------------------------------------

SEND_WAIT_TELEGRAM = False

# Tata Motors excluded as requested
EXCLUDED_SYMBOLS = {
    "TATAMOTORS"
}


# ================================================================
# NIFTY 50 UNIVERSE
# ================================================================

NIFTY50 = [
    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJAJFINSV",
    "BAJFINANCE",
    "BEL",
    "BHARTIARTL",
    "CIPLA",
    "COALINDIA",
    "DRREDDY",
    "EICHERMOT",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "INDUSINDBK",
    "INFY",
    "ITC",
    "JIOFIN",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "NESTLEIND",
    "NTPC",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SBIN",
    "SHRIRAMFIN",
    "SUNPHARMA",
    "TATACONSUM",
    "TATASTEEL",
    "TECHM",
    "TITAN",
    "TRENT",
    "ULTRACEMCO",
    "WIPRO",
]


# ================================================================
# GLOBAL STATE
# ================================================================

last_alerted_signal = {}

active_trades = {}


# ================================================================
# SYMBOL CONVERSION
# ================================================================

def yahoo_symbol(symbol):

    return symbol.replace("&", "%26") + ".NS"


# ================================================================
# IST TIME
# ================================================================

def now_ist():

    return datetime.now(TIMEZONE)


def format_ist(dt):

    if dt is None:
        return ""

    if not isinstance(dt, datetime):

        try:
            dt = pd.Timestamp(dt).to_pydatetime()
        except Exception:
            return str(dt)

    if dt.tzinfo is None:

        dt = dt.replace(tzinfo=TIMEZONE)

    return dt.astimezone(TIMEZONE).strftime(
        "%d-%m-%Y %H:%M:%S"
    )


# ================================================================
# DOWNLOAD DATA
# ================================================================

def download_data(symbol):

    ticker = yahoo_symbol(symbol)

    try:

        df = yf.download(
            ticker,
            period=PERIOD,
            interval=TIMEFRAME,
            auto_adjust=False,
            progress=False,
            threads=False
        )

    except Exception as e:

        print(
            f"Error downloading {symbol}: {e}"
        )

        return None

    if df is None or df.empty:

        return None

    # ------------------------------------------------------------
    # Fix Yahoo multi-index columns
    # ------------------------------------------------------------

    if isinstance(df.columns, pd.MultiIndex):

        try:

            df.columns = df.columns.get_level_values(0)

        except Exception:

            df.columns = [
                c[0] if isinstance(c, tuple) else c
                for c in df.columns
            ]

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for col in required:

        if col not in df.columns:

            return None

    df = df[required].copy()

    # ------------------------------------------------------------
    # Convert columns to numeric Series
    # ------------------------------------------------------------

    for col in required:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close"
        ]
    )

    # ------------------------------------------------------------
    # Remove duplicate timestamps
    # ------------------------------------------------------------

    df = df[~df.index.duplicated(keep="last")]

    df = df.sort_index()

    if len(df) < 200:

        return None

    # ------------------------------------------------------------
    # Timezone
    # ------------------------------------------------------------

    try:

        if df.index.tz is None:

            df.index = df.index.tz_localize(
                "UTC"
            )

        df.index = df.index.tz_convert(
            TIMEZONE
        )

    except Exception:

        pass

    return df


# ================================================================
# DOWNLOAD HIGHER-TIMEFRAME DATA
# ================================================================

def download_timeframe_data(symbol, timeframe, period="10d"):
    """Download OHLCV data for a specific higher timeframe."""
    ticker = yahoo_symbol(symbol)

    try:
        df = yf.download(
            ticker,
            period=period,
            interval=timeframe,
            auto_adjust=False,
            progress=False,
            threads=False
        )
    except Exception as e:
        print(f"Error downloading {symbol} {timeframe}: {e}")
        return None

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df.columns = [
                c[0] if isinstance(c, tuple) else c
                for c in df.columns
            ]

    df.columns = [str(c).strip().lower() for c in df.columns]

    required = ["open", "high", "low", "close", "volume"]

    for col in required:
        if col not in df.columns:
            return None

    df = df[required].copy()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    if len(df) < 200:
        return None

    try:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(TIMEZONE)
    except Exception:
        pass

    return df


# ================================================================
# HEIKIN ASHI
# ================================================================

def heikin_ashi(df):

    if df is None or df.empty:

        return None

    ha = pd.DataFrame(
        index=df.index
    )

    # ------------------------------------------------------------
    # Normal OHLC
    # ------------------------------------------------------------

    o = pd.to_numeric(
        df["open"],
        errors="coerce"
    ).astype(float)

    h = pd.to_numeric(
        df["high"],
        errors="coerce"
    ).astype(float)

    l = pd.to_numeric(
        df["low"],
        errors="coerce"
    ).astype(float)

    c = pd.to_numeric(
        df["close"],
        errors="coerce"
    ).astype(float)

    # ------------------------------------------------------------
    # HA Close
    # ------------------------------------------------------------

    ha_close = (
        o + h + l + c
    ) / 4.0

    # ------------------------------------------------------------
    # HA Open
    # ------------------------------------------------------------

    ha_open = np.zeros(
        len(df),
        dtype=float
    )

    ha_open[0] = (
        o.iloc[0] + c.iloc[0]
    ) / 2.0

    for i in range(1, len(df)):

        ha_open[i] = (
            ha_open[i - 1]
            + ha_close.iloc[i - 1]
        ) / 2.0

    # ------------------------------------------------------------
    # HA High / Low
    # ------------------------------------------------------------

    ha_high = np.maximum.reduce([
        h.to_numpy(),
        ha_open,
        ha_close.to_numpy()
    ])

    ha_low = np.minimum.reduce([
        l.to_numpy(),
        ha_open,
        ha_close.to_numpy()
    ])

    ha["open"] = pd.Series(
        ha_open,
        index=df.index
    )

    ha["high"] = pd.Series(
        ha_high,
        index=df.index
    )

    ha["low"] = pd.Series(
        ha_low,
        index=df.index
    )

    ha["close"] = ha_close

    ha["volume"] = pd.to_numeric(
        df["volume"],
        errors="coerce"
    ).fillna(0.0)

    return ha


# ================================================================
# TRUE RANGE
# ================================================================

def true_range(df):

    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    tr1 = high - low

    tr2 = (
        high - previous_close
    ).abs()

    tr3 = (
        low - previous_close
    ).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return tr


# ================================================================
# ATR
# ================================================================

def calculate_atr(df, period=14):

    tr = true_range(df)

    atr = tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    return atr


# ================================================================
# RSI
# ================================================================

def calculate_rsi(series, period=14):

    # Force Series.
    series = pd.Series(
        np.asarray(series, dtype=float),
        index=series.index if isinstance(
            series,
            pd.Series
        ) else None
    )

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    rsi = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    # Handle special cases
    rsi = rsi.where(
        avg_loss != 0,
        100
    )

    rsi = rsi.where(
        avg_gain != 0,
        0
    )

    rsi = rsi.astype(float)

    return rsi


# ================================================================
# WMA
# ================================================================

def calculate_wma(series, length):

    weights = np.arange(
        1,
        length + 1,
        dtype=float
    )

    weight_sum = weights.sum()

    return series.rolling(
        length
    ).apply(
        lambda x: np.dot(
            x,
            weights
        ) / weight_sum,
        raw=True
    )


# ================================================================
# SUPERTREND BASELINE
# ================================================================

def calculate_trend_line(
    df,
    factor=12.0,
    atr_period=90,
    wma_length=40,
    ema_length=14
):

    src = (
        df["high"]
        + df["low"]
    ) / 2.0

    atr = calculate_atr(
        df,
        atr_period
    )

    upper = (
        src
        + factor * atr
    )

    lower = (
        src
        - factor * atr
    )

    upper_values = upper.to_numpy(
        dtype=float
    )

    lower_values = lower.to_numpy(
        dtype=float
    )

    close_values = df[
        "close"
    ].to_numpy(
        dtype=float
    )

    final_upper = np.copy(
        upper_values
    )

    final_lower = np.copy(
        lower_values
    )

    for i in range(1, len(df)):

        if np.isnan(
            final_lower[i - 1]
        ):

            final_lower[i] = lower_values[i]

        elif (
            lower_values[i]
            > final_lower[i - 1]
            or close_values[i - 1]
            < final_lower[i - 1]
        ):

            final_lower[i] = lower_values[i]

        else:

            final_lower[i] = final_lower[i - 1]

        if np.isnan(
            final_upper[i - 1]
        ):

            final_upper[i] = upper_values[i]

        elif (
            upper_values[i]
            < final_upper[i - 1]
            or close_values[i - 1]
            > final_upper[i - 1]
        ):

            final_upper[i] = upper_values[i]

        else:

            final_upper[i] = final_upper[i - 1]

    midpoint = (
        final_lower
        + final_upper
    ) / 2.0

    midpoint_series = pd.Series(
        midpoint,
        index=df.index
    )

    wma = calculate_wma(
        midpoint_series,
        wma_length
    )

    trend_line = wma.ewm(
        span=ema_length,
        adjust=False,
        min_periods=ema_length
    ).mean()

    return trend_line


# ================================================================
# VWAP
# ================================================================

def calculate_vwap(df):

    typical_price = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3.0

    if df.index.tz is not None:

        date_values = df.index.date

    else:

        date_values = pd.to_datetime(
            df.index
        ).date

    volume = pd.to_numeric(
        df["volume"],
        errors="coerce"
    ).fillna(0)

    tp_volume = (
        typical_price
        * volume
    )

    cumulative_pv = (
        tp_volume
        .groupby(date_values)
        .cumsum()
    )

    cumulative_volume = (
        volume
        .groupby(date_values)
        .cumsum()
    )

    vwap = (
        cumulative_pv
        / cumulative_volume.replace(
            0,
            np.nan
        )
    )

    return vwap


# ================================================================
# TREND STATE
# ================================================================

def calculate_trend_state(
    trend_line
):

    trend = np.zeros(
        len(trend_line),
        dtype=int
    )

    values = trend_line.to_numpy(
        dtype=float
    )

    for i in range(
        1,
        len(values)
    ):

        if (
            np.isnan(values[i])
            or np.isnan(values[i - 1])
        ):

            trend[i] = (
                trend[i - 1]
            )

        elif values[i] > values[i - 1]:

            trend[i] = 1

        elif values[i] < values[i - 1]:

            trend[i] = -1

        else:

            trend[i] = (
                trend[i - 1]
            )

    return pd.Series(
        trend,
        index=trend_line.index
    )


# ================================================================
# HIGHER-TIMEFRAME TREND
# ================================================================

def get_timeframe_trend(symbol, timeframe):
    """
    Calculate the same trend-line methodology used by the 5M scanner
    and return the latest completed candle trend.
    """
    period = "10d" if timeframe == "15m" else "30d"

    raw = download_timeframe_data(
        symbol,
        timeframe,
        period=period
    )

    if raw is None:
        return {
            "trend": "NA",
            "trend_state": 0,
            "candle_time": None
        }

    # Remove currently-forming candle.
    if len(raw) > 2:
        raw = raw.iloc[:-1].copy()

    ha = heikin_ashi(raw)

    if ha is None or len(ha) < 200:
        return {
            "trend": "NA",
            "trend_state": 0,
            "candle_time": None
        }

    trend_line = calculate_trend_line(
        ha,
        ST_FACTOR,
        ST_ATR_PERIOD,
        WMA_LENGTH,
        EMA_LENGTH
    )

    trend_state_series = calculate_trend_state(trend_line)
    state = int(trend_state_series.iloc[-1])

    if state == 1:
        trend = "BULLISH"
    elif state == -1:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    return {
        "trend": trend,
        "trend_state": state,
        "candle_time": ha.index[-1]
    }


def higher_timeframe_confirmation(
    signal,
    trend_15m_state,
    trend_1h_state
):
    """Require higher timeframes to agree with the 5M signal direction."""

    if signal == "BUY":
        return (
            (not USE_15M_FILTER or trend_15m_state == 1)
            and
            (not USE_1H_FILTER or trend_1h_state == 1)
        )

    if signal == "SELL":
        return (
            (not USE_15M_FILTER or trend_15m_state == -1)
            and
            (not USE_1H_FILTER or trend_1h_state == -1)
        )

    return False


# ================================================================
# SIGNAL DETECTION
# ================================================================

def get_signal(
    df,
    trend_15m_state=0,
    trend_1h_state=0,
    trend_15m_text="NA",
    trend_1h_text="NA"
):

    if df is None or len(df) < 200:

        return None

    # ------------------------------------------------------------
    # Trend line
    # ------------------------------------------------------------

    df["TrendLine"] = calculate_trend_line(
        df,
        ST_FACTOR,
        ST_ATR_PERIOD,
        WMA_LENGTH,
        EMA_LENGTH
    )

    # ------------------------------------------------------------
    # Trend state
    # ------------------------------------------------------------

    df["TrendState"] = calculate_trend_state(
        df["TrendLine"]
    )

    # ------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------

    df["RSI"] = calculate_rsi(
        df["close"],
        RSI_LENGTH
    )

    # ------------------------------------------------------------
    # VWAP
    # ------------------------------------------------------------

    df["VWAP"] = calculate_vwap(
        df
    )

    latest = df.iloc[-1]

    previous = df.iloc[-2]

    price = float(
        latest["close"]
    )

    rsi = float(
        latest["RSI"]
    ) if pd.notna(
        latest["RSI"]
    ) else np.nan

    vwap = float(
        latest["VWAP"]
    ) if pd.notna(
        latest["VWAP"]
    ) else np.nan

    trend_state = int(
        latest["TrendState"]
    )

    previous_trend_state = int(
        previous["TrendState"]
    )

    # ------------------------------------------------------------
    # Trend
    # ------------------------------------------------------------

    if trend_state == 1:

        trend_text = "BULLISH"

    elif trend_state == -1:

        trend_text = "BEARISH"

    else:

        trend_text = "NEUTRAL"

    # ------------------------------------------------------------
    # Cross confirmation
    # ------------------------------------------------------------

    bullish_cross = (
        trend_state == 1
        and previous_trend_state != 1
    )

    bearish_cross = (
        trend_state == -1
        and previous_trend_state != -1
    )

    # ------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------

    rsi_buy = (
        pd.notna(rsi)
        and rsi >= 50
    )

    rsi_sell = (
        pd.notna(rsi)
        and rsi <= 50
    )

    # ------------------------------------------------------------
    # VWAP
    # ------------------------------------------------------------

    vwap_buy = (
        not USE_VWAP_FILTER
        or (
            pd.notna(vwap)
            and price > vwap
        )
    )

    vwap_sell = (
        not USE_VWAP_FILTER
        or (
            pd.notna(vwap)
            and price < vwap
        )
    )

    # ------------------------------------------------------------
    # FINAL SIGNAL
    # ------------------------------------------------------------

    signal = "WAIT"

    active = False

    # A signal requires trend transition +
    # RSI + VWAP confirmation.
    #
    # This prevents the dashboard from saying
    # BULLISH while immediately producing SELL.
    # ------------------------------------------------------------

    if (
        bullish_cross
        and rsi_buy
        and vwap_buy
    ):

        signal = "BUY"

        active = True

    elif (
        bearish_cross
        and rsi_sell
        and vwap_sell
    ):

        signal = "SELL"

        active = True

    # ------------------------------------------------------------
    # HIGHER-TIMEFRAME CONFIRMATION
    # ------------------------------------------------------------

    htf_confirmed = higher_timeframe_confirmation(
        signal,
        trend_15m_state,
        trend_1h_state
    )

    if signal in ("BUY", "SELL") and not htf_confirmed:
        signal = "WAIT"
        active = False

    return {
        "price": price,
        "rsi": rsi,
        "vwap": vwap,
        "trend": trend_text,
        "trend_state": trend_state,
        "trend_15m": trend_15m_text,
        "trend_15m_state": trend_15m_state,
        "trend_1h": trend_1h_text,
        "trend_1h_state": trend_1h_state,
        "htf_confirmed": htf_confirmed,
        "signal": signal,
        "active": active,
        "bullish_cross": bullish_cross,
        "bearish_cross": bearish_cross,
        "candle_time": df.index[-1],
    }


# ================================================================
# TRADE LEVELS
# ================================================================

def calculate_trade_levels(
    signal,
    price
):

    if signal == "BUY":

        entry = price

        sl = (
            entry
            * (1 - SL_PERCENT / 100)
        )

        tp = (
            entry
            * (1 + TP_PERCENT / 100)
        )

    elif signal == "SELL":

        entry = price

        sl = (
            entry
            * (1 + SL_PERCENT / 100)
        )

        tp = (
            entry
            * (1 - TP_PERCENT / 100)
        )

    else:

        return None

    return {
        "entry": entry,
        "sl": sl,
        "tp": tp,
    }


# ================================================================
# TELEGRAM
# ================================================================

def send_telegram(
    message
):

    if (
        not TELEGRAM_TOKEN
        or TELEGRAM_TOKEN
        == "PASTE_YOUR_NEW_TELEGRAM_BOT_TOKEN_HERE"
    ):

        print(
            "Telegram token not configured."
        )

        return False

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=15
        )

        if response.ok:

            return True

        print(
            "Telegram error:",
            response.text
        )

    except Exception as e:

        print(
            "Telegram connection error:",
            e
        )

    return False


# ================================================================
# TELEGRAM SIGNAL MESSAGE
# ================================================================

def create_signal_message(
    symbol,
    result,
    levels
):

    signal = result["signal"]

    price = levels["entry"]

    sl = levels["sl"]

    tp = levels["tp"]

    rsi = result["rsi"]

    vwap = result["vwap"]

    candle_time = format_ist(
        result["candle_time"]
    )

    if signal == "BUY":

        emoji = "🟢"

        action = "BUY"

    else:

        emoji = "🔴"

        action = "SELL"

    message = (
        f"{emoji} <b>JJ ULTIMATE TREND</b>\n"
        f"\n"
        f"<b>{action} SIGNAL</b>\n"
        f"\n"
        f"📊 <b>Stock:</b> {symbol}\n"
        f"⏱ <b>Timeframe:</b> 5M\n"
        f"🕯 <b>Candle:</b> Heikin Ashi\n"
        f"🕐 <b>Candle Time:</b> {candle_time} IST\n"
        f"\n"
        f"💰 <b>Entry:</b> {price:.2f}\n"
        f"🛑 <b>SL:</b> {sl:.2f}\n"
        f"🎯 <b>TP:</b> {tp:.2f}\n"
        f"\n"
        f"📈 <b>5M Trend:</b> {result['trend']}\n"
        f"⏱ <b>15M Trend:</b> {result['trend_15m']}\n"
        f"🕐 <b>1H Trend:</b> {result['trend_1h']}\n"
        f"✅ <b>HTF Confirm:</b> {'YES' if result['htf_confirmed'] else 'NO'}\n"
        f"📊 <b>RSI:</b> {rsi:.2f}\n"
        f"📍 <b>VWAP:</b> {vwap:.2f}\n"
        f"\n"
        f"⚖️ <b>Risk : Reward:</b> 1 : 2\n"
        f"📉 <b>SL:</b> {SL_PERCENT:.2f}%\n"
        f"📈 <b>TP:</b> {TP_PERCENT:.2f}%\n"
        f"\n"
        f"⚠️ Educational / scanner signal only."
    )

    return message


# ================================================================
# ACTIVE TRADE MANAGEMENT
# ================================================================

def update_active_trade(
    symbol,
    result
):

    if symbol not in active_trades:

        return

    trade = active_trades[symbol]

    price = result["price"]

    signal = trade["signal"]

    entry = trade["entry"]

    sl = trade["sl"]

    tp = trade["tp"]

    candle_time = result[
        "candle_time"
    ]

    exit_reason = None

    exit_price = None

    if signal == "BUY":

        if price <= sl:

            exit_reason = "STOP LOSS"

            exit_price = sl

        elif price >= tp:

            exit_reason = "TARGET"

            exit_price = tp

    elif signal == "SELL":

        if price >= sl:

            exit_reason = "STOP LOSS"

            exit_price = sl

        elif price <= tp:

            exit_reason = "TARGET"

            exit_price = tp

    if exit_reason is None:

        return

    if signal == "BUY":

        pnl_percent = (
            (exit_price - entry)
            / entry
        ) * 100

    else:

        pnl_percent = (
            (entry - exit_price)
            / entry
        ) * 100

    if pnl_percent >= 0:

        result_emoji = "✅"

    else:

        result_emoji = "❌"

    message = (
        f"{result_emoji} <b>JJ TRADE EXIT</b>\n"
        f"\n"
        f"📊 <b>Stock:</b> {symbol}\n"
        f"📌 <b>Side:</b> {signal}\n"
        f"📍 <b>Entry:</b> {entry:.2f}\n"
        f"🚪 <b>Exit:</b> {exit_price:.2f}\n"
        f"📋 <b>Reason:</b> {exit_reason}\n"
        f"📊 <b>P&L:</b> {pnl_percent:+.2f}%\n"
        f"🕐 <b>Exit Time:</b> "
        f"{format_ist(candle_time)} IST"
    )

    send_telegram(
        message
    )

    del active_trades[symbol]


# ================================================================
# ANALYZE SYMBOL
# ================================================================

def analyze_symbol(
    symbol
):

    raw = download_data(symbol)

    if raw is None:
        return None

    # ------------------------------------------------------------
    # Use ONLY completed 5M candles
    # ------------------------------------------------------------

    if len(raw) > 2:
        raw = raw.iloc[:-1].copy()

    ha = heikin_ashi(raw)

    if ha is None:
        return None

    # ------------------------------------------------------------
    # Higher timeframe confirmation
    # ------------------------------------------------------------

    trend_15m = get_timeframe_trend(symbol, HTF_15M)
    trend_1h = get_timeframe_trend(symbol, HTF_1H)

    result = get_signal(
        ha,
        trend_15m_state=trend_15m["trend_state"],
        trend_1h_state=trend_1h["trend_state"],
        trend_15m_text=trend_15m["trend"],
        trend_1h_text=trend_1h["trend"]
    )

    if result is None:
        return None

    return result


# ================================================================
# PRINT HEADER
# ================================================================

def print_header():

    print()
    print("=" * 80)
    print(
        "JJ ULTIMATE TREND PYTHON SCANNER"
    )
    print("=" * 80)
    print(
        f"Timeframe       : {TIMEFRAME.replace('m', ' Minutes')}"
    )
    print(
        "Candle          : Heikin Ashi"
    )
    print(
        "Timezone        : Asia/Kolkata"
    )
    print(
        f"SL              : {SL_PERCENT:.2f}%"
    )
    print(
        f"Risk : Reward   : 1 : {RR:.0f}"
    )
    print(
        "RSI Filter      : >=50 BUY / <=50 SELL"
    )
    print(
        "HTF Confirmation : 15M + 1H must agree with 5M"
    )
    print(
        "VWAP Filter     : ABOVE BUY / BELOW SELL"
    )
    print(
        "Tata Motors     : EXCLUDED"
    )
    print(
        "Universe        : NIFTY 50"
    )
    print("=" * 80)


# ================================================================
# PRINT SCAN HEADER
# ================================================================

def print_scan_header():

    current = now_ist()

    print()
    print("=" * 80)

    print(
        f"JJ SCANNER | "
        f"{current.strftime('%d-%m-%Y %H:%M:%S')} IST"
    )

    print(
        "5M HEIKIN ASHI | NSE | NIFTY 50"
    )

    print("=" * 80)


# ================================================================
# PRINT RESULT
# ================================================================

def print_result(
    symbol,
    result
):

    trend = result[
        "trend"
    ]

    rsi = result[
        "rsi"
    ]

    price = result[
        "price"
    ]

    signal = result[
        "signal"
    ]

    active = result[
        "active"
    ]

    if pd.isna(rsi):

        rsi_text = "NA"

    else:

        rsi_text = (
            f"{rsi:6.2f}"
        )

    trend_15m = result.get("trend_15m", "NA")
    trend_1h = result.get("trend_1h", "NA")
    htf = "YES" if result.get("htf_confirmed", False) else "NO"

    print(
        f"{symbol:<15}"
        f"5M={trend:<9} "
        f"15M={trend_15m:<9} "
        f"1H={trend_1h:<9} "
        f"RSI={rsi_text:>6} "
        f"Price={price:>10.2f} "
        f"HTF={htf:<3} "
        f"Signal={signal:<5} "
        f"Active={str(active):<5}"
    )


# ================================================================
# PROCESS SIGNAL
# ================================================================

def process_signal(
    symbol,
    result
):

    signal = result[
        "signal"
    ]

    if signal not in (
        "BUY",
        "SELL"
    ):

        return

    levels = calculate_trade_levels(
        signal,
        result["price"]
    )

    if levels is None:

        return

    # ------------------------------------------------------------
    # Existing active trade
    # ------------------------------------------------------------

    if symbol in active_trades:

        existing = active_trades[
            symbol
        ]

        # Do not create another trade
        # while one is active.
        return

    # ------------------------------------------------------------
    # Avoid duplicate alert for same candle
    # ------------------------------------------------------------

    candle_time = result[
        "candle_time"
    ]

    signal_key = (
        signal,
        str(candle_time)
    )

    if last_alerted_signal.get(
        symbol
    ) == signal_key:

        return

    # ------------------------------------------------------------
    # Store signal
    # ------------------------------------------------------------

    active_trades[
        symbol
    ] = {
        "signal": signal,
        "entry": levels["entry"],
        "sl": levels["sl"],
        "tp": levels["tp"],
        "candle_time": candle_time,
    }

    last_alerted_signal[
        symbol
    ] = signal_key

    # ------------------------------------------------------------
    # Telegram
    # ------------------------------------------------------------

    message = create_signal_message(
        symbol,
        result,
        levels
    )

    send_telegram(
        message
    )

    # ------------------------------------------------------------
    # Console
    # ------------------------------------------------------------

    print()
    print(
        ">>> NEW SIGNAL"
    )

    print(
        f"{symbol} | {signal} | "
        f"Entry={levels['entry']:.2f} | "
        f"SL={levels['sl']:.2f} | "
        f"TP={levels['tp']:.2f}"
    )


# ================================================================
# SCAN ALL
# ================================================================

def scan_all():

    print_scan_header()

    processed = 0

    signals_found = 0

    for symbol in NIFTY50:

        # --------------------------------------------------------
        # Exclusions
        # --------------------------------------------------------

        if symbol in EXCLUDED_SYMBOLS:

            continue

        print(
            f"Scanning {symbol}..."
        )

        try:

            result = analyze_symbol(
                symbol
            )

            if result is None:

                print(
                    f"{symbol:<15}"
                    "DATA ERROR"
                )

                continue

            processed += 1

            # ----------------------------------------------------
            # Update existing trade
            # ----------------------------------------------------

            update_active_trade(
                symbol,
                result
            )

            # ----------------------------------------------------
            # Print requested format
            # ----------------------------------------------------

            print_result(
                symbol,
                result
            )

            # ----------------------------------------------------
            # New signal
            # ----------------------------------------------------

            if result[
                "active"
            ]:

                signals_found += 1

                process_signal(
                    symbol,
                    result
                )

        except KeyboardInterrupt:

            raise

        except Exception as e:

            print(
                f"Error analyzing "
                f"{symbol}: {e}"
            )

            # Uncomment for full debugging:
            #
            # traceback.print_exc()

    print("=" * 80)

    print(
        f"Scan completed | "
        f"Stocks={processed} | "
        f"Signals={signals_found} | "
        f"Active Trades={len(active_trades)}"
    )

    print("=" * 80)


# ================================================================
# MARKET HOURS
# ================================================================

def market_is_open():

    now = now_ist()

    if now.weekday() >= 5:
        return False

    minutes = now.hour * 60 + now.minute

    return (
        9 * 60 + 15
        <= minutes
        <= 15 * 60 + 30
    )


# ================================================================
# SINGLE-RUN MAIN
# ================================================================

def main():

    print_header()

    now = now_ist()

    print()
    print(
        f"Run time        : {now.strftime('%d-%m-%Y %H:%M:%S')} IST"
    )

    if not TEST_MODE and not market_is_open():

        print(
            "Market is closed. Scanner finished."
        )

        return

    if TEST_MODE:
        print("TEST MODE: Market-hours check bypassed.")

    # GitHub Actions starts this program again every 5 minutes.
    # Do not wait/sleep here.
    scan_all()


# ================================================================
# START
# ================================================================

if __name__ == "__main__":

    main()
