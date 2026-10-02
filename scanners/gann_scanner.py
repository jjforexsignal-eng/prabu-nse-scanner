# ============================================================
# PRABU GANN PRICE ACTION NSE SCANNER
# GitHub Actions - SINGLE RUN VERSION
# ============================================================
#
# 15 MIN
# GANN + PRICE ACTION + SUPERTREND (10,2) + RSI
#
# This version runs ONE scan and exits.
# GitHub Actions will start it again every 5 minutes.
# ============================================================

import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import math

from datetime import datetime
from zoneinfo import ZoneInfo


# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# ============================================================
# SETTINGS
# ============================================================

TIMEZONE = ZoneInfo("Asia/Kolkata")

TIMEFRAME = "15m"
PERIOD = "5d"

GANN_ZONE_PERCENT = 0.15

RSI_PERIOD = 14

BUY_RSI_MIN = 30
BUY_RSI_MAX = 60

SELL_RSI_MIN = 40
SELL_RSI_MAX = 70

SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 2.0

GANN_RISK_DIVISOR = 3.0


# ============================================================
# NSE SYMBOLS
# ============================================================

SYMBOLS = [

    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "INDUSINDBK.NS",

    "INFY.NS",
    "TCS.NS",
    "WIPRO.NS",
    "HCLTECH.NS",
    "TECHM.NS",

    "LT.NS",
    "BHARTIARTL.NS",
    "ITC.NS",

    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "MARUTI.NS",
    "M&M.NS",


    "SUNPHARMA.NS",
    "CIPLA.NS",
    "DRREDDY.NS",

    "TATASTEEL.NS",
    "JSWSTEEL.NS",
    "HINDALCO.NS",

]


# ============================================================
# RSI
# ============================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


# ============================================================
# ATR
# ============================================================

def calculate_atr(df, period=10):

    high_low = df["High"] - df["Low"]

    high_close = (
        df["High"] - df["Close"].shift()
    ).abs()

    low_close = (
        df["Low"] - df["Close"].shift()
    ).abs()

    tr = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# ============================================================
# SUPERTREND
# ============================================================

def calculate_supertrend(
    df,
    period=10,
    multiplier=2.0
):

    atr = calculate_atr(
        df,
        period
    )

    hl2 = (
        df["High"] +
        df["Low"]
    ) / 2

    upperband = (
        hl2 +
        multiplier * atr
    )

    lowerband = (
        hl2 -
        multiplier * atr
    )

    final_upper = upperband.copy()
    final_lower = lowerband.copy()

    trend = pd.Series(
        index=df.index,
        dtype=int
    )

    trend.iloc[0] = 1

    for i in range(1, len(df)):

        if (
            upperband.iloc[i]
            < final_upper.iloc[i - 1]
            or
            df["Close"].iloc[i - 1]
            > final_upper.iloc[i - 1]
        ):
            final_upper.iloc[i] = upperband.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if (
            lowerband.iloc[i]
            > final_lower.iloc[i - 1]
            or
            df["Close"].iloc[i - 1]
            < final_lower.iloc[i - 1]
        ):
            final_lower.iloc[i] = lowerband.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if trend.iloc[i - 1] == -1:

            if df["Close"].iloc[i] > final_upper.iloc[i]:
                trend.iloc[i] = 1
            else:
                trend.iloc[i] = -1

        else:

            if df["Close"].iloc[i] < final_lower.iloc[i]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = 1

    supertrend = pd.Series(
        np.where(
            trend == 1,
            final_lower,
            final_upper
        ),
        index=df.index
    )

    return supertrend, trend


# ============================================================
# GANN LEVELS
# ============================================================

def get_nearest_gann_levels(price):

    if price <= 0:
        return None, None, None

    root = math.sqrt(price)

    lower_root = math.floor(root)
    upper_root = lower_root + 1

    support = lower_root ** 2
    resistance = upper_root ** 2

    return (
        support,
        resistance,
        root
    )


# ============================================================
# GANN ZONE
# ============================================================

def near_level(price, level):

    if level is None or price <= 0:
        return False

    distance_percent = (
        abs(price - level)
        / price
        * 100
    )

    return (
        distance_percent
        <= GANN_ZONE_PERCENT
    )


# ============================================================
# CANDLE PATTERNS
# ============================================================

def is_hammer(candle):

    body = abs(
        candle["Close"] -
        candle["Open"]
    )

    candle_range = (
        candle["High"] -
        candle["Low"]
    )

    if candle_range <= 0:
        return False

    upper = (
        candle["High"] -
        max(
            candle["Open"],
            candle["Close"]
        )
    )

    lower = (
        min(
            candle["Open"],
            candle["Close"]
        ) -
        candle["Low"]
    )

    return (
        lower >= body * 2
        and
        upper <= body
    )


def is_inverted_hammer(candle):

    body = abs(
        candle["Close"] -
        candle["Open"]
    )

    candle_range = (
        candle["High"] -
        candle["Low"]
    )

    if candle_range <= 0:
        return False

    upper = (
        candle["High"] -
        max(
            candle["Open"],
            candle["Close"]
        )
    )

    lower = (
        min(
            candle["Open"],
            candle["Close"]
        ) -
        candle["Low"]
    )

    return (
        upper >= body * 2
        and
        lower <= body
    )


def bullish_engulfing(
    previous,
    current
):

    return (
        previous["Close"]
        < previous["Open"]
        and
        current["Close"]
        > current["Open"]
        and
        current["Open"]
        <= previous["Close"]
        and
        current["Close"]
        >= previous["Open"]
    )


def bearish_engulfing(
    previous,
    current
):

    return (
        previous["Close"]
        > previous["Open"]
        and
        current["Close"]
        < current["Open"]
        and
        current["Open"]
        >= previous["Close"]
        and
        current["Close"]
        <= previous["Open"]
    )


# ============================================================
# GET DATA
# ============================================================

def get_data(symbol):

    try:

        df = yf.download(
            symbol,
            period=PERIOD,
            interval=TIMEFRAME,
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            print(
                f"{symbol}: NO DATA"
            )
            return None

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):
            df.columns = (
                df.columns
                .get_level_values(0)
            )

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        missing = [
            x for x in required
            if x not in df.columns
        ]

        if missing:
            print(
                f"{symbol}: "
                f"Missing {missing}"
            )
            return None

        df = (
            df[required]
            .dropna()
            .copy()
        )

        if len(df) < 50:
            return None

        return df

    except Exception as e:

        print(
            f"{symbol}: DATA ERROR: {e}"
        )

        return None


# ============================================================
# BUY LEVELS
# ============================================================

def calculate_buy_levels(price):

    root = math.sqrt(price)

    base = math.floor(root)

    previous_square = base ** 2
    next_square = (base + 1) ** 2

    target = (
        previous_square + 1
    )

    distance = (
        next_square - price
    )

    risk = (
        distance /
        GANN_RISK_DIVISOR
    )

    sl = (
        price - risk
    )

    return (
        sl,
        target,
        previous_square,
        next_square,
        risk
    )


# ============================================================
# SELL LEVELS
# ============================================================

def calculate_sell_levels(price):

    root = math.sqrt(price)

    base = math.floor(root)

    previous_square = base ** 2
    next_square = (base + 1) ** 2

    target = (
        next_square - 1
    )

    distance = (
        price - previous_square
    )

    risk = (
        distance /
        GANN_RISK_DIVISOR
    )

    sl = (
        price + risk
    )

    return (
        sl,
        target,
        previous_square,
        next_square,
        risk
    )


# ============================================================
# ANALYZE SYMBOL
# ============================================================

def analyze_symbol(symbol):

    df = get_data(symbol)

    if df is None:
        return None

    df["RSI"] = calculate_rsi(
        df["Close"],
        RSI_PERIOD
    )

    (
        df["Supertrend"],
        df["ST_Trend"]
    ) = calculate_supertrend(
        df,
        SUPERTREND_PERIOD,
        SUPERTREND_MULTIPLIER
    )

    # Latest CLOSED candle
    current = df.iloc[-2]
    previous = df.iloc[-3]

    timestamp = df.index[-2]

    price = float(
        current["Close"]
    )

    rsi = float(
        current["RSI"]
    )

    st_trend = int(
        current["ST_Trend"]
    )

    (
        support,
        resistance,
        gann
    ) = get_nearest_gann_levels(
        price
    )

    hammer = is_hammer(
        current
    )

    inverted_hammer = (
        is_inverted_hammer(
            current
        )
    )

    bullish_engulf = (
        bullish_engulfing(
            previous,
            current
        )
    )

    bearish_engulf = (
        bearish_engulfing(
            previous,
            current
        )
    )

    # ========================================================
    # BUY
    # ========================================================

    buy_gann = near_level(
        price,
        support
    )

    buy_pattern = (
        hammer
        or
        bullish_engulf
    )

    buy_supertrend = (
        st_trend == 1
    )

    buy_rsi = (
        BUY_RSI_MIN
        <= rsi
        <= BUY_RSI_MAX
    )

    if (
        buy_gann
        and
        buy_pattern
        and
        buy_supertrend
        and
        buy_rsi
    ):

        if hammer:
            pattern = "HAMMER"
        else:
            pattern = "BULLISH ENGULFING"

        (
            sl,
            target,
            previous_square,
            next_square,
            risk
        ) = calculate_buy_levels(
            price
        )

        return {

            "symbol": symbol,
            "signal": "BUY",
            "pattern": pattern,
            "entry": price,
            "gann": support,
            "previous_square":
                previous_square,
            "next_square":
                next_square,
            "sl": sl,
            "target": target,
            "risk": risk,
            "rsi": rsi,
            "supertrend": "BULLISH",
            "timestamp": timestamp
        }


    # ========================================================
    # SELL
    # ========================================================

    sell_gann = near_level(
        price,
        resistance
    )

    sell_pattern = (
        inverted_hammer
        or
        bearish_engulf
    )

    sell_supertrend = (
        st_trend == -1
    )

    sell_rsi = (
        SELL_RSI_MIN
        <= rsi
        <= SELL_RSI_MAX
    )

    if (
        sell_gann
        and
        sell_pattern
        and
        sell_supertrend
        and
        sell_rsi
    ):

        if inverted_hammer:
            pattern = "INVERTED HAMMER"
        else:
            pattern = "BEARISH ENGULFING"

        (
            sl,
            target,
            previous_square,
            next_square,
            risk
        ) = calculate_sell_levels(
            price
        )

        return {

            "symbol": symbol,
            "signal": "SELL",
            "pattern": pattern,
            "entry": price,
            "gann": resistance,
            "previous_square":
                previous_square,
            "next_square":
                next_square,
            "sl": sl,
            "target": target,
            "risk": risk,
            "rsi": rsi,
            "supertrend": "BEARISH",
            "timestamp": timestamp
        }

    return None


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    if (
        not TELEGRAM_BOT_TOKEN
        or
        not TELEGRAM_CHAT_ID
    ):
        print(
            "Telegram credentials "
            "not configured."
        )
        return

    url = (
        "https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}"
        "/sendMessage"
    )

    payload = {
        "chat_id":
            TELEGRAM_CHAT_ID,
        "text":
            message,
        "parse_mode":
            "HTML"
    }

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=10
        )

        print(
            "Telegram:",
            response.status_code
        )

    except Exception as e:

        print(
            "Telegram error:",
            e
        )


# ============================================================
# FORMAT ALERT
# ============================================================

def format_alert(signal):

    if signal["signal"] == "BUY":
        emoji = "🟢"
    else:
        emoji = "🔴"

    entry = signal["entry"]
    sl = signal["sl"]
    target = signal["target"]
    risk = signal["risk"]

    reward = abs(
        target - entry
    )

    rr = (
        reward / risk
        if risk > 0
        else 0
    )

    return f"""
<b>🔔 GANN V2 PRICE ACTION SIGNAL</b>

<b>📌 SYMBOL:</b> {signal["symbol"]}

<b>⏱ TIMEFRAME:</b> 15 MIN

{emoji} <b>{signal["signal"]}</b>

🕯 <b>Pattern:</b> {signal["pattern"]}

💰 <b>Entry:</b> ₹{entry:.2f}

📐 <b>Gann Level:</b> ₹{signal["gann"]:.2f}

⬇️ <b>Previous Square:</b> ₹{signal["previous_square"]:.2f}

⬆️ <b>Next Square:</b> ₹{signal["next_square"]:.2f}

🛑 <b>Stop Loss:</b> ₹{sl:.2f}

🎯 <b>Target:</b> ₹{target:.2f}

📏 <b>Risk:</b> ₹{risk:.2f}

📊 <b>Reward:</b> ₹{reward:.2f}

⚖️ <b>R:R:</b> 1:{rr:.2f}

📈 <b>Supertrend:</b> {signal["supertrend"]}

📊 <b>RSI:</b> {signal["rsi"]:.2f}

⏰ <b>Candle:</b>
{signal["timestamp"]}

⚠️ Closed-candle confirmation.
""".strip()


# ============================================================
# MARKET HOURS
# ============================================================

def market_is_open():

    now = datetime.now(
        TIMEZONE
    )

    # Monday-Friday
    if now.weekday() >= 5:
        return False

    minutes = (
        now.hour * 60 +
        now.minute
    )

    return (
        9 * 60 + 15
        <= minutes
        <
        15 * 60 + 30
    )


# ============================================================
# ONE SCAN
# ============================================================

def scan_market():

    now = datetime.now(
        TIMEZONE
    )

    print()
    print("=" * 80)
    print(
        "PRABU GANN PRICE ACTION NSE SCANNER"
    )
    print(
        "Time:",
        now.strftime(
            "%Y-%m-%d %H:%M:%S IST"
        )
    )
    print("=" * 80)

  #  if not market_is_open():

  #      print(
  #          "Market is closed."
  #      )

  #      return


    # TEST MODE
    # Allow manual GitHub Actions testing outside NSE hours.
    # Remove this test override before production scheduling.

    print("TEST MODE: Market-hours check bypassed.")

    signals = 0

    for symbol in SYMBOLS:

        print(
            f"Scanning {symbol}...",
            end=" "
        )

        try:

            signal = analyze_symbol(
                symbol
            )

            if signal is None:

                print(
                    "NO SIGNAL"
                )

                continue

            print(
                f'{signal["signal"]} | '
                f'{signal["pattern"]} | '
                f'RSI={signal["rsi"]:.2f}'
            )

            message = format_alert(
                signal
            )

            send_telegram(
                message
            )

            signals += 1

        except Exception as e:

            print(
                "ERROR:",
                e
            )

    print()
    print(
        f"Signals generated: {signals}"
    )

    print(
        "Scanner finished."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    scan_market()
