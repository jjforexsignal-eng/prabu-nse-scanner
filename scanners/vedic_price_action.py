"""
GANN + VEDIC ASTROLOGY + PRICE ACTION
NSE Scanner V1

Purpose:
- Quantitatively test a combined Gann + Vedic astrology + price-action model.
- Astrology is a timing/filter input, NOT a standalone buy/sell claim.
- Designed for NSE cash/index data with 5M, 15M and 1H confirmation.

Data:
- yfinance is used for market OHLCV.
- Swiss Ephemeris (pyswisseph) is used for planetary positions.
- Lahiri ayanamsa is used for sidereal Moon/Nakshatra calculations.

Install:
    pip install pandas numpy yfinance pyswisseph pytz

Optional Telegram:
    Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID below.

Notes:
- NSE equity regular session is 09:15-15:30 IST.
- This scanner deliberately waits for price-action confirmation.
- Backtest the rules before using real money.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

try:
    import swisseph as swe
except ImportError:
    swe = None

IST = pytz.timezone("Asia/Kolkata")

# ---------------- USER SETTINGS ----------------

SYMBOLS = [
    "^NSEI",       # NIFTY 50
    "^NSEBANK",    # BANK NIFTY
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "INFY.NS",
    "TCS.NS",
    "LT.NS",
    "BHARTIARTL.NS",
]

INTERVALS = {
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
}

# Gann settings
GANN_LOOKBACK = 120
GANN_NEAR_PCT = 0.35       # price must be within this % of a Gann level

# Price action settings
ATR_PERIOD = 14
RSI_PERIOD = 14
VOLUME_LOOKBACK = 20

# Confluence thresholds
MIN_SIGNAL_SCORE = 7
STRONG_SIGNAL_SCORE = 9

# Risk model
RISK_REWARD_1 = 2.0
RISK_REWARD_2 = 4.0

# Optional Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-1003556534364")

# Continuous scanner settings
SCAN_INTERVAL_MINUTES = 5
MARKET_OPEN = (9, 15)
MARKET_CLOSE = (15, 30)

# NSE 2026 equity holidays. 02-Oct-2026 is Mahatma Gandhi Jayanti.
NSE_HOLIDAYS_2026 = {
    "2026-01-15", "2026-01-26", "2026-03-03", "2026-03-26",
    "2026-03-31", "2026-04-03", "2026-04-14", "2026-05-01",
    "2026-05-28", "2026-06-26", "2026-09-14", "2026-10-02",
    "2026-10-20", "2026-11-10", "2026-11-24", "2026-12-25",
}



# ---------------- DATA STRUCTURES ----------------

@dataclass
class AstroInfo:
    timestamp: datetime
    moon_longitude_tropical: float
    moon_longitude_sidereal: float
    nakshatra: str
    nakshatra_pada: int
    moon_sign: str
    tithi: Optional[int]
    sun_longitude_sidereal: float


@dataclass
class GannInfo:
    price: float
    sqrt_price: float
    support: float
    resistance: float
    levels: List[float]
    near_support: bool
    near_resistance: bool


# ---------------- INDICATORS ----------------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)
    df["RSI"] = rsi(df["Close"], RSI_PERIOD)
    df["ATR"] = atr(df, ATR_PERIOD)
    df["VOL_AVG"] = df["Volume"].rolling(VOLUME_LOOKBACK).mean()
    return df


# ---------------- GANN ENGINE ----------------

def gann_square_levels(price: float, levels_each_side: int = 4) -> List[float]:
    """
    Square-root price levels.

    Example:
        price ~ 400
        sqrt(400)=20
        next square=21^2=441
        previous square=19^2=361

    We also generate fractional square levels at 0.25 increments
    to create a denser price map.
    """
    if price <= 0:
        return []

    root = math.sqrt(price)
    base = math.floor(root)

    roots = []
    for i in range(-levels_each_side, levels_each_side + 1):
        roots.append(base + i)

    # Quarter-root levels add intermediate Gann-style price zones.
    for i in range(-levels_each_side, levels_each_side):
        roots.append(base + i + 0.25)
        roots.append(base + i + 0.50)
        roots.append(base + i + 0.75)

    levels = sorted(set(round(x * x, 4) for x in roots if x > 0))
    return levels


def calculate_gann(price: float) -> GannInfo:
    levels = gann_square_levels(price)

    below = [x for x in levels if x <= price]
    above = [x for x in levels if x >= price]

    support = max(below) if below else min(levels)
    resistance = min(above) if above else max(levels)

    near_support = abs(price - support) / price * 100 <= GANN_NEAR_PCT
    near_resistance = abs(resistance - price) / price * 100 <= GANN_NEAR_PCT

    return GannInfo(
        price=price,
        sqrt_price=math.sqrt(price),
        support=support,
        resistance=resistance,
        levels=levels,
        near_support=near_support,
        near_resistance=near_resistance,
    )


# ---------------- PRICE ACTION ENGINE ----------------

def candle_features(df: pd.DataFrame) -> Dict[str, bool]:
    if len(df) < 4:
        return {}

    a = df.iloc[-1]
    p = df.iloc[-2]

    body = abs(a["Close"] - a["Open"])
    rng = max(a["High"] - a["Low"], 1e-9)
    upper = a["High"] - max(a["Open"], a["Close"])
    lower = min(a["Open"], a["Close"]) - a["Low"]

    bullish = a["Close"] > a["Open"]
    bearish = a["Close"] < a["Open"]

    bullish_engulf = (
        bullish and
        p["Close"] < p["Open"] and
        a["Open"] <= p["Close"] and
        a["Close"] >= p["Open"]
    )

    bearish_engulf = (
        bearish and
        p["Close"] > p["Open"] and
        a["Open"] >= p["Close"] and
        a["Close"] <= p["Open"]
    )

    hammer = lower >= body * 2 and upper <= body
    shooting_star = upper >= body * 2 and lower <= body

    strong_body = body / rng >= 0.60

    return {
        "bullish": bullish,
        "bearish": bearish,
        "bullish_engulf": bullish_engulf,
        "bearish_engulf": bearish_engulf,
        "hammer": hammer,
        "shooting_star": shooting_star,
        "strong_body": strong_body,
    }


def structure_signal(df: pd.DataFrame) -> str:
    if len(df) < 10:
        return "NEUTRAL"

    recent = df.tail(6)

    highs = recent["High"].values
    lows = recent["Low"].values

    higher_high = highs[-1] > highs[-3]
    higher_low = lows[-1] > lows[-3]

    lower_high = highs[-1] < highs[-3]
    lower_low = lows[-1] < lows[-3]

    if higher_high and higher_low:
        return "BULLISH"
    if lower_high and lower_low:
        return "BEARISH"
    return "NEUTRAL"


def price_action_score(df5: pd.DataFrame, df15: pd.DataFrame, df1h: pd.DataFrame):
    d5 = add_indicators(df5)
    d15 = add_indicators(df15)
    d1h = add_indicators(df1h)

    a5 = d5.iloc[-1]
    a15 = d15.iloc[-1]
    a1h = d1h.iloc[-1]

    c = candle_features(d5)

    score_buy = 0
    score_sell = 0
    reasons_buy = []
    reasons_sell = []

    # 1H trend
    if a1h["EMA50"] > a1h["EMA200"]:
        score_buy += 2
        reasons_buy.append("1H EMA50>EMA200")
    elif a1h["EMA50"] < a1h["EMA200"]:
        score_sell += 2
        reasons_sell.append("1H EMA50<EMA200")

    # 15M trend
    if a15["EMA20"] > a15["EMA50"]:
        score_buy += 1
        reasons_buy.append("15M EMA20>EMA50")
    elif a15["EMA20"] < a15["EMA50"]:
        score_sell += 1
        reasons_sell.append("15M EMA20<EMA50")

    # 5M trend
    if a5["EMA20"] > a5["EMA50"]:
        score_buy += 1
        reasons_buy.append("5M EMA20>EMA50")
    elif a5["EMA20"] < a5["EMA50"]:
        score_sell += 1
        reasons_sell.append("5M EMA20<EMA50")

    # RSI regime
    if 50 < a5["RSI"] < 70:
        score_buy += 1
        reasons_buy.append(f"5M RSI={a5['RSI']:.1f}")
    elif 30 < a5["RSI"] < 50:
        score_sell += 1
        reasons_sell.append(f"5M RSI={a5['RSI']:.1f}")

    # Volume expansion
    if pd.notna(a5["VOL_AVG"]) and a5["Volume"] > a5["VOL_AVG"] * 1.20:
        if a5["Close"] > a5["Open"]:
            score_buy += 1
            reasons_buy.append("Bullish volume expansion")
        elif a5["Close"] < a5["Open"]:
            score_sell += 1
            reasons_sell.append("Bearish volume expansion")

    # Candle confirmation
    if c.get("bullish_engulf") or c.get("hammer"):
        score_buy += 2
        reasons_buy.append("Bullish reversal candle")

    if c.get("bearish_engulf") or c.get("shooting_star"):
        score_sell += 2
        reasons_sell.append("Bearish reversal candle")

    # Structure
    s15 = structure_signal(d15)
    s5 = structure_signal(d5)

    if s15 == "BULLISH" and s5 == "BULLISH":
        score_buy += 2
        reasons_buy.append("5M+15M bullish structure")

    if s15 == "BEARISH" and s5 == "BEARISH":
        score_sell += 2
        reasons_sell.append("5M+15M bearish structure")

    return {
        "buy": score_buy,
        "sell": score_sell,
        "buy_reasons": reasons_buy,
        "sell_reasons": reasons_sell,
        "structure_5m": s5,
        "structure_15m": s15,
    }


# ---------------- VEDIC ASTROLOGY ENGINE ----------------

NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira",
    "Ardra", "Punarvasu", "Pushya", "Ashlesha", "Magha",
    "Purva Phalguni", "Uttara Phalguni", "Hasta", "Chitra",
    "Swati", "Vishakha", "Anuradha", "Jyeshtha", "Mula",
    "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta",
    "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
]

SIGNS = [
    "Mesha", "Vrishabha", "Mithuna", "Karka",
    "Simha", "Kanya", "Tula", "Vrischika",
    "Dhanu", "Makara", "Kumbha", "Meena"
]


def normalize_deg(x: float) -> float:
    return x % 360.0


def get_planet_longitude(jd_ut: float, planet: int) -> float:
    flags = swe.FLG_SWIEPH | swe.FLG_SPEED
    result, _ = swe.calc_ut(jd_ut, planet, flags)
    return normalize_deg(result[0])


def vedic_astro(timestamp_ist: datetime) -> AstroInfo:
    if swe is None:
        raise RuntimeError(
            "pyswisseph is not installed. Run: pip install pyswisseph"
        )

    ts = timestamp_ist.astimezone(pytz.UTC)
    hour_decimal = (
        ts.hour + ts.minute / 60 + ts.second / 3600
    )

    jd = swe.julday(ts.year, ts.month, ts.day, hour_decimal)

    # Lahiri sidereal mode
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    swe.set_topo(78.4867, 17.3850, 0)  # approximate India reference only

    sun_tropical = get_planet_longitude(jd, swe.SUN)
    moon_tropical = get_planet_longitude(jd, swe.MOON)

    ayanamsa = swe.get_ayanamsa_ut(jd)

    sun_sidereal = normalize_deg(sun_tropical - ayanamsa)
    moon_sidereal = normalize_deg(moon_tropical - ayanamsa)

    nak_size = 360 / 27
    nak_index = int(moon_sidereal / nak_size)
    nak_position = moon_sidereal % nak_size
    pada = int(nak_position / (nak_size / 4)) + 1

    moon_sign = SIGNS[int(moon_sidereal / 30)]

    # Approximate tithi:
    # Tithi = 12-degree separation of Moon from Sun.
    elongation = normalize_deg(moon_sidereal - sun_sidereal)
    tithi = int(elongation / 12) + 1

    return AstroInfo(
        timestamp=timestamp_ist,
        moon_longitude_tropical=moon_tropical,
        moon_longitude_sidereal=moon_sidereal,
        nakshatra=NAKSHATRAS[nak_index],
        nakshatra_pada=pada,
        moon_sign=moon_sign,
        tithi=tithi,
        sun_longitude_sidereal=sun_sidereal,
    )


def astro_filter(astro: AstroInfo) -> Dict:
    """
    Deliberately conservative astro component.

    It does NOT claim a particular Nakshatra causes price direction.
    Instead it creates a repeatable timing classification that can
    be statistically tested.

    The first version gives a neutral astro score. This is intentional:
    directional planetary rules should only be added after backtesting.
    """
    return {
        "score_buy": 0,
        "score_sell": 0,
        "timing_state": "NEUTRAL",
        "reason": (
            f"Moon {astro.moon_sign} / {astro.nakshatra} "
            f"P{astro.nakshatra_pada}, Tithi {astro.tithi}"
        ),
    }


# ---------------- COMBINATION ENGINE ----------------

def build_signal(
    symbol: str,
    df5: pd.DataFrame,
    df15: pd.DataFrame,
    df1h: pd.DataFrame,
    astro: AstroInfo,
) -> Dict:

    price = float(df5["Close"].iloc[-1])
    g = calculate_gann(price)
    pa = price_action_score(df5, df15, df1h)
    af = astro_filter(astro)

    d5 = add_indicators(df5)
    d15 = add_indicators(df15)
    d1h = add_indicators(df1h)

    a5 = d5.iloc[-1]
    a15 = d15.iloc[-1]
    a1h = d1h.iloc[-1]

    trend_1h = (
        "BULLISH" if a1h["EMA50"] > a1h["EMA200"]
        else "BEARISH" if a1h["EMA50"] < a1h["EMA200"]
        else "NEUTRAL"
    )
    trend_15m = (
        "BULLISH" if a15["EMA20"] > a15["EMA50"]
        else "BEARISH" if a15["EMA20"] < a15["EMA50"]
        else "NEUTRAL"
    )
    trend_5m = pa["structure_5m"]

    buy_score = pa["buy"] + af["score_buy"]
    sell_score = pa["sell"] + af["score_sell"]

    if g.near_support:
        buy_score += 2
    if g.near_resistance:
        sell_score += 2

    # HARD DIRECTIONAL GATE:
    # Never issue BUY unless all three timeframes are bullish.
    # Never issue SELL unless all three timeframes are bearish.
    bullish_alignment = (
        trend_1h == "BULLISH"
        and trend_15m == "BULLISH"
        and trend_5m == "BULLISH"
    )
    bearish_alignment = (
        trend_1h == "BEARISH"
        and trend_15m == "BEARISH"
        and trend_5m == "BEARISH"
    )

    direction = "WAIT"

    if bullish_alignment and buy_score >= MIN_SIGNAL_SCORE and buy_score > sell_score:
        direction = "BUY"
    elif bearish_alignment and sell_score >= MIN_SIGNAL_SCORE and sell_score > buy_score:
        direction = "SELL"

    score = max(buy_score, sell_score)

    atr_value = float(d5["ATR"].iloc[-1])
    if not np.isfinite(atr_value) or atr_value <= 0:
        atr_value = max(price * 0.005, 0.01)

    if direction == "BUY":
        entry = price
        sl_candidates = [x for x in (g.support, price - atr_value) if x < price]
        sl = max(sl_candidates) if sl_candidates else price - atr_value
        risk = max(entry - sl, atr_value * 0.50)
        tp1 = entry + risk * RISK_REWARD_1
        tp2 = entry + risk * RISK_REWARD_2

    elif direction == "SELL":
        entry = price
        sl_candidates = [x for x in (g.resistance, price + atr_value) if x > price]
        sl = min(sl_candidates) if sl_candidates else price + atr_value
        risk = max(sl - entry, atr_value * 0.50)
        tp1 = entry - risk * RISK_REWARD_1
        tp2 = entry - risk * RISK_REWARD_2

    else:
        entry = sl = tp1 = tp2 = np.nan

    return {
        "SYMBOL": symbol,
        "TIME": df5.index[-1],
        "PRICE": round(price, 2),
        "SIGNAL": direction,
        "SCORE": score,
        "BUY_SCORE": buy_score,
        "SELL_SCORE": sell_score,
        "TREND_1H": trend_1h,
        "TREND_15M": trend_15m,
        "STRUCTURE_5M": trend_5m,
        "GANN_SUPPORT": round(g.support, 2),
        "GANN_RESISTANCE": round(g.resistance, 2),
        "SQRT_PRICE": round(g.sqrt_price, 4),
        "NEAR_GANN_SUPPORT": g.near_support,
        "NEAR_GANN_RESISTANCE": g.near_resistance,
        "ASTRO": af["reason"],
        "NAKSHATRA": astro.nakshatra,
        "NAKSHATRA_PADA": astro.nakshatra_pada,
        "MOON_SIGN": astro.moon_sign,
        "TITHI": astro.tithi,
        "ENTRY": None if np.isnan(entry) else round(entry, 2),
        "SL": None if np.isnan(sl) else round(sl, 2),
        "TARGET_1": None if np.isnan(tp1) else round(tp1, 2),
        "TARGET_2": None if np.isnan(tp2) else round(tp2, 2),
        "PA_BUY_REASONS": "; ".join(pa["buy_reasons"]),
        "PA_SELL_REASONS": "; ".join(pa["sell_reasons"]),
    }


# ---------------- DOWNLOAD ----------------

def download_data(symbol: str, period: str = "60d") -> Dict[str, pd.DataFrame]:
    result = {}

    for label, interval in INTERVALS.items():
        try:
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if df is None or df.empty:
                print(f"{symbol} {label}: NO DATA")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            required = ["Open", "High", "Low", "Close", "Volume"]
            df = df[[c for c in required if c in df.columns]].dropna()

            if df.empty:
                continue

            # Normalize timezone to IST.
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert(IST)
            else:
                df.index = df.index.tz_convert(IST)

            result[label] = df

        except Exception as e:
            print(f"{symbol} {label}: {e}")

    return result


# ---------------- TELEGRAM ----------------

def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    import requests

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }

    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print("Telegram error:", e)


def format_signal(row: Dict) -> str:
    return f"""
<b>GANN + VEDIC + PRICE ACTION</b>

<b>{row['SYMBOL']}</b>
Signal: <b>{row['SIGNAL']}</b>
Score: {row['SCORE']} | BUY {row['BUY_SCORE']} / SELL {row['SELL_SCORE']}

Price: {row['PRICE']}
Gann Support: {row['GANN_SUPPORT']}
Gann Resistance: {row['GANN_RESISTANCE']}

Entry: {row['ENTRY']}
SL: {row['SL']}
Target 1: {row['TARGET_1']}
Target 2: {row['TARGET_2']}

1H Trend: {row['TREND_1H']}
15M Trend: {row['TREND_15M']}
5M Structure: {row['STRUCTURE_5M']}

Moon: {row['MOON_SIGN']}
Nakshatra: {row['NAKSHATRA']} P{row['NAKSHATRA_PADA']}
Tithi: {row['TITHI']}

Astro: {row['ASTRO']}
"""


# ---------------- MAIN SCANNER ----------------

def is_market_day(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    return now.strftime("%Y-%m-%d") not in NSE_HOLIDAYS_2026


def market_is_open(now: datetime) -> bool:
    if not is_market_day(now):
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 15 <= minutes < 15 * 60 + 30


def next_scan_boundary(now: datetime) -> datetime:
    """Next exact 5-minute boundary, e.g. 12:13 -> 12:15."""
    base = now.replace(second=0, microsecond=0)
    next_minute = ((now.minute // SCAN_INTERVAL_MINUTES) + 1) * SCAN_INTERVAL_MINUTES
    if next_minute >= 60:
        return base.replace(minute=0) + timedelta(hours=1)
    return base.replace(minute=next_minute)


def scan_once(sent_signals: set):
    now = datetime.now(IST)

    print("\n" + "=" * 100)
    print("GANN + VEDIC ASTROLOGY + PRICE ACTION | CONTINUOUS SCANNER")
    print("Scan:", now.strftime("%d-%m-%Y %H:%M:%S IST"))
    print("=" * 100)

    if not market_is_open(now):
        print("Market is closed. No scan.")
        return

    astro = vedic_astro(now)
    print(
        f"ASTRO | Moon={astro.moon_sign} | "
        f"Nakshatra={astro.nakshatra} P{astro.nakshatra_pada} | "
        f"Tithi={astro.tithi}"
    )

    rows = []

    for symbol in SYMBOLS:
        print(f"\nScanning {symbol}...")

        data = download_data(symbol)

        if not all(k in data for k in ("5m", "15m", "1h")):
            print("  Missing one or more timeframes.")
            continue

        if min(len(data[k]) for k in ("5m", "15m", "1h")) < 50:
            print("  Insufficient bars.")
            continue

        try:
            row = build_signal(symbol, data["5m"], data["15m"], data["1h"], astro)
            rows.append(row)

            print(
                f"  {row['SIGNAL']:4s} | Score={row['SCORE']:2d} | "
                f"1H={row['TREND_1H']} | 15M={row['TREND_15M']} | "
                f"5M={row['STRUCTURE_5M']} | Price={row['PRICE']}"
            )

            # Only alert once for a symbol/signal/5M candle.
            signal_key = (symbol, row["SIGNAL"], str(row["TIME"]))

            if row["SIGNAL"] in ("BUY", "SELL") and signal_key not in sent_signals:
                send_telegram(format_signal(row))
                sent_signals.add(signal_key)
                print("  >>> TELEGRAM ALERT SENT")

        except Exception as e:
            print("  ERROR:", e)

    if not rows:
        print("\nNo results.")
        return

    out = pd.DataFrame(rows)
    signal_order = {"BUY": 0, "SELL": 1, "WAIT": 2}
    out["_order"] = out["SIGNAL"].map(signal_order).fillna(9)
    out = out.sort_values(["_order", "SCORE"], ascending=[True, False]).drop(columns="_order")

    print("\n" + "=" * 100)
    print("FINAL SCAN")
    print("=" * 100)

    display_cols = [
        "SYMBOL", "SIGNAL", "SCORE", "PRICE",
        "TREND_1H", "TREND_15M", "STRUCTURE_5M",
        "GANN_SUPPORT", "GANN_RESISTANCE",
        "NAKSHATRA", "ENTRY", "SL", "TARGET_1", "TARGET_2"
    ]
    print(out[display_cols].to_string(index=False))

    now = datetime.now(IST)
    daily_log = f"astro_gann_scan_{now.strftime('%Y%m%d')}.csv"
    snapshot = f"astro_gann_scan_{now.strftime('%Y%m%d_%H%M%S')}.csv"

    write_header = not Path(daily_log).exists()
    out.to_csv(daily_log, mode="a", header=write_header, index=False)
    out.to_csv(snapshot, index=False)

    print(f"\nDaily log: {daily_log}")
    print(f"Snapshot : {snapshot}")


def continuous_scanner():
    import time

    print("=" * 100)
    print("GANN + VEDIC ASTROLOGY + PRICE ACTION | CONTINUOUS MODE")
    print("=" * 100)
    print("Interval : 5 minutes")
    print("Session  : 09:15-15:30 IST")
    print("Press CTRL+C to stop.")
    print("=" * 100)

    sent_signals = set()

    while True:
        now = datetime.now(IST)

        if not is_market_day(now):
            print(f"{now.strftime('%d-%m-%Y')} is NSE holiday/weekend. Waiting...")
            time.sleep(300)
            continue

        if market_is_open(now):
            scan_once(sent_signals)

            target = next_scan_boundary(datetime.now(IST))
            wait_seconds = max(1, (target - datetime.now(IST)).total_seconds())

            print(
                f"\nNext scan: {target.strftime('%H:%M:%S IST')} | "
                f"waiting {int(wait_seconds)} sec..."
            )
            time.sleep(wait_seconds)
        else:
            if now.hour < 9 or (now.hour == 9 and now.minute < 15):
                target = now.replace(hour=9, minute=15, second=0, microsecond=0)
                wait_seconds = max(30, (target - now).total_seconds())
                print(f"Market opens at {target.strftime('%H:%M:%S IST')}. Waiting...")
                time.sleep(wait_seconds)
            else:
                print("Market closed for today. Waiting for next trading day...")
                time.sleep(300)


if __name__ == "__main__":
    try:
        continuous_scanner()
    except KeyboardInterrupt:
        print("\nScanner stopped by user.")
