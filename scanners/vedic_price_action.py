    return 9 * 60 + 15 <= minutes < 15 * 60 + 30


def scan_once(sent_signals: set):
    now = datetime.now(IST)

    print("\n" + "=" * 100)
    print("GANN + VEDIC ASTROLOGY + PRICE ACTION | GITHUB ACTIONS SCANNER")
    print("Scan:", now.strftime("%d-%m-%Y %H:%M:%S IST"))
    print("=" * 100)

    if not market_is_open(now):
        print("Market is closed. Scanner finished.")
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
            row = build_signal(
                symbol,
                data["5m"],
                data["15m"],
                data["1h"],
                astro,
            )
            rows.append(row)

            print(
                f"  {row['SIGNAL']:4s} | Score={row['SCORE']:2d} | "
                f"1H={row['TREND_1H']} | 15M={row['TREND_15M']} | "
                f"5M={row['STRUCTURE_5M']} | Price={row['PRICE']}"
            )

            # A GitHub Actions run is one scan, so this set prevents
            # duplicate alerts within the current run.
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
    out = out.sort_values(
        ["_order", "SCORE"],
        ascending=[True, False]
    ).drop(columns="_order")

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


if __name__ == "__main__":
    sent_signals = set()

    try:
        scan_once(sent_signals)
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        raise
