"""
Resumable fetcher for SPY historical options data across all 3 endpoints.

Endpoints fetched (one call each per trading day):
  1. HISTORICAL_OPTIONS                    -> full option chain
  2. HISTORICAL_PUT_CALL_RATIO             -> P/C ratio (full chain + by expiration)
  3. HISTORICAL_VOLUME_OPEN_INTEREST_RATIO -> V/OI per contract

Default: 5 years back (1260 trading days) x 3 endpoints = 3,780 calls.
At 75 req/min premium tier -> ~51 minutes.

Usage:
    python fetch_spy_1yr.py                              # fetch everything
    python fetch_spy_1yr.py --days 252                   # 1 year
    python fetch_spy_1yr.py --endpoints chains,pcr       # subset
    python fetch_spy_1yr.py --combine-only               # skip fetch, just build CSVs

Output CSVs land in pilot_data/:
    spy_chains.csv        (one row per contract per day)
    spy_pc_ratio.csv      (one row per (date, expiration) + a full-chain row per date)
    spy_vol_oi_ratio.csv  (one row per contract per day)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
CACHE_ROOT = HERE / "pilot_data" / "spy_chains"
DEFAULT_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")

ENDPOINTS = {
    "chains": {
        "function": "HISTORICAL_OPTIONS",
        "cache_subdir": "HISTORICAL_OPTIONS",
        "csv_name": "spy_chains.csv",
    },
    "pcr": {
        "function": "HISTORICAL_PUT_CALL_RATIO",
        "cache_subdir": "HISTORICAL_PUT_CALL_RATIO",
        "csv_name": "spy_pc_ratio.csv",
    },
    "voi": {
        "function": "HISTORICAL_VOLUME_OPEN_INTEREST_RATIO",
        "cache_subdir": "HISTORICAL_VOLUME_OPEN_INTEREST_RATIO",
        "csv_name": "spy_vol_oi_ratio.csv",
    },
}


def fetch_one(function: str, symbol: str, date: str, api_key: str,
              cache_dir: Path, delay: float) -> str:
    """Return 'cached' | 'fetched' | 'rate_limited' | 'empty' | 'error'."""
    cache_path = cache_dir / symbol / f"{date}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return "cached"

    url = (
        "https://www.alphavantage.co/query"
        f"?function={function}&symbol={symbol}&date={date}&apikey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=60)
        data = resp.json()
    except Exception as e:
        print(f"  [error] {function} {date}: {e}")
        return "error"

    if "Information" in data or "Note" in data:
        msg = data.get("Information") or data.get("Note")
        print(f"  [rate-limit] {msg[:200]}")
        return "rate_limited"

    has_data = bool(data.get("data")) or bool(data.get("put_call_ratio_full_chain"))
    if not has_data:
        cache_path.write_text(json.dumps(data))
        return "empty"

    cache_path.write_text(json.dumps(data))
    time.sleep(delay)
    return "fetched"


def combine_chains(symbol: str, out: Path) -> None:
    """HISTORICAL_OPTIONS -> flatten data[] list, add fetch_date."""
    src = CACHE_ROOT / "HISTORICAL_OPTIONS" / symbol
    files = sorted(src.glob("*.json"))
    if not files:
        print(f"[chains] no files in {src}")
        return
    first = True
    total = 0
    chunk: list[pd.DataFrame] = []
    for i, f in enumerate(files, 1):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        rows = data.get("data") or []
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["fetch_date"] = f.stem
        chunk.append(df)
        if i % 50 == 0 or i == len(files):
            out_df = pd.concat(chunk, ignore_index=True)
            out_df.to_csv(out, mode="w" if first else "a", header=first, index=False)
            total += len(out_df)
            chunk = []
            first = False
            print(f"  [chains] {i}/{len(files)} files, {total:,} rows written")
    print(f"[chains] done -> {out} ({total:,} rows)")


def combine_pcr(symbol: str, out: Path) -> None:
    """
    HISTORICAL_PUT_CALL_RATIO schema per file:
        {symbol, date, put_call_ratio_full_chain, put_call_ratio_by_expiration: [{date, value}, ...]}
    Output rows: one per expiration + one summary row with expiration='FULL_CHAIN'.
    """
    src = CACHE_ROOT / "HISTORICAL_PUT_CALL_RATIO" / symbol
    files = sorted(src.glob("*.json"))
    if not files:
        print(f"[pcr] no files in {src}")
        return
    rows = []
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        fetch_date = f.stem
        full = data.get("put_call_ratio_full_chain")
        if full is not None:
            rows.append({"fetch_date": fetch_date, "expiration": "FULL_CHAIN", "pc_ratio": full})
        for row in data.get("put_call_ratio_by_expiration") or []:
            rows.append({
                "fetch_date": fetch_date,
                "expiration": row.get("date"),
                "pc_ratio": row.get("value"),
            })
    if not rows:
        print("[pcr] no data rows extracted")
        return
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"[pcr] done -> {out} ({len(df):,} rows)")


def combine_voi(symbol: str, out: Path) -> None:
    """HISTORICAL_VOLUME_OPEN_INTEREST_RATIO -> flatten data[] list, add fetch_date."""
    src = CACHE_ROOT / "HISTORICAL_VOLUME_OPEN_INTEREST_RATIO" / symbol
    files = sorted(src.glob("*.json"))
    if not files:
        print(f"[voi] no files in {src}")
        return
    first = True
    total = 0
    chunk: list[pd.DataFrame] = []
    for i, f in enumerate(files, 1):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        rows = data.get("data") or []
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["fetch_date"] = f.stem
        chunk.append(df)
        if i % 50 == 0 or i == len(files):
            out_df = pd.concat(chunk, ignore_index=True)
            out_df.to_csv(out, mode="w" if first else "a", header=first, index=False)
            total += len(out_df)
            chunk = []
            first = False
            print(f"  [voi] {i}/{len(files)} files, {total:,} rows written")
    print(f"[voi] done -> {out} ({total:,} rows)")


COMBINERS = {"chains": combine_chains, "pcr": combine_pcr, "voi": combine_voi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--end", default=None, help="end date YYYY-MM-DD (default: today)")
    ap.add_argument("--days", type=int, default=1260, help="trading days back (default 1260 ~ 5y)")
    ap.add_argument("--api-key", default=DEFAULT_KEY)
    ap.add_argument("--delay", type=float, default=0.85,
                    help="seconds between calls (75/min premium -> 0.85)")
    ap.add_argument("--endpoints", default="chains,pcr,voi",
                    help="comma-separated subset: chains,pcr,voi")
    ap.add_argument("--combine-only", action="store_true", help="skip fetch, only combine")
    ap.add_argument("--out-dir", default=str(HERE / "pilot_data"),
                    help="where to write combined CSVs")
    args = ap.parse_args()

    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip() in ENDPOINTS]
    if not endpoints:
        print(f"No valid endpoints. Choose from: {list(ENDPOINTS.keys())}")
        return 1

    if not args.combine_only and not args.api_key:
        print("ERROR: no API key. Set ALPHAVANTAGE_API_KEY env var or pass --api-key.")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.combine_only:
        end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
        dates = pd.bdate_range(end=end, periods=args.days).strftime("%Y-%m-%d").tolist()

        print(f"Symbol={args.symbol}  range={dates[0]} ... {dates[-1]}")
        print(f"Endpoints={endpoints}  total calls={len(dates) * len(endpoints):,}")
        print(f"Cache root: {CACHE_ROOT}")

        totals = {k: {"cached": 0, "fetched": 0, "empty": 0, "rate_limited": 0, "error": 0}
                  for k in endpoints}
        stopped = False
        for i, d in enumerate(dates, 1):
            for ep in endpoints:
                cfg = ENDPOINTS[ep]
                cache_dir = CACHE_ROOT / cfg["cache_subdir"]
                status = fetch_one(cfg["function"], args.symbol, d, args.api_key,
                                   cache_dir, args.delay)
                totals[ep][status] += 1
                if status == "rate_limited":
                    print(f"Stopping. Progress: {i}/{len(dates)} days. Re-run to resume.")
                    stopped = True
                    break
            if i % 25 == 0:
                print(f"[{i:4d}/{len(dates)}] {d} done across {endpoints}")
            if stopped:
                break

        print("\nFetch summary:")
        for ep, c in totals.items():
            print(f"  {ep}: {c}")

    # Combine phase
    for ep in endpoints:
        out_path = out_dir / ENDPOINTS[ep]["csv_name"]
        COMBINERS[ep](args.symbol, out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
