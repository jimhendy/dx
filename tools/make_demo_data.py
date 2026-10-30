"""Make the demo dataset used for the README GIFs: a year of synthetic trades.

    uv run python tools/make_demo_data.py            # demo/trades.parquet, 100 M rows (~1.3 GB)
    uv run python tools/make_demo_data.py --rows 1e6 # something smaller

Prices follow a random walk per symbol, so plots look like real markets. The
file is written in row groups with a dated index, like a real tick store.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SYMBOLS = {  # symbol: (start price, yearly volatility, share of trades)
    "NVDA": (120, 0.55, 0.16), "AAPL": (190, 0.25, 0.13), "MSFT": (410, 0.22, 0.11), "TSLA": (240, 0.60, 0.11),
    "AMZN": (180, 0.30, 0.09), "META": (500, 0.35, 0.08), "GOOG": (165, 0.28, 0.08), "AMD": (150, 0.50, 0.07),
    "JPM": (200, 0.20, 0.05), "XOM": (110, 0.22, 0.05), "KO": (62, 0.12, 0.04), "BRK.B": (420, 0.15, 0.03),
}
VENUES = (["NASDAQ", "NYSE", "ARCA", "BATS", "IEX", "EDGX"], [0.34, 0.24, 0.14, 0.12, 0.08, 0.08])
ORDER_TYPES = (["limit", "market", "iceberg", "stop"], [0.71, 0.22, 0.05, 0.02])
NOTES = ["manual override", "client amend", "late report", "price improvement", "split fill", "corrected venue"]
START = pd.Timestamp("2025-01-02 09:30")
SPAN_S = 365 * 86400
GRID = 20_000  # points on each symbol's price path


def price_paths(rng):
    """A smooth-ish geometric random walk per symbol over the year (with a few jumps)."""
    paths = {}
    dt = 1 / GRID
    for sym, (p0, vol, _) in SYMBOLS.items():
        steps = rng.normal(0.08 * dt, vol * np.sqrt(dt), GRID)
        jumps = rng.random(GRID) < 3 / GRID
        steps[jumps] += rng.normal(0, 0.06, jumps.sum())
        paths[sym] = p0 * np.exp(np.cumsum(steps))
    return paths


def chunk(rng, n, t0, t1, paths, syms, probs):
    ts = np.sort(rng.uniform(t0, t1, n))
    sym_idx = rng.choice(len(syms), n, p=probs)
    grid_pos = np.minimum((ts / SPAN_S * (GRID - 1)).astype(np.int64), GRID - 1)
    base = np.empty(n)
    for i, s in enumerate(syms):
        m = sym_idx == i
        base[m] = paths[s][grid_pos[m]]
    price = np.round(base * (1 + rng.normal(0, 0.0007, n)), 2)
    qty = np.maximum(1, rng.lognormal(4.2, 1.3, n).astype(np.int64))
    notional = price * qty
    fee = np.round(notional * rng.uniform(0.00002, 0.0001, n), 4)
    fee[rng.random(n) < 0.03] = np.nan
    note = np.full(n, None, dtype=object)
    has = rng.random(n) < 0.004
    note[has] = rng.choice(NOTES, has.sum())
    df = pd.DataFrame({
        "symbol": pd.Categorical.from_codes(sym_idx, syms).astype(str),
        "side": np.where(rng.random(n) < 0.51, "buy", "sell"),
        "price": price,
        "qty": qty,
        "venue": rng.choice(VENUES[0], n, p=VENUES[1]),
        "order_type": rng.choice(ORDER_TYPES[0], n, p=ORDER_TYPES[1]),
        "fee": fee,
        "latency_us": np.round(rng.lognormal(5.5, 0.6, n), 1),
        "trader_id": rng.integers(1000, 6000, n),
        "flagged": rng.random(n) < 0.0015,
        "note": pd.array(note, dtype="str"),
    }, index=pd.DatetimeIndex(START + pd.to_timedelta(ts, unit="s"), name="ts").as_unit("ms"))
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=float, default=100e6)
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "demo", "trades.parquet"))
    p.add_argument("--row-group", type=int, default=1_000_000)
    a = p.parse_args()
    rows = int(a.rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rng = np.random.default_rng(42)
    paths = price_paths(rng)
    syms = list(SYMBOLS)
    probs = np.array([v[2] for v in SYMBOLS.values()])
    probs /= probs.sum()
    nchunks = max(1, -(-rows // a.row_group))
    edges = np.linspace(0, SPAN_S, nchunks + 1)
    writer = None
    t = time.time()
    for i in range(nchunks):
        n = min(a.row_group, rows - i * a.row_group)
        table = pa.Table.from_pandas(chunk(rng, n, edges[i], edges[i + 1], paths, syms, probs), preserve_index=True)
        if writer is None:
            writer = pq.ParquetWriter(a.out, table.schema, compression="zstd")
        writer.write_table(table)
        print(f"\r{(i + 1) * 100 // nchunks}%", end="", flush=True)
    writer.close()
    print(f"\n{a.out}: {rows:,} rows, {os.path.getsize(a.out) / 1e9:.2f} GB in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
