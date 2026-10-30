"""Glue between the TUI and the explorer engine (explorer.py).

The engine is shared with the browser version, pyodide-data-explorer: both
load, sample, filter and evaluate through the same code. This module adds what
a terminal needs on top: per-column display formatting, sort orders, and plot
data reduced to the terminal's resolution.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import explorer as ex  # the engine shared with pyodide-data-explorer (see DEVELOPMENT_NOTES.md)

UserError = ex.UserError
STATE = ex.STATE
FORMATS_BY_EXT = ex.FORMATS_BY_EXT
DUCK_FORMATS = ex.DUCK_FORMATS
DEFAULT_BUDGET_MB = 512  # native pandas has far more room than the browser's 200 MB
MB = ex.MB

MAX_CELL = 200          # characters kept per cell in the table
MAX_COL_WIDTH = 40
MIN_COL_WIDTH = 3


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------

def is_numeric(dtype) -> bool:
    return ex._is_numeric_dtype(dtype)


def is_dates(values) -> bool:
    return bool(ex._is_dates(values))


def _clean_text(s: str) -> str:
    if len(s) > MAX_CELL:
        s = s[:MAX_CELL] + "…"
    if "\n" in s or "\t" in s or "\r" in s:
        s = s.replace("\r\n", "↵").replace("\n", "↵").replace("\r", "↵").replace("\t", "→")
    return s


class ColumnFormat:
    """How one column (or the index) is shown: alignment and a vectorised
    formatter, decided once from the whole column so a column's numbers line
    up on the decimal point however it is scrolled."""

    def __init__(self, values, thousands: bool = False):
        self.right = False
        self.thousands = thousands
        self.kind = "text"
        dtype = getattr(values, "dtype", None)
        if dtype is not None and pd.api.types.is_bool_dtype(dtype):
            self.kind = "bool"
        elif dtype is not None and pd.api.types.is_float_dtype(dtype):
            self.kind, self.right = "float", True
            self.spec = _float_spec(np.asarray(values, dtype="float64"))
        elif dtype is not None and pd.api.types.is_integer_dtype(dtype):
            self.kind, self.right = "int", True
        elif dtype is not None and pd.api.types.is_complex_dtype(dtype):
            self.kind, self.right = "complex", True
        elif dtype is not None and pd.api.types.is_datetime64_any_dtype(dtype):
            self.kind = "datetime"
            idx = pd.DatetimeIndex(values)
            valid = idx[~idx.isna()]
            self.date_only = bool(len(valid)) and bool((valid == valid.normalize()).all())
        elif dtype is not None and pd.api.types.is_timedelta64_dtype(dtype):
            self.kind, self.right = "timedelta", True

    def format(self, values) -> tuple[list[str], np.ndarray]:
        """Display strings and a null mask for ``values`` (a slice of the column)."""
        nulls = np.asarray(pd.isna(values)) if not isinstance(values, pd.MultiIndex) else np.zeros(len(values), bool)
        if nulls.ndim != 1:  # cells holding lists/arrays
            nulls = np.zeros(len(values), dtype=bool)
        k = self.kind
        if k == "float":
            arr = np.asarray(values, dtype="float64")
            spec = self.spec
            out = ["NaN" if m else ("inf" if v > 0 else "-inf") if math.isinf(v) else format(v, spec)
                   for v, m in zip(arr.tolist(), nulls)]
        elif k == "int":
            spec = "," if self.thousands else ""
            out = ["<NA>" if m else format(v, spec) for v, m in zip(np.asarray(values, dtype=object).tolist(), nulls)]
        elif k == "datetime":
            idx = pd.DatetimeIndex(values)
            strs = idx.strftime("%Y-%m-%d") if self.date_only else idx.astype(str)
            out = ["NaT" if m else s for s, m in zip(strs, nulls)]
        elif k == "bool":
            out = ["<NA>" if m else str(bool(v)) for v, m in zip(np.asarray(values, dtype=object).tolist(), nulls)]
        else:
            out = []
            for v, m in zip(np.asarray(values, dtype=object).tolist() if not isinstance(values, pd.MultiIndex)
                            else list(values), nulls):
                if m:
                    out.append("NaT" if v is pd.NaT else "<NA>" if v is pd.NA else "None" if v is None else "NaN")
                elif isinstance(v, str):
                    out.append(_clean_text(v))
                elif isinstance(v, bytes):
                    out.append(_clean_text(repr(v)))
                else:
                    out.append(_clean_text(str(v)))
        return out, nulls


def _float_spec(arr: np.ndarray) -> str:
    """A format spec that suits a float column: fixed decimals (as few as the
    values need, up to 6), or general/scientific for very large or tiny values."""
    if len(arr) > 3000:
        arr = np.concatenate([arr[:1000], arr[-1000:], arr[np.random.default_rng(0).integers(0, len(arr), 1000)]])
    finite = arr[np.isfinite(arr)]
    if not len(finite):
        return ".6g"
    big = np.abs(finite).max()
    nz = np.abs(finite[finite != 0])
    if big >= 1e15 or (len(nz) and nz.max() < 1e-4):
        return ".6g"
    group = "," if big >= 1e5 else ""  # 1,234,567.89 reads far better than 1234567.89
    with np.errstate(all="ignore"):
        for d in range(7):
            scaled = finite * 10 ** d
            if np.all(np.abs(scaled - np.round(scaled)) < 1e-6 * np.maximum(1, np.abs(scaled))):
                return f"{group}.{d}f"
    return f"{group}.6f"


def column_width(name: str, fmt: ColumnFormat, values) -> int:
    """Width for a column from its header and a sample of its values."""
    n = len(values)
    if n > 120:
        pos = np.unique(np.concatenate([np.arange(40), np.arange(n - 40, n),
                                        np.random.default_rng(0).integers(0, n, 40)]))
        values = values[pos] if isinstance(values, (np.ndarray, pd.Index)) else values.iloc[pos]
    strs, _ = fmt.format(values)
    w = max([len(s) for s in strs] + [len(name), MIN_COL_WIDTH])
    return min(w, MAX_COL_WIDTH)


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def sort_order(values, ascending: bool) -> np.ndarray:
    """Positions that sort ``values`` (missing values last)."""
    s = pd.Series(np.asarray(values) if not isinstance(values, pd.Series) else values.to_numpy(),
                  dtype=getattr(values, "dtype", None)).reset_index(drop=True)
    try:
        return s.sort_values(ascending=ascending, kind="stable", na_position="last").index.to_numpy()
    except TypeError:  # mixed types: compare as text
        return s.astype(str).sort_values(ascending=ascending, kind="stable", na_position="last").index.to_numpy()


# ---------------------------------------------------------------------------
# Column stats / details
# ---------------------------------------------------------------------------

_stats_cache: dict = {"df": None, "frame": None}

STAT_COLUMNS = ["name", "dtype", "count", "nulls", "null %", "unique", "min", "max", "mean", "std",
                "idxmin", "idxmax"]


def column_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column stats (the web app's vectorised implementation) as a frame,
    one row per column; cached until the view changes."""
    if _stats_cache["df"] is df:
        return _stats_cache["frame"]
    rows = ex._compute_colstats(df)
    frame = pd.DataFrame({
        "name": [r["name"] for r in rows],
        "dtype": [r["dtype"] for r in rows],
        "count": [r["count"] for r in rows],
        "nulls": [r["nulls"] for r in rows],
        "null %": [None if r["null_frac"] is None else round(100 * r["null_frac"], 2) for r in rows],
        "unique": pd.array([r.get("unique") for r in rows], dtype="Int64"),
        "min": [r.get("min") for r in rows],
        "max": [r.get("max") for r in rows],
        "mean": [r["_raw"].get("mean") if r["_raw"] else _num(r.get("mean")) for r in rows],
        "std": [r["_raw"].get("std") if r["_raw"] else _num(r.get("std")) for r in rows],
        "idxmin": [r.get("idxmin") for r in rows],
        "idxmax": [r.get("idxmax") for r in rows],
    }, index=pd.RangeIndex(len(rows), name="#"))
    frame.attrs["thousands"] = True
    frame["mean"] = pd.to_numeric(frame["mean"], errors="coerce")
    frame["std"] = pd.to_numeric(frame["std"], errors="coerce")
    # numeric min/max sort as numbers, not text
    frame.attrs["sort_keys"] = {
        "min": pd.Series([r["_raw"].get("min") if r["_raw"] else None for r in rows], dtype="float64"),
        "max": pd.Series([r["_raw"].get("max") if r["_raw"] else None for r in rows], dtype="float64"),
    }
    _stats_cache.update(df=df, frame=frame)
    return frame


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def column_detail(s: pd.Series) -> dict:
    """Everything the column screen shows for one column."""
    st = ex._col_stats(s)
    out = {"stats": st, "quantiles": None, "top": None}
    valid = s.dropna()
    numeric = is_numeric(s.dtype)
    if len(valid) and (numeric or pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s)):
        qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["quantiles"] = [(f"{int(q * 100)}%", ex._cell(v)) for q, v in zip(qs, valid.quantile(qs))]
    try:
        vc = s.value_counts(dropna=False).head(50)
    except TypeError:
        vc = s.astype(str).value_counts(dropna=False).head(50)
    n = len(s)
    out["top"] = pd.DataFrame({
        "value": vc.index.astype(object) if not isinstance(vc.index, pd.MultiIndex) else vc.index.to_flat_index(),
        "count": vc.to_numpy(),
        "%": np.round(100 * vc.to_numpy() / n, 3) if n else np.nan,
    })
    out["top"].attrs["thousands"] = True
    return out


# ---------------------------------------------------------------------------
# Plot data, reduced to what the terminal can show (every row contributes)
# ---------------------------------------------------------------------------

OTHER = ex.OTHER
MAX_GROUPS = 8


def as_axis(values):
    """(float64 array, kind) for numeric / date / timedelta values, else (None, None)."""
    dtype = getattr(values, "dtype", None)
    if dtype is None:
        return None, None
    if pd.api.types.is_datetime64_any_dtype(dtype):
        idx = pd.DatetimeIndex(values)
        if idx.tz is not None:
            idx = idx.tz_convert(None)
        arr = idx.as_unit("ns").asi8.astype("float64")
        arr[np.asarray(idx.isna())] = np.nan
        return arr / 1e9, "datetime"
    if pd.api.types.is_timedelta64_dtype(dtype):
        td = pd.TimedeltaIndex(values)
        arr = td.as_unit("ns").asi8.astype("float64")
        arr[np.asarray(td.isna())] = np.nan
        return arr / 1e9, "timedelta"
    if pd.api.types.is_bool_dtype(dtype):
        return np.asarray(pd.Series(values).astype("float64")), "number"
    if is_numeric(dtype):
        try:
            return np.asarray(pd.Series(values).astype("float64")), "number"
        except (TypeError, ValueError):
            return None, None
    return None, None


def m4(x: np.ndarray, y: np.ndarray, buckets: int):
    """First / min / max / last point of each x bucket: at this resolution a
    line through these looks the same as one through every point. Linear time
    (no sort) when x is already in order, as it usually is."""
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.all():
        x, y = x[ok], y[ok]
    if len(x) <= 4 * buckets:
        return x, y
    if not (x[1:] >= x[:-1]).all():
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]
    lo, hi = x[0], x[-1]
    if hi == lo:
        return x[[0, -1]], y[[0, -1]]
    b = np.minimum(((x - lo) * (buckets / (hi - lo))).astype(np.int64), buckets - 1)
    starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
    ends = np.r_[starts[1:], len(x)] - 1
    ymin, ymax = np.minimum.reduceat(y, starts), np.maximum.reduceat(y, starts)
    # min and max share the bucket's dot column, so their exact x doesn't show
    xs = np.column_stack([x[starts], x[starts], x[ends], x[ends]]).ravel()
    ys = np.column_stack([y[starts], ymin, ymax, y[ends]]).ravel()
    return xs, ys


def occupied(x: np.ndarray, y: np.ndarray, w: int, h: int):
    """Centres of the (w × h) grid cells that hold at least one point."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) <= 2000:
        return x, y
    xr = (x.min(), x.max()) if x.max() > x.min() else (x.min() - 0.5, x.max() + 0.5)
    yr = (y.min(), y.max()) if y.max() > y.min() else (y.min() - 0.5, y.max() + 0.5)
    H, xe, ye = np.histogram2d(x, y, bins=[w, h], range=[xr, yr])
    ix, iy = np.nonzero(H)
    return (xe[ix] + xe[ix + 1]) / 2, (ye[iy] + ye[iy + 1]) / 2


def factorize(values):
    """(codes, labels): a code per value (-1 for missing) and a display label per code."""
    codes, uniques = pd.factorize(values if isinstance(values, (pd.Series, pd.Index)) else np.asarray(values))
    return codes, [str(u) for u in uniques]


def groups(values, limit=MAX_GROUPS):
    """(labels, codes) for colouring by a column: the most common ``limit - 1``
    values, the rest (and missing values) folded into "(other)"."""
    raw, names = factorize(values)
    counts = np.bincount(raw[raw >= 0], minlength=len(names))
    order = np.argsort(-counts, kind="stable")
    folded = len(names) > limit or (raw < 0).any()
    keep = order[: limit - 1] if folded else order
    labels = [names[i] for i in keep] + ([OTHER] if folded else [])
    lookup = np.full(len(names) + 1, len(labels) - 1, dtype=np.int64)  # last slot: missing (-1)
    lookup[keep] = np.arange(len(keep))
    return labels, lookup[raw]


def nice_ticks(lo: float, hi: float, kind: str, n: int = 6):
    """Tick positions and labels for an axis (dates and durations get readable labels)."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return None
    if kind == "datetime":
        span = hi - lo
        pos = np.linspace(lo, hi, n)
        stamps = pd.to_datetime(pos, unit="s")
        if span > 3 * 86400 * 365:
            fmt = "%Y"
        elif span > 60 * 86400:
            fmt = "%Y-%m"
        elif span > 2 * 86400:
            fmt = "%Y-%m-%d"
        elif span > 3600:
            fmt = "%m-%d %H:%M"
        else:
            fmt = "%H:%M:%S"
        return list(pos), [t.strftime(fmt) for t in stamps]
    if kind == "timedelta":
        pos = np.linspace(lo, hi, n)
        return list(pos), [str(pd.Timedelta(seconds=float(p))).replace("0 days ", "") for p in pos]
    # numbers: round steps (1, 2 or 5 × 10^k)
    raw = (hi - lo) / max(1, n)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    pos = np.arange(math.ceil(lo / step) * step, hi + step * 1e-9, step)
    decimals = max(0, -math.floor(math.log10(step)))
    if round(step, decimals) != step:
        decimals += 1
    if decimals == 0 and max(abs(lo), abs(hi)) < 1e12:
        return list(pos), [f"{p:,.0f}" for p in pos]
    return list(pos), [f"{p:.3g}" if max(abs(lo), abs(hi)) >= 1e6 else f"{p:.{decimals}f}" for p in pos]


def short(v, n=24) -> str:
    s = str(v)
    return s if len(s) <= n else s[:n - 1] + "…"


# ---------------------------------------------------------------------------
# Steps as a script
# ---------------------------------------------------------------------------

STEP_LABELS = {"update": "changed df", "view": "result became the view", "define": "defined names"}

READERS = {
    "csv": lambda n, k: f"pd.read_csv({n!r})",
    "parquet": lambda n, k: f"pd.read_parquet({n!r})",
    "jsonl": lambda n, k: f"pd.read_json({n!r}, lines=True)",
    "json": lambda n, k: f"pd.read_json({n!r})",
    "xlsx": lambda n, k: f"pd.read_excel({n!r}" + (f", sheet_name={k!r})" if k else ")"),
    "h5": lambda n, k: f"pd.read_hdf({n!r}" + (f", key={k!r})" if k else ")"),
}


def steps_script() -> str:
    """The applied steps as a runnable Python script (as the web app's "Copy as Python")."""
    i = STATE.info
    sel = i.get("selection")
    lines = [f"# Steps applied in dx to {i['name']}" + (f" ({i['key']})" if i.get("key") else "")]
    if sel and sel.get("sql"):
        lines += ["# Loaded with SQL (the file is the table \"data\"):"] + [f"# {x}" for x in sel["sql"].splitlines()]
    if sel and not sel.get("sql") and sel.get("columns") is not None:
        lines.append(f"# Loaded {sel['columns']} of {sel['of']} columns")
    if sel and not sel.get("sql") and sel.get("where"):
        lines += ["# Loaded rows where:"] + [f"# {x}" for x in sel["where"].splitlines()]
    if i.get("sampled"):
        lines.append(f"# Sampled to {i['rows_loaded']} rows to fit the memory budget")
    reader = READERS.get(i["format"], READERS["csv"])(STATE.path or i["name"], i.get("key"))
    lines += ["import numpy as np", "import pandas as pd", "",
              f"df = {reader}  # approximately: dx also picked the index ({i.get('index_mode')}) and parsed dates",
              "df0 = df"]
    for n, s in enumerate(STATE.steps, 1):
        lines += ["", f"# {n}. {STEP_LABELS.get(s['kind'], s['kind'])}", s["code"]]
    return "\n".join(lines) + "\n"


def human_bytes(n) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return str(n)


def save_view(df: pd.DataFrame, path: str) -> str:
    """Write the current view; the format follows the extension."""
    p = Path(path).expanduser()
    ext = p.suffix.lower()
    if p.exists() and p.is_dir():
        raise UserError(f"{p} is a folder")
    if ext in (".parquet", ".pq"):
        df.to_parquet(p)
    elif ext in (".csv", ".txt"):
        df.to_csv(p)
    elif ext == ".tsv":
        df.to_csv(p, sep="\t")
    elif ext in (".jsonl", ".ndjson"):
        df.reset_index().to_json(p, orient="records", lines=True, date_format="iso")
    elif ext == ".json":
        df.reset_index().to_json(p, orient="records", date_format="iso")
    elif ext in (".xlsx",):
        df.to_excel(p)
    elif ext in (".h5", ".hdf5"):
        df.to_hdf(p, key="data")
    elif ext in (".pkl", ".pickle"):
        df.to_pickle(p)
    else:
        raise UserError("Use a .parquet, .csv, .tsv, .json, .jsonl, .xlsx, .h5 or .pkl file name")
    return str(p)
