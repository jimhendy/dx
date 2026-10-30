"""Terminal plots (plotext). Every row contributes, but each series is first
reduced to the plot's resolution: lines keep each column's min / max / first /
last point, scatter plots keep every occupied cell, histograms and bars are
binned / aggregated with numpy and pandas."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Input, Label, Select, SelectionList, Static
from textual_plotext import PlotextPlot

from . import engine as E

KINDS = [("Line", "line"), ("Scatter", "scatter"), ("Histogram", "hist"), ("Bar", "bar")]
AGGS = [("count", "count"), ("mean", "mean"), ("sum", "sum"), ("median", "median"), ("min", "min"),
        ("max", "max")]
INDEX_KEY = "\x00index"
NONE = Select.NULL


class PlotPane(Horizontal):
    DEFAULT_CSS = """
    PlotPane #plot-controls { width: 30; padding: 0 1; border-right: vkey $panel-lighten-2; }
    PlotPane #plot-controls Label { margin-top: 1; color: $text-muted; }
    PlotPane #plot-controls Select { width: 100%; }
    PlotPane #plot-controls Input, PlotPane #plot-controls Select > SelectCurrent { background: $panel; }
    PlotPane #ys { height: auto; max-height: 12; background: $panel; }
    PlotPane #plot-note { height: auto; color: $text-muted; padding: 0 1; }
    PlotPane #plot-area { width: 1fr; }
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.df: pd.DataFrame | None = None
        self._populating = False
        self._ys: list = []
        self._drawn = None  # what the plot currently shows, to skip redundant redraws

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="plot-controls"):
            yield Label("Kind")
            yield Select(KINDS, value="line", allow_blank=False, id="kind", compact=True)
            yield Label("X")
            yield Select([], id="x", prompt="(index)", compact=True)
            yield Label("Y  (space ticks)")
            yield Input(placeholder="filter columns…", id="yfilter", compact=True)
            yield SelectionList(id="ys", compact=True)
            yield Label("Colour by")
            yield Select([], id="color", prompt="(none)", compact=True)
            yield Label("Bar value")
            yield Select(AGGS, value="count", allow_blank=False, id="agg", compact=True)
            yield Label("Histogram bins")
            yield Input("40", id="bins", type="integer", compact=True)
        with Vertical(id="plot-area"):
            yield PlotextPlot(id="plot")
            yield Static("", id="plot-note")

    # ----------------------------------------------------------- options

    def set_frame(self, df: pd.DataFrame | None) -> None:
        """New data: refresh the column choices, keeping what still applies."""
        self.df = df
        self._drawn = None
        if df is None:
            return
        self._populating = True
        try:
            names = [str(c) for c in df.columns]
            xsel = self.query_one("#x", Select)
            old_x = xsel.value
            xopts = [(Text(n), i) for i, n in enumerate(names)]
            xsel.set_options(xopts)
            if isinstance(old_x, int) and old_x < len(names):
                xsel.value = old_x
            csel = self.query_one("#color", Select)
            old_c = csel.value
            csel.set_options(xopts)
            if isinstance(old_c, int) and old_c < len(names):
                csel.value = old_c
            numeric = [i for i, t in enumerate(df.dtypes) if E.as_axis(pd.Series([], dtype=t))[1] is not None]
            self._ys = [y for y in self._ys if y in numeric]
            if not self._ys and numeric:
                floats = [i for i in numeric if pd.api.types.is_float_dtype(df.dtypes.iloc[i])]
                self._ys = [(floats or numeric)[0]]
            self._numeric = numeric
            self._fill_ys()
        finally:
            self._populating = False
        self.draw()

    def _fill_ys(self):
        flt = self.query_one("#yfilter", Input).value.strip().lower()
        df = self.df
        sl = self.query_one("#ys", SelectionList)
        sl.clear_options()
        shown = [i for i in self._numeric if flt in str(df.columns[i]).lower()][:500]
        sl.add_options([(Text(E.short(df.columns[i], 28)), i, i in self._ys) for i in shown])

    def configure(self, kind: str, y: int | None = None, x: int | None = None) -> None:
        """Set up a plot from elsewhere (the table's "plot this column")."""
        self._populating = True
        try:
            self.query_one("#kind", Select).value = kind
            xs = self.query_one("#x", Select)
            if x is None:
                xs.clear()
            else:
                xs.value = x
            if y is not None:
                self._ys = [y]
                self._fill_ys()
        finally:
            self._populating = False
        self.draw()

    @on(Input.Changed, "#yfilter")
    def _yfilter(self) -> None:
        if self.df is not None:
            self._populating = True
            try:
                self._fill_ys()
            finally:
                self._populating = False

    @on(SelectionList.SelectedChanged, "#ys")
    def _ys_changed(self, event: SelectionList.SelectedChanged) -> None:
        if self._populating:
            return
        sl = event.selection_list
        visible = {sl.get_option_at_index(i).value for i in range(sl.option_count)}
        chosen = set(sl.selected)
        self._ys = [y for y in self._ys if y not in visible] + [y for y in sorted(chosen)]
        self.draw()

    @on(Select.Changed)
    @on(Input.Submitted, "#bins")
    def _changed(self) -> None:
        if not self._populating:
            self.draw()

    def on_resize(self) -> None:
        self.draw()

    # -------------------------------------------------------------- drawing

    def _settings(self):
        g = lambda i: self.query_one(i, Select).value  # noqa: E731
        try:
            bins = max(2, min(500, int(self.query_one("#bins", Input).value or 40)))
        except ValueError:
            bins = 40
        x, color = g("#x"), g("#color")
        return dict(kind=g("#kind"), x=x if isinstance(x, int) else None, color=color if isinstance(color, int) else None,
                    agg=g("#agg"), bins=bins, ys=list(self._ys))

    def draw(self) -> None:
        if self.df is None or not self.is_mounted:
            return
        plot = self.query_one("#plot", PlotextPlot)
        w, h = plot.size.width, plot.size.height
        if w < 10 or h < 5:
            return
        settings = self._settings()
        key = (id(self.df), repr(settings), w, h)
        if key == self._drawn:
            return
        self._drawn = key
        self._reduce(self.df, settings, w, h)

    @work(thread=True, exclusive=True, group="plot")
    def _reduce(self, df, s, w, h) -> None:
        try:
            spec = build_spec(df, s, w, h)
        except Exception as e:  # noqa: BLE001
            spec = {"error": f"{type(e).__name__}: {e}"}
        self.app.call_from_thread(self._show, df, spec)

    def _show(self, df, spec) -> None:
        if df is not self.df:
            return
        plot = self.query_one("#plot", PlotextPlot)
        plt = plot.plt
        plt.clear_figure()
        note = spec.get("note", "")
        if "error" in spec or not spec.get("series") and not spec.get("bars"):
            plt.title(spec.get("error") or spec.get("empty") or "Nothing to plot")
            plot.refresh()
            self.query_one("#plot-note", Static).update(Text(note))
            return
        if spec.get("bars"):
            labels, values = spec["bars"]
            plt.bar(labels, values, orientation="h" if spec.get("horizontal") else "v", width=0.6)
        for sr in spec.get("series", []):
            if sr["kind"] == "scatter":
                plt.scatter(sr["x"], sr["y"], marker="braille", label=sr.get("label"))
            elif sr["kind"] == "bar":
                plt.bar(sr["x"], sr["y"], width=sr.get("width", 1), label=sr.get("label"))
            else:
                plt.plot(sr["x"], sr["y"], marker="braille", label=sr.get("label"))
        if spec.get("xticks"):
            plt.xticks(*spec["xticks"])
        if spec.get("yticks"):
            plt.yticks(*spec["yticks"])
        plt.title(spec.get("title", ""))
        if spec.get("xlabel"):
            plt.xlabel(spec["xlabel"])
        plot.refresh()
        self.query_one("#plot-note", Static).update(Text(note))


def build_spec(df: pd.DataFrame, s: dict, w: int, h: int) -> dict:
    """What to draw, reduced to a plot about ``w`` × ``h`` cells (runs in a thread)."""
    n = len(df)
    kind = s["kind"]
    if n == 0:
        return {"empty": "No rows"}
    pw, ph = max(10, (w - 12) * 2), max(4, (h - 4) * 4)  # braille: 2 × 4 dots per cell
    x_name = "index" if s["x"] is None else str(df.columns[s["x"]])
    xs = df.index if s["x"] is None else df.iloc[:, s["x"]]
    ys = [y for y in s["ys"] if y < df.shape[1]]
    notes = []

    if kind == "bar":
        return _bars(df, xs, x_name, ys, s["agg"], h)

    if not ys:
        return {"empty": "Tick a Y column (numeric, date or boolean)"}
    labels, codes = (None, None)
    if s["color"] is not None and kind != "hist" or (kind == "hist" and s["color"] is not None and len(ys) == 1):
        labels, codes = E.groups(df.iloc[:, s["color"]])
        if E.OTHER in labels:
            notes.append(f"colour: top {len(labels) - 1} values shown, the rest as {E.OTHER}")

    if kind == "hist":
        series = []
        arrays = []
        for y in ys:
            a, k = E.as_axis(df.iloc[:, y])
            arrays.append((str(df.columns[y]), a, k))
        finite = np.concatenate([a[np.isfinite(a)] for _, a, _ in arrays])
        if not len(finite):
            return {"empty": "No finite values"}
        lo, hi = finite.min(), finite.max()
        edges = np.histogram_bin_edges(finite, bins=s["bins"], range=(lo, hi) if hi > lo else (lo - .5, hi + .5))
        centres = (edges[:-1] + edges[1:]) / 2
        parts = [(name, a, k) for name, a, k in arrays]
        if labels is not None:
            name, a, k = arrays[0]
            parts = [(lab, a[codes == i], k) for i, lab in enumerate(labels)]
        for name, a, _ in parts:
            counts, _ = np.histogram(a[np.isfinite(a)], bins=edges)
            series.append({"kind": "bar" if len(parts) == 1 else "line", "x": centres.tolist(),
                           "y": counts.tolist(), "label": name if len(parts) > 1 else None,
                           "width": 1})
        kind0 = arrays[0][2]
        top = max(max(sr["y"]) for sr in series)
        spec = {"series": series, "title": f"Distribution of {', '.join(p[0] for p in arrays)}"
                + (f" by {df.columns[s['color']]}" if labels is not None else ""),
                "xticks": E.nice_ticks(edges[0], edges[-1], kind0),
                "yticks": E.nice_ticks(0, max(1, top), "number", 5)}
        spec["note"] = f"{n:,} rows · {s['bins']} bins" + ("" if not notes else " · " + "; ".join(notes))
        return spec

    xa, xk = E.as_axis(xs)
    if xa is None:  # text x: plot against position
        xa, xk = np.arange(n, dtype="float64"), "number"
        notes.append(f"{x_name} isn't numeric: plotted against row position")
    series = []
    for y in ys:
        ya, _ = E.as_axis(df.iloc[:, y])
        name = str(df.columns[y])
        parts = [(name, np.ones(n, bool))] if labels is None else \
            [(f"{name}: {lab}" if len(ys) > 1 else str(lab), codes == i) for i, lab in enumerate(labels)]
        for label, mask in parts:
            if kind == "scatter":
                px, py = E.occupied(xa[mask], ya[mask], pw, ph)
            else:
                px, py = E.m4(xa[mask], ya[mask], pw)
            if len(px):
                series.append({"kind": kind, "x": px.tolist(), "y": py.tolist(),
                               "label": label if (len(ys) > 1 or labels is not None) else None})
    if not series:
        return {"empty": "No finite values to plot"}
    allx = np.concatenate([np.asarray(sr["x"]) for sr in series])
    ally = np.concatenate([np.asarray(sr["y"]) for sr in series])
    spec = {"series": series, "xticks": E.nice_ticks(allx.min(), allx.max(), xk),
            "yticks": E.nice_ticks(ally.min(), ally.max(), "number", 6),
            "title": f"{', '.join(str(df.columns[y]) for y in ys)} vs {x_name}"}
    detail = ("every occupied cell is drawn" if kind == "scatter" else "min/max of every column of dots is kept")
    spec["note"] = f"{n:,} rows · {detail}" + ("" if not notes else " · " + "; ".join(notes))
    return spec


def _bars(df, xs, x_name, ys, agg, h):
    codes, names = E.factorize(xs)
    names = names + ["NaN"]  # code -1
    limit = max(5, min(60, h - 4))
    if agg == "count" or not ys:
        counts = np.bincount(codes + 1, minlength=len(names))  # missing values count too
        vals = pd.Series(np.r_[counts[1:], counts[:1]], index=names)
        vals = vals[vals > 0].sort_values(ascending=False)
        what = f"rows per {x_name}"
    else:
        y = df.iloc[:, ys[0]]
        ya, _ = E.as_axis(y)
        if ya is None:
            return {"empty": f"{df.columns[ys[0]]} isn't numeric"}
        vals = pd.Series(ya).groupby(codes).agg(agg).sort_values(ascending=False)
        vals.index = [names[c] for c in vals.index]
        what = f"{agg} of {df.columns[ys[0]]} by {x_name}"
    total = len(vals)
    vals = vals.head(limit)
    note = f"{len(df):,} rows · {total:,} distinct {x_name}" + (f" · top {limit} shown" if total > limit else "")
    labels = [E.short(k, 30) for k in vals.index][::-1]
    return {"bars": (labels, [float(v) for v in vals.to_numpy()][::-1]), "horizontal": True,
            "title": what, "note": note}
