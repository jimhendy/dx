"""Modal screens: load options, column details, a row's values, steps, file
picker and small prompts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rich.markup import escape
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (Button, DirectoryTree, Input, Label, OptionList, Select, SelectionList, Static,
                             TextArea)
from textual.widgets.option_list import Option
from textual_plotext import PlotextPlot

from . import engine as E
from .grid import INDEX, DataGrid


class Dialog(ModalScreen):
    """A centred panel; Esc closes it."""

    BINDINGS = [Binding("escape", "close", "Close")]
    DEFAULT_CSS = """
    Dialog { align: center middle; }
    Dialog > #panel {
        width: 90%; height: 90%; max-width: 160;
        border: round $accent; background: $surface; padding: 0 1;
    }
    Dialog .title { text-style: bold; color: $text-accent; width: 1fr; }
    Dialog .muted { color: $text-muted; }
    Dialog .buttons { height: auto; align-horizontal: right; margin-top: 1; }
    Dialog .buttons Button { margin-left: 2; }
    Dialog .hint { width: 1fr; }
    Dialog Input.-textual-compact, Dialog Select.-textual-compact > SelectCurrent { background: $panel; }
    Dialog Input.-textual-compact:focus { background: $panel-lighten-1; }
    """

    def action_close(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# Load options
# ---------------------------------------------------------------------------

class LoadScreen(Dialog):
    """Choose what to load before anything is read: sheet / key, columns, a
    row filter or an SQL query, index and memory budget."""

    BINDINGS = [Binding("escape", "close", "Cancel"), Binding("ctrl+l", "load", "Load", priority=True)]
    DEFAULT_CSS = """
    LoadScreen #cols { height: 1fr; min-height: 4; background: $panel; }
    LoadScreen .row { height: auto; }
    LoadScreen .row Button { min-width: 6; margin-left: 2; }
    LoadScreen .row Input#colfilter { width: 1fr; }
    LoadScreen Label { margin-top: 1; }
    LoadScreen #opts > Vertical { width: 1fr; height: auto; }
    LoadScreen #opts { height: auto; }
    """

    def __init__(self, path: str, info: dict, current: dict | None = None):
        super().__init__()
        self.path, self.info = path, info
        self.cur = current or {}
        schema = info.get("schema") or {}
        self.columns = [c for c in schema.get("columns", []) if not c.get("index")]
        self.index_cols = [c["name"] for c in schema.get("columns", []) if c.get("index")]
        chosen = self.cur.get("columns")
        self.chosen = set(c["name"] for c in self.columns) if chosen is None else set(chosen)

    def compose(self) -> ComposeResult:
        i = self.info
        schema = i.get("schema") or {}
        bits = [i["format"], E.human_bytes(i["size"])]
        if schema.get("rows") is not None:
            bits.append(f"{schema['rows']:,} rows")
        if schema:
            bits.append(f"{schema['total_cols']:,} columns")
        with Vertical(id="panel"):
            yield Static(f"Load [b]{escape(Path(self.path).name)}[/b]   [dim]{' · '.join(bits)}[/dim]", classes="title")
            if i.get("keys"):
                yield Label("Dataset / sheet")
                opts = [(Text(f"{k['key']}  ({k['kind']}, {k['rows']:,} rows)" if k.get("rows") is not None
                              else k["key"]), k["key"]) for k in i["keys"]]
                yield Select(opts, value=self.cur.get("key") or i["keys"][0]["key"], allow_blank=False, id="key",
                             compact=True)
            if self.columns:
                yield Label(self._cols_label(), id="cols-label")
                if self.index_cols:
                    yield Static(Text(f"Always loaded as the index: {', '.join(self.index_cols)}", style="dim"))
                with Horizontal(classes="row"):
                    yield Input(placeholder="search columns…", id="colfilter", compact=True)
                    yield Button("All", id="all", compact=True)
                    yield Button("None", id="none", compact=True)
                    yield Button("Only these", id="only", compact=True)
                yield SelectionList(id="cols", compact=True)
                yield Label("Filter rows: SQL WHERE condition, e.g. region = 'south' AND price > 100")
                yield Input(self.cur.get("where") or "", placeholder="(all rows)", id="where", compact=True)
                yield Label("…or an SQL query (DuckDB; the file is the table data). Replaces the two above.")
                yield Input(self.cur.get("sql") or "", placeholder="SELECT … FROM data …", id="sql", compact=True)
            with Horizontal(id="opts"):
                with Vertical():
                    yield Label("Index")
                    yield Select([("Auto (stored index, dates, saved index)", "auto"), ("First column", "first"),
                                  ("Row numbers", "none")], value=self.cur.get("index", "auto"),
                                 allow_blank=False, id="index", compact=True)
                with Vertical():
                    yield Label("Memory budget (MB): larger files are sampled to fit")
                    yield Input(str(self.cur.get("budget_mb", E.DEFAULT_BUDGET_MB)), type="integer", id="budget",
                                compact=True)
            with Horizontal(classes="buttons"):
                yield Static("Enter in a field or ctrl+l loads · Esc cancels", classes="muted hint")
                yield Button("Cancel", id="cancel", compact=True)
                yield Button("Load", variant="primary", id="load", compact=True)

    def on_mount(self) -> None:
        if self.columns:
            self._fill()
        self.query_one("#load", Button).focus()

    def _cols_label(self) -> str:
        return f"Columns: {len(self.chosen):,} of {len(self.columns):,} chosen  (space ticks)"

    def _matches(self) -> list[str]:
        f = self.query_one("#colfilter", Input).value.strip().lower() if self.is_mounted else ""
        return [c["name"] for c in self.columns if f in c["name"].lower()]

    def _fill(self) -> None:
        sl = self.query_one("#cols", SelectionList)
        sl.clear_options()
        types = {c["name"]: c["dtype"] for c in self.columns}
        names = self._matches()[:2000]
        sl.add_options([(Text.assemble(n, ("  " + types[n], "dim")), n, n in self.chosen) for n in names])
        self.query_one("#cols-label", Label).update(self._cols_label())

    @on(Input.Changed, "#colfilter")
    def _filter(self) -> None:
        self._fill()

    @on(SelectionList.SelectionToggled, "#cols")
    def _toggled(self, event: SelectionList.SelectionToggled) -> None:
        name = event.selection.value
        self.chosen.symmetric_difference_update({name})
        self.query_one("#cols-label", Label).update(self._cols_label())

    @on(Button.Pressed, "#all")
    def _all(self) -> None:
        self.chosen |= set(self._matches())
        self._fill()

    @on(Button.Pressed, "#none")
    def _none(self) -> None:
        self.chosen -= set(self._matches())
        self._fill()

    @on(Button.Pressed, "#only")
    def _only(self) -> None:
        self.chosen = set(self._matches())
        self._fill()

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#load")
    @on(Input.Submitted, "#where,#sql,#budget")
    def action_load(self) -> None:
        q = self.query
        opts = {"index": q("#index").first(Select).value}
        try:
            opts["budget_mb"] = max(1, int(q("#budget").first(Input).value or E.DEFAULT_BUDGET_MB))
        except ValueError:
            self.notify("Memory budget must be a number of MB", severity="error")
            return
        if self.info.get("keys"):
            opts["key"] = q("#key").first(Select).value
        if self.columns:
            names = [c["name"] for c in self.columns]
            if not self.chosen:
                self.notify("Choose at least one column", severity="error")
                return
            opts["columns"] = None if len(self.chosen) == len(names) else [n for n in names if n in self.chosen]
            opts["where"] = q("#where").first(Input).value.strip() or None
            opts["sql"] = q("#sql").first(Input).value.strip() or None
        self.dismiss(opts)


# ---------------------------------------------------------------------------
# Column details
# ---------------------------------------------------------------------------

class ColumnScreen(Dialog):
    DEFAULT_CSS = """
    ColumnScreen #body { height: 1fr; }
    ColumnScreen #stats { width: 44; height: 100%; }
    ColumnScreen #right { width: 1fr; }
    ColumnScreen #top { height: 1fr; }
    ColumnScreen #hist { height: 1fr; min-height: 8; }
    """

    def __init__(self, df: pd.DataFrame, col: int):
        super().__init__()
        self.df, self.col = df, col
        self.series = df.index.to_series() if col == INDEX else df.iloc[:, col]
        raw = df.index.name if col == INDEX else df.columns[col]
        self.name_ = "index" if raw is None else str(raw)

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            yield Static(f"Column [b]{escape(self.name_)}[/b]  [dim]{self.series.dtype} · Esc closes[/dim]", classes="title")
            with Horizontal(id="body"):
                yield VerticalScroll(Static("Computing…", id="stats-text"), id="stats")
                with Vertical(id="right"):
                    yield Label("Most common values", classes="muted")
                    yield DataGrid(id="top", max_width=60)
                    yield PlotextPlot(id="hist")

    def on_mount(self) -> None:
        self.query_one("#top", DataGrid).loading = True
        self._compute()

    @work(thread=True)
    def _compute(self) -> None:
        detail = E.column_detail(self.series)
        a, kind = E.as_axis(self.series)
        hist = None
        if a is not None and pd.api.types.is_bool_dtype(self.series.dtype) is False:
            fin = a[np.isfinite(a)]
            if len(fin):
                lo, hi = fin.min(), fin.max()
                counts, edges = np.histogram(fin, bins=40, range=(lo, hi) if hi > lo else (lo - .5, hi + .5))
                hist = ((edges[:-1] + edges[1:]) / 2, counts, kind, edges)
        self.app.call_from_thread(self._show, detail, hist)

    def _show(self, detail: dict, hist) -> None:
        st = detail["stats"]
        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim")
        t.add_column()
        n = st["count"] + st["nulls"]
        for k, v in [("rows", f"{n:,}"), ("non-null", f"{st['count']:,}"),
                     ("nulls", f"{st['nulls']:,}" + (f"  ({100 * st['null_frac']:.2f}%)" if st.get("null_frac") else "")),
                     ("unique", f"{st['unique']:,}" if st.get("unique") is not None else "–"),
                     ("min", st.get("min")), ("max", st.get("max")), ("mean", st.get("mean")), ("std", st.get("std")),
                     ("at min", st.get("idxmin")), ("at max", st.get("idxmax"))]:
            if v is not None:
                t.add_row(k, Text(str(v)))
        if detail["quantiles"]:
            t.add_row("", "")
            for q, v in detail["quantiles"]:
                t.add_row(q, Text(str(v)))
        self.query_one("#stats-text", Static).update(t)
        top = self.query_one("#top", DataGrid)
        top.loading = False
        top.set_frame(detail["top"])
        plot = self.query_one("#hist", PlotextPlot)
        plt = plot.plt
        plt.clear_figure()
        if hist is not None:
            centres, counts, kind, edges = hist
            plt.bar(centres.tolist(), counts.tolist(), width=1)
            ticks = E.nice_ticks(edges[0], edges[-1], kind)
            if ticks:
                plt.xticks(*ticks)
            plt.yticks(*E.nice_ticks(0, max(1, int(counts.max())), "number", 5))
            plt.title("Distribution")
        else:
            vc = detail["top"].head(15)
            plt.bar([E.short(v, 20) for v in vc["value"]][::-1], vc["count"].tolist()[::-1], orientation="h",
                    width=0.6)
            plt.title("Most common values")
        plot.refresh()


# ---------------------------------------------------------------------------
# One row, as a list of fields
# ---------------------------------------------------------------------------

class RowScreen(Dialog):
    DEFAULT_CSS = """
    RowScreen #fields { height: 1fr; }
    """

    def __init__(self, df: pd.DataFrame, pos: int, col: int):
        super().__init__()
        self.df, self.pos, self.col = df, pos, col
        self.label = E.ex._fmt_values(df.index[[pos]])[0]

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            yield Static(f"Row [b]{escape(str(self.label))}[/b]  [dim]position {self.pos:,} of {len(self.df):,} · "
                         f"/ filters fields · Esc closes[/dim]", classes="title")
            yield Input(placeholder="filter fields…", id="filter", compact=True)
            yield DataGrid(id="fields", max_width=200)

    def on_mount(self) -> None:
        df = self.df
        row = df.iloc[self.pos]
        values = [_full(v) for v in row.tolist()]
        self.frame = pd.DataFrame({"value": values, "dtype": [str(t) for t in df.dtypes]},
                                  index=pd.Index([str(c) for c in df.columns], name="field"))
        self._show("")
        grid = self.query_one("#fields", DataGrid)
        if self.col >= 0:
            grid.move_to(row=self.col, col=0)
        grid.focus()

    def _show(self, flt: str) -> None:
        f = self.frame
        if flt:
            f = f[f.index.str.lower().str.contains(flt.lower(), regex=False)]
        self.query_one("#fields", DataGrid).set_frame(f)

    @on(Input.Changed, "#filter")
    def _filter(self, event: Input.Changed) -> None:
        self._show(event.value.strip())

    def key_slash(self) -> None:
        self.query_one("#filter", Input).focus()


def _full(v) -> str:
    if isinstance(v, float) and v != v:
        return "NaN"
    if v is None:
        return "None"
    s = v if isinstance(v, str) else repr(v) if isinstance(v, bytes) else str(v)
    return s if len(s) <= 5000 else s[:5000] + "…"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

class StepsScreen(Dialog):
    DEFAULT_CSS = """
    StepsScreen #script { height: 1fr; }
    """
    BINDINGS = [Binding("escape", "close", "Close"), Binding("y", "copy", "Copy as Python")]

    def compose(self) -> ComposeResult:
        n = len(E.STATE.steps)
        with Vertical(id="panel"):
            yield Static(f"Steps  [dim]{n} applied since loading · the script below reproduces the view[/dim]",
                         classes="title")
            yield TextArea(E.steps_script(), language="python", read_only=True, id="script")
            with Horizontal(classes="buttons"):
                yield Button("Copy as Python (y)", id="copy", compact=True)
                yield Button("Reset to original (R)", id="reset", variant="warning", disabled=not n, compact=True)
                yield Button("Close", id="close", variant="primary", compact=True)

    def action_copy(self) -> None:
        self.app.copy_to_clipboard(E.steps_script())
        self.notify("Copied (via your terminal's clipboard support)")

    @on(Button.Pressed, "#copy")
    def _copy(self) -> None:
        self.action_copy()

    @on(Button.Pressed, "#reset")
    def _reset(self) -> None:
        self.dismiss("reset")

    @on(Button.Pressed, "#close")
    def _close(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# File picker and prompts
# ---------------------------------------------------------------------------

class DataTree(DirectoryTree):
    def filter_paths(self, paths):
        return [p for p in paths if not p.name.startswith(".") and
                (p.is_dir() or p.suffix.lower() in E.FORMATS_BY_EXT)]


class OpenScreen(Dialog):
    DEFAULT_CSS = """
    OpenScreen #tree { height: 1fr; }
    """

    def __init__(self, start: str | None = None):
        super().__init__()
        self.start = Path(start or ".").expanduser().resolve()

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            yield Static(f"Open a data file  [dim]{', '.join(sorted(set(E.FORMATS_BY_EXT.values())))} · "
                         f"Enter opens · Backspace goes up · Esc cancels[/dim]", classes="title")
            yield Input(str(self.start), id="dir", compact=True)
            yield DataTree(str(self.start), id="tree")

    def on_mount(self) -> None:
        self.query_one("#tree").focus()

    @on(DirectoryTree.FileSelected)
    def _picked(self, event: DirectoryTree.FileSelected) -> None:
        self.dismiss(str(event.path))

    @on(Input.Submitted, "#dir")
    def _goto(self, event: Input.Submitted) -> None:
        p = Path(event.value).expanduser()
        if p.is_file():
            self.dismiss(str(p))
        elif p.is_dir():
            self._set_root(p)
        else:
            self.notify(f"No such file or folder: {p}", severity="error", markup=False)

    def _set_root(self, p: Path) -> None:
        self.start = p.resolve()
        tree = self.query_one("#tree", DataTree)
        tree.path = str(self.start)
        self.query_one("#dir", Input).value = str(self.start)
        tree.focus()

    def key_backspace(self) -> None:
        if self.query_one("#tree").has_focus:
            self._set_root(self.start.parent)


class PromptScreen(Dialog):
    """Ask for one line of text."""

    DEFAULT_CSS = """
    PromptScreen > #panel { height: auto; width: 80; }
    """

    def __init__(self, title: str, value: str = "", placeholder: str = ""):
        super().__init__()
        self.title_, self.value, self.placeholder = title, value, placeholder

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            yield Static(Text(self.title_), classes="title")
            yield Input(self.value, placeholder=self.placeholder, id="answer", compact=True)

    @on(Input.Submitted)
    def _done(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)


class PickScreen(Dialog):
    """Type to filter a list (columns), Enter picks."""

    DEFAULT_CSS = """
    PickScreen > #panel { width: 80; height: 70%; }
    PickScreen OptionList { height: 1fr; }
    """
    BINDINGS = [Binding("escape", "close", "Close"), Binding("down", "down", show=False),
                Binding("up", "up", show=False)]

    def __init__(self, title: str, names: list[str], extra: list[str] | None = None):
        super().__init__()
        self.title_, self.names = title, names
        self.extra = extra or [""] * len(names)

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            yield Static(Text(self.title_), classes="title")
            yield Input(placeholder="type to filter…", id="q", compact=True)
            yield OptionList(id="opts")

    def on_mount(self) -> None:
        self._fill("")

    def _fill(self, q: str) -> None:
        ol = self.query_one(OptionList)
        ol.clear_options()
        q = q.lower()
        hits = [i for i, n in enumerate(self.names) if q in n.lower()]
        # names starting with the text first
        hits.sort(key=lambda i: not self.names[i].lower().startswith(q))
        ol.add_options([Option(Text.assemble(self.names[i], ("  " + self.extra[i], "dim")), id=str(i))
                        for i in hits[:1000]])
        if hits:
            ol.highlighted = 0

    @on(Input.Changed, "#q")
    def _changed(self, event: Input.Changed) -> None:
        self._fill(event.value)

    def action_down(self) -> None:
        self.query_one(OptionList).action_cursor_down()

    def action_up(self) -> None:
        self.query_one(OptionList).action_cursor_up()

    @on(Input.Submitted, "#q")
    def _submit(self) -> None:
        ol = self.query_one(OptionList)
        if ol.highlighted is not None:
            self.dismiss(int(ol.get_option_at_index(ol.highlighted).id))

    @on(OptionList.OptionSelected)
    def _selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(int(event.option.id))
