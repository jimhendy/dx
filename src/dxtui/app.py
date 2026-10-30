"""dx: a terminal data explorer (Textual) on the web app's pandas engine.

    dx data.parquet                 # pick columns / filter, then explore
    dx big.csv -y                   # load straight away (sampled to the budget)
    dx data.h5 -k /prices -w "price > 100"
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from rich.markup import escape
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.suggester import Suggester
from textual.widgets import Footer, Input, Static, TabbedContent, TabPane

from . import engine as E
from .grid import INDEX, DataGrid
from .plot import PlotPane
from .screens import ColumnScreen, LoadScreen, OpenScreen, PickScreen, PromptScreen, RowScreen, StepsScreen

PICK_FIRST_BYTES = 64 * E.MB  # bigger files show the load options first (unless -y)
HISTORY_FILE = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "dx" / "history"
MAX_HISTORY = 1000


class ExprSuggester(Suggester):
    """Inline completion (→ accepts): earlier expressions, then column names
    after ``df["`` or ``df.``."""

    def __init__(self, app: "DxApp"):
        super().__init__(use_cache=False, case_sensitive=True)
        self.dx = app

    async def get_suggestion(self, value: str) -> str | None:
        for h in reversed(self.dx.history):
            if h.startswith(value) and h != value:
                return h
        df = E.STATE.df
        if df is None:
            return None
        m = re.search(r"""\[(["'])([^"']*)$""", value)
        if m:
            q, part = m.group(1), m.group(2)
            for c in map(str, df.columns):
                if c.startswith(part) and c != part:
                    return value + c[len(part):] + q + "]"
        m = re.search(r"\bdf\.(\w+)$", value)
        if m:
            part = m.group(1)
            for c in map(str, df.columns):
                if c.isidentifier() and c.startswith(part) and c != part:
                    return value + c[len(part):]
        return None


class DxApp(App):
    TITLE = "dx"
    CSS = """
    Screen { layout: vertical; }
    #status { height: 1; background: $panel; color: $text; padding: 0 1; }
    #status #file { width: 1fr; }
    #status #pos { width: auto; color: $text-muted; }
    TabbedContent { height: 1fr; }
    TabPane { padding: 0; }
    #stats-filter { dock: top; background: $panel; margin-bottom: 1; }
    #overview { padding: 1 2; }
    #output { height: auto; max-height: 40%; border-top: solid $panel-lighten-2; display: none; padding: 0 1; }
    #output.shown { display: block; }
    #output.error Static { color: $text-error; }
    #expr { display: none; border: none; height: 1; padding: 0 1; background: $panel; }
    #expr.shown { display: block; }
    #empty { content-align: center middle; height: 1fr; color: $text-muted; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("colon,e", "expr", "Expression"),
        Binding("o", "load_options", "Load options"),
        Binding("ctrl+o", "open", "Open file"),
        Binding("c", "column", "Column info"),
        Binding("p", "plot_column", "Plot column"),
        Binding("slash", "find_column", "Find column"),
        Binding("number_sign", "goto_row", "Go to row", show=False),
        Binding("S", "steps", "Steps"),
        Binding("R", "reset", "Reset", show=False),
        Binding("w", "save", "Save view", show=False),
        Binding("N", "count_rows", "Count rows", show=False),
        Binding("1", "tab('table')", "Table", show=False),
        Binding("2", "tab('columns')", "Columns", show=False),
        Binding("3", "tab('overview')", "Overview", show=False),
        Binding("4", "tab('plot')", "Plot", show=False),
        Binding("question_mark", "help", "Help"),
        Binding("escape", "dismiss_output", show=False),
    ]

    def __init__(self, path: str | None = None, opts: dict | None = None, pick: bool | None = None):
        super().__init__()
        self.start_path, self.start_opts, self.start_pick = path, opts or {}, pick
        self.load_opts: dict = {}
        self.busy = False
        self.history = _read_history()
        self.hist_pos: int | None = None

    # ------------------------------------------------------------- layout

    def compose(self) -> ComposeResult:
        with Horizontal(id="status"):
            yield Static("dx · no file", id="file")
            yield Static("", id="pos")
        with TabbedContent(initial="table"):
            with TabPane("Table [dim]1[/]", id="table"):
                yield DataGrid(id="grid")
            with TabPane("Columns [dim]2[/]", id="columns"):
                yield Input(placeholder="filter columns by name…  (Enter: details · s: sort by stat)",
                            id="stats-filter", compact=True)
                yield DataGrid(id="stats")
            with TabPane("Overview [dim]3[/]", id="overview-tab"):
                yield VerticalScroll(Static("", id="overview"), id="overview-scroll")
            with TabPane("Plot [dim]4[/]", id="plot"):
                yield PlotPane(id="plotpane")
        with VerticalScroll(id="output"):
            yield Static("", id="output-text", markup=False)
        yield Input(placeholder="pandas expression on df, e.g. df.describe() · df['x2'] = df.x * 2 · "
                                "→ accepts suggestion · ↑↓ history · Esc leaves",
                    id="expr", suggester=ExprSuggester(self))
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#grid", DataGrid).focus()
        if self.start_path:
            self.open_path(self.start_path, self.start_opts, self.start_pick)
        else:
            self.action_open()

    # ------------------------------------------------------------ loading

    def open_path(self, path: str, opts: dict | None = None, pick: bool | None = None) -> None:
        path = os.path.abspath(Path(path).expanduser())
        if not os.path.isfile(path):
            self.notify(f"No such file: {path}", severity="error", markup=False)
            return
        try:
            info = E.ex.inspect(path, os.path.basename(path))
        except E.UserError as e:
            self.notify(str(e), severity="error", markup=False)
            return
        except Exception as e:  # noqa: BLE001
            self.notify(f"Can't read {path}: {type(e).__name__}: {e}", severity="error", markup=False)
            return
        opts = dict(opts or {})
        several = len(info.get("keys") or []) > 1
        if pick is None:  # ask when there's a dataset / sheet to choose, or the file is big
            pick = several and "key" not in opts or bool(info.get("schema")) and info["size"] > PICK_FIRST_BYTES
        if pick and (info.get("schema") or info.get("keys")):
            def done(result):
                if result is not None:
                    self.load(path, result)
            self.push_screen(LoadScreen(path, info, opts), done)
        else:
            self.load(path, opts)

    def load(self, path: str, opts: dict) -> None:
        if self.busy:
            self.notify("Busy, try again in a moment", severity="warning")
            return
        self._set_busy(True, f"Opening {Path(path).name}…")
        self._load(path, opts)

    @work(thread=True, exclusive=True, group="engine")
    def _load(self, path: str, opts: dict) -> None:
        E.ex.set_progress(lambda msg: self.call_from_thread(self._progress, msg))
        t0 = time.time()
        try:
            E.ex.load(path, os.path.basename(path), key=opts.get("key"),
                      budget_mb=opts.get("budget_mb", E.DEFAULT_BUDGET_MB), index=opts.get("index", "auto"),
                      columns=opts.get("columns"), where=opts.get("where"), sql=opts.get("sql"))
        except (E.UserError, MemoryError, Exception) as e:  # noqa: BLE001
            msg = str(e) if isinstance(e, E.UserError) else f"{type(e).__name__}: {e}"
            self.call_from_thread(self._load_failed, path, msg)
            return
        self.call_from_thread(self._loaded, path, opts, time.time() - t0)

    def _load_failed(self, path: str, msg: str) -> None:
        self._set_busy(False)
        self._show_output(f"Couldn't load {path}:\n{msg}", error=True)
        if E.STATE.df is None:
            self.query_one("#file", Static).update(Text(f"dx · {Path(path).name} didn't load"))

    def _loaded(self, path: str, opts: dict, seconds: float) -> None:
        self.load_opts = dict(opts, path=path)
        self._set_busy(False)
        self._hide_output()
        self.sub_title = Path(path).name
        self.refresh_views(new=True)
        self._focus_main()
        self.notify(f"Loaded in {seconds:.1f}s", timeout=2)

    def _progress(self, msg: str) -> None:
        self.query_one("#file", Static).update(Text(f"⏳ {msg}"))

    def _set_busy(self, busy: bool, msg: str = "") -> None:
        self.busy = busy
        self.query_one(TabbedContent).loading = busy
        if busy:
            self._progress(msg)
        else:
            self._update_status()

    # -------------------------------------------------------------- views

    def refresh_views(self, new: bool = False) -> None:
        """The view (STATE.df) changed: redraw everything that shows it."""
        df = E.STATE.df
        self.query_one("#grid", DataGrid).set_frame(df, keep_position=not new)
        self._stats_shown_for = None
        if self.query_one(TabbedContent).active == "columns":
            self._show_stats()
        self._update_overview()
        self.query_one("#plotpane", PlotPane).set_frame(df)
        self._update_status()

    def _update_status(self) -> None:
        i, df = E.STATE.info, E.STATE.df
        if df is None:
            return
        rows = f"{len(df):,} rows"
        if E.STATE.df is E.STATE.df0:
            total = i.get("rows_total")
            if i.get("sampled") or i.get("filtered"):
                tot = f"{total:,}" if total is not None else "?"
                matched = i.get("rows_matched")
                rows = (f"{len(df):,} of {matched:,} matching" if i.get("filtered") and matched is not None
                        else f"{len(df):,} of {'~' if not i.get('rows_exact') else ''}{tot} rows")
                if i.get("sampled"):
                    rows += " [b $warning]sample[/]"
        bits = [f"[b]{escape(i['name'])}[/b]" + (f" [dim]{escape(i['key'])}[/dim]" if i.get("key") else ""),
                f"{i['format']} {E.human_bytes(i.get('size'))}", rows, f"{df.shape[1]:,} cols",
                E.human_bytes(E.ex.mem_bytes(df))]
        if E.STATE.df is not E.STATE.df0:
            bits.append(f"[b $accent]view: {len(E.STATE.steps)} step{'s' * (len(E.STATE.steps) != 1)}[/] (S)")
        self.query_one("#file", Static).update(" · ".join(bits))

    @on(DataGrid.CursorMoved, "#grid")
    def _cursor(self, event: DataGrid.CursorMoved) -> None:
        g = event.grid
        if g.df is None or not g.nrows:
            self.query_one("#pos", Static).update("")
            return
        name = g.column_name(event.col) if event.col != INDEX else (g.column_name(INDEX) or "index")
        colpos = "index" if event.col == INDEX else f"col {event.col + 1:,}/{g.ncols:,}"
        self.query_one("#pos", Static).update(
            Text(f"row {event.row + 1:,}/{g.nrows:,} · {colpos} {E.short(name, 30)}"))

    @on(TabbedContent.TabActivated)
    def _tab(self, event: TabbedContent.TabActivated) -> None:
        if not self.screen.query("#grid"):  # shutting down (or a dialog is open)
            return
        pane = event.pane.id
        if pane == "columns":
            self._show_stats()
        elif pane == "plot":
            self.query_one("#plotpane", PlotPane).draw()
        self._focus_tab(pane)

    # what gets focus on each tab: TabbedContent shows whichever pane holds the
    # focus, so focus left in a hidden pane would drag that pane back
    TAB_FOCUS = {"table": "#grid", "columns": "#stats", "overview-tab": "#overview-scroll", "plot": "#plot-controls"}

    def _focus_tab(self, pane: str) -> None:
        def go():
            tabs = self.screen.query(TabbedContent)
            if not tabs or tabs.first().active != pane:
                return
            focused = self.focused
            if focused is not None and any(a.id == pane for a in focused.ancestors_with_self):
                return  # already inside this tab (e.g. its filter box)
            target = self.screen.query(self.TAB_FOCUS[pane])
            if target:
                target.first().focus()
        self.call_after_refresh(go)

    def action_tab(self, tab: str) -> None:
        self.query_one(TabbedContent).active = "overview-tab" if tab == "overview" else tab

    # ----------------------------------------------------- column stats tab

    _stats_shown_for = None

    def _show_stats(self) -> None:
        df = E.STATE.df
        if df is None or self._stats_shown_for is df:
            return
        self._stats_shown_for = df
        grid = self.query_one("#stats", DataGrid)
        grid.loading = True
        self._compute_stats(df)

    @work(thread=True, exclusive=True, group="stats")
    def _compute_stats(self, df) -> None:
        try:
            frame = E.column_stats(df)
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self.notify, f"Column stats failed: {e}", severity="error", markup=False)
            return
        self.call_from_thread(self._stats_ready, df, frame)

    def _stats_ready(self, df, frame) -> None:
        if df is not E.STATE.df:
            return
        self._stats_frame = frame
        grid = self.query_one("#stats", DataGrid)
        grid.loading = False
        self._filter_stats(self.query_one("#stats-filter", Input).value)
        if self.query_one(TabbedContent).active == "columns" and not isinstance(self.focused, Input):
            grid.focus()  # a widget can't take focus while it shows the loading indicator

    @on(Input.Changed, "#stats-filter")
    def _stats_filter_changed(self, event: Input.Changed) -> None:
        self._filter_stats(event.value)

    def _filter_stats(self, text: str) -> None:
        frame = getattr(self, "_stats_frame", None)
        if frame is None:
            return
        text = text.strip().lower()
        if text:
            keep = frame["name"].str.lower().str.contains(text, regex=False).to_numpy()
            sub = frame[keep]
            sub.attrs["sort_keys"] = {k: v[keep].reset_index(drop=True) for k, v in frame.attrs["sort_keys"].items()}
            frame = sub
        self.query_one("#stats", DataGrid).set_frame(frame)

    @on(Input.Submitted, "#stats-filter")
    def _stats_filter_done(self) -> None:
        self.query_one("#stats", DataGrid).focus()

    # ----------------------------------------------------------- actions

    def _need_data(self) -> bool:
        if E.STATE.df is None:
            self.notify("Open a file first (ctrl+o)", severity="warning")
            return False
        if self.busy:
            self.notify("Busy, try again in a moment", severity="warning")
            return False
        return True

    def _current_column(self) -> int | None:
        """The column the user is pointing at (table cursor, or a row of the Columns tab)."""
        tab = self.query_one(TabbedContent).active
        if tab == "columns":
            g = self.query_one("#stats", DataGrid)
            if not g.nrows:
                return None
            return int(g.df.index[g.frame_row(g.cursor_row)])
        return self.query_one("#grid", DataGrid).cursor_col

    @on(DataGrid.Selected, "#grid")
    def _row(self, event: DataGrid.Selected) -> None:
        self.push_screen(RowScreen(E.STATE.df, event.row, event.col))

    @on(DataGrid.Selected, "#stats")
    def _stats_selected(self, event: DataGrid.Selected) -> None:
        self.action_column()

    def action_column(self) -> None:
        if not self._need_data():
            return
        col = self._current_column()
        if col is not None:
            self.push_screen(ColumnScreen(E.STATE.df, col))

    def action_plot_column(self) -> None:
        if not self._need_data():
            return
        col = self._current_column()
        if col is None or col == INDEX:
            return
        df = E.STATE.df
        s = df.iloc[:, col]
        pane = self.query_one("#plotpane", PlotPane)
        numeric = E.as_axis(s)[0] is not None and not pd.api.types.is_bool_dtype(s.dtype)
        if numeric and (E.is_dates(df.index) or E.is_numeric(df.index.dtype)):
            kind = "line" if df.index.is_monotonic_increasing or E.is_dates(df.index) else "scatter"
            pane.configure(kind, y=col)
        elif numeric:
            pane.configure("hist", y=col)
        else:
            pane.query_one("#agg").value = "count"
            pane.configure("bar", x=col)
        self.action_tab("plot")

    def action_find_column(self) -> None:
        if not self._need_data():
            return
        df = E.STATE.df
        names = [str(c) for c in df.columns]

        def go(i):
            if i is None:
                return
            if self.query_one(TabbedContent).active == "columns":
                g = self.query_one("#stats", DataGrid)
                rows = list(g.df.index)
                if i in rows:
                    g.move_to(row=rows.index(i))
                return
            self.action_tab("table")
            self.query_one("#grid", DataGrid).move_to(col=i)

        self.push_screen(PickScreen("Go to column", names, [str(t) for t in df.dtypes]), go)

    def action_goto_row(self) -> None:
        if not self._need_data():
            return

        def go(v):
            if not v:
                return
            try:
                n = int(v.replace(",", "").replace("_", ""))
            except ValueError:
                self.notify("Type a row number", severity="error")
                return
            self.action_tab("table")
            g = self.query_one("#grid", DataGrid)
            g.move_to(row=(n - 1) if n > 0 else g.nrows + n)

        self.push_screen(PromptScreen(f"Go to row (1–{len(E.STATE.df):,}, negative counts from the end)"), go)

    def action_open(self) -> None:
        start = os.path.dirname(E.STATE.path) if E.STATE.path else os.getcwd()

        def done(path):
            if path:
                self.open_path(path)
            elif E.STATE.df is None:
                self.query_one("#file", Static).update("dx · no file · ctrl+o opens one")

        self.push_screen(OpenScreen(start), done)

    def action_load_options(self) -> None:
        if not self.load_opts:
            self.action_open()
            return
        path = self.load_opts["path"]
        info = E.ex.inspect(path, os.path.basename(path))

        def done(result):
            if result is not None:
                self.load(path, result)

        self.push_screen(LoadScreen(path, info, self.load_opts), done)

    def action_steps(self) -> None:
        if E.STATE.df is None:
            return

        def done(r):
            if r == "reset":
                self.action_reset()

        self.push_screen(StepsScreen(), done)

    def action_reset(self) -> None:
        if not self._need_data() or E.STATE.df is E.STATE.df0 and not E.STATE.steps:
            return
        E.ex.reset()
        self._hide_output()
        self.refresh_views()
        self.notify("Back to the loaded data")

    def action_save(self) -> None:
        if not self._need_data():
            return
        stem = Path(E.STATE.name).stem

        def done(path):
            if path:
                self._save(E.STATE.df, path)

        self.push_screen(PromptScreen("Save the current view as (.parquet .csv .tsv .json .jsonl .xlsx .h5 .pkl)",
                                      f"{stem}_view.parquet"), done)

    @work(thread=True, group="save")
    def _save(self, df, path) -> None:
        try:
            out = E.save_view(df, path)
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self.notify, f"Couldn't save: {e}", severity="error", markup=False)
            return
        self.call_from_thread(self.notify, f"Saved {len(df):,} rows to {out}", markup=False)

    def action_count_rows(self) -> None:
        if not self._need_data():
            return
        if E.STATE.format not in ("csv", "jsonl") or E.STATE.info.get("rows_exact"):
            self.notify("The row count is already exact")
            return
        self._set_busy(True, "Counting rows…")
        self._count()

    @work(thread=True, group="engine")
    def _count(self) -> None:
        try:
            E.ex.scan_rows()
        finally:
            self.call_from_thread(self._counted)

    def _counted(self) -> None:
        self._set_busy(False)
        self._update_overview()

    def action_help(self) -> None:
        if self.screen.query("HelpPanel"):
            self.action_hide_help_panel()
        else:
            self.action_show_help_panel()

    # --------------------------------------------------------- expressions

    def action_expr(self) -> None:
        if E.STATE.df is None:
            return
        box = self.query_one("#expr", Input)
        box.add_class("shown")
        box.focus()
        self.hist_pos = None

    @on(Input.Submitted, "#expr")
    def _run_expr(self, event: Input.Submitted) -> None:
        expr = event.value.strip()
        if not expr or not self._need_data():
            return
        if not self.history or self.history[-1] != expr:
            self.history.append(expr)
            _write_history(self.history)
        self.hist_pos = None
        event.input.value = ""
        self._set_busy(True, f"Running {E.short(expr, 60)}…")
        self._evaluate(expr)

    @work(thread=True, exclusive=True, group="engine")
    def _evaluate(self, expr: str) -> None:
        try:
            r = E.ex.evaluate(expr)
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self.call_from_thread(self._evaluated, r)

    def _evaluated(self, r: dict) -> None:
        self._set_busy(False)
        out = r.get("stdout") or ""
        if not r["ok"]:
            self._show_output((out + "\n" if out else "") + r["error"], error=True)
            return
        if r["kind"] == "frame":
            self.refresh_views()
            msg = r.get("message") or f"View: {len(E.STATE.df):,} rows × {E.STATE.df.shape[1]:,} columns"
            if out:
                self._show_output(out + "\n" + msg)
            else:
                self.notify(msg, markup=False)
                self._hide_output()
            if self.query_one(TabbedContent).active in ("overview-tab",):
                self.action_tab("table")
        else:
            text = r.get("text")
            if text:  # numpy 2 scalars repr as np.float64(1.5): show the value
                text = re.sub(r"^np\.(?:float|int|uint|bool|complex|str)\w*\((.*)\)$", r"\1", text, flags=re.S)
            body = (out + ("\n" if out and text else "") + (text or "")) or "(no output)"
            self._show_output(body)
            self._update_status()

    def _show_output(self, text: str, error: bool = False) -> None:
        panel = self.query_one("#output", VerticalScroll)
        self.query_one("#output-text", Static).update(text)
        panel.set_class(error, "error")
        panel.add_class("shown")
        panel.scroll_home(animate=False)

    def _hide_output(self) -> None:
        self.query_one("#output").remove_class("shown")

    def on_key(self, event) -> None:
        box = self.query_one("#expr", Input)
        if not box.has_focus:
            return
        if event.key in ("up", "down") and self.history:
            n = len(self.history)
            pos = self.hist_pos if self.hist_pos is not None else n
            pos = max(0, pos - 1) if event.key == "up" else pos + 1
            if pos >= n:
                self.hist_pos, box.value = None, ""
            else:
                self.hist_pos, box.value = pos, self.history[pos]
            box.cursor_position = len(box.value)
            event.stop()
            event.prevent_default()
        elif event.key == "escape":
            box.remove_class("shown")
            self._focus_main()
            event.stop()
            event.prevent_default()

    def action_dismiss_output(self) -> None:
        self._hide_output()

    def _focus_main(self) -> None:
        self._focus_tab(self.query_one(TabbedContent).active)

    # ------------------------------------------------------------ overview

    def _update_overview(self) -> None:
        if E.STATE.df is None:
            return
        o = E.ex.overall()
        i = o["info"]
        t = Table.grid(padding=(0, 3))
        t.add_column(style="bold", no_wrap=True)
        t.add_column()
        t.add_row("File", escape(str(E.STATE.path)) + (f"  [dim]key {escape(i['key'])}[/dim]" if i.get("key") else ""))
        t.add_row("Format", f"{i['format']} · {E.human_bytes(i.get('size'))} on disk")
        total = i.get("rows_total")
        if total is not None:
            t.add_row("Rows in file", f"{total:,}" + ("" if i.get("rows_exact") else
                                                      "  [dim](estimated · N counts exactly)[/dim]"))
        elif E.STATE.format in ("csv", "jsonl"):
            t.add_row("Rows in file", "[dim]unknown · N counts exactly[/dim]")
        t.add_row("Loaded", f"{i['rows_loaded']:,} rows × {i['cols']:,} columns in {i['load_seconds']}s "
                            f"[dim](budget {i['budget_mb']} MB)[/dim]")
        if i.get("sampled"):
            t.add_row(Text("Sample", style="bold yellow"), Text(i.get("sample_method", "a sample, to fit the memory budget")))
        sel = i.get("selection")
        if sel:
            if sel.get("sql"):
                t.add_row("SQL", Text(sel["sql"]))
            if sel.get("columns") is not None:
                t.add_row("Columns", f"{sel['columns']:,} of {sel['of'] or '?'} chosen")
            if sel.get("where"):
                t.add_row("Rows where", escape(sel["where"]) + (f"  [dim]{i['rows_matched']:,} matched[/dim]"
                                                        if i.get("rows_matched") is not None else ""))
        t.add_row("", "")
        shape = o["shape"]
        t.add_row("Current view", f"{shape[0]:,} rows × {shape[1]:,} columns · {E.human_bytes(o['memory_bytes'])}"
                  + ("" if o["is_original"] else f"  [bold]{len(E.STATE.steps)} steps applied (S)[/]"))
        nf = o.get("null_frac")
        t.add_row("Missing", f"{o['total_nulls']:,} cells" + (f" ({100 * nf:.2f}%)" if nf else ""))
        t.add_row("dtypes", ", ".join(f"{d['dtype']} × {d['count']}" for d in o["dtypes"]))
        ix = o["index"]
        desc = [ix["dtype"]]
        if ix.get("name"):
            desc.insert(0, f"[b]{escape(ix['name'])}[/b]")
        if ix.get("min") is not None:
            desc.append(escape(f"{ix['min']} → {ix['max']}"))
        if ix.get("freq"):
            desc.append(f"freq {ix['freq']}")
        desc.append(ix["monotonic"] or "unordered")
        desc.append("unique" if ix["unique"] else "[yellow]has duplicates[/]")
        t.add_row("Index", " · ".join(desc))
        self.query_one("#overview", Static).update(t)


# ---------------------------------------------------------------------------

def _read_history() -> list[str]:
    try:
        return [line.replace("\\n", "\n") for line in HISTORY_FILE.read_text().splitlines() if line][-MAX_HISTORY:]
    except OSError:
        return []


def _write_history(hist: list[str]) -> None:
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text("\n".join(h.replace("\n", "\\n") for h in hist[-MAX_HISTORY:]) + "\n")
    except OSError:
        pass


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="dx", description="Explore a data file in the terminal "
                                "(parquet, csv/tsv, json, jsonl, h5, xlsx).")
    p.add_argument("file", nargs="?", help="file to open (omit to browse)")
    p.add_argument("-k", "--key", help="HDF5 dataset / Excel sheet")
    p.add_argument("-c", "--columns", help="comma-separated columns to load (parquet, csv, jsonl)")
    p.add_argument("-w", "--where", help="SQL WHERE condition rows must match (parquet, csv, jsonl)")
    p.add_argument("-q", "--sql", help="SQL query over the file as table `data` (parquet, csv, jsonl)")
    p.add_argument("-b", "--budget", type=int, default=E.DEFAULT_BUDGET_MB,
                   help=f"memory budget in MB; bigger files are sampled (default {E.DEFAULT_BUDGET_MB})")
    p.add_argument("-i", "--index", choices=["auto", "first", "none"], default="auto")
    g = p.add_mutually_exclusive_group()
    g.add_argument("-y", "--yes", action="store_true", help="load straight away, without the load options")
    g.add_argument("-p", "--pick", action="store_true", help="always show the load options first")
    a = p.parse_args(argv)
    opts = {"budget_mb": a.budget, "index": a.index}
    if a.key:
        opts["key"] = a.key
    if a.columns:
        opts["columns"] = [c.strip() for c in a.columns.split(",") if c.strip()]
    if a.where:
        opts["where"] = a.where
    if a.sql:
        opts["sql"] = a.sql
    explicit = any(k in opts for k in ("key", "columns", "where", "sql"))
    pick = True if a.pick else False if (a.yes or explicit) else None
    if a.file and not os.path.isfile(a.file):
        sys.exit(f"dx: no such file: {a.file}")
    DxApp(a.file, opts, pick).run()


if __name__ == "__main__":
    main()
