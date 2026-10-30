"""A virtual DataFrame grid: only the cells on screen are ever formatted, so
millions of rows and thousands of columns scroll instantly.

Rows scroll with the scrollbar / mouse wheel; columns scroll one column at a
time with the cursor. The index stays pinned at the left and the header
(column names and dtypes) at the top.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rich.cells import cell_len, set_cell_size
from rich.segment import Segment
from rich.style import Style
from textual import events, work
from textual.binding import Binding
from textual.cache import LRUCache
from textual.geometry import Size
from textual.message import Message
from textual.scroll_view import ScrollView
from textual.strip import Strip

from . import engine as E

BLOCK = 64          # rows formatted together
HEADER = 2          # name line + dtype line
INDEX = -1          # column number of the index


class DataGrid(ScrollView, can_focus=True):
    """Shows a DataFrame. ``sort_keys`` (in ``df.attrs``) can give columns
    alternative values to sort by (e.g. numeric min/max of a text column)."""

    COMPONENT_CLASSES = {"grid--header", "grid--dtype", "grid--index", "grid--cursor", "grid--cursor-row",
                         "grid--cursor-col", "grid--null", "grid--sep"}

    DEFAULT_CSS = """
    DataGrid {
        overflow-x: hidden;
        overflow-y: auto;
        scrollbar-size-vertical: 1;
        background: $surface;
    }
    DataGrid > .grid--header { text-style: bold; color: $text; background: $panel; }
    DataGrid > .grid--dtype { color: $text-muted; background: $panel; }
    DataGrid > .grid--index { color: $text-accent; text-style: bold; }
    DataGrid > .grid--cursor { background: $accent; color: $background; text-style: bold; }
    DataGrid:blur > .grid--cursor { background: $accent-darken-3; color: $text; }
    DataGrid > .grid--cursor-row { background: $panel; }
    DataGrid > .grid--cursor-col { color: $text-accent; text-style: bold underline; }
    DataGrid > .grid--null { color: $text-disabled; text-style: italic; }
    DataGrid > .grid--sep { color: $panel-lighten-2; }
    """

    BINDINGS = [
        Binding("up,k", "move(-1, 0)", "Up", show=False),
        Binding("down,j", "move(1, 0)", "Down", show=False),
        Binding("left,h", "move(0, -1)", "Left", show=False),
        Binding("right,l", "move(0, 1)", "Right", show=False),
        Binding("pageup,ctrl+b", "page(-1)", "Page up", show=False),
        Binding("pagedown,ctrl+f,space", "page(1)", "Page down", show=False),
        Binding("g,ctrl+home", "edge_row(0)", "Top", show=False),
        Binding("G,ctrl+end", "edge_row(1)", "Bottom", show=False),
        Binding("home,0", "edge_col(0)", "First column", show=False),
        Binding("end,dollar_sign", "edge_col(1)", "Last column", show=False),
        Binding("ctrl+left,H", "move(0, -10)", "10 columns left", show=False),
        Binding("ctrl+right,L", "move(0, 10)", "10 columns right", show=False),
        Binding("s", "sort", "Sort"),
        Binding("less_than_sign", "resize(-4)", "Narrower", show=False),
        Binding("greater_than_sign", "resize(4)", "Wider", show=False),
    ]

    class CursorMoved(Message):
        def __init__(self, grid: "DataGrid", row: int, col: int) -> None:
            super().__init__()
            self.grid, self.row, self.col = grid, row, col

        @property
        def control(self):
            return self.grid

    class Selected(Message):
        """Enter on a cell: ``row`` is the row's position in the frame (after sorting is undone)."""
        def __init__(self, grid: "DataGrid", row: int, col: int) -> None:
            super().__init__()
            self.grid, self.row, self.col = grid, row, col

        @property
        def control(self):
            return self.grid

    def __init__(self, *, max_width: int = E.MAX_COL_WIDTH, id: str | None = None, classes: str | None = None):
        super().__init__(id=id, classes=classes)
        self.max_width = max_width
        self.df: pd.DataFrame | None = None
        self.order: np.ndarray | None = None
        self.sort_col: int | None = None
        self.sort_asc = True
        self.cursor_row = 0
        self.cursor_col = 0
        self.col_offset = 0
        self.show_index = True
        self._reset_caches()
        self._styles: dict[str, Style] = {}
        self.busy = False

    # ------------------------------------------------------------------ data

    def _reset_caches(self):
        self._fmts: dict[int, E.ColumnFormat] = {}
        self._widths: dict[int, int] = {}
        self._cols: dict[int, object] = {}
        self._cells: LRUCache = LRUCache(4096)
        self._names: dict[int, str] = {}

    def set_frame(self, df: pd.DataFrame | None, keep_position: bool = False) -> None:
        """Show ``df`` (keeps the cursor where it was if asked and still valid)."""
        same_shape = self.df is not None and df is not None and keep_position
        same_cols = same_shape and self.df.columns.equals(df.columns)
        self.df = df
        self.order = None
        self.sort_col = None
        self._reset_caches()
        if not same_shape:
            self.cursor_row = self.cursor_col = self.col_offset = 0
            self.scroll_to(y=0, animate=False)
        self.show_index = df is not None and not (isinstance(df.index, pd.RangeIndex) and df.index.name is None
                                                  and df.attrs.get("hide_index"))
        if same_shape and not same_cols:  # different columns: scroll back to the left
            self.col_offset = 0
        self._clamp()
        self._update_virtual_size()
        self._ensure_visible()
        self.refresh()
        self._post_cursor()

    @property
    def nrows(self) -> int:
        return 0 if self.df is None else len(self.df)

    @property
    def ncols(self) -> int:
        return 0 if self.df is None else self.df.shape[1]

    def frame_row(self, row: int) -> int:
        """The frame position shown on display row ``row``."""
        return int(self.order[row]) if self.order is not None else row

    def _column(self, col: int):
        c = self._cols.get(col)
        if c is None:
            c = self.df.index if col == INDEX else self.df.iloc[:, col]
            self._cols[col] = c
        return c

    def column_name(self, col: int) -> str:
        name = self._names.get(col)
        if name is None:
            raw = self.df.index.name if col == INDEX else self.df.columns[col]
            if col == INDEX and isinstance(self.df.index, pd.MultiIndex):
                raw = ", ".join(str(n) for n in self.df.index.names)
            name = "" if raw is None else str(raw)
            self._names[col] = name
        return name

    def _fmt(self, col: int) -> E.ColumnFormat:
        f = self._fmts.get(col)
        if f is None:
            f = self._fmts[col] = E.ColumnFormat(self._column(col), thousands=self.df.attrs.get("thousands", False))
        return f

    def width_of(self, col: int) -> int:
        w = self._widths.get(col)
        if w is None:
            vals = self._column(col)
            dtype = str(vals.dtype)
            w = min(max(E.column_width(self.column_name(col), self._fmt(col), vals), min(len(dtype), 12)),
                    self.max_width)
            if col == self.sort_col:
                w = max(w, cell_len(self.column_name(col)) + 2)
            self._widths[col] = w
        return w

    def _cell_block(self, col: int, block: int):
        key = (block, col)
        hit = self._cells.get(key)
        if hit is None:
            lo, hi = block * BLOCK, min((block + 1) * BLOCK, self.nrows)
            pos = self.order[lo:hi] if self.order is not None else np.arange(lo, hi)
            vals = self._column(col)
            vals = vals[pos] if isinstance(vals, pd.Index) else vals.iloc[pos]
            hit = self._fmt(col).format(vals)
            self._cells[key] = hit
        return hit

    def cell_text(self, row: int, col: int) -> tuple[str, bool]:
        strs, nulls = self._cell_block(col, row // BLOCK)
        return strs[row % BLOCK], bool(nulls[row % BLOCK])

    # ---------------------------------------------------------------- layout

    def _update_virtual_size(self):
        self.virtual_size = Size(self.size.width, self.nrows + HEADER)

    def on_resize(self, event: events.Resize) -> None:
        self._update_virtual_size()
        self._ensure_visible()

    @property
    def body_height(self) -> int:
        return max(1, self.size.height - HEADER)

    def _index_width(self) -> int:
        return self.width_of(INDEX) + 1 if (self.show_index and self.df is not None) else 0

    def visible_cols(self) -> list[int]:
        if self.df is None:
            return []
        avail = self.size.width - self._index_width() - 1
        cols, used = [], 0
        for c in range(self.col_offset, self.ncols):
            w = self.width_of(c) + 1
            if cols and used + w > avail:
                break
            cols.append(c)
            used += w
        return cols

    def _clamp(self):
        self.cursor_row = max(0, min(self.cursor_row, self.nrows - 1)) if self.nrows else 0
        lo = INDEX if self.show_index else 0
        self.cursor_col = max(lo, min(self.cursor_col, self.ncols - 1)) if self.ncols else lo
        self.col_offset = max(0, min(self.col_offset, max(0, self.ncols - 1)))

    def _ensure_visible(self):
        if self.df is None or not self.size.width:
            return  # hidden (e.g. behind the loading indicator): on_resize runs this again
        top = round(self.scroll_y)
        if self.cursor_row < top:
            self.scroll_to(y=self.cursor_row, animate=False)
        elif self.cursor_row >= top + self.body_height:
            self.scroll_to(y=self.cursor_row - self.body_height + 1, animate=False)
        c = self.cursor_col
        if c >= 0:
            if c < self.col_offset:
                self.col_offset = c
            else:
                while self.col_offset < c and c not in self.visible_cols():
                    self.col_offset += 1

    # --------------------------------------------------------------- render

    def notify_style_update(self) -> None:
        self._styles = {}
        super().notify_style_update()

    def _style(self, name: str) -> Style:
        s = self._styles.get(name)
        if s is None:
            s = self._styles[name] = self.get_component_rich_style(name, partial=True)
        return s

    def render_line(self, y: int) -> Strip:
        width = self.size.width
        base = self.rich_style
        if self.df is None:
            return Strip.blank(width, base)
        cols = self.visible_cols()
        segs: list[Segment] = []
        sep = Segment("│", base + self._style("grid--sep"))
        gap = Segment(" ", base)
        if y < HEADER:
            st = base + self._style("grid--header" if y == 0 else "grid--dtype")
            cur = st + self._style("grid--cursor-col")

            def head(col):
                if y == 0:
                    t = self.column_name(col)
                    if col == self.sort_col:
                        t = ("▲ " if self.sort_asc else "▼ ") + t
                else:
                    t = str(self.df.index.dtype if col == INDEX else self.df.dtypes.iloc[col])
                return t

            if self._index_width():
                w = self.width_of(INDEX)
                segs += [Segment(_fit(head(INDEX), w, False), cur if self.cursor_col == INDEX else st), sep]
            for c in cols:
                w = self.width_of(c)
                segs += [Segment(_fit(head(c), w, self._fmt(c).right), cur if c == self.cursor_col else st),
                         Segment(" ", st)]
            return Strip(segs).crop_extend(0, width, st)

        row = round(self.scroll_y) + y - HEADER
        if row >= self.nrows:
            return Strip.blank(width, base)
        is_cur = row == self.cursor_row
        row_style = base + self._style("grid--cursor-row") if is_cur else base
        null_style = self._style("grid--null")
        cursor = self._style("grid--cursor")
        if self._index_width():
            text, null = self.cell_text(row, INDEX)
            st = row_style + self._style("grid--index") + (null_style if null else Style())
            if is_cur and self.cursor_col == INDEX:
                st += cursor
            segs += [Segment(_fit(text, self.width_of(INDEX), self._fmt(INDEX).right), st), sep]
        for c in cols:
            text, null = self.cell_text(row, c)
            st = row_style + null_style if null else row_style
            if is_cur and c == self.cursor_col:
                st += cursor
            segs += [Segment(_fit(text, self.width_of(c), self._fmt(c).right), st), Segment(" ", row_style)]
        return Strip(segs).crop_extend(0, width, row_style if is_cur else base)

    # -------------------------------------------------------------- actions

    def _post_cursor(self):
        self.post_message(self.CursorMoved(self, self.cursor_row, self.cursor_col))

    def move_to(self, row: int | None = None, col: int | None = None) -> None:
        if row is not None:
            self.cursor_row = row
        if col is not None:
            self.cursor_col = col
        self._clamp()
        self._ensure_visible()
        self.refresh()
        self._post_cursor()

    def action_move(self, dr: int, dc: int) -> None:
        self.move_to(self.cursor_row + dr, self.cursor_col + dc)

    def action_page(self, direction: int) -> None:
        step = self.body_height
        self.scroll_to(y=self.scroll_y + direction * step, animate=False)
        self.move_to(self.cursor_row + direction * step)

    def action_edge_row(self, end: int) -> None:
        self.move_to(self.nrows - 1 if end else 0)

    def action_edge_col(self, end: int) -> None:
        if end:
            self.move_to(col=self.ncols - 1)
        else:
            self.col_offset = 0
            self.move_to(col=INDEX if self.show_index else 0)

    def action_resize(self, delta: int) -> None:
        c = self.cursor_col
        if self.df is None:
            return
        self._widths[c] = max(E.MIN_COL_WIDTH, min(self.width_of(c) + delta, 400))
        self.refresh()

    def key_enter(self) -> None:
        if self.nrows:
            self.post_message(self.Selected(self, self.frame_row(self.cursor_row), self.cursor_col))

    def on_click(self, event: events.Click) -> None:
        if self.df is None:
            return
        x, y = event.x, event.y
        col = None
        iw = self._index_width()
        if iw and x < iw:
            col = INDEX
        else:
            pos = iw
            for c in self.visible_cols():
                w = self.width_of(c) + 1
                if pos <= x < pos + w:
                    col = c
                    break
                pos += w
        if y < HEADER:
            if col is not None:
                self.move_to(col=col)
                self.action_sort()
            return
        row = round(self.scroll_y) + y - HEADER
        if row < self.nrows:
            if event.chain >= 2 and row == self.cursor_row and col == self.cursor_col:
                self.key_enter()
            self.move_to(row, col)

    def action_sort(self) -> None:
        """Sort by the cursor column: ascending, then descending, then file order."""
        if self.df is None or self.busy:
            return
        col = self.cursor_col
        if self.sort_col == col and self.sort_asc:
            asc = False
        elif self.sort_col == col:
            self.order, self.sort_col = None, None
            self._cells.clear()
            self._widths.pop(col, None)
            self.move_to(0)
            return
        else:
            asc = True
        self.busy = True
        if self.nrows > 1_000_000:
            self.notify(f"Sorting by {self.column_name(col) or 'index'}…", timeout=2, markup=False)
        self._sort(self.df, col, asc)

    @work(thread=True, exclusive=True, group="sort")
    def _sort(self, df, col: int, asc: bool) -> None:
        keys = df.attrs.get("sort_keys", {})
        name = None if col == INDEX else df.columns[col]
        values = keys.get(name) if name in keys else (df.index if col == INDEX else df.iloc[:, col])
        try:
            order = E.sort_order(values, asc)
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self.notify, f"Couldn't sort: {e}", severity="error", markup=False)
            self.busy = False
            return
        self.app.call_from_thread(self._apply_sort, df, col, asc, order)

    def _apply_sort(self, df, col, asc, order):
        self.busy = False
        if df is not self.df:
            return
        self.order, self.sort_col, self.sort_asc = order, col, asc
        self._cells.clear()
        self._widths.pop(col, None)
        self.move_to(0)


def _fit(text: str, width: int, right: bool) -> str:
    """Pad or truncate ``text`` to exactly ``width`` cells."""
    if text.isascii():
        n = len(text)
        if n > width:
            return text[: width - 1] + "…" if width > 1 else text[:width]
        return text.rjust(width) if right else text.ljust(width)
    n = cell_len(text)
    if n > width:
        return set_cell_size(text, width - 1) + "…" if width > 1 else set_cell_size(text, width)
    pad = " " * (width - n)
    return pad + text if right else text + pad
