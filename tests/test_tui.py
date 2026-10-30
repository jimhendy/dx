"""Tests for the terminal app (tui/dxtui), driven headlessly with Textual's pilot.

The engine itself is covered by test_explorer.py and friends; these check the
TUI's own pieces (grid, sorting, formatting, plot reduction) and the flows
through the app: load options, expressions, steps, stats, dialogs.
"""
import asyncio
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))  # also runs without installing
from dxtui import engine as E  # noqa: E402
from dxtui.app import DxApp  # noqa: E402
from dxtui.grid import INDEX, DataGrid  # noqa: E402
from dxtui.plot import build_spec  # noqa: E402
from dxtui.screens import ColumnScreen, LoadScreen, PickScreen, RowScreen, StepsScreen  # noqa: E402

FIX = os.path.join(HERE, "fixtures")
SIZE = (150, 45)


def fx(name):
    return os.path.join(FIX, name)


@pytest.fixture(autouse=True)
def fresh_state(tmp_path, monkeypatch):
    E.ex.STATE.__init__()
    E._stats_cache.update(df=None, frame=None)
    monkeypatch.setattr("dxtui.app.HISTORY_FILE", tmp_path / "history")


async def settle(app, pilot):
    await pilot.pause(0.05)
    for _ in range(600):
        if not app.busy and not any(w.state.name in ("PENDING", "RUNNING") for w in app.workers):
            break
        await pilot.pause(0.05)
    await pilot.pause(0.05)


def run(path, steps, opts=None, pick=False):
    """Start the app on ``path`` and run ``steps(app, pilot)`` once it has loaded."""
    async def main():
        app = DxApp(path, opts or {}, pick)
        async with app.run_test(size=SIZE) as pilot:
            await settle(app, pilot)
            await steps(app, pilot)
    asyncio.run(main())


async def expr(app, pilot, text):
    await pilot.press("colon")
    box = app.query_one("#expr")
    box.value = text
    await pilot.press("enter")
    await settle(app, pilot)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,key", [
    ("sample.parquet", None), ("indexed.parquet", None), ("sample.csv", None), ("sample.tsv", None),
    ("lines.jsonl", None), ("records.json", None), ("sample.xlsx", "Data"), ("sample.h5", "/pandas_fixed"),
    ("wide.csv", None), ("wide_keyed.parquet", None),
])
def test_every_format_loads(name, key):
    async def steps(app, pilot):
        df = E.STATE.df
        assert df is not None and len(df) > 0
        grid = app.query_one("#grid", DataGrid)
        assert grid.df is df
        # every visible line renders
        for y in range(grid.size.height):
            grid.render_line(y)
        assert name in str(app.query_one("#file").render())
    run(fx(name), steps, {"key": key} if key else None)


def test_load_screen_filters_rows():
    async def steps(app, pilot):
        assert isinstance(app.screen, LoadScreen)
        app.screen.query_one("#where").value = "value > 10"
        await pilot.press("ctrl+l")
        await settle(app, pilot)
        df = E.STATE.df
        assert len(df) and (df["value"] > 10).all()
        assert E.STATE.info["selection"]["where"] == "value > 10"
        assert "matching" in str(app.query_one("#file").render())
    run(fx("sample.parquet"), steps, pick=True)


def test_load_screen_columns():
    async def steps(app, pilot):
        scr = app.screen
        assert isinstance(scr, LoadScreen)
        scr.query_one("#colfilter").value = "val"
        await pilot.pause(0.1)
        scr.query_one("#only").press()
        await pilot.pause(0.1)
        assert scr.chosen == {"value"}
        await pilot.press("ctrl+l")
        await settle(app, pilot)
        assert list(E.STATE.df.columns) == ["value"]
    run(fx("sample.csv"), steps, pick=True)


def test_several_datasets_ask_first():
    async def steps(app, pilot):
        scr = app.screen
        assert isinstance(scr, LoadScreen)
        scr.query_one("#key").value = "/raw/matrix"
        await pilot.press("ctrl+l")
        await settle(app, pilot)
        assert E.STATE.df.shape == (20, 3)
        assert E.STATE.info["key"] == "/raw/matrix"
    run(fx("sample.h5"), steps, pick=None)


def test_bad_sql_reports_error():
    async def steps(app, pilot):
        assert "Couldn't load" in str(app.query_one("#output-text").render())
        assert app.query_one("#output").has_class("shown")
    run(fx("sample.parquet"), steps, {"where": "no_such_column > 1"})


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def test_sort_cycles_and_keeps_nans_last():
    async def steps(app, pilot):
        grid = app.query_one("#grid", DataGrid)
        col = list(E.STATE.df.columns).index("value")
        grid.move_to(col=col)
        await pilot.press("s")
        await settle(app, pilot)
        vals = E.STATE.df["value"].to_numpy()[grid.order]
        finite = vals[~np.isnan(vals)]
        assert (np.diff(finite) >= 0).all() and np.isnan(vals[-1])
        await pilot.press("s")
        await settle(app, pilot)
        vals = E.STATE.df["value"].to_numpy()[grid.order]
        assert vals[0] == np.nanmax(vals) and not grid.sort_asc
        await pilot.press("s")
        await settle(app, pilot)
        assert grid.order is None and grid.sort_col is None
    run(fx("sample.parquet"), steps)


def test_navigation_and_row_view():
    async def steps(app, pilot):
        grid = app.query_one("#grid", DataGrid)
        await pilot.press("G")
        assert grid.cursor_row == grid.nrows - 1
        await pilot.press("dollar_sign")
        assert grid.cursor_col == grid.ncols - 1
        await pilot.press("home")
        assert grid.cursor_col == INDEX
        await pilot.press("g", "j", "j", "l")
        assert (grid.cursor_row, grid.cursor_col) == (2, 0)
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert isinstance(app.screen, RowScreen) and app.screen.pos == 2
        fields = app.screen.query_one("#fields", DataGrid)
        assert fields.nrows == E.STATE.df.shape[1]
        app.screen.query_one("#filter").value = "val"
        await pilot.pause(0.1)
        assert list(fields.df.index) == ["value"]
        await pilot.press("escape")
        assert not isinstance(app.screen, RowScreen)
    run(fx("sample.parquet"), steps)


def test_wide_frame_scrolls_by_column():
    async def steps(app, pilot):
        grid = app.query_one("#grid", DataGrid)
        assert grid.ncols > 500
        await pilot.press("dollar_sign")
        assert grid.cursor_col == grid.ncols - 1
        assert grid.cursor_col in grid.visible_cols()
        assert grid.col_offset > 0
        await pilot.press("slash")
        await pilot.pause(0.1)
        assert isinstance(app.screen, PickScreen)
        target = str(E.STATE.df.columns[123])
        app.screen.query_one("#q").value = target
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert str(E.STATE.df.columns[grid.cursor_col]) == target
    run(fx("wide.csv"), steps)


def test_goto_row():
    async def steps(app, pilot):
        await pilot.press("number_sign")
        await pilot.pause(0.1)
        app.screen.query_one("#answer").value = "-1"
        await pilot.press("enter")
        await pilot.pause(0.1)
        grid = app.query_one("#grid", DataGrid)
        assert grid.cursor_row == grid.nrows - 1
    run(fx("sample.csv"), steps)


# ---------------------------------------------------------------------------
# Expressions and steps
# ---------------------------------------------------------------------------

def test_expressions_change_view_and_reset():
    async def steps(app, pilot):
        n = E.STATE.df.shape[1]
        await expr(app, pilot, "df['v2'] = df['value'] * 2")
        assert E.STATE.df.shape[1] == n + 1
        assert app.query_one("#grid", DataGrid).ncols == n + 1
        assert "1 step" in str(app.query_one("#file").render())
        grid = app.query_one("#grid", DataGrid)
        grid.move_to(col=n)  # on the new column, at the far right
        await expr(app, pilot, "df.describe()")
        assert list(E.STATE.df.index[:2]) == ["count", "mean"]
        assert grid.col_offset == 0 and len(grid.visible_cols()) > 1  # new columns: scrolled back left
        assert len(E.STATE.steps) == 2
        await pilot.press("escape")  # leave the expression box
        await pilot.press("S")
        await pilot.pause(0.1)
        assert isinstance(app.screen, StepsScreen)
        script = app.screen.query_one("#script").text
        assert "df['v2'] = df['value'] * 2" in script and "df = df.describe()" in script
        await pilot.press("escape")
        await pilot.press("R")
        await settle(app, pilot)
        assert E.STATE.df is E.STATE.df0 and not E.STATE.steps
        assert app.query_one("#grid", DataGrid).ncols == n
    run(fx("sample.parquet"), steps)


def test_value_and_error_output():
    async def steps(app, pilot):
        await expr(app, pilot, "round(df['value'].mean(), 3)")
        out = str(app.query_one("#output-text").render())
        assert out.strip() == str(round(E.STATE.df["value"].mean(), 3))
        assert not app.query_one("#output").has_class("error")
        await expr(app, pilot, "df['nope']")
        assert app.query_one("#output").has_class("error")
        assert "KeyError" in str(app.query_one("#output-text").render())
        assert app.history[-2:] == ["round(df['value'].mean(), 3)", "df['nope']"]
    run(fx("sample.parquet"), steps)


def test_expression_suggestions():
    async def steps(app, pilot):
        sug = app.query_one("#expr").suggester
        assert await sug.get_suggestion("df['val") == "df['value']"
        assert await sug.get_suggestion("df.gro") == "df.group"
        app.history.append("df.groupby('group').value.mean()")
        assert await sug.get_suggestion("df.groupby(") == "df.groupby('group').value.mean()"
    run(fx("sample.parquet"), steps)


# ---------------------------------------------------------------------------
# Columns tab, column details, overview, plot
# ---------------------------------------------------------------------------

def test_columns_tab_stats_filter_and_details():
    async def steps(app, pilot):
        await pilot.press("2")
        await settle(app, pilot)
        stats = app.query_one("#stats", DataGrid)
        assert stats.nrows == E.STATE.df.shape[1]
        frame = stats.df
        row = frame[frame["name"] == "value"].iloc[0]
        assert row["nulls"] == E.STATE.df["value"].isna().sum()
        assert row["mean"] == pytest.approx(E.STATE.df["value"].mean())
        assert app.focused is stats  # keys work straight away on the Columns tab
        app.query_one("#stats-filter").value = "val"
        await pilot.pause(0.1)
        assert stats.nrows == 1
        stats.focus()
        await pilot.press("enter")
        await settle(app, pilot)
        assert isinstance(app.screen, ColumnScreen) and app.screen.name_ == "value"
        await settle(app, pilot)
        top = app.screen.query_one("#top", DataGrid)
        assert top.nrows > 0 and list(top.df.columns) == ["value", "count", "%"]
    run(fx("sample.parquet"), steps)


def test_stats_sort_uses_numeric_min():
    df = pd.DataFrame({"a": [5.0, 6.0], "b": [10.0, 11.0], "c": ["x", "y"]})
    frame = E.column_stats(df)
    order = E.sort_order(frame.attrs["sort_keys"]["min"], True)
    assert list(frame["name"].to_numpy()[order]) == ["a", "b", "c"]  # 5 < 10 (not "10" < "5"), text last


def test_overview_and_plot_column():
    async def steps(app, pilot):
        await pilot.press("3")
        labels = [str(c) for c in app.query_one("#overview").content.columns[0]._cells]
        assert {"File", "Rows in file", "Loaded", "Index"} <= set(labels)
        await pilot.press("1")
        grid = app.query_one("#grid", DataGrid)
        grid.move_to(col=list(E.STATE.df.columns).index("value"))
        await pilot.press("p")
        await settle(app, pilot)
        assert app.query_one("TabbedContent").active == "plot"
        await pilot.pause(0.1)
        assert app.focused.id == "plot-controls"  # focus moves with the tab (else TabbedContent jumps back)
        for key, tab in [("1", "table"), ("4", "plot"), ("2", "columns"), ("4", "plot"), ("3", "overview-tab")]:
            await pilot.press(key)
            await pilot.pause(0.2)
            assert app.query_one("TabbedContent").active == tab  # digits switch tabs from every tab
        pane = app.query_one("#plotpane")
        assert pane._ys == [list(E.STATE.df.columns).index("value")]
    run(fx("indexed.parquet"), steps)


def test_build_spec_kinds():
    rng = np.random.default_rng(1)
    n = 50_000
    df = pd.DataFrame({"t": pd.date_range("2024-01-01", periods=n, freq="min"),
                       "y": rng.normal(size=n), "g": rng.choice(list("abcdefghijk"), n)})
    base = dict(x=0, ys=[1], color=None, agg="mean", bins=30)
    line = build_spec(df, dict(base, kind="line"), 120, 40)
    assert line["series"] and len(line["series"][0]["x"]) < n  # reduced
    assert max(line["series"][0]["y"]) == df["y"].max()        # extremes kept
    assert line["xticks"] and line["xticks"][1][0].startswith("2024")
    scatter = build_spec(df, dict(base, kind="scatter", color=2), 120, 40)
    assert len(scatter["series"]) == E.MAX_GROUPS and "(other)" in scatter["note"]
    hist = build_spec(df, dict(base, kind="hist"), 120, 40)
    assert sum(hist["series"][0]["y"]) == n
    bar = build_spec(df, dict(base, kind="bar", x=2, agg="count"), 120, 40)
    labels, values = bar["bars"]
    assert sum(values) == n and len(labels) == 11
    empty = build_spec(df, dict(base, kind="line", ys=[]), 120, 40)
    assert "empty" in empty


def test_m4_and_occupied_keep_extremes():
    x = np.arange(1_000_000, dtype=float)
    y = np.sin(x / 5000) + np.random.default_rng(0).random(len(x))
    y[123_456] = 50
    px, py = E.m4(x, y, 300)
    assert len(px) <= 4 * 300 and py.max() == 50 and py.min() == y.min()
    ox, oy = E.occupied(x, y, 200, 100)
    assert len(ox) < 200 * 100 and oy.max() > 49


# ---------------------------------------------------------------------------
# Formatting, saving
# ---------------------------------------------------------------------------

def test_column_formats():
    f = E.ColumnFormat(pd.Series([1.5, np.nan, 2.25]))
    assert f.right and f.format(pd.Series([1.5, np.nan, 2.25]))[0] == ["1.50", "NaN", "2.25"]
    f = E.ColumnFormat(pd.Series([1e20, 3.0]))
    assert f.format(pd.Series([1e20]))[0] == ["1e+20"]
    d = pd.Series(pd.to_datetime(["2024-01-01", None]))
    assert E.ColumnFormat(d).format(d)[0] == ["2024-01-01", "NaT"]
    t = pd.Series(["a\nb", "x" * 500])
    out = E.ColumnFormat(t).format(t)[0]
    assert out[0] == "a↵b" and out[1].endswith("…") and len(out[1]) == E.MAX_CELL + 1


@pytest.mark.parametrize("ext", ["parquet", "csv", "jsonl", "xlsx", "h5"])
def test_save_view_roundtrip(tmp_path, ext):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = E.save_view(df, str(tmp_path / f"v.{ext}"))
    back = {"parquet": pd.read_parquet, "csv": pd.read_csv, "xlsx": pd.read_excel, "h5": pd.read_hdf,
            "jsonl": lambda p: pd.read_json(p, lines=True)}[ext](out)
    assert list(back["b"]) == ["x", "y", "z"]
    with pytest.raises(E.UserError):
        E.save_view(df, str(tmp_path / "v.unknown"))


def test_escape_sequences_are_not_rendered():
    # text from the file is drawn as plain text: no markup, no terminal escapes interpreted by us
    async def steps(app, pilot):
        grid = app.query_one("#grid", DataGrid)
        for y in range(grid.size.height):
            strip = grid.render_line(y)
            assert "\x1b" not in "".join(seg.text for seg in strip)
    run(fx("xss.csv"), steps)
