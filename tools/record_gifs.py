"""Record the README GIFs by driving the real app headlessly.

    uv run python tools/make_demo_data.py      # once: demo/trades.parquet (100 M rows)
    uv run --group demo python tools/record_gifs.py [scene ...]

Each scene runs dx under Textual's test pilot and presses real keys. Every
frame is the app's own SVG screenshot, rendered by headless Chrome
(Playwright, system Chrome) with a caption naming the keys just pressed.
Pillow assembles the frames into assets/<scene>.gif with one shared palette.
Waits for loading are shortened in the GIF; captions quote the real times.
"""
import asyncio
import io
import os
import sys
import time
from html import escape

from PIL import Image
from playwright.async_api import async_playwright
from textual.widgets import DirectoryTree, Input, Select

import tempfile
from pathlib import Path

import dxtui.app
from dxtui import engine as E
from dxtui.app import DxApp
from dxtui.grid import DataGrid

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "demo", "trades.parquet")
OUT = os.path.join(ROOT, "assets")
SIZE = (120, 32)  # terminal columns × rows
WIDTH = 1200      # rendered width of the terminal window (px)

PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
body { margin: 0; background: #0d1117; }
#wrap { display: inline-block; padding: 16px 16px 0; }
#wrap svg { display: block; width: %dpx; height: auto; }
.cap { height: 58px; display: flex; align-items: center; justify-content: center; gap: 12px;
       font: 500 21px 'DejaVu Sans', system-ui, sans-serif; color: #c9d1d9; }
.cap b { color: #f0f6fc; font-weight: 600; }
kbd { font: 700 18px 'DejaVu Sans Mono', monospace; background: #f0f6fc; color: #0d1117; border-radius: 6px;
      padding: 3px 10px; box-shadow: 0 2px 0 #8b949e; }
.prompt { font: 600 20px 'DejaVu Sans Mono', monospace; color: #7ee787; }
.prompt span { color: #f0f6fc; }
code { font: 17px 'DejaVu Sans Mono', monospace; color: #7ee787; }
</style></head><body><div id="wrap">%s<div class="cap">%s</div></div></body></html>"""


def caption(keys=(), text="", prompt=None):
    if prompt:
        return f'<span class="prompt">$ <span>{escape(prompt)}</span></span>' + (
            f"<span>{escape(text)}</span>" if text else "")
    return "".join(f"<kbd>{escape(k)}</kbd>" for k in keys) + (f"<span>{text}</span>" if text else "")


class Recorder:
    def __init__(self, page, name):
        self.page, self.name = page, name
        self.frames: list[tuple[bytes, int]] = []
        self.app = self.pilot = None

    async def shot(self, hold_ms, cap=""):
        hold_ms = max(hold_ms, 60)  # browsers stretch 0–10 ms GIF frames to 100 ms
        svg = self.app.export_screenshot()
        await self.page.set_content(PAGE % (WIDTH, svg, cap))
        png = await (await self.page.query_selector("#wrap")).screenshot()
        if self.frames and self.frames[-1][0] == png:
            self.frames[-1] = (png, self.frames[-1][1] + hold_ms)
        else:
            self.frames.append((png, hold_ms))

    def busy(self):
        if self.app.busy:
            return True
        return any(w.state.name in ("PENDING", "RUNNING") and not isinstance(w.node, DirectoryTree)
                   for w in self.app.workers)

    async def settle(self, cap="", progress_ms=0):
        """Wait for loading / background work; optionally film it (shortened)."""
        await self.pilot.pause(0.15)
        last = 0.0
        while self.busy():
            if progress_ms and time.time() - last > 0.5:
                last = time.time()
                await self.shot(progress_ms, cap)
            await self.pilot.pause(0.05)
        await self.pilot.pause(0.25)

    async def press(self, *keys, text="", hold=1100, label=None, progress_ms=0):
        for k in keys:
            await self.pilot.press(k)
            await self.pilot.pause(0.03)
        cap = caption(label or keys, text)
        await self.settle(cap, progress_ms)
        await self.shot(hold, cap)

    async def type(self, selector, text, label=(), cap_text="", every=2, frame_ms=70, hold=900):
        box = self.app.screen.query_one(selector, Input)
        cap = caption(label, cap_text)
        for i in range(1, len(text) + 1):
            box.value = text[:i]
            box.cursor_position = i
            await self.pilot.pause(0.02)
            if i % every == 0 or i == len(text):
                await self.pilot.pause(0.04)  # let the suggestion appear
                await self.shot(frame_ms, cap)
        await self.shot(hold, cap)

    def grid(self):
        return self.app.query_one("#grid", DataGrid)

    def col(self, name):
        return list(E.STATE.df.columns).index(name)

    def save(self):
        imgs = [Image.open(io.BytesIO(png)).convert("RGB") for png, _ in self.frames]
        # one palette for the whole GIF, from a sample of frames
        step = max(1, len(imgs) // 12)
        sample = imgs[::step][:12]
        w, h = sample[0].size
        mosaic = Image.new("RGB", (w, h * len(sample)))
        for i, im in enumerate(sample):
            mosaic.paste(im, (0, i * h))
        pal = mosaic.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
        frames = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
        path = os.path.join(OUT, f"{self.name}.gif")
        frames[0].save(path, save_all=True, append_images=frames[1:], loop=0, optimize=True,
                       duration=[d for _, d in self.frames], disposal=1)
        total = sum(d for _, d in self.frames) / 1000
        print(f"{path}: {len(frames)} frames, {total:.1f}s, {os.path.getsize(path) / 1e6:.1f} MB")


async def session(rec, path, opts=None, pick=False, scene=None):
    """Run ``scene`` against the app with ``path`` open."""
    app = DxApp(path, opts or {}, pick)
    async with app.run_test(size=SIZE) as pilot:
        rec.app, rec.pilot = app, pilot
        await pilot.pause(0.3)
        await scene(rec)


def info_time():
    return E.STATE.info.get("load_seconds", 0)


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

async def hero(r):
    await r.settle()
    await r.shot(2600, caption(prompt="dx demo/trades.parquet", text="  100 million rows · choose what to load first"))
    await r.press("enter", text="load", hold=200, progress_ms=180)
    await r.shot(2200, caption(["Enter"], f"<b>5 million row sample</b> of 100,000,000 in <b>{info_time():.1f}s</b>"))
    for _ in range(3):
        await r.press("pagedown", text="scroll", hold=350, label=["PgDn"])
    await r.press("G", text="last row, instantly", hold=1300)
    await r.press("g", text="back to the top", hold=600)
    await r.press("l", "l", text="", hold=300, label=["→", "→"])
    await r.press("s", text="sort by <b>price</b> (5M rows)", hold=1200)
    await r.press("s", text="descending", hold=1300)
    await r.press("2", text="stats for every column", hold=300, progress_ms=150)
    await r.shot(2400, caption(["2"], "stats for every column"))
    await r.press("1", text="", hold=200)
    r.grid().move_to(col=r.col("venue"))
    await r.press("p", text="plot the column: rows per <b>venue</b>", hold=2200, progress_ms=150)
    pane = r.app.query_one("#plotpane")
    pane.query_one("#kind", Select).value = "hist"
    pane._ys = [r.col("latency_us")]
    pane._fill_ys()
    pane.query_one("#color", Select).value = r.col("order_type")
    await r.settle()
    await r.shot(3200, caption([], "histogram of <b>latency_us</b> by order type · every row counted"))


async def table(r):
    await r.settle()
    await r.shot(1200, caption([], "<b>trades.parquet</b> · the index and header stay pinned"))
    await r.press("j", "j", "j", text="move", hold=300, label=["↓", "↓", "↓"])
    await r.press("enter", text="the row as a list of fields", hold=2200, label=["Enter"])
    await r.press("escape", text="", hold=300, label=["Esc"])
    await r.press("slash", text="find a column", hold=500, label=["/"])
    await r.type("#q", "lat", label=["/"], cap_text="find a column", every=1, frame_ms=150, hold=600)
    await r.press("enter", text="jump to <b>latency_us</b>", hold=1000, label=["Enter"])
    await r.press("s", text="sort", hold=1200)
    await r.press("greater_than_sign", "greater_than_sign", text="widen the column", hold=900, label=[">", ">"])
    await r.press("number_sign", text="go to a row", hold=400, label=["#"])
    await r.type("#answer", "2500000", label=["#"], cap_text="go to a row", every=2, frame_ms=110, hold=300)
    await r.press("enter", text="row 2,500,000", hold=1600, label=["Enter"])


async def columns(r):
    await r.settle()
    await r.press("2", text="per-column stats", hold=300, progress_ms=150)
    await r.shot(2200, caption(["2"], "per-column stats: nulls, unique, min/max, mean, std, where the extremes are"))
    g = r.app.query_one("#stats", DataGrid)
    g.move_to(col=list(g.df.columns).index("nulls"))
    await r.press("s", "s", text="sort by <b>nulls</b>, most first", hold=1800, label=["s", "s"])
    await r.press("enter", text="column details: a text column", hold=3000, label=["Enter"])
    await r.press("escape", text="", hold=300, label=["Esc"])
    g.move_to(row=list(g.df["name"].to_numpy()[g.order]).index("price"), col=0)
    await r.press("c", text="column details: quantiles, top values, histogram", hold=3400)
    await r.press("escape", text="", hold=300, label=["Esc"])
    r.app.query_one("#stats-filter", Input).focus()
    await r.type("#stats-filter", "id", label=[], cap_text="filter columns by name", every=1, frame_ms=200, hold=1600)


PLOT_WHERE = "symbol IN ('NVDA', 'TSLA', 'MSFT', 'AAPL', 'META') AND qty > 1500"


async def plots(r):
    await r.settle()
    await r.press("4", text="plot", hold=300, progress_ms=150)
    pane = r.app.query_one("#plotpane")
    sel = lambda i: pane.query_one(i, Select)  # noqa: E731
    pane.configure("line", y=r.col("price"))
    sel("#color").value = r.col("symbol")
    await r.settle()
    await r.shot(2800, caption(["4"], f"line · colour by <b>symbol</b> · {len(E.STATE.df):,} trades from "
                                       "<code>dx trades.parquet -w \"symbol IN (…) AND qty &gt; 1500\"</code>"))
    sel("#kind").value = "hist"
    sel("#color").clear()
    pane._ys = [r.col("latency_us")]
    pane._fill_ys()
    pane.draw()
    await r.settle()
    await r.shot(2400, caption([], "histogram of <b>latency_us</b>"))
    sel("#kind").value = "scatter"
    sel("#x").value = r.col("qty")
    pane._ys = [r.col("fee")]
    pane._fill_ys()
    sel("#color").value = r.col("venue")
    await r.settle()
    await r.shot(2600, caption([], "scatter <b>fee</b> vs <b>qty</b> by venue · every occupied cell is drawn"))
    sel("#color").clear()
    sel("#kind").value = "bar"
    sel("#x").value = r.col("symbol")
    pane._ys = [r.col("price")]
    pane._fill_ys()
    sel("#agg").value = "mean"
    await r.settle()
    await r.shot(2600, caption([], "bar · <b>mean price</b> by symbol"))


async def expressions(r):
    await r.settle()
    await r.press("colon", text="pandas expression", hold=500, label=[":"])
    await r.type("#expr", "df['notional'] = df.pr", label=[":"], cap_text="type… suggestions come from your columns",
                 every=2, frame_ms=90, hold=1200)
    await r.press("right", text="accept the suggestion", hold=700, label=["→"])
    box = r.app.query_one("#expr", Input)
    await r.type("#expr", box.value + " * df.qty", label=[":"], cap_text="", every=3, frame_ms=80, hold=500)
    await r.press("enter", text="a new column, everywhere", hold=1600, label=["Enter"])
    r.grid().move_to(col=r.col("notional"))
    await r.shot(1400, caption([], "<b>notional</b> is in the table, stats and plots"))
    await r.press("colon", text="", hold=200, label=[":"])
    await r.type("#expr", "df.groupby('symbol').notional.agg(['sum', 'mean', 'count'])", label=[":"],
                 cap_text="", every=4, frame_ms=70, hold=500)
    await r.press("enter", text="the result becomes the view", hold=2200, label=["Enter"])
    await r.press("escape", text="", hold=1, label=["Esc"])
    r.grid().move_to(col=0)
    await r.press("s", "s", text="sort it", hold=1400, label=["s", "s"])
    await r.press("S", text="the steps, as a Python script to copy", hold=3200)
    await r.press("escape", text="", hold=200, label=["Esc"])
    await r.press("R", text="back to the loaded data", hold=1600)


async def load(r):
    await r.settle()
    await r.shot(1800, caption(prompt="dx demo/trades.parquet", text="  pick columns, filter rows or write SQL first"))
    scr = r.app.screen
    await r.type("#where", "symbol = 'NVDA' AND qty > 2000", label=[], cap_text="filter rows with an SQL WHERE",
                 every=3, frame_ms=80, hold=900)
    t = time.time()
    await r.press("ctrl+l", text="DuckDB scans the whole file…", hold=200, progress_ms=250)
    await r.shot(2400, caption(["ctrl+l"], f"all <b>100,000,000</b> rows filtered in <b>{info_time():.1f}s</b> "
                                           f"→ {len(E.STATE.df):,} matches"))
    r.grid().move_to(col=r.col("price"))
    await r.press("p", text="plot NVDA's price", hold=2800, progress_ms=150)
    await r.press("1", text="", hold=1)
    await r.press("o", text="load options again: columns, SQL, index, memory budget", hold=600)
    await r.type("#sql", "SELECT symbol, venue, count(*) AS trades, sum(price * qty) AS notional FROM data "
                         "GROUP BY ALL ORDER BY notional DESC", label=["o"], cap_text="…or any SQL query",
                 every=6, frame_ms=70, hold=700)
    await r.press("ctrl+l", text="DuckDB runs it over the file", hold=200, progress_ms=250)
    await r.shot(2800, caption(["ctrl+l"], f"query over 100M rows in <b>{info_time():.1f}s</b>"))


SCENES = {
    "hero": (hero, dict(pick=None)),
    "table": (table, dict(pick=False)),
    "columns": (columns, dict(pick=False)),
    "plots": (plots, dict(pick=False, opts={"where": PLOT_WHERE})),
    "expressions": (expressions, dict(pick=False)),
    "load": (load, dict(pick=True)),
}


async def main(names):
    if not os.path.exists(DATA):
        sys.exit("Run tools/make_demo_data.py first")
    os.makedirs(OUT, exist_ok=True)
    # a fresh expression history, so suggestions come from the columns (and yours is untouched)
    dxtui.app.HISTORY_FILE = Path(tempfile.mkdtemp()) / "history"
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome")
        page = await browser.new_page(viewport={"width": WIDTH + 200, "height": 1200})
        for name in names:
            scene, kw = SCENES[name]
            E.STATE.__init__()
            rec = Recorder(page, name)
            t = time.time()
            await session(rec, DATA, scene=scene, **kw)
            rec.save()
            print(f"  recorded in {time.time() - t:.0f}s")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or list(SCENES)))
