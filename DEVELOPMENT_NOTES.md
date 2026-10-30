# Development notes

This file records design decisions and the problems hit while building dx: what
caused each one and how it was fixed. Read it before making big changes.

## Layout

| Path | What |
|---|---|
| `src/dxtui/explorer.py` | The engine: format detection, loading and sampling, DuckDB filters and SQL, index choice, column stats, expressions and steps. **Shared** with [pyodide-data-explorer](https://github.com/jimhendy/pyodide-data-explorer) (there `src/explorer.py`, running in Pyodide). Keep the two in step: change it in one repo, copy it to the other, run both test suites. |
| `src/dxtui/engine.py` | TUI-only glue: per-column display formats, sort orders, plot reduction (M4 lines, occupied-cell scatter, groups), steps script, save. Native defaults (512 MB budget) live here, not in the engine. |
| `src/dxtui/grid.py` | `DataGrid`, the virtual table (Line API `ScrollView`). |
| `src/dxtui/plot.py` | The Plot tab and `build_spec` (all data reduction runs in a worker thread). |
| `src/dxtui/screens.py` | Dialogs: load options, column details, row view, steps, file picker, prompts. |
| `src/dxtui/app.py` | The app, the key bindings and the CLI. |
| `tests/test_tui.py` | App tests through Textual's pilot. `test_explorer.py`, `test_wide.py` and `test_load_options.py` are the engine's tests, shared with the web app. |
| `tools/record_gifs.py`, `tools/make_demo_data.py` | The README GIFs and their 100 M-row demo file. |

## Decisions

- **Reuse the engine instead of rewriting it.** The browser version already
  solved sampling of ~5 GB files, DuckDB filtering, index inference and
  expression tracking, and its tests run natively. The TUI imports it unchanged.
- **The engine's DuckDB path is already the fast one natively.** Measured on
  5 GB files: pyarrow streams into a single DuckDB thread, taking 11 s for a
  parquet WHERE and 30 s for a CSV one. DuckDB's own multi-threaded readers
  were slower (13 s and 58 s), because the row-number window and
  `preserve_insertion_order` serialise them. Using all threads saved only
  about 15%.
- **A virtual grid, not `DataTable`.** `DataTable` builds every row up front.
  `DataGrid` formats 64-row blocks of only the visible columns (LRU cache). It
  scrolls rows with the scrollbar and columns one at a time with the cursor,
  and keeps the index and header pinned. Each column's format (decimals,
  date-only, thousands separators for big floats) is decided once per column,
  so numbers line up however you scroll.
- **Plots reduce, not sample.** Lines keep first / min / max / last per pixel
  column (`np.minimum/maximum.reduceat` takes 0.12 s for 10 M points; the
  first version sorted everything with lexsort and took 2.5 s). Scatter is a
  `histogram2d` at braille resolution. Grouping uses `pd.factorize` +
  `bincount` (0.2 s for 5 M strings; a Python `.map` took seconds).
- **Ask before loading big files.** Files over 64 MB, or with several
  datasets / sheets, open the load options first. The user of the web version
  wanted options offered before data is read. `-y` skips them.

## Problems hit

- **duckdb 2.x wheels need AVX2.** On the dev machine (i7-2600) `import duckdb`
  dies with SIGILL ("Illegal instruction") and no Python traceback.
  `pyproject.toml` pins `duckdb<2`.
- **Markup injection.** Textual `Static` strings and Rich `Table` cells are
  parsed as markup, so a column called `[/]` crashed the app. Everything that
  comes from data goes through `rich.markup.escape` or `Text`, and
  `notify(..., markup=False)` is used for it too. Rich `Table` cells also don't
  understand Textual's `$warning` variables. Use plain Rich colours there.
- **Rich styles can't hold alpha.** `$boost` and `$accent 40%` turned the cursor
  row solid white. Use opaque theme colours (`$panel`, `$accent-darken-3`).
- **`TabbedContent` follows focus.** If the focused widget ends up in a hidden
  pane, that pane comes back to the front. Pressing `p` would sometimes land on
  the Columns tab. Every tab now gives focus to a known widget
  (`App.TAB_FOCUS`), checked again after refresh.
- **A focused `Select` eats digit keys** (type-to-search), so `1`–`4` stopped
  switching tabs on the Plot tab. That tab focuses the controls container
  instead.
- **Widgets are zero-sized behind the loading indicator.** Laying out columns
  against width 0 scrolled the grid to the last column after an expression
  changed the columns. `_ensure_visible` skips while hidden, and `on_resize`
  runs it again.
- **A widget can't take focus while `loading` is set.** The Columns tab
  focuses its grid again when the stats arrive.
- **Focus after a modal.** After the load dialog closed, focus didn't return to
  the grid, so keys went nowhere. `_loaded` focuses the current tab.
- Textual 8 renamed `Select.BLANK` to `Select.NULL`. Messages used with
  `@on(..., "#id")` need a `control` property.
- numpy 2 scalars repr as `np.float64(9.9)`. Expression results strip the
  wrapper.
- **Line plots of sampled parquet look broken.** The sample is 16 contiguous
  slices, so a time-series line jumps between them. That's honest, but for the
  README the plots scene loads a filtered subset that covers the whole year.

## Testing and the GIFs

- `tests/test_tui.py` drives the app headlessly (`App.run_test`). Wait for
  `app.busy` and the workers to finish. `DirectoryTree` keeps a worker running,
  so an "all workers idle" wait never ends while the file picker is open.
- The tests and the GIF recorder point `dxtui.app.HISTORY_FILE` at a temporary
  file. Otherwise they write to the user's expression history.
- `tools/record_gifs.py`: `app.export_screenshot()` gives an SVG. It is put in a
  page with a caption strip and screenshotted with Playwright (system Chrome),
  and Pillow builds the GIF with one palette shared across frames (about
  0.2–1 MB per GIF). Long waits are shortened in the GIF; the captions quote
  the real times from the app.
- Check new frames by building a contact sheet of the long-held frames.
  Problems that are invisible in the code (focus bugs, layout at 120×32) show
  up straight away.
