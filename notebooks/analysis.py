import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import duckdb
    from pathlib import Path

    return Path, duckdb, mo, pl


@app.cell
def _():
    # Hailer's period utilities (YY-MM filename convention, schema-tolerant loading).
    from hailer.periods import (
        describe_periods,
        duckdb_periods_view,
        load_periods,
        scan_period_files,
        scan_periods,
    )

    return describe_periods, duckdb_periods_view, load_periods, scan_period_files, scan_periods


@app.cell
def _(Path, mo):
    # Workspace layout: this notebook lives in <workspace>/notebooks/, data in <workspace>/data/.
    _notebook_dir = mo.notebook_dir()
    WORKSPACE = (_notebook_dir.parent if _notebook_dir is not None else Path.cwd()).resolve()
    DATA_DIR = WORKSPACE / "data"
    return DATA_DIR, WORKSPACE


@app.cell
def _(DATA_DIR, scan_period_files):
    period_files = scan_period_files(DATA_DIR)
    return (period_files,)


@app.cell
def _(DATA_DIR, WORKSPACE, describe_periods, mo, period_files, pl):
    # Welcome / status cell. Hailer adds analysis cells below this one.
    _header = mo.md(
        f"""
    # Hailer workspace

    **Workspace:** `{WORKSPACE}`
    **Data:** `{DATA_DIR}`

    Chat in the terminal (`uv run hailer`); results, tables and charts appear here.
    """
    )
    if period_files:
        _table = mo.ui.table(
            pl.DataFrame(
                {
                    "file": [f.path.name for f in period_files],
                    "dataset": [f.stem for f in period_files],
                    "period": [f.period.label for f in period_files],
                    "size_kb": [round(f.path.stat().st_size / 1024, 1) for f in period_files],
                }
            ),
            selection=None,
            label="Period files",
        )
        _summary = mo.md("```text\n" + describe_periods(period_files) + "\n```")
        welcome = mo.vstack([_header, _table, _summary])
    else:
        welcome = mo.vstack(
            [
                _header,
                mo.callout(
                    mo.md(
                        "No period files found. Add files named like `25-01 pra101.parquet` to the data folder, "
                        "or run `uv run python scripts/make_sample_data.py` for a synthetic example."
                    ),
                    kind="info",
                ),
            ]
        )
    welcome
    return (welcome,)


if __name__ == "__main__":
    app.run()
