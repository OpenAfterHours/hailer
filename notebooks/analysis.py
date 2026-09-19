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
    # Hailer's data helpers: list_data_files finds every data file; the others handle monthly
    # files named "YY-MM <dataset>.parquet" (schema-tolerant loading with a period column).
    from hailer.periods import (
        describe_periods,
        duckdb_periods_view,
        list_data_files,
        load_periods,
        scan_period_files,
        scan_periods,
    )

    return (
        describe_periods,
        duckdb_periods_view,
        list_data_files,
        load_periods,
        scan_period_files,
        scan_periods,
    )


@app.cell
def _(Path, mo):
    # Workspace layout: this notebook lives in <workspace>/notebooks/, data in <workspace>/data/.
    _notebook_dir = mo.notebook_dir()
    WORKSPACE = (_notebook_dir.parent if _notebook_dir is not None else Path.cwd()).resolve()
    DATA_DIR = WORKSPACE / "data"
    return DATA_DIR, WORKSPACE


@app.cell
def _(DATA_DIR, list_data_files, scan_period_files):
    data_files = list_data_files(DATA_DIR)  # CSV, Parquet, JSON, Excel ... with any name
    period_files = scan_period_files(DATA_DIR)  # the monthly "YY-MM <dataset>.parquet" ones
    return data_files, period_files


@app.cell
def _(DATA_DIR, WORKSPACE, data_files, describe_periods, mo, period_files, pl):
    # Welcome / status cell. Hailer adds analysis cells below this one.
    _header = mo.md(
        f"""
    # Hailer workspace

    Chat with your data in the terminal (`uv run hailer`): ask a question in plain English and
    Hailer loads the files, explores them and puts the tables, charts and summaries here.

    **Workspace:** `{WORKSPACE}`

    **Data:** `{DATA_DIR}`
    """
    )
    _parts = [_header]
    if data_files:
        _periods = {pf.path.name: pf.period.label for pf in period_files}
        _columns = {
            "file": [f.name for f in data_files],
            "type": [f.suffix.lstrip(".").lower() for f in data_files],
            "size_kb": [round(f.stat().st_size / 1024, 1) for f in data_files],
        }
        if _periods:
            _columns["period"] = [_periods.get(f.name, "") for f in data_files]
        _parts.append(mo.ui.table(pl.DataFrame(_columns), selection=None, label="Data files"))
        _parts.append(
            mo.md(
                "Try asking: *what is in these files?* · *show the ten largest values* · "
                "*chart the totals by month*"
            )
        )
        if period_files:
            _parts.append(mo.md("**Monthly files**\n\n```text\n" + describe_periods(period_files) + "\n```"))
    else:
        _parts.append(
            mo.callout(
                mo.md(
                    "No data files yet. Put the files you want to analyse (CSV, Parquet, JSON or Excel, any "
                    "name) in the data folder above, or run `uv run python scripts/make_sample_data.py` "
                    "for six months of synthetic sales data."
                ),
                kind="info",
            )
        )
    welcome = mo.vstack(_parts)
    welcome
    return (welcome,)


if __name__ == "__main__":
    app.run()
