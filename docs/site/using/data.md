# Working with your data {#data-conventions}

Any CSV, Parquet, JSON or Excel file in `data/` can be analysed, whatever it is called: the starter notebook lists
them in `data_files`, and the agent loads them with Polars or queries them with DuckDB when you ask.

Excel reading is included in both the Docker kernel and Hailer's own installation (the unsafe-local runtime) through
[Polars' Calamine engine (`fastexcel`)](https://docs.pola.rs/api/python/stable/reference/api/polars.read_excel.html).
Put an `.xlsx`, `.xls` or `.xlsb` workbook in `data/` and ask, for example, *Explore the Sales worksheet in
sales.xlsx and chart revenue by region*. The agent can inspect worksheet names and load the relevant sheet:

```python
sales = pl.read_excel(DATA_DIR / "sales.xlsx", sheet_name="Sales")
```

`pl.read_excel(path, sheet_id=0)` reads all sheets into a dictionary keyed by worksheet name. Check
inferred column types before analysing; `schema_overrides` can keep identifiers as text. Formula cells
use results saved in the workbook; reading does not recalculate formulas.

One naming convention is optional. When the same dataset arrives every month, name the files
`YY-MM <dataset>.parquet`, for example `25-03 sales.parquet` for March 2025. The data itself then needs no
period column; Hailer derives it from the filename, combines the months into one table and copes with
columns that only appear in later months.

`hailer.periods` (imported by the starter notebook, so the agent reuses it):

| Function | Purpose |
|---|---|
| `list_data_files(data_dir)` | Every data file (CSV, Parquet, JSON, Excel, ...) directly in the folder, whatever its name, sorted by name. |
| `parse_period(name)` | `"25-03 sales.parquet"` → `Period(2025, 3)`; `None` if the name does not match. |
| `scan_period_files(data_dir, name=None)` | Sorted `PeriodFile`s for one dataset (or all) in a folder. |
| `load_periods(files, columns=None)` | One Polars DataFrame with a leading `period` column (`YYYY-MM`); schema evolution handled with a relaxed diagonal concat. |
| `scan_periods(files)` | Lazy variant of `load_periods`. |
| `duckdb_periods_view(con, files, view_name="periods")` | Registers a DuckDB view over all files (`union_by_name`) with `period` derived from the filename. |
| `describe_periods(files)` | Compact text: months found, common columns, columns present only in some months. |

A monthly file that cannot be read raises `MalformedParquetError` naming the file. The starter notebook
(`notebooks/analysis.py`) defines `WORKSPACE`, `DATA_DIR`, `data_files` and `period_files` and shows a
table of the data files it found.
