# Data folder

Put the files you want to analyse here: CSV, Parquet or JSON files, with any names. The starter
notebook lists them, and the agent loads them with Polars or DuckDB when you ask about them. Files are ignored by git (only this README and `.gitkeep` are tracked),
so datasets stay local.

## Monthly files (optional)

If the same dataset arrives every month, name the files with the period first, as `YY-MM`, followed by
a space and the dataset name:

```text
25-01 sales.parquet
25-02 sales.parquet
25-03 sales.parquet
```

`hailer.periods` reads this convention:

- `parse_period("25-03 sales.parquet")` → `Period(2025, 3)` (label `2025-03`)
- `scan_period_files(DATA_DIR, "sales")` → the files for one dataset, sorted by period
- `load_periods(files)` → one Polars DataFrame with a leading `period` column; columns that only exist in later months are filled with nulls for earlier ones
- `duckdb_periods_view(con, files)` → a DuckDB view `periods` over all files (`union_by_name`), with `period` derived from the filename

The parquet data itself is not expected to contain a period column.

## Synthetic example

```bash
uv run python scripts/make_sample_data.py
```

writes six months of sample sales orders (`order_id`, `order_date`, `region`, `category`, `channel`, `units`, `revenue`, `returned`; from the fourth month also `discount`).
