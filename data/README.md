# Data folder

Put monthly data files here. Files are ignored by git (only this README and `.gitkeep` are tracked), so datasets stay local.

## Naming convention

The reporting period is encoded in the filename as `YY-MM`, followed by a space and the dataset name:

```text
25-01 pra101.parquet
25-02 pra101.parquet
25-03 pra101.parquet
```

`hailer.periods` reads this convention:

- `parse_period("25-03 pra101.parquet")` → `Period(2025, 3)` (label `2025-03`)
- `scan_period_files(DATA_DIR, "pra101")` → the files for one dataset, sorted by period
- `load_periods(files)` → one Polars DataFrame with a leading `period` column; columns that only exist in later months are filled with nulls for earlier ones
- `duckdb_periods_view(con, files)` → a DuckDB view `periods` over all files (`union_by_name`), with `period` derived from the filename

The parquet data itself is not expected to contain a period column.

## Synthetic example

```bash
uv run python scripts/make_sample_data.py
```

writes six months of PRA101-style data (`counterparty_id`, `exposure_class`, `country`, `default_flag`, `exposure_value`, `rwa`; from the fourth month also `risk_weight`).
