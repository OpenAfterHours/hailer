# Sales analysis: team context

> This file is sent to the model endpoint at the start of every Hailer session.
> Keep it factual and short. No credentials, no customer names, no row-level data.
> It describes the sample data from `scripts/make_sample_data.py`; replace it with what your own
> data means.

## What the data is

- Monthly order extracts, one Parquet file per month in `data/`.
- Filenames encode the period as `YY-MM`, e.g. `25-03 sales.parquet` is March 2025.
  The data itself has **no** period column; Hailer adds one when loading several months.
- Later months may add columns. Treat a missing column in an older month as null, not an error.

## Column meanings

| column | meaning |
|---|---|
| `order_id` | Order key, unique within a month |
| `order_date` | Date the order was placed |
| `region` | Sales region: North, South, East, West |
| `category` | Product category: Electronics, Home, Clothing, Sports, Books |
| `channel` | `Online` or `Store` |
| `units` | Items in the order |
| `revenue` | Net revenue in GBP, after any discount |
| `returned` | `true` when the order was returned. "Exclude returns" means `returned == false` |
| `discount` | Discount rate applied (0.1 = 10%); present from 2025-04 onwards |

## Conventions

- "Latest month" means the most recent period present in `data/`, not the calendar month.
- Month-on-month movement is `latest - previous`; percentages are relative to the previous month.
- Report revenue in £k with one decimal (e.g. £41.1k). Keep tables to the top 10 rows unless asked.
- Prefer Polars; use DuckDB SQL when joining across many files.
- Charts: bar for breakdowns by region or category, line for trends over time. Label axes with units.
