# PRA101 analysis: team context

> This file is sent to the model endpoint at the start of every Hailer session.
> Keep it factual and short. No credentials, no customer names, no row-level data.

## What the data is

- Monthly extracts of the PRA101 (capital+) return, one Parquet file per month in `data/`.
- Filenames encode the period as `YY-MM`, e.g. `25-03 pra101.parquet` is March 2025.
  The data itself has **no** period column; Hailer adds one when loading several months.
- Later months may add columns. Treat a missing column in an older month as null, not an error.

## Column meanings

| column | meaning |
|---|---|
| `exposure_class` | Regulatory exposure class: Corporate, Retail, Institution, Sovereign, Equity, Other |
| `counterparty_id` | Anonymised counterparty key; stable across months |
| `default_flag` | `true` when the exposure is in default. "Exclude defaults" means `default_flag == false` |
| `exposure_value` | Exposure at default in GBP |
| `rwa` | Risk-weighted assets in GBP |
| `risk_weight` | Effective risk weight (`rwa / exposure_value`); present from 2025-04 onwards |

## Conventions

- "Latest month" means the most recent period present in `data/`, not the calendar month.
- Month-on-month movement is `latest - previous`; percentages are relative to the previous month.
- Report RWA in £m with one decimal (e.g. £1,234.5m). Keep tables to the top 10 rows unless asked.
- Prefer Polars; use DuckDB SQL when joining across many monthly files.
- Charts: bar for breakdowns by class, line for trends over periods. Label axes with units.
