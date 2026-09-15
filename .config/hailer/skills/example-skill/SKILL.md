---
name: pra101-month-on-month
description: >-
  Compare the two latest PRA101 months and explain what drives the RWA movement,
  producing a movement table, an exposure-class breakdown and a chart in the marimo
  notebook. Use when the user asks what changed, what is driving RWA, or for a
  month-on-month comparison.
---

# PRA101 month-on-month movement

## When to use

The user asks any of: "compare the latest two months", "what's driving the RWA increase",
"break the movement down by exposure class", "show me the biggest movers".

## Steps

1. `list_periods` to confirm which months exist and whether columns differ between them.
2. In the notebook, reuse the existing `files`/`df` variables if a previous turn created them
   (`notebook_cells` shows what exists). Otherwise create one cell that loads all months with
   `load_periods(scan_period_files(DATA_DIR, "pra101"))` and keeps the `period` column.
3. Create (or edit) a cell `mom` that aggregates `rwa` and `exposure_value` by `period` and
   `exposure_class`, pivots the latest two periods side by side, and adds `movement` and
   `movement_pct` columns. Sort by absolute movement, descending.
4. Create a cell that shows `mom` with `mo.ui.table`, and a cell with a bar chart of
   `movement` by `exposure_class` (Altair via `mo.ui.altair_chart`, or `df.plot`).
5. If the user says "exclude defaults", filter `default_flag == False` in the loading cell and
   let the dependent cells re-run; do not create a second copy of the pipeline.
6. Run the checks in `reference/checks.md` before summarising.
7. In the terminal, give a two-sentence summary (largest driver, total movement in £m and %),
   then list what was added to the notebook.

## Notebook hygiene

- One owning cell per public name (`files`, `df`, `mom`, `chart`); edit the owner, never redefine.
- Use `_` prefixed names for intermediates.
- Keep `hide_code=False` for analysis cells so the user can read the code.
