---
name: sales-month-on-month
description: >-
  Compare a month of sales (the latest by default) with the month before and explain what
  drives the revenue movement, producing a movement table, a breakdown by region or category
  and a chart in a marimo notebook that re-runs for any month. Use when the user asks what
  changed, what is driving revenue, or for a month-on-month comparison.
---

# Sales month-on-month movement

## When to use

The user asks any of: "compare the latest two months", "what's driving the revenue increase",
"break the movement down by region", "show me the biggest movers".

## Steps

1. `list_periods` to confirm which months exist and whether columns differ between them.
2. In the notebook, reuse the cells a previous turn created (`notebook_cells` shows what exists).
   Otherwise create the settings first:
   - a `settings` cell with the sales periods (`scan_period_files(DATA_DIR, "sales")`), the
     breakdown column (`region` unless the user asks for another, such as `category`) and a
     `period_picker` dropdown over every period but the first, defaulting to
     `mo.cli_args().get("period")` or the latest period;
   - a `report_period` cell that reads `period_picker.value` into `report_period` and the period
     before it into `previous_period`.
3. Create one cell `sales` that loads those two periods with `load_periods` and keeps the
   `period` column.
4. Create (or edit) a cell `mom` that aggregates `revenue` and `units` by `period` and the
   breakdown column, pivots the two periods side by side as `previous` and `current` (named for
   their role, not their month, so the cell works for any period), and adds `movement` and
   `movement_pct` columns. Sort by absolute movement, descending.
5. Create a `headline` cell with the numbers the summary quotes (`total_movement`,
   `total_movement_pct`, `largest_driver`) and a `summary` cell above the table that is an
   f-string over them: the direction word ("rose" or "fell") and the driver are computed too.
   Then a cell that shows `mom` with `mo.ui.table`, and a bar chart of `movement` by the
   breakdown column whose title names `report_period`.
6. If the user says "exclude returns", filter `returned == False` in the `sales` cell and let
   the dependent cells re-run; do not create a second copy of the pipeline. If they want the
   same comparison for another month, switch `period_picker` rather than copying cells.
7. Run the checks in `reference/checks.md` before summarising.
8. In the terminal, give a two-sentence summary (largest driver, total movement in £k and %),
   then list what was added to the notebook.

## Notebook hygiene

- One owning cell per public name (`settings`, `sales`, `mom`, `headline`, `chart`); edit the
  owner, never redefine.
- No month, number or file name typed into the analysis cells or the summary text; they come
  from the settings and the data, so the notebook re-runs next month.
- Use `_` prefixed names for intermediates.
- Keep `hide_code=False` for analysis cells so the user can read the code.
