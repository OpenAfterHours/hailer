# Reconciliation checks for a month-on-month comparison

Run these in the scratchpad (not as notebook cells) before summarising. Print only the
check name and pass/fail with the numbers involved.

1. **Totals reconcile.** Sum of `movement` across exposure classes equals
   `total_rwa(latest) - total_rwa(previous)` to within £1.
2. **No period leakage.** The `period` column only contains the two periods being compared.
3. **Class coverage.** Every `exposure_class` present in either month appears in the table;
   a class missing from one month shows 0 for that month, not null.
4. **Defaults handled consistently.** If defaults are excluded, both months are filtered with
   the same predicate (`default_flag == False`); state the number of rows excluded per month.
5. **Schema evolution.** If a column exists only in the latest month (e.g. `risk_weight`),
   do not use it in the comparison unless the user asks; mention it once in the summary.
6. **Sign convention.** `movement > 0` means RWA increased in the latest month.
