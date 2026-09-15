Produce the monthly PRA101 pack for period {{args}} (use the latest period if none is given).

In the marimo notebook, add or update these sections, reusing existing variables and cells
where they already exist:

1. **Periods available**: a small table of the months found in `data/` and their row counts.
2. **Headline**: total RWA and exposure value for the period and the previous month, with
   movement in £m and %.
3. **Movement by exposure class**: table sorted by absolute movement, plus a bar chart.
4. **Top 10 movers**: counterparties with the largest absolute RWA movement between the two
   months (join on `counterparty_id`), showing both months and the movement.
5. **Data quality notes**: rows with null `exposure_class`, negative `rwa`, or
   `exposure_value == 0`, counted per month.

Then reply in the terminal with a five-line summary: total movement, the largest driver, the
second-largest driver, anything unusual in the data-quality notes, and what you added to the
notebook. Do not paste tables into the terminal.
