Take a first look at {{args}} (every data file in `data/` if nothing is named).

In the marimo notebook, add one section per dataset, reusing existing variables and cells
where they already exist:

1. **Overview**: rows and columns, and the first rows in a `mo.ui.table`.
2. **Columns**: type, null count, distinct values and min / max (or the most common values)
   for each column.
3. **Charts**: the distribution of the main numeric columns, the largest categories as a bar
   chart, and a line chart over time when there is a date or period column.
4. **Things to check**: duplicate rows, mostly empty columns, and values that look wrong
   (negative amounts, dates in the future, numbers stored as text).

Then reply in the terminal with at most five lines: what the data is, how big it is, the most
interesting pattern in the charts, anything that looks wrong, and what you added to the
notebook. Do not paste tables into the terminal.
