# Everyday workflow

Hailer has two places to work: chat in your terminal, and a live marimo notebook in your browser.
Use the conversation to explore and refine an analysis. Use the notebook to inspect the results and Python code.

## Start with a question

Run `uvx hailer` from your analysis folder. It opens the active notebook and resumes your conversation.
Keep the notebook tab open while you work; that browser session lets the notebook execute code.

For the [quickstart's sample data](../getting-started/quickstart.md), ask:

```text
What data do I have? Summarise the columns and any missing values.
Chart total revenue by month, split by region.
```

Tables and charts appear in the notebook. You can follow up without repeating the dataset or the entire question:

```text
How much did total revenue grow from January to February?
Add a short summary below the chart.
```

With the supplied sample, January totals 2,100 and February totals 2,600: growth of 500, or 23.8%.
Responses and presentation can vary with the model you configure.

## Inspect the work

The notebook opens in marimo's app view, which presents the results. Press **Ctrl+.** (**Cmd+.** on macOS),
or choose **Toggle app view** in the notebook menu, to see and edit the code. Use the same shortcut to return to the results.

Durable tables, charts and summaries live in the notebook's Python cells. Hailer also uses temporary
scratchpad calculations while exploring; those are not automatically saved as notebook cells. Ask it
to add a calculation or explanation to the notebook when you want to keep it.

## Keep going, or come back later

Type `/exit` to finish. Run `uvx hailer` from the same folder to resume your conversation and active notebook.
`/new` starts a fresh conversation; it does not clear your notebook.

For a different analysis, ask Hailer to create a notebook or use:

```text
/notebook new regional-sales
/notebook list
/notebook open analysis
```

See [notebooks and sessions](notebooks.md) for switching, reopening and managing several notebooks.

## Useful shortcuts

| Action | Shortcut or command |
|---|---|
| Send a message | Enter |
| Insert a newline | Alt+Enter |
| Cancel the current turn | Ctrl+C |
| See chat commands | `/help` |
| See the active model and notebook | `/status` |
| Show the notebook link | `/notebook` |
| Diagnose startup problems | `uvx hailer doctor` in your terminal |

Cancelling a turn does not undo completed actions. Python already running in the notebook may continue;
interrupt or restart the kernel in marimo if needed. See the [command reference](../reference/commands.md)
for the full interaction details and [data handling](../security/data-handling.md) for what reaches the model endpoint.
