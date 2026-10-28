# Notebooks and sessions {#notebooks-and-sessions}

`uvx hailer init` creates the starter notebook (`[hailer].notebook`) and the data folder (`[hailer].data_dir`)
that `hailer.toml` names. It never overwrites anything except `hailer.toml` itself, and only with
`--force`, so running it again just fills in what is missing (a deleted notebook, say). Put the files you
want to analyse in `data/`: CSV, Parquet, JSON or Excel, with any names (see [Data conventions](data.md#data-conventions)).
The notebook lists them when it opens, and you can start asking straight away.

`uvx hailer notebook` runs the startup checks, starts a marimo server on the **notebooks folder** in the
background, opens the active notebook's URL so the kernel gets a session, then runs the chat in the same
terminal. One server hosts every notebook in the folder, and marimo's own home page (the server URL
without `?file=`) lists them all. When you leave the chat (`/exit`, Ctrl+Z Enter, or Ctrl+C at the
prompt) it stops the marimo server it started.

- **One server per session.** Every `uvx hailer` or `uvx hailer notebook` starts its own server and stops
  it when the chat ends. Hailer never looks for, or attaches to, a server another session started, so
  several sessions can run side by side in one workspace (each on its own port). Restarting the chat
  re-runs the notebook in a fresh kernel.
- **The server needs a token.** Hailer gives every server it starts a random token and keeps it in
  memory, so other programs on the machine cannot send code to the kernel. The link Hailer opens or prints
  carries `access_token=...`, which signs the browser in. Links the agent quotes in the chat leave the token
  out, because tool results go to the model endpoint; `/notebook` prints the signed-in link.
- **Log.** The kernel's log is `docker logs hailer-kernel-<id>-<suffix>`. When a start fails, its last
  lines are in the error message; Hailer masks the token in them. With `unsafe-local`, marimo's output goes
  to `.hailer/marimo-<pid>.log`, one per session, deleted when the kernel stops.
- **If Hailer is killed.** Docker kernels left behind are removed by the next start or
  `uvx hailer kernel stop`. Nothing records an unsafe-local kernel: its marimo keeps running (other programs
  still need its token) until you end the `python -m marimo` process with Task Manager or `kill`.
- **Where the code runs.** By default the kernel runs in an isolated Docker container (see
  [Isolated kernel (Docker)](../security/docker.md#isolated-kernel-docker)). `unsafe-local` runs it in
  Hailer's own Python, as you (see [Kernel runtimes](../security/runtimes.md#kernel-runtimes)).
- **Docker works on copies.** A docker kernel has a notebooks folder of its own. Hailer copies your
  notebooks into it when it starts and copies the ones that changed back after every turn, every 15
  seconds and when the session ends; only marimo notebooks with plain names travel, never other files
  (see [Notebook copies](../security/docker.md#notebook-copies)). If you edit a notebook in another editor
  while a docker session runs, and the kernel changes it too, your version is kept in
  `.hailer/notebook-backups/`. Files notebook code writes next to the notebooks (an export, say) stay in
  the container and are gone when it stops.

The notebook opens in marimo's **app view**: you see the results, tables and charts the agent produces,
not the code behind them (the URL carries `view-as=present`). To see or edit the code, press `Ctrl+.`
(`Cmd+.` on macOS) in the notebook, or choose *Toggle app view* from the notebook menu; the same shortcut
switches back. It is a normal edit session either way, so the agent works in it exactly the same.

Flags: `--port N` (default 2718; a free port is chosen when it is busy), `--no-browser` (print the URL
instead of opening it), `--foreground` (just run marimo attached to this terminal, no chat, until Ctrl+C; it
opens marimo's home page unless `--no-browser` is given), `--new` (start a fresh conversation), `--kernel docker|unsafe-local`
(where notebook code runs for this run; the default comes from `[kernel] runtime`, which defaults to `docker`).

Bare `uvx hailer` does the same as `uvx hailer notebook` with its defaults: it starts its own marimo
server, opens the notebook, chats, and stops the server when the chat ends:

```bash
uvx hailer
```

To run code in the kernel yourself, use `/exec <code>` in the chat: it runs in the active notebook's
kernel session and prints the output, and nothing is sent to the model. The code may start on the line
after `/exec`; an indented pasted block is dedented first.

## Working with several notebooks

One analysis, one notebook: the agent can create a fresh notebook or go back to an earlier one, and you
can do the same with slash commands. Everything happens on the one marimo server started on the notebooks
folder, so nothing restarts.

In the chat, just say it:

```text
You > start a new notebook for the Q2 churn review and load the churn files
You > open the regional sales notebook we did last week and add a chart by category
You > which notebooks do we have?
```

The agent lists the folder, creates `q2_churn_review.py` in it (the name is slugified) or opens the
existing file, and the browser tab appears by itself. Notebooks are named by their path inside the
notebooks folder (`q2_churn_review.py`, `q3/review.py`). Whatever the agent switches to becomes the
**active notebook**: the one every later cell edit, `/status` line and `/exec` call refers to. The chat
shows `Active notebook is now q2_churn_review.py.` when the agent switched.

The same from the prompt, without a model round-trip:

| Command | What it does |
|---|---|
| `/notebook` | Active notebook, folder, marimo state and the signed-in URL. |
| `/notebook list` | Every notebook in the folder with `active` and `open` (has a kernel session) markers. |
| `/notebook new <name> [--empty]` | Create `<slug>.py` from the starter template (or an empty marimo notebook with `--empty`), open it, make it active. |
| `/notebook open <name>` | Switch to an existing notebook by name, filename or path; opens it in the browser when it has no session. |
| `/notebook close [name]` | Shut down that notebook's kernel session (the browser tab disconnects); it can be reopened any time. |

After a slash-command switch the next message you send carries a one-line notice such as
`[Hailer] The active notebook is now q2_churn_review.py (reopened, 7 cells). Call notebook_cells
before editing.` so the agent inspects the notebook before touching it. The conversation itself continues;
use `/new` if you want a clean thread as well.

Where things live:

- Notebooks go in `[hailer].notebooks_dir` (default: the folder of `[hailer].notebook`, i.e. `notebooks/`).
  Only files in that folder can be opened; names are matched case-insensitively and `q2 churn`,
  `q2_churn`, `q2_churn.py` and `notebooks/q2_churn.py` all mean the same file. The agent and the chat
  reach notebook files only through the kernel (marimo's own file API), never by reading the folder
  themselves, so the same commands work wherever the kernel runs.
- The **starter template** is the same set of cells as `notebooks/analysis.py` (imports, the
  `hailer.periods` helpers, `WORKSPACE`, `DATA_DIR`, `data_files`, `period_files`, a welcome cell with the
  notebook's title and the data files it found), so the agent can start analysing straight away. `--empty` gives marimo's plain empty notebook.
- The active notebook is remembered by name in `.hailer/notebook.json` (git-ignored, next to
  `session.json`), so the next `uvx hailer` or `uvx hailer notebook` resumes where you left off; if that
  notebook is gone, the session starts on `[hailer].notebook`. Delete the file to go back to
  `[hailer].notebook`.
- `/notebook` also lists the notebooks you worked in recently (`Recent:`), most recent first.

```toml
[hailer]
notebook      = "notebooks/analysis.py"   # the default (and first) notebook
notebooks_dir = "notebooks"               # where /notebook new and the agent's notebook_create put files
```

## Re-running a notebook for a new period {#re-running}

Hailer writes an analysis so that you can run it again on other data without editing it: next month's
report, or a dashboard you keep open. Three habits make that work, and you will see them in the code:

- **A settings cell.** Every choice the analysis depends on (the file, the reporting period, filters,
  top-N) sits in one cell named `settings`, above the analysis. Its defaults come from the data: the
  reporting period is the latest one in the data folder unless you pick another. No other cell has a
  month, file name or threshold typed into it.
- **Commentary from the numbers.** A summary is an f-string over values the notebook works out, such as
  `total_revenue`, `revenue_change_pct` and `top_region`, so the text changes with the figures, including
  words like *rose* or *fell* and which region leads. Interpretation the data cannot give (a cause you
  told Hailer about, say) is labelled with the period it was written for, and shows a warning when you
  look at another period. A check backs this up: see [Formatting and code checks](#code-checks).
- **Simple code.** One step per cell, named after what it works out, in plain Polars, without helper
  functions or defensive code the data does not need, so you can read the notebook from top to bottom.

To see another period, pick it in the **Reporting period** dropdown at the top of the notebook, or ask in
the chat:

```text
You > show the same for March
```

Every cell below the settings recalculates; Hailer switches the dropdown rather than copying cells. When
next month's file arrives (`25-07 sales.parquet`, say), put it in the data folder and open the notebook:
the dropdown now includes July and opens on it.

To make an older notebook re-runnable, ask for it; Hailer finds the typed-in values with its checks:

```text
You > make this notebook re-runnable for other months
```

The settings cell also reads a period from marimo's command line (`-- --period 2025-04`). Running a
notebook with marimo yourself, outside Hailer, runs its code on your machine as you, not in the Docker
kernel; see [Kernel runtimes](../security/runtimes.md#kernel-runtimes).

## Formatting and code checks {#code-checks}

The code the agent writes into your notebook is formatted and checked as it goes in:

- **Formatting.** New and edited cells are formatted with [ruff](https://docs.astral.sh/ruff/) before
  marimo runs them, using marimo's line length (`formatting.line_length` in your marimo settings, 79 by
  default). Hailer switches this on in the running kernel only; your marimo settings file is not changed,
  and cells you edit yourself in the browser follow your own *format on save* setting as before.
- **Checks.** After each call that creates or edits cells, Hailer checks the changed cells with ruff (lint)
  and [ty](https://docs.astral.sh/ty/) (types) and hands the findings back to the agent, which fixes the ones
  that are real mistakes before it carries on: an undefined or misspelt name, a method the dataframe does
  not have, a wrong argument, a mutable default argument. ty sees the whole notebook in the order marimo
  runs it, so it knows that `sales` from another cell is a Polars `DataFrame`.
- **Typed-in values.** Hailer's own check looks at the text a reader sees, `mo.md(...)` and chart titles,
  for numbers, dates and month names typed in rather than formatted from a variable: `"Revenue rose 5.6%
  in March"` is reported, `f"Revenue rose {change_pct:.1%} in {report_period}"` is not. The agent rewrites
  such text so that it follows the data when the notebook is [re-run](#re-running). Numbered lists and
  names such as `Q2` are left alone.

Ask the agent to check the whole notebook at any time (it has a `notebook_check` tool for that):

```text
You > check the notebook for mistakes
```

Findings about what cells do on purpose are left out: an import meant for a later cell, top-level `await`,
a cell whose last line is just a value to display, a missing package (the cell fails when it runs anyway).
ruff runs with Hailer's own small rule set, so a `ruff.toml` or `pyproject.toml` in the workspace does not
change the checks. ruff and ty come with Hailer, and all three checks run on this machine, whichever
kernel runtime you use.
The checks usually take a fraction of a second and do not change the notebook; the agent fixes findings through
ordinary cell edits.

To switch parts off, add a `[checks]` section to `hailer.toml`:

```toml
[checks]
lint      = true    # ruff
typecheck = true    # ty
hardcoded = true    # numbers, dates and months typed into notebook text
format    = false   # leave the agent's cells as it wrote them
```
