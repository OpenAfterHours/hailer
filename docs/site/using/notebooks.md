# Notebooks and sessions {#notebooks-and-sessions}

`hailer init` creates the starter notebook (`[hailer].notebook`) and the data folder (`[hailer].data_dir`)
that `hailer.toml` names. It never overwrites anything except `hailer.toml` itself, and only with
`--force`, so running it again just fills in what is missing (a deleted notebook, say). Put the files you
want to analyse in `data/`: CSV, Parquet, JSON or Excel, with any names (see [Data conventions](data.md#data-conventions)).
The notebook lists them when it opens, and you can start asking straight away.

`hailer notebook` runs the startup checks, starts a marimo server on the **notebooks folder** in the
background, opens the active notebook's URL so the kernel gets a session, then runs the chat in the same
terminal. One server hosts every notebook in the folder, and marimo's own home page (the server URL
without `?file=`) lists them all. When you leave the chat (`/exit`, Ctrl+Z Enter, or Ctrl+C at the
prompt) it stops the marimo server it started.

- **The server needs a token.** Hailer gives every server it starts a random token and records it in
  `.hailer/kernel.json`, so other programs on the machine cannot send code to the kernel. The link Hailer
  opens or prints carries `access_token=...`, which signs the browser in. Links the agent quotes in the
  chat leave the token out, because tool results go to the model endpoint; `/notebook` prints the signed-in
  link.
- **Reuse.** A server Hailer started for this workspace and left running (`--keep-marimo`, or `--foreground`
  in another terminal) is reused and left running. So is a marimo server you started yourself with a
  notebook from the folder open (local runtime only).
- **Log.** marimo's output goes to `.hailer/marimo.log`, emptied at each start. Hailer masks the token in
  the lines it prints from it. In docker mode the log is `docker logs hailer-kernel-<id>`.
- **Where the code runs.** By default the kernel runs in Hailer's own Python, as you. To run it in an
  isolated container instead, add `--kernel docker` or see [Isolated kernel (Docker)](../security/docker.md#isolated-kernel-docker).

The notebook opens in marimo's **app view**: you see the results, tables and charts the agent produces,
not the code behind them (the URL carries `view-as=present`). To see or edit the code, press `Ctrl+.`
(`Cmd+.` on macOS) in the notebook, or choose *Toggle app view* from the notebook menu; the same shortcut
switches back. It is a normal edit session either way, so the agent works in it exactly the same.

Flags: `--port N` (default 2718; a free port is chosen when it is busy), `--no-browser` (print the URL
instead of opening it), `--keep-marimo` (leave the server running after the chat; `uvx hailer kernel stop`
stops it later), `--foreground` (just run marimo attached to this terminal, no chat; it opens marimo's home
page unless `--no-browser` is given), `--new` (start a fresh conversation), `--kernel local|docker` (where
notebook code runs for this run; the default comes from `[kernel] runtime`).

Bare `uvx hailer` does the same as `uvx hailer notebook` with its defaults: it reuses this workspace's
marimo server or starts one, opens the notebook, chats, and stops only a server it started. A kept server
is found through `.hailer/kernel.json`:

```bash
uvx hailer
```

A pinned server (`[hailer].marimo_url` in `hailer.toml`, or `HAILER_MARIMO_URL`) is used as is, whichever
folder it serves. If it does not answer, Hailer says so and exits; it never starts a server in its place.

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

The agent lists the folder, creates `notebooks/q2_churn_review.py` (the name is slugified) or opens the
existing file, and the browser tab appears by itself. Whatever it switches to becomes the **active
notebook**: the one every later cell edit, `/status` line and `hailer exec` call refers to. The chat shows
`Active notebook is now notebooks/q2_churn_review.py.` when the agent switched.

The same from the prompt, without a model round-trip:

| Command | What it does |
|---|---|
| `/notebook` | Active notebook, folder, marimo state, the signed-in URL and the launch command. |
| `/notebook list` | Every notebook in the folder with `active` and `open` (has a kernel session) markers. |
| `/notebook new <name> [--empty]` | Create `<slug>.py` from the starter template (or an empty marimo notebook with `--empty`), open it, make it active. |
| `/notebook open <name>` | Switch to an existing notebook by name, filename or path; opens it in the browser when it has no session. |
| `/notebook close [name]` | Shut down that notebook's kernel session (the browser tab disconnects); it can be reopened any time. |

After a slash-command switch the next message you send carries a one-line notice such as
`[Hailer] The active notebook is now notebooks/q2_churn_review.py (reopened, 7 cells). Call notebook_cells
before editing.` so the agent inspects the notebook before touching it. The conversation itself continues;
use `/new` if you want a clean thread as well.

Where things live:

- Notebooks go in `[hailer].notebooks_dir` (default: the folder of `[hailer].notebook`, i.e. `notebooks/`).
  Only files in that folder can be opened; names are matched case-insensitively and `q2 churn`,
  `q2_churn`, `q2_churn.py` and `notebooks/q2_churn.py` all mean the same file.
- The **starter template** is the same set of cells as `notebooks/analysis.py` (imports, the
  `hailer.periods` helpers, `WORKSPACE`, `DATA_DIR`, `data_files`, `period_files`, a welcome cell with the
  notebook's title and the data files it found), so the agent can start analysing straight away. `--empty` gives marimo's plain empty notebook.
- The active notebook is remembered in `.hailer/notebook.json` (git-ignored, next to `session.json`), so
  the next `uvx hailer` or `uvx hailer notebook` resumes where you left off. Delete the file to go
  back to `[hailer].notebook`. `HAILER_NOTEBOOK=<path>` makes that notebook the active one for this and
  later sessions: every Hailer command (`hailer`, `hailer notebook`, `hailer exec`, `status`, `doctor`)
  writes it to the state file at startup, so the agent's tools see the same notebook.
- `/notebook` also lists the notebooks you worked in recently (`Recent:`), most recent first.

```toml
[hailer]
notebook      = "notebooks/analysis.py"   # the default (and first) notebook
notebooks_dir = "notebooks"               # where /notebook new and the agent's notebook_create put files
```
