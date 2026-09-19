# Hailer

Hailer is a Python command-line package for chatting with your data. Ask questions about local CSV,
Parquet or JSON files in plain English, and it explores the data and adds tables, charts and summaries
to a live [marimo](https://marimo.io/) notebook in your browser. You chat in the terminal; the notebook
keeps the analysis and its Python code.

## Quick start

You'll need:

- [uv](https://docs.astral.sh/uv/getting-started/installation/), which provides the `uvx` command.
  Install it, then reopen your terminal.
- An [OpenAI API key](https://platform.openai.com/api-keys) for the default setup.
- A web browser.

These steps work in PowerShell on Windows and in a terminal on macOS or Linux. `uvx` downloads Hailer,
its dependencies and a suitable Python version when needed. You do not need to clone this repository
or install Docker.

### 1. Create a workspace and sign in

Run these commands in your terminal:

```bash
mkdir my-analysis
cd my-analysis
uvx hailer init
uvx hailer login openai
```

`init` creates your settings (`hailer.toml`), a starter notebook and a `data/` folder. At the login prompt,
paste your API key; input is hidden and the key is saved in your operating system's credential store.
The starter settings use OpenAI with `gpt-5.5`.

For a company or another model endpoint, follow [custom endpoint setup](#a-custom-or-internal-endpoint)
after `init`, using that provider's login command, then continue below.

### 2. Add your data

Copy the CSV, Parquet or JSON files you want to analyse into the `data/` folder inside `my-analysis`.
Any filenames work.

For a small example, save this as `data/sales.csv` in your workspace:

```csv
month,region,revenue
2026-01,North,1200
2026-01,South,900
2026-02,North,1500
2026-02,South,1100
```

### 3. Start chatting

From the same terminal, run:

```bash
uvx hailer
```

Hailer opens the notebook in your browser and starts the chat in your terminal. **Keep the notebook tab
open while you work**; it provides the live session that runs the analysis.

Ask a question in the terminal, for example:

```text
What data do I have? Summarise the columns and any missing values.
```

With the sample CSV, try:

```text
Chart total revenue by month, split by region.
```

Tables and charts appear in the notebook. Type `/exit` to finish. To continue later, open a terminal
in `my-analysis` and run `uvx hailer` again; it resumes your conversation and active notebook.
Use `/new` for a fresh conversation, or `/help` to see the chat commands.

If startup fails, run `uvx hailer doctor` for checks and suggested fixes, or see
[Troubleshooting](#troubleshooting). A missing marimo server before your first session is expected:
`uvx hailer` starts it for you.

**Data and code:** files are read locally, but your messages, code and notebook tool outputs (which can
include data samples) go to the configured model endpoint. By default, notebook code runs with your
account's file and network access. See [Security](#security) and the optional
[Docker runtime](#isolated-kernel-docker) for details.

## Next steps

The sections below are reference material; the quick start above is enough for a first analysis.

- **Everyday use:** [notebooks and sessions](#notebooks-and-sessions),
  [chat and CLI commands](#slash-commands), [several notebooks](#working-with-several-notebooks).
- **Make it your own:** [model configuration](#model-configuration),
  [project context, skills and prompts](#project-context-skills-and-prompts),
  [data conventions](#data-conventions).
- **Setup options:** [installation and upgrades](#installation),
  [Docker](#isolated-kernel-docker), [abeam](#running-inside-abeam).
- **Reference:** [troubleshooting](#troubleshooting), [security](#security),
  [architecture](#architecture), [development](#development).

## Notebooks and sessions

`hailer init` creates the starter notebook (`[hailer].notebook`) and the data folder (`[hailer].data_dir`)
that `hailer.toml` names. It never overwrites anything except `hailer.toml` itself, and only with
`--force`, so running it again just fills in what is missing (a deleted notebook, say). Put the files you
want to analyse in `data/`: CSV, Parquet or JSON, with any names (see [Data conventions](#data-conventions)).
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
  isolated container instead, add `--kernel docker` or see [Isolated kernel (Docker)](#isolated-kernel-docker).

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

## Slash commands

| Command | What it does |
|---|---|
| `/help` | Show the command list. |
| `/status` | Model, provider, credentials source, thread id, token usage, marimo state, kernel runtime, web allowlist. |
| `/new` | Start a new conversation thread (context files are re-read). |
| `/model <name>` or `/model <provider>:<name>` | Switch model (and provider); starts a new thread. |
| `/notebook`, `/notebook list`, `/notebook new <name> [--empty]`, `/notebook open <name>`, `/notebook close [name]` | Show or switch the active notebook (see *Working with several notebooks*). |
| `/context` | List loaded context files, skills, prompts and the web allowlist. |
| `/skill <name> [message]` | Run a turn with a project skill attached. |
| `/prompt <name> [args]` | Send a saved prompt from `.config/hailer/prompts`. |
| `/reload` | Re-read `.config/hailer`; applies from your next message. |
| `/clear` | Clear the screen. |
| `/exit`, `/quit` | Exit Hailer (Ctrl+C at the prompt, Ctrl+D on an empty line, or Ctrl+Z then Enter, also exit). |

Interactive chat keeps a framed input box below the conversation. Your submitted messages, Hailer's
replies and command results appear above it. The context line shows the active notebook, model and
number of loaded context files; the activity line shows what Hailer is doing.

Enter sends; Alt+Enter inserts a newline. Pasting several lines keeps them in one editable message.
Up and Down move through multiline input and recall earlier messages at its boundaries (history stays
in memory for this session). You can draft your next message while Hailer works; Enter preserves that
draft until the current operation finishes or is cancelled. Requests are not queued.

The interface uses normal terminal scrollback and text selection. It does not capture the mouse or
switch to an alternate screen. Use `hailer --plain` or `hailer notebook --plain` for the line-oriented
`You >` interface. Pipes and terminals reporting `TERM=dumb` or `TERM=unknown` use plain input automatically.

Ctrl+C during a turn cancels the model request and keeps the composer open. A notebook command already
performing a blocking operation finishes its cleanup before another command can start. Cancellation does
not undo tool actions that have already happened. Ctrl+C while idle exits; Ctrl+D on empty input or
Ctrl+Z then Enter also exits. The conversation is stored in `.hailer/threads.sqlite`, which is how
`hailer` resumes it after a restart; `/new` (or `hailer --new`) discards it and starts another.
Completed replies are displayed once, with Markdown formatting; live activity does not print interim
model commentary. `/clear` clears the display without resetting the conversation.

Other subcommands:

| Command | What it does |
|---|---|
| `hailer` | Start or reuse this workspace's notebook server, open the notebook and chat in the terminal. |
| `hailer notebook [--port N] [--no-browser] [--keep-marimo] [--foreground] [--new] [--plain] [--kernel RUNTIME]` | The same session with extra startup options; marimo runs on the notebooks folder. `RUNTIME` is `local` or `docker`. |
| `hailer exec -c "code"` (or `hailer exec script.py`, `hailer exec -` for stdin) | Run Python in the active notebook's kernel yourself. |
| `hailer status`, `hailer doctor` | Configuration and state; the startup checks with fixes. |
| `hailer login <provider>`, `hailer logout <provider>` | Store or remove a provider's API key. |
| `hailer init [--force] [--kernel RUNTIME]` | Set up a workspace (see Quick start); `--kernel` writes `[kernel] runtime`. |
| `hailer kernel pull` | Download the kernel image for this Hailer (or `[kernel].image`). |
| `hailer kernel build [--tag <name>]` | Build the kernel image on this machine. |
| `hailer kernel stop` | Stop this workspace's kernel (local or docker) and remove its containers and network. |

Global options go before the subcommand: `--verbose`, `--config <path>`, `--workspace <path>`, `--new`,
`--plain`, `--version` (for example `uvx hailer --workspace C:\projects\sales kernel stop`).

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

## Data conventions

Any CSV, Parquet or JSON file in `data/` can be analysed, whatever it is called: the starter notebook lists
them in `data_files`, and the agent loads them with Polars or queries them with DuckDB when you ask.

One naming convention is optional. When the same dataset arrives every month, name the files
`YY-MM <dataset>.parquet`, for example `25-03 sales.parquet` for March 2025. The data itself then needs no
period column; Hailer derives it from the filename, combines the months into one table and copes with
columns that only appear in later months.

`hailer.periods` (imported by the starter notebook, so the agent reuses it):

| Function | Purpose |
|---|---|
| `list_data_files(data_dir)` | Every data file (CSV, Parquet, JSON, ...) directly in the folder, whatever its name, sorted by name. |
| `parse_period(name)` | `"25-03 sales.parquet"` → `Period(2025, 3)`; `None` if the name does not match. |
| `scan_period_files(data_dir, name=None)` | Sorted `PeriodFile`s for one dataset (or all) in a folder. |
| `load_periods(files, columns=None)` | One Polars DataFrame with a leading `period` column (`YYYY-MM`); schema evolution handled with a relaxed diagonal concat. |
| `scan_periods(files)` | Lazy variant of `load_periods`. |
| `duckdb_periods_view(con, files, view_name="periods")` | Registers a DuckDB view over all files (`union_by_name`) with `period` derived from the filename. |
| `describe_periods(files)` | Compact text: months found, common columns, columns present only in some months. |

A monthly file that cannot be read raises `MalformedParquetError` naming the file. The starter notebook
(`notebooks/analysis.py`) defines `WORKSPACE`, `DATA_DIR`, `data_files` and `period_files` and shows a
table of the data files it found.

## Model configuration

All model settings live in `hailer.toml` (repo root, or `.config/hailer/hailer.toml`). `uvx hailer init`
writes a commented starter file.

### OpenAI

```toml
[model]
name = "gpt-5.5"
provider = "openai"
```

Store an API key once with `uvx hailer login openai`, or set `OPENAI_API_KEY` in the terminal. Hailer
talks to `https://api.openai.com/v1` over the Responses API. To use Chat Completions instead, or to turn
streaming off, declare the provider without a `base_url`:

```toml
[model_providers.openai]
wire_api = "chat"
```

### A custom or internal endpoint

Hailer supports endpoints that implement the OpenAI Responses API or Chat Completions with function
calling, including compatible internal gateways and local model servers. Use the model name, URL and
protocol required by your endpoint.

```toml
[model]
name     = "analyst-v3"               # whatever model id the endpoint expects
provider = "internal"
# reasoning_effort = "medium"         # minimal | low | medium | high | xhigh; "" sends no reasoning effort
# summarize_after_tokens = 100000     # summarise older turns past this size; lower it for small context windows

[model_providers.internal]
base_url         = "https://llm.example.internal/v1"
wire_api         = "responses"               # or "chat" for a Chat Completions endpoint
env_key          = "INTERNAL_MODEL_API_KEY"  # env var name; value from `hailer login internal` or the terminal
# stream         = true                      # false if the endpoint rejects stream = true
# stream_options = true                      # false omits stream_options (token counts may be lost)
# name             = "Internal"
# http_headers     = { "X-Team" = "data-analytics" }
# env_http_headers = { "X-Client-Id" = "INTERNAL_CLIENT_ID" }
# query_params     = { "api-version" = "2025-04-01-preview" }
```

Then:

```bash
uvx hailer login internal     # stores the key in the OS credential store (hidden prompt)
uvx hailer doctor             # config / notebook / credentials / marimo / session, with fixes
uvx hailer
```

Or set `INTERNAL_MODEL_API_KEY` in the terminal instead of logging in; an environment variable always wins
over the credential store. The startup panel and `status` show the provider with its base URL; `doctor` and
`status` show where the key came from (`from keyring` or `from env`), never the value.

**`wire_api` chooses the protocol the endpoint speaks:**

| `wire_api` | What Hailer sends | Use it when |
|---|---|---|
| `"responses"` (default) | `POST {base_url}/responses` | The endpoint implements the OpenAI Responses API. |
| `"chat"` | `POST {base_url}/chat/completions` | The endpoint implements Chat Completions (vLLM, Ollama, LiteLLM, most internal gateways). |

Hailer uses LangChain's `ChatOpenAI` client to send requests to `base_url`: the key as
`Authorization: Bearer ...`, your `http_headers`, each `env_http_headers` entry whose variable is set,
and your `query_params`. The body follows the selected protocol: Chat Completions uses `messages`,
while Responses uses `input`. Both carry the configured model, conversation and Hailer's tool definitions.
Every request names the model you configured; there are no side requests under other model names.
`HTTP_PROXY`, `HTTPS_PROXY` and `NO_PROXY` are honoured.

Two per-provider switches help with strict endpoints; both default to `true` and apply to either
`wire_api`:

- `stream = false` sends `stream: false` and reads one JSON reply, for endpoints that reject streamed
  requests or cannot deliver server-sent events (some gateways and proxies buffer or refuse them). The
  answer then appears when the turn finishes rather than as it is generated. It is also the workaround
  for an endpoint that streams tool calls in a shape the client does not understand. The provider shows
  as `internal (https://..., chat completions, no streaming)` in the startup panel and `/status`.
- `stream_options = false` omits the `stream_options` field from streamed requests (some Azure API
  versions and proxies reject it). Token counts are then whatever the final chunk carries, often nothing.

Independently of the provider, `reasoning_effort = ""` under `[model]` stops the reasoning effort being
sent at all, for endpoints or models that reject it.

Long conversations are kept inside the model's context window by summarising older turns once the
conversation passes `[model].summarize_after_tokens` (default 100000; the last 20 messages are always kept
as they are). The summary is written by the model you configured. Lower the number for a model with a
small context window; `0` turns summarising off.

Several providers can be declared; switch inside a session with `/model <name>` or
`/model <provider>:<name>` (this starts a new thread). One-off overrides: `HAILER_MODEL`,
`HAILER_MODEL_PROVIDER`.

## Project context, skills and prompts

`.config/hailer/` holds what the agent knows about *this* project. Commit it with the notebook.

```
.config/hailer/
├── context/      always-on: every *.md here is sent to the model at the start of each session
├── skills/       on-demand: <name>/SKILL.md (+ reference/, scripts/) in the Agent Skills format
└── prompts/      reusable prompts: /prompt <name> [args]   ({{args}} is substituted)
```

- **context/**: short Markdown files with column meanings, conventions and house style, loaded in filename
  order and concatenated up to `max_context_bytes`. Hailer warns at startup if a file looks like it contains
  a key or token.
- **skills/**: only the `name` and `description` from each `SKILL.md` frontmatter go into the agent's
  instructions; the body and bundled files are loaded when the task matches (the agent calls `load_skill`
  and `read_skill_file`) or when you type `/skill <name> [message]`.
- **prompts/**: `/prompt first-look customers.csv` sends `prompts/first-look.md` with `{{args}}` replaced.
- `/context` lists what is loaded; `/reload` re-reads the folder and applies from your next message, in the
  same conversation.

Everything in `context/`, and any skill you load, is sent to the configured model endpoint. Keep
credentials, customer names and row-level data out of it.

## Installation

`uvx hailer ...` fetches Hailer from PyPI the first time, keeps it in uv's cache with an environment of
its own (marimo, Polars, DuckDB and the rest), and runs it from there. No `pyproject.toml` or `.venv`
is needed in your analysis folder.

To use the latest release, run `uvx hailer@latest`. Use `uvx hailer@X.Y.Z` to select a specific version.

For a permanent `hailer` command:

```bash
uv tool install hailer
```

Then use `hailer` wherever these instructions say `uvx hailer`. Upgrade that installation with
`uv tool upgrade hailer`. You can also use `pip install hailer` in your own Python 3.12+ environment.
See [uv's tools guide](https://docs.astral.sh/uv/guides/tools/) for more installation options.

### Requirements

- Windows 11 is the primary target; macOS and Linux work too.
- Hailer needs Python 3.12 or newer; uv can download it automatically.
- Keep the notebook open in a web browser while using Hailer.
- Every model provider needs an API key (see [Model configuration](#model-configuration)).
- Docker Desktop (Windows, macOS) or Docker Engine (Linux) is needed only for the
  [optional Docker runtime](#isolated-kernel-docker).

For source checkouts and sample data generation, see [Development](#development).

## Running inside abeam

[abeam](https://github.com/OpenAfterHours/abeam) runs coding-agent CLIs in a terminal pane beside git status,
a file viewer and a shell. It starts `hailer` from your `PATH`, so install Hailer as a tool first:

```bash
uv tool install hailer
abeam +hailer
```

abeam forwards every argument, so `abeam +hailer notebook --no-browser` or `abeam +hailer --new` behave as
they do in a terminal. abeam starts Hailer in a git worktree; commit `hailer.toml` (and your notebooks) so
each worktree is a workspace of its own, since Hailer uses the nearest folder holding `hailer.toml` or
`pyproject.toml`. Each workspace then gets its own marimo server, `.hailer/` stays out of the git pane, and a
pasted block of several lines arrives as one message.

## Isolated kernel (Docker)

By default the notebook kernel runs in Hailer's own Python, as you: the code the agent writes can read and
change every file your account can, and reach the network (see [Security](#security)). If Docker is
installed, you can run the kernel in a container instead. The agent, the conversation and the API key
stay on your machine; only marimo and the notebook code move into the container.

| | `local` (the default) | `docker` |
|---|---|---|
| Where marimo and notebook code run | Hailer's own Python, as you | A Linux container, as a non-root user |
| Files notebook code can read | Everything your account can | The notebooks folder and the data folder, nothing else |
| Files it can change | Everything your account can | The notebooks folder only; data is read-only |
| Network | Yours | None, unless `network = true` |
| Your environment variables | All except secret-looking names (see [Security](#security)) | None |
| Needs | Nothing | Docker Desktop (Windows, macOS) or Docker Engine (Linux), and the kernel image |

### Choosing the runtime

```toml
[kernel]
runtime  = "docker"   # "local" (default) or "docker"
# image   = ""        # docker: default ghcr.io/openafterhours/hailer-kernel:<hailer version>
# memory  = "4g"      # docker: memory limit, no swap on top
# cpus    = 2         # docker: CPU limit (lowered to what Docker has, with a note)
# network = false     # docker: true lets notebook code reach the internet, this machine and other containers
# pass_env = []       # local: secret-looking variables notebook code may still read (see Security)
```

- **At setup:** `uvx hailer init --kernel docker` writes a `hailer.toml` with this section switched on.
  Plain `uvx hailer init` writes it commented out, and when Docker is found on the machine its next steps
  say how to switch it on. `init` never rewrites an existing `hailer.toml` without `--force`, and says so
  when `--kernel` could not be applied.
- **In an existing file:** uncomment the `[kernel]` line as well as `runtime`. A `runtime` line that ends up
  under `[model]` or `[hailer]` is ignored with a warning that says so.
- **For one run:** `uvx hailer notebook --kernel docker` (or `--kernel local`), or `HAILER_KERNEL=docker` in
  the environment. The flag wins over the variable, the variable over the file. `HAILER_KERNEL_IMAGE`
  overrides `image`.
- **Which one is in use** shows in the startup panel, `uvx hailer status`, `/status` and `uvx hailer doctor`
  (`X.Y.Z` is Hailer's version):

  ```text
  Kernel:     local (runs as you; not isolated)
  Kernel:     docker (hailer-kernel X.Y.Z; no network; data read-only)
  Kernel:     docker (hailer-kernel X.Y.Z; network on: the internet and this machine; data read-only)
  ```

  A docker kernel Hailer started for this workspace is the one in use whatever the file says: `uvx hailer` attaches to it when its mounted folders match and the line says `docker`.
- **Docker mode fails closed.** Docker missing or not running, Docker Desktop set to Windows containers,
  an image for another Hailer version, or a folder layout it refuses (below): Hailer stops and says how to
  fix it (see [Troubleshooting](#troubleshooting)). It never runs notebook code on this machine instead.

### First run: the kernel image

The first `uvx hailer notebook` in docker mode downloads `ghcr.io/openafterhours/hailer-kernel:<version>`,
where the tag is Hailer's own version (about 200 MB to download, 900 MB on disk), with docker's progress
in the terminal. After that a start takes a few seconds (4.3 s measured on Windows 11 with Docker
Desktop). `uvx hailer kernel pull` downloads it ahead of time.

- `uvx hailer kernel build` builds the image on this machine instead, from the Dockerfile that ships inside
  Hailer, with the marimo, Polars and DuckDB versions this Hailer runs. It needs Docker Hub (the base image)
  and PyPI (the packages) once. Use it where the registry is out of reach, or for a Hailer version that has
  no published image (development versions never do). `--tag <name>` builds under another name; Hailer
  then says to set `image` under `[kernel]` to use it.
- `[kernel].image` (or `HAILER_KERNEL_IMAGE`) points Hailer at another copy, such as a company mirror.
- The image's version label must equal Hailer's version, because a different marimo would break the
  notebook API Hailer drives. After upgrading Hailer, the next start downloads the matching image; a copy
  for another version stops the start with the pull and build commands.

### What notebook code can and cannot do in the container

- **Two folders.** It sees the notebooks folder as `/work/notebooks` (read-write, the only place files
  persist) and the data folder as `/work/data` (read-only). Nothing else of your machine: not the
  workspace, `.hailer/`, `.config/hailer/`, your home folder or the Docker socket. The data files are read
  in place through the mount, never copied. The starter notebook finds `WORKSPACE = /work` and
  `DATA_DIR = /work/data`, the model is told those paths, and notebooks need no changes.
- **No network.** No internet, no DNS, no route to this machine. Installing packages (`ctx.packages.add()`)
  and DuckDB's `INSTALL` fail. The image has marimo, Polars, DuckDB, altair and plotly; anything else
  needs an image of your own (see [Known limitations](#known-limitations)).
- **No secrets.** None of your environment variables reach the container, and nothing of Hailer's own
  runs in it but the notebook helpers (`hailer.periods`) and the forwarder: no LangChain, no keyring, no
  API key.
- **Limits.** A non-root user with no Linux capabilities, a read-only root file system, at most 256
  processes, and the memory and CPU limits from `[kernel]` (memory without extra swap). On a Linux host
  the kernel runs with your user and group id, so files marimo saves stay yours.
- **Scratch space.** `/tmp` and the home folder are in memory, shared by every notebook in the container,
  and wiped when the kernel stops.
- **Cells run when a notebook opens**, so the starter notebook's `DATA_DIR`, `data_files` and
  `period_files` exist before anyone runs a cell.

`network = true` puts the kernel on Docker's default network with its port published directly on
`127.0.0.1`. Notebook code can then reach the internet, services on this machine (`host.docker.internal`)
and other containers. Everything else above still holds, and the `Kernel:` line says
`network on: the internet and this machine`.

### Which folders can be mounted

In docker mode Hailer refuses a layout that would hand notebook code Hailer's own files, your
credentials, or a way to run code on this machine later. `uvx hailer doctor` reports it in the `config`
row, and every start stops on it, `--foreground` included (the same rules, in one place):

- Neither the notebooks folder nor the data folder may be, or contain, a drive root, your home folder,
  the workspace folder, `.hailer/`, the config file, or the context, skills and prompts folders.
- Neither may be, contain or sit inside a folder that holds credentials: `~/.config`, `~/.ssh`, `~/.aws`,
  `~/.azure`, `~/.gnupg`, `~/.docker`, `~/.kube`, and on Windows `%APPDATA%` and `%LOCALAPPDATA%` (the
  temporary folder inside it is fine).
- The notebooks folder may not sit inside `.hailer/` or those folders either, and the data folder may not
  sit inside the notebooks folder (it would become writable).
- The notebooks folder may not contain `.git` or `hailer.toml`, including in subfolders: notebook code
  could change repository hooks or another workspace's settings. Keep notebooks in a plain subfolder of
  your repository, such as `notebooks/`, and move nested repositories and workspaces outside that folder.
  Hailer checks names without following symlinks or junctions; an unreadable subtree stops startup.
- The default layout, `notebooks/` and `data/` in the workspace, passes. The data folder can be anywhere
  else on the machine: set `[hailer].data_dir` to an absolute path and it is mounted read-only where it is.
- Windows: a folder on a network share (a UNC path such as `\\server\share\sales`) cannot be mounted: it
  is an error in `doctor`'s `config` row and stops the start, with the same message. A mapped network drive
  gets a warning (`doctor` rows `data` and `notebooks`): Docker Desktop usually cannot see it. Symlinks and
  junctions inside the data folder that point outside it do not resolve in the container; `doctor` lists
  them. Copy such data to a folder on a local disk.
- `[hailer].marimo_url` (or `HAILER_MARIMO_URL`) cannot be combined with docker: Hailer starts and finds its
  own containers.

### Starting, reusing and stopping

- **What runs.** `uvx hailer notebook` creates three things, each labelled with the workspace:
  `hailer-net-<id>`, an internal network with no route out; `hailer-kernel-<id>`, the kernel, on that
  network only; and `hailer-fwd-<id>`, a small forwarder from the same image, published on
  `127.0.0.1:<port>` only, which carries the browser's and Hailer's requests to the kernel (Docker
  publishes no port for a container on an internal network). `<id>` comes from the workspace path, so two
  workspaces can run side by side. With `network = true` there is only the kernel, published directly.
  When the chat ends, Hailer removes the kernel it started. Reused kernels are left running.
- **Reuse.** `--keep-marimo` leaves the kernel running. Both `uvx hailer` and `uvx hailer notebook`
  reuse it only when its mounted folders match. When Docker is configured, image, network, memory and
  CPU settings must also match; otherwise the command asks you to run `uvx hailer kernel stop` first.
  `--kernel local` can attach to a running Docker kernel with matching folders. The chat keeps that
  server's token and path mapping in memory, including across notebook and model switches.
- **`--foreground`** starts the kernel and follows its log in this terminal. Ctrl+C stops and removes it;
  when it ends for another reason, Hailer says why (out of memory, removed from outside, exited).
- **`uvx hailer kernel stop`** stops this workspace's kernel, docker or local, and removes every container
  and network labelled with the workspace, running or not. It prints what it removed, says when Docker is
  not running (so leftovers could not be checked), and exits 1 when a recorded Docker kernel cannot be stopped or a removal failed (a container another
  terminal is removing at the same moment counts as removed, and a network whose containers are still
  detaching is retried for a few seconds). If Docker is unreachable, the recorded Docker kernel is kept
  for retry. An unanswered health check alone never proves those containers are gone.
- **Files the kernel may have planted.** When a docker kernel stops (the chat ends, `--foreground` ends,
  `uvx hailer kernel stop`), Hailer scans the notebooks folder and its subfolders for `.git`, `.vscode`,
  `.idea`, `.devcontainer` and `hailer.toml` and prints a loud `WARNING` naming them: notebook code can
  write there, and git or an editor would run commands from them. Delete them (unless you put them there
  yourself) before you run git in that folder or open it in an editor. `uvx hailer doctor` shows the
  same as a `notebooks` warning. Nothing is removed for you.
- **Leftovers.** If Hailer itself is killed, the containers keep running and `.hailer/kernel.json` still
  records them, so the next `uvx hailer notebook` or `uvx hailer` attaches to them and
  `uvx hailer kernel stop` removes them. A start never removes a running kernel container that Hailer has
  no working record of, because it may still be in use: it stops and points at `uvx hailer kernel stop`.
  A start also refuses while the recorded kernel (docker or local) does not answer but is not provably
  gone (its containers or its process still exist: busy, stuck or suspended), and keeps its record so
  `uvx hailer kernel stop` can still find it. Stopped leftovers are cleaned up by the next start.
- **Logs.** `docker logs hailer-kernel-<id>`. A failed start prints its last lines.

### Platforms

- **Windows 11 with Docker Desktop** (WSL2 engine, Linux containers): tested live and by the integration
  test.
- **Linux**: the integration test passes in WSL Ubuntu, and a CI job runs it on Ubuntu (Docker Engine)
  for every pull request.
- **macOS**: Docker Desktop. The release publishes the image for `linux/amd64` and `linux/arm64` (Apple
  Silicon). The CI integration test runs on amd64; it does not cover arm64.
- **Podman** is not supported.

## Allowed web domains

By default the agent's **`fetch_page` tool is disabled**: it refuses every URL. To let that tool read
specific sites for context:

```toml
[web]
allowed_domains = ["docs.pola.rs", "duckdb.org", "**.marimo.io"]
max_page_bytes = 200000
```

- Exact hosts match themselves; `*.example.com` matches subdomains only; `**.example.com` matches the apex
  and its subdomains. Matching is case-insensitive and ignores ports. Loopback and private addresses are
  only allowed when listed literally (for example `127.0.0.1`).
- `fetch_page` checks the host before any network I/O and on every redirect, converts HTML to readable
  text, caps the size, and returns the text to the model. A refused URL tells the agent which domains are
  allowed so it can ask you to extend the list.

Text fetched from an allowed site becomes a tool result and is sent to the model endpoint like any other
result. `/context`, `/status` and the startup panel show the active allowlist. The allowlist governs the
agent's own web tool, not what notebook code can do: with the default local kernel, code the agent runs
in the notebook is ordinary Python on your machine with your network access; with the docker kernel it
has no network at all unless `[kernel] network = true` (see [Security](#security)).

## Secrets

`uvx hailer login <provider>` stores the API key in the operating system's credential store through
`keyring` (Windows Credential Manager on Windows) under the service name `hailer`. Hailer reads it when the
agent starts and hands it to the HTTP client inside its own process, which sends it only as the
`Authorization` header of requests to that provider's endpoint. It is never written to `hailer.toml`, the
session file, command history or logs, and never passed on a command line or to another process.
`uvx hailer logout <provider>` removes it.

An environment variable named by `env_key` takes precedence over the credential store, which keeps scripted
and CI use simple. Hailer leaves that variable (and other secret-looking ones) out of the environment of the
marimo server it starts, so it is not in notebook code's environment (see [Security](#security)). Note that any
process running as the same user can read both credential-store entries and environment variables; the
gain is keeping the key out of files and history, not isolation from your own account.

Every provider needs a key, including the built-in `openai` one. A missing key stops `hailer` at startup
with the `login` command to run.

## Security

**Sent to the configured model endpoint:** your messages; Hailer's system prompt plus everything in
`.config/hailer/context`; the name and description of each skill, and a skill's full body once loaded; every
tool call argument (that is, the Python the agent writes) and the tool results, truncated to
`max_tool_output_chars`; and pages fetched from allowed domains. The instructions name the workspace,
notebooks and data folders by path (with the docker kernel, the container's `/work` paths). With the `openai` provider this goes to OpenAI; with a custom provider,
to the `base_url` you configured, and nowhere else: when older turns are summarised, the same endpoint and
model write the summary.

**Not sent:** your data files or any dataframe, unless code explicitly prints or returns it (the agent is
instructed to inspect schemas, samples and aggregates and to keep outputs compact); API keys or other
secrets (the key stays inside the Hailer process and is masked in logs).

**What the agent can do:** it has Hailer's eleven tools and nothing else: no shell, no file tools, no
tools from other software on your machine. Nothing asks for approval before a tool runs. The tool that
matters is `marimo_execute`: it runs the Python the model writes in the notebook kernel. That is what
makes the analysis possible. What the code can reach depends on the kernel runtime:

- **`local` (the default): not sandboxed.** The kernel runs in Hailer's own Python, as you, with your file
  and network access. Hailer closes two gaps around it:
  - *Other programs.* The marimo server Hailer starts requires a random token, handed to marimo on stdin
    (never on a command line) and kept in `.hailer/kernel.json`, so another program on the machine cannot
    send code to the kernel. A marimo server you start yourself with `--no-token` has no such protection.
  - *Secrets in the environment.* The server's environment leaves out every provider's `env_key` and
    `OPENAI_API_KEY`, the variables named in `env_http_headers`, `HAILER_MARIMO_TOKEN`, the names
    `PASSWORD`, `SECRET`, `TOKEN`, `PGPASSWORD` and `MYSQL_PWD`, and every name ending in `_KEY`, `_TOKEN`,
    `_SECRET`, `_PASSWORD`, `_PASSWD`, `_PWD`, `_CREDENTIALS`, `_CONNECTION_STRING` or `APIKEY` (any case).
    `uvx hailer doctor` shows how many were withheld. A notebook that needs one of them (a database
    password, say) gets it through `[kernel] pass_env = ["DB_PASSWORD"]`; naming a provider's key or
    header variable, or `HAILER_MARIMO_TOKEN`, there is a `config` warning.

  Notebook code can still read the OS credential store and every file you can, which is why the `Kernel:`
  line says `not isolated`.
- **`docker`: isolated.** The kernel runs in a container that sees only the notebooks folder (read-write)
  and the data folder (read-only), with no network, none of your environment variables, as a non-root user
  with resource limits (see [Isolated kernel (Docker)](#isolated-kernel-docker)). Hailer refuses to mount
  a folder that would expose its own files or your credentials (`~/.ssh`, `~/.aws`, `%APPDATA%`, ...), a
  data folder inside the notebooks folder, and a notebooks folder that is a git repository, on every start
  path (see [Which folders can be mounted](#which-folders-can-be-mounted)).

The instructions forbid destructive file operations and sending data anywhere, but instructions are not an
enforcement boundary: use an endpoint and model you trust, keep the data folder to data the agent may read,
and press Ctrl+C if a turn goes somewhere you did not intend.

**What the docker runtime does not protect against:**

- *Output goes to the model.* Anything notebook code prints or returns is a tool result and is sent to the
  model endpoint, in either runtime.
- *Notebooks are code.* A notebook the container wrote runs on your machine, as you, if it is later opened
  with the local runtime (or with marimo directly). Hailer warns on the first local start after a docker
  kernel used the notebooks folder. Keep the notebooks folder inside your project's git repository (a plain
  subfolder, not a repository of its own), so every change is a diff you can review.
- *Other files in the notebooks folder can run code later.* Notebook code can write anything into the
  notebooks folder, including a `.git` folder (hooks and settings such as `core.fsmonitor` run when git
  or an editor that scans nested repositories, like VS Code, touches it), `.vscode`, `.idea` or
  `.devcontainer` settings, or a `hailer.toml` that makes the folder look like another workspace. Hailer
  refuses existing repositories and nested Hailer workspaces anywhere in the notebooks tree. At stop
  and in `doctor`, it warns about repository, editor and workspace controls in that tree. These checks
  do not prevent an editor from acting on a new file while the kernel is running. Disable automatic
  repository discovery and automatic editor tasks for folders holding untrusted notebooks. Review new
  control files before opening them; Hailer never removes them for you.
- *The token is on the container's command line*, so `docker inspect` shows it. Anyone who can use Docker
  on the machine can already `docker exec` into the container, so hiding it would gain nothing.
- *The image is trusted by name.* Hailer checks its tag and version label, not a signature, and runs
  whatever `[kernel].image` names. The image pins marimo, Polars, DuckDB, altair, plotly and its base
  image, but not their dependencies.
- *It is a choice, not a policy.* Anyone can switch back to `local`. An organisation that must enforce
  isolation should run Hailer itself in a managed virtual machine or dev container.

**Links and tokens:** the notebook link Hailer opens in your browser, and every link the CLI prints
(`/notebook`, `hailer status`, `doctor`), carries `access_token=<token>`, which signs the browser in. It
works like a password for the kernel while the server runs, so do not paste it anywhere. URLs in tool results
leave the token out, because tool results go to the model endpoint; when the agent asks you to open a
notebook, run `/notebook` for the signed-in link.

**Tracing:** Hailer's agent library (LangChain) can send full conversation traces to the LangSmith service
when variables such as `LANGSMITH_TRACING` are set in the environment. Hailer switches that off for its own
process at startup, so a variable left over from another project cannot send your prompts and results to a
third party. Set `HAILER_TRACING=1` if you do want the tracing variables in your environment to apply.

**On disk:** the conversation (your messages, the agent's replies, tool calls and their truncated results)
is stored unencrypted in `.hailer/threads.sqlite` inside the workspace until `/new` or `hailer --new`
replaces it. `.hailer/kernel.json` records the marimo server Hailer started (runtime, URL, token and, for
docker, the container and network ids and the settings it was started with) and is deleted after successful cleanup; failed Docker cleanup keeps the record for retry; `.hailer/last-kernel.json` notes which runtime last used the notebooks folder. `.hailer/marimo.log`
holds the local server's output, including its signed-in URL, and is emptied at each start. On macOS and
Linux these three files are readable by you only. `.hailer/` is never mounted into a docker kernel and is
kept out of git by its own `.gitignore` containing `*`; Hailer never edits your repository's `.gitignore`.

## Troubleshooting

`uvx hailer doctor` runs the startup checks (config, notebook, credentials, kernel, marimo, session; in
docker mode also docker, image, data and, when there is something to say, notebooks) and, when a session
exists, confirms that marimo's code-mode API and Hailer's notebook helpers (`hailer.periods`) are
available in the kernel.

| Message | Meaning and fix |
|---|---|
| `No marimo server is running for this workspace.` (`doctor` or `exec`; `status` says `none for this workspace`) | Run `uvx hailer` or `uvx hailer notebook` to start or reuse this workspace's kernel. Hailer finds its own servers through `.hailer/kernel.json`; local registry servers must serve this notebooks folder. Docker mode never attaches to a local server. |
| `Marimo is not running at <url>.` with a hint about `[hailer].marimo_url` | The pinned URL does not answer. Start marimo there or remove the setting; Hailer never starts a replacement for an explicit pin. |
| `Could not finish stopping the Docker kernel` or `Could not stop the Docker kernel` | The record is kept and the command exits 1. Restore access to Docker and run `uvx hailer kernel stop` again. |
| `Could not record the Docker kernel` or `Could not record the local kernel` | Writing `.hailer/kernel.json` failed. Hailer stops what it started; check disk space and permissions before retrying. |
| `Marimo exited early (code N)` or `Marimo did not answer on http://127.0.0.1:2718 within 60 s`, followed by `Log: <workspace>\.hailer\marimo.log` and its last lines | `hailer notebook` could not start marimo. The log tail usually names the cause (port in use by something else, a syntax error in the notebook, marimo not installed in the environment). |
| `the notebook is not open in a browser` followed by `Open http://... in your browser.` | The server is up but has no kernel session. Open the URL; Hailer opens it for you once at startup. The URL ends in `&view-as=present` (app view) and, for a server Hailer started, `&access_token=...`; `Ctrl+.` in the notebook shows the code. |
| The browser shows marimo's sign-in page instead of the notebook | The link has no `access_token`: links in the agent's replies and tool results leave the token out, because they go to the model endpoint. Run `/notebook` in the chat for the signed-in link. |
| `Docker is not installed.` | Docker mode needs the `docker` command. Install Docker Desktop (Windows, macOS) or Docker Engine (Linux) and run the command again, or set `[kernel] runtime = "local"` to run notebook code on this machine without isolation. |
| `Docker is not running.` | The `docker` command is there but the engine does not answer (Docker Desktop is closed or still starting). Start Docker Desktop (or the Docker service), wait until it is running, and run the command again. `uvx hailer kernel stop` still stops a local server and says that it could not check for containers. When the hint says `The docker command cannot reach an engine through its current context`, `DOCKER_CONTEXT` or `docker context use` names a context that does not work: `docker context use default` switches back. |
| `Docker runs windows containers; Hailer's kernel image needs Linux containers.` | Docker Desktop is set to Windows containers. Choose *Switch to Linux containers* from its tray icon menu, or set `[kernel] runtime = "local"`. |
| `The kernel image for Hailer X.Y.Z is not published (or not visible to you): ghcr.io/openafterhours/hailer-kernel:X.Y.Z.` | The registry refused the download: this version has no published image (a development version, or a release whose image job has not finished), or the image is not public. `uvx hailer kernel build` builds it on this machine; or set `[kernel].image` (or `HAILER_KERNEL_IMAGE`) to a copy you can reach, such as a company mirror. `Could not download the kernel image ...`, with docker's own error above it, is a network, proxy or registry problem with the same fixes. |
| `The kernel image <image> is for Hailer A.B.C; this is Hailer X.Y.Z.` (or `(no version label)`) | `[kernel].image` names an image built for another Hailer, and a different marimo would break code mode. `uvx hailer kernel pull` downloads the matching one, `uvx hailer kernel build` builds it; or point `image` at the matching tag. |
| `The notebooks folder (<path>) is the workspace folder. With [kernel] runtime = "docker" it is mounted writable into the container, so notebook code could change Hailer's own files.` (also `is a whole drive`, `contains your home folder`, `is inside Hailer's .hailer folder`, `is inside C:\Users\<you>\.aws, a folder that holds credentials`, the same for `The data folder`, and `[hailer].data_dir (...) is inside the notebooks folder`) | Docker mode refuses, on every start (`--foreground` too), a mount that would hand notebook code Hailer's own files or your credentials (see [Which folders can be mounted](#which-folders-can-be-mounted)). Keep the notebooks and the data in folders of their own, such as `notebooks/` and `data/`, and point `[hailer].notebooks_dir` and `[hailer].data_dir` at them. |
| `The notebooks folder (<path>) is a git repository (it has .git at its top level).` or `contains repository or workspace controls: ...` | In docker mode notebook code could add git hooks or settings (`core.fsmonitor`) there that run on this machine the next time git or an editor touches the repository. Keep the notebooks in a plain subfolder of your repository (such as `notebooks/`), or remove the nested repository. |
| `WARNING: the notebooks folder <path> contains .git, project/.vscode.` (any of `.git`, `.vscode`, `.idea`, `.devcontainer`, `hailer.toml`), when a docker kernel stops or in `doctor` | The docker kernel can write to the notebooks folder, and git, editors and Hailer read these. Unless you put them there yourself, delete them before you run git in that folder, open it in an editor, or run Hailer from inside it. Hailer never removes them for you. |
| `The data folder \\server\share\sales is on a network share (UNC path), which Docker cannot mount.` (an error in `doctor`'s `config` row and at every start), or in `doctor`: `the container may not see all of the data folder` with `... is on a mapped network drive (Z:); Docker Desktop usually cannot mount it.` (a warning) | Docker Desktop cannot mount network locations. Copy the data to a folder on a local disk and point `[hailer].data_dir` at it (the same for the notebooks folder and `[hailer].notebooks_dir`). `N link(s) in the data folder point outside it` means symlinks or junctions the container cannot follow: copy those files into the folder. |
| `The running kernel was started with other settings (memory 4g, hailer.toml: 2g).` | A kernel left running (`--keep-marimo`, `--foreground`) was started with other folders, network, image, memory or cpus than `hailer.toml` asks for now, so `uvx hailer notebook` does not reuse it. Run `uvx hailer kernel stop`, then the command again. Bare `uvx hailer` follows the same reuse rules. |
| `A kernel container for this workspace is still running, but Hailer cannot reach it: hailer-kernel-<id>.` | A docker kernel from an earlier run (a crash, a closed terminal) that Hailer has no working record of. Hailer never removes a running kernel on its own, since it may be in use. `uvx hailer kernel stop` removes it. |
| `A docker kernel Hailer started for this workspace is already running at <url>.` or `A local marimo server Hailer started for this workspace is still running at <url>.` | Starting another server would orphan the running one (a second `--foreground`, or `--kernel docker` while a local server Hailer started is still up). `uvx hailer notebook` and `uvx hailer` attach to a running docker kernel. Otherwise end the session that started it, or run `uvx hailer kernel stop`. |
| `A docker kernel Hailer started for this workspace (hailer-kernel-<id>, hailer-fwd-<id>) may still be running, but it does not answer at <url>.` (or `A local marimo server ... (process N) may still be running, ...`) | The recorded kernel does not answer, but its containers or its process still exist (busy, stuck or suspended), so a new start would orphan it. Wait and try again, or run `uvx hailer kernel stop`. For a local server whose process still runs, `kernel stop` clears the record and says so (`Process N is still running: ...`); end that process yourself if it is the stuck server. |
| `marimo stopped (exit code N): it was ended from outside this terminal (uvx hailer kernel stop, Task Manager or kill), or it failed; its own output is above.` | Printed by a local `--foreground` when its marimo ended without Ctrl+C. |
| `The kernel container exited early (code N).`, `The forwarder container exited early (code N).` or `Marimo did not answer on http://127.0.0.1:<port> within 60 s.`, followed by `Last lines of docker logs hailer-kernel-<id>:` | The docker kernel did not come up; the log lines usually name the cause. Nothing is left running. |
| `The kernel ran out of memory and Docker stopped it; [kernel].memory in hailer.toml raises the limit.` | Printed by `--foreground`. Raise `[kernel].memory` (for example `"8g"`) and start again. In a chat the same event shows as marimo no longer answering. |
| `Note: [kernel].cpus = 8 is more than the 4 CPUs Docker has; the kernel gets 4.` | Information only. Docker Desktop's resource settings decide how many CPUs its engine has. |
| A notebook that used to work fails with `KeyError: 'DB_PASSWORD'` (or a database refuses a login) in the local runtime | Hailer withholds secret-looking environment variables from the local kernel (`doctor` shows how many). Name the ones the notebook needs in `[kernel] pass_env = ["DB_PASSWORD"]`. A docker kernel gets no environment variables at all, and `pass_env` does not change that. |
| `The running kernel cannot see <notebook>: it mounts other folders.` | The active notebook is outside the folders the running docker kernel mounted (the notebooks folder changed after it started), so the link opens marimo's home page instead. Keep notebooks in the notebooks folder, or run `uvx hailer kernel stop` so the next start mounts the current folders. |
| `Warning: [model].runtime in <path> is ignored: runtime belongs under [kernel].` | The `[kernel]` line is still commented out, so `runtime` landed in the table above it. Uncomment `[kernel]` as well (or try docker once with `uvx hailer notebook --kernel docker`). |
| `[hailer].marimo_url (or HAILER_MARIMO_URL) cannot be used with [kernel] runtime = "docker"` | Hailer starts and finds its own container in docker mode. Remove `marimo_url`, or set `runtime = "local"` to use that server. |
| `Invalid [kernel].runtime "..." (or HAILER_KERNEL)` or `--kernel must be "local" or "docker"`; also `Invalid [kernel].memory` / `cpus` / `pass_env` | Fix the value: `runtime` is `"local"` or `"docker"`, `memory` a size such as `"4g"` or `"512m"`, `cpus` a number above 0, `pass_env` a list of variable names. |
| `warn  notebook helpers: hailer.periods did not import in the kernel` | The kernel's Python lacks Hailer's notebook helpers, which the starter notebook imports. In docker mode `[kernel].image` names an image without them: `uvx hailer kernel pull` or `uvx hailer kernel build` gets the right one. |
| `Could not remove hailer-kernel-<id>: docker said: ...` after `uvx hailer kernel stop` (exit code 1) | Docker refused a removal and the container (or network) is still there. Run `uvx hailer kernel stop` again; if it keeps failing, remove the one named in the message with `docker rm -f` (or `docker network rm`). |
| `Warning: [kernel].pass_env lets notebook code read CORP_API_KEY (the API key of provider "corp"); ...` | `pass_env` names one of Hailer's own secrets (a provider's key or header variable, or `HAILER_MARIMO_TOKEN`), so code the model writes could use or leak it. Remove it unless notebooks truly need it. |
| `not found: <path>` for the notebook | The configured notebook (`[hailer].notebook`, or `HAILER_NOTEBOOK`) does not exist. `uvx hailer init` creates it from the starter template (your `hailer.toml` is kept), or fix the path; a deleted *active* notebook is not the cause, because Hailer already falls back to the configured one when the remembered notebook is gone. New notebooks are created from the chat with `/notebook new <name>`. |
| `No notebook named '...' in notebooks.` or `... is outside the notebooks folder.` | `/notebook open` (or the agent's `notebook_open`) only opens marimo notebooks inside `[hailer].notebooks_dir`; the hint lists the available names. Move the file into the folder or point `notebooks_dir` at it. |
| `INTERNAL_MODEL_API_KEY is not set (required by provider 'internal')`, or the same for `OPENAI_API_KEY` and provider `'openai'` | Every provider needs a key. Run `uvx hailer login <provider>` or set the variable in this terminal. Releases up to 0.2.2 could use a ChatGPT login for the `openai` provider; that is gone, so create an API key. |
| `The model endpoint rejected the API key for provider '...'` | The endpoint returned 401. Re-run `hailer login <provider>`. |
| `Unknown model '...' for provider '...'` | The endpoint does not know `[model].name` (or the name given to `/model`). Its reply is quoted after `The endpoint said:`. |
| `Could not reach the model endpoint at <base_url>` | Check `base_url`, VPN or proxy (`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`), and that the endpoint is running. |
| `The endpoint at <base_url> did not accept the request.` | The endpoint answered 404/405. The hint names the request Hailer sent (`POST <base_url>/responses` or `POST <base_url>/chat/completions`): either `base_url` is wrong, or the endpoint implements the other API, in which case switch `wire_api`. The endpoint's own reply is quoted on the last line, after `The endpoint said:`. |
| `The endpoint at <base_url> rejected the request (HTTP 400)` (or 422) | The endpoint understood the request but refused a field or a value. The line `The endpoint said:` quotes its reply: fix what it names (`stream = false` or `stream_options = false` on the provider, `reasoning_effort = ""` under `[model]` to stop sending reasoning effort). `hailer --verbose` logs the HTTP traffic. |
| `The endpoint at <base_url> refused access (HTTP 403).` | The key was accepted but is not allowed for this model, route or organisation. Check the endpoint's access policy and the provider's `http_headers` / `env_http_headers`. |
| `The endpoint at <base_url> is unavailable (HTTP 429)` (or 5xx) | Rate limit or an outage on the endpoint's side. Retry in a moment. |
| `The conversation no longer fits the model's context window.` | Start again with `/new`, and lower `[model].summarize_after_tokens` so older turns are summarised before the model's limit is reached. |
| `Warning: unknown key [model_providers.x].merge_messages ... is ignored` (also `parallel_tool_calls`, `requires_openai_auth`, `[hailer].codex_home`, `[web].allow_shell_network`) | Settings from releases up to 0.2.2 that no longer mean anything. The file still loads; delete the keys to silence the warning. |
| `Previous conversation could not be resumed; started a new one.` after upgrading from 0.2.2 or earlier | Conversations from before 0.2.3 (the Codex releases) were stored elsewhere and do not carry over. |
| A tool result starting with `ERROR:` inside the conversation | The agent hit a marimo or allowlist problem; the text contains the fix (for example the URL to open). |
| `Marimo at <url> rejected the request (HTTP 401).` (or 403) and the hint mentions `.hailer/kernel.json` and `HAILER_MARIMO_TOKEN` | The server was started with a token. If Hailer started it, run Hailer in the same workspace, so it reads the token from `.hailer/kernel.json`. For a server you started yourself, export its token as `HAILER_MARIMO_TOKEN` (kept in memory only), or restart it with `--no-token`. |

## Known limitations

- `marimo._code_mode` is a private API; Hailer pins marimo 0.24.2 and may need changes for other versions.
- The notebook must be open in a browser; a headless server without a tab has nothing to execute against.
  Switching notebooks opens a new tab; the old one stays open until you close it or `/notebook close` it.
  When the agent switches notebooks and the browser is slow to load, the CLI may open the same notebook a
  second time after the turn (it opens the URL once more when it still sees no kernel session); the extra
  tab is harmless, close it.
- If you press Ctrl+C while the agent is in the middle of `notebook_create` / `notebook_open`, that tool call
  may still finish and switch the active notebook a moment later. Hailer re-reads the state after every
  turn and before `/notebook` and `/status`, so the next command shows the right notebook; a `/notebook
  new` typed in that instant can still be overtaken by the late switch.
- Both `uvx hailer` and `uvx hailer notebook` start or reuse this workspace's marimo server and stop only
  a server they started. An explicitly pinned server must already be running.
- Every request carries about 10 KB of instructions and 5 KB of tool definitions (measured with an empty
  project context) plus the conversation; an endpoint with a strict request-size limit needs room for that.
- The endpoint must support function calling in the standard OpenAI shape. An endpoint that streams tool
  calls in a non-standard way may lose their arguments; `stream = false` on the provider avoids that.
- After Ctrl+C during a long `marimo_execute`, the turn ends at once but the code keeps running in the
  kernel until it finishes, or until you interrupt or restart the kernel from the notebook.
- The docker kernel needs Docker, which is not always an option: Docker Desktop is free for personal use
  and small businesses, but larger organisations need a paid subscription (check Docker's current terms),
  and managed machines often block Docker Desktop, WSL2 or Hyper-V. That is why `local` stays the default
  and fully supported.
- Reading large files through a Windows bind mount is slower than reading them from a local folder; how
  much slower for large Parquet files has not been measured yet.
- The release workflow builds the `linux/arm64` kernel image (Apple Silicon, ARM Linux), but the CI
  integration test only covers amd64. Podman is not supported.
- The docker kernel has only the packages in the image (marimo, Polars, DuckDB, altair, plotly); anything
  else needs an image of your own, built `FROM` Hailer's (so it keeps the version label) and named in
  `[kernel].image`. There is one kernel per workspace, and
  changing `[kernel]` settings while one is kept running needs `uvx hailer kernel stop` first.

## Architecture

```
                   Model endpoint
                (OpenAI, or your own endpoint:
                 Responses API or Chat Completions)
                         ▲
                         │ HTTPS
                ┌────────┴────────┐
                │  Hailer         │   one Python process: a LangChain agent
Terminal ─────► │  conversation   │   (create_agent + ChatOpenAI), Hailer's system
 uvx hailer     │  + tools        │   prompt, the conversation kept in .hailer/
                │                 │
                │  marimo_execute, marimo_status, notebook_cells, notebook_list,
                │  notebook_create, notebook_open, notebook_close,
                │  list_periods, load_skill, read_skill_file, fetch_page
                └────────┬────────┘
                         │ HTTP + SSE  (/api/sessions, /api/kernel/execute),
                         │ with the server's token
                ┌────────▼────────┐
                │ Marimo runtime  │   scratchpad over the kernel globals +
                │ Polars · DuckDB │   marimo._code_mode for durable cells
                │ Python          │
                └────────┬────────┘   local (default): Hailer's own Python, as you
                         │            docker: a Linux container that sees only the
                    Browser UI        notebooks folder and, read-only, the data folder
```

The CLI runs the agent in its own process, with the model and provider taken from `hailer.toml`. The agent
has exactly the eleven tools above and nothing else: no shell, no file editing. The tools talk to the
running marimo server over plain HTTP. Code from the agent runs in marimo's *scratchpad*, a temporary
namespace that can read every notebook variable, and durable changes (new cells, edits, runs) go through
marimo's code-mode API so they appear immediately in the browser. The notebook file on disk is written by
marimo itself, never edited behind the kernel's back.

Where marimo and its kernel run is the *kernel runtime*, set by `[kernel] runtime`: `local` (the default)
starts marimo in Hailer's own Python, `docker` starts it in a container (see
[Isolated kernel (Docker)](#isolated-kernel-docker)). The agent, the conversation and the API key stay in
the Hailer process either way.

## Logging

Normal runs print only warnings. `uvx hailer --verbose` (or `HAILER_LOG_LEVEL=DEBUG`) logs provider,
session, marimo and tool activity and shows full tracebacks. Log output passes through a redaction filter
that masks bearer tokens and the values of environment variables whose names end in `_KEY`, `_TOKEN`,
`_SECRET` or `_PASSWORD`.

## Development

To work on Hailer itself:

```bash
git clone https://github.com/OpenAfterHours/hailer.git
cd hailer
uv sync --locked
uv run hailer login openai
uv run hailer
```

Use `uv run hailer ...` in the checkout so you run the code you are changing. The repository already
includes `hailer.toml`, a starter notebook and the data folder. Configure another provider in
`hailer.toml` before logging in if needed.

The checkout also has optional sample data: six months of synthetic sales orders, with an extra column
in later months. Generate it before starting Hailer:

```bash
uv run python scripts/make_sample_data.py          # writes data/25-01 sales.parquet ... 25-06
uv run python scripts/make_sample_data.py --help   # --out, --months, --rows, --start, --seed, --dataset
```

Hailer pins `marimo==0.24.2` because it drives the private `marimo._code_mode` API, which has no stability
guarantee. The agent uses `langchain`, `langchain-openai` and `langgraph-checkpoint-sqlite` within their
declared major-version bounds. Upgrade deliberately and re-run the tests.

## Tests

```bash
uv run pytest
```

The suite is offline: it needs no API key, no marimo server, no model endpoint, no network and no Docker.
The marimo protocol is exercised against a local fake server (`tests/fake_marimo.py`, token-checked like a
server Hailer starts), the agent against a scripted chat model and against a strict Chat-Completions-only
fake gateway on loopback (`tests/fake_gateway.py`, which rejects unknown request fields and model names the
way internal gateways do), and the credential store against an in-memory backend. The kernel runtimes are
tested without starting anything: `tests/test_kernel.py` (path map, `kernel.json`, the kernel's
environment, the local runtime with its processes faked), `tests/test_kernel_docker.py` (the docker runtime
against `tests/fake_docker.py`, a scripted `docker` CLI that keeps containers and networks with ids,
labels and `--filter`, like Docker 29), `tests/test_kernel_image.py`, `tests/test_forward.py` (the forwarder
on real loopback sockets) and `tests/test_build_kernel_image.py`. `tests/fake_kernel.py` holds the shared
pieces.

`tests/test_docker_integration.py` drives the real docker runtime: it starts a kernel for a temporary
workspace with sample sales data, opens the notebook in headless Chrome, and checks that code runs as a
non-root user in `/work`, a code-mode cell lands in the host notebook, writes to the data folder and the
network are refused, no host secret or the server token is visible, and stopping leaves nothing behind.
It is opt-in and never pulls the image:

```bash
uv run python -m scripts.build_kernel_image --load     # the image for this checkout, into local Docker
HAILER_DOCKER_TESTS=1 uv run pytest tests/test_docker_integration.py -rs
```

(In PowerShell, set the variable first: `$env:HAILER_DOCKER_TESTS = "1"`.)
`HAILER_DOCKER_TESTS=1` skips, with the reason, when Docker, the image or a browser is missing;
`HAILER_DOCKER_TESTS=strict` fails instead. `HAILER_TEST_CHROME` picks the browser (otherwise
`google-chrome` or `chromium` on PATH, or Chrome's usual install folder on Windows and macOS), and
`HAILER_KERNEL_IMAGE` another image.

`.github/workflows/test.yml` runs the suite on Ubuntu and Windows with Python 3.12 and 3.13 for every
push to `main` and every pull request; the repository requires those four checks by name. A fifth job,
*Docker kernel*, builds the image from the checkout on Ubuntu and runs the integration test with
`HAILER_DOCKER_TESTS=strict` (Linux only: GitHub's Windows runners only run Windows containers). It is
not a required check, but the release waits for it.

## Releasing

```bash
uv run python -m scripts.release            # patch bump: 0.1.0 -> 0.1.1
uv run python -m scripts.release minor      # 0.1.0 -> 0.2.0
uv run python -m scripts.release major      # 0.1.0 -> 1.0.0
uv run python -m scripts.release --dry-run  # preflight checks and the plan, nothing changed
```

Run it from a clean `main` that matches `origin/main`. The script writes the new version to `pyproject.toml`,
`src/hailer/__init__.py` and `uv.lock`, then runs the test suite. If the tests fail, the version files are
restored and nothing is committed. If they pass, it commits `Release vX.Y.Z`, tags `vX.Y.Z` and pushes the
branch and tag atomically. The tag triggers `.github/workflows/release.yml`, which:

1. runs every job in `test.yml` again (the four platforms and the Docker kernel job) and builds the sdist
   and wheel;
2. builds the kernel image for `linux/amd64` and `linux/arm64` with `scripts/build_kernel_image.py` and
   pushes it as `ghcr.io/openafterhours/hailer-kernel:X.Y.Z` (the job checks that the tag, `pyproject.toml`
   and `hailer.__version__` agree, since the image's version label must match the package);
3. only when all of that succeeds, publishes to PyPI through the `pypi` environment (trusted publishing,
   no token to store) and creates the GitHub release with the files attached. The image goes first so no
   released Hailer points at a missing image.

`--no-push` stops after the local commit and tag, `--version X.Y.Z` releases an exact version (pre-releases
such as `1.2.0rc1` are accepted), and arguments after `--` are passed to pytest.

**One-time step for the kernel image.** GHCR creates the `hailer-kernel` package as private on the first
push. After the first release, make it public in the organisation's package settings
(github.com/orgs/OpenAfterHours/packages, `hailer-kernel`, *Package settings*, *Change visibility*);
otherwise `uvx hailer kernel pull` and the first docker start fail for everyone outside the organisation
(they see `The kernel image for Hailer X.Y.Z is not published (or not visible to you)`). If a
`hailer-kernel` package was ever pushed by hand, also give this repository the *Write* role under *Manage
Actions access* on the same page, or the workflow's push is refused.

**Development versions have no published image.** Build one for the checkout into your local Docker with
`uv run python -m scripts.build_kernel_image --load`. It uses the build context `hailer kernel build` and
the release use (the packaged Dockerfile and the `hailer` package this checkout runs); `--dry-run` prints
the `docker buildx build` command, `--tag` renames the image, and `--help` lists the rest.

Repository rulesets restrict this: `main` cannot be force-pushed or deleted and changes to it must come
through a pull request with the test checks green, and `v*` tags can only be created by repository admins,
who also bypass the pull-request rule so the release script can push directly. The `pypi` environment only
deploys from `v*` tags, and `.github/workflows/members-only.yml` closes pull requests opened from forks by
people outside the OpenAfterHours organization. Open your own pull requests from a branch in this repository;
those are always kept.
