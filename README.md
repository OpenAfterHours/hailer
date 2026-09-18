# Hailer

Hailer lets you chat with your data. Put CSV, Parquet or JSON files in a folder, ask questions in plain
English in the terminal, and the agent explores the data with Polars and DuckDB inside a live marimo
notebook, putting tables, charts and summaries in the notebook open in your browser. It is built for
analysts who want to go from a file to a chart in minutes, whatever the data is about: sales, operations,
finance, survey results, logs. Datasets stay on your machine: the model endpoint receives your messages,
code, and compact results, never the raw files.

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
                         │ HTTP + SSE  (/api/sessions, /api/kernel/execute)
                ┌────────▼────────┐
                │ Marimo runtime  │   scratchpad over the kernel globals +
                │ Polars · DuckDB │   marimo._code_mode for durable cells
                │ Python          │
                └────────┬────────┘
                         │
                    Browser UI
```

The CLI runs the agent in its own process, with the model and provider taken from `hailer.toml`. The agent
has exactly the eleven tools above and nothing else: no shell, no file editing. The tools talk to the
running marimo server over plain HTTP. Code from the agent runs in marimo's *scratchpad*, a temporary
namespace that can read every notebook variable, and durable changes (new cells, edits, runs) go through
marimo's code-mode API so they appear immediately in the browser. The notebook file on disk is written by
marimo itself, never edited behind the kernel's back.

## Requirements

- Windows 11 is the primary target; macOS and Linux work too. No Git Bash, curl, jq or WSL is needed.
- [uv](https://docs.astral.sh/uv/). Hailer runs with `uvx`, so no project or virtual environment is needed;
  uv fetches Python 3.12 or newer if the machine has none.
- A web browser. **A kernel session only exists while the notebook is open in a browser tab**; Hailer
  tells you the URL to open when it is not.
- An API key for the provider you configure: OpenAI, or your own OpenAI-compatible endpoint.
- Pinned dependency: `marimo==0.24.2` (Hailer drives its private `marimo._code_mode` API, which has no
  stability guarantee). The agent uses `langchain`, `langchain-openai` and `langgraph-checkpoint-sqlite`
  within their current major versions. Upgrade deliberately and re-run the tests. Nothing Hailer installs
  is a standalone executable; it is all Python packages.

## Installation

Nothing to install beyond uv. `uvx hailer ...` fetches Hailer from PyPI the first time, keeps it in uv's
cache with an environment of its own (marimo, Polars, DuckDB and the rest), and runs it from there. No
`pyproject.toml` or `.venv` is needed in your folder, which suits ad-hoc analysis. `uvx hailer@latest ...`
picks up a newer release; `uvx hailer@X.Y.Z ...` pins one.

For a permanent `hailer` command instead, `uv tool install hailer` (or `pip install hailer` into an
environment of your own) and drop the `uvx` from the commands below.

Working on Hailer itself: from a checkout of this repository, `uv sync` and then `uv run hailer ...`, so
the code you are changing is what runs. The checkout also has optional sample data (six months of
synthetic sales orders, later months gain a column):

```bash
uv run python scripts/make_sample_data.py          # writes data/25-01 sales.parquet ... 25-06
uv run python scripts/make_sample_data.py --help   # --out, --months, --rows, --start, --seed, --dataset
```

## Quick start

In an empty folder, or the one that holds your data, one terminal is enough:

```bash
uvx hailer init           # hailer.toml, .config/hailer/{context,skills,prompts}, notebooks/analysis.py, data/
uvx hailer login openai   # stores your API key in the OS credential store (or set OPENAI_API_KEY)
uvx hailer notebook       # starts marimo, opens the notebook in your browser, and chats right here
```

`hailer init` creates the starter notebook (`[hailer].notebook`) and the data folder (`[hailer].data_dir`)
that `hailer.toml` names. It never overwrites anything except `hailer.toml` itself, and only with
`--force`, so running it again just fills in what is missing (a deleted notebook, say). Put the files you
want to analyse in `data/`: CSV, Parquet or JSON, with any names (see [Data conventions](#data-conventions)).
The notebook lists them when it opens, and you can start asking straight away.

`hailer notebook` runs the startup checks, starts a token-less marimo server on the **notebooks folder**
in the background (log in `.hailer/marimo.log`), opens the active notebook's URL so the kernel gets a
session, then runs the chat in the same terminal. One server hosts every notebook in the folder, and
marimo's own home page (the server URL without `?file=`) lists them all. When you leave the chat (`/exit`,
Ctrl+Z Enter, or Ctrl+C at the prompt) it stops the marimo server it started. If this workspace's marimo
server is already running (started on this notebooks folder, or with one of its notebooks open), it is
reused and left running. A server for another folder, such as another git worktree of the same repository,
is never used, so several workspaces can each run their own.

The notebook opens in marimo's **app view**: you see the results, tables and charts the agent produces,
not the code behind them (the URL carries `view-as=present`). To see or edit the code, press `Ctrl+.`
(`Cmd+.` on macOS) in the notebook, or choose *Toggle app view* from the notebook menu; the same shortcut
switches back. It is a normal edit session either way, so the agent works in it exactly the same.

Flags: `--port N` (default 2718; a free port is chosen when it is busy), `--no-browser` (print the URL
instead of opening it), `--keep-marimo` (leave the server running after the chat), `--foreground` (just run
marimo attached to this terminal, no chat; it opens marimo's home page unless `--no-browser` is given),
`--new` (start a fresh conversation).

Bare `uvx hailer` does the same as `uvx hailer notebook` with its defaults: it reuses this workspace's
marimo server or starts one, opens the notebook, chats, and stops only a server it started.

```bash
uvx hailer
```

A pinned server (`[hailer].marimo_url` in `hailer.toml`, or `HAILER_MARIMO_URL`) is used as is, whichever
folder it serves. If it does not answer, Hailer says so and exits; it never starts a server in its place.

```text
+--------------------- Hailer ----------------------+
| Model:      gpt-5.5                               |
| Provider:   openai                                |
| Notebook:   notebooks\analysis.py                 |
| Notebooks:  notebooks                             |
| Workspace:  C:\projects\hailer                    |
| Web access: none                                  |
+---------------------------------------------------+
Type /help for commands.

You > what data do I have?

Hailer >
Six monthly sales files, 2025-01 to 2025-06, 400 orders each (discount appears from 2025-04).
I loaded them into one table, df, with a period column and added a summary to the notebook.

You > chart revenue by month and region

Hailer >
Revenue grew 8% from January to June, almost all of it in the North (+33%); the other regions are flat.
Added: revenue_by_month (table) and a line chart by region.

You > what is behind the North's growth?

Hailer >
Home (+101%) and Sports (+85%); Electronics fell 12%.
Added a category breakdown for the North with a bar chart.

You > exclude returns

Hailer >
Done. Returns were 2.7% of orders; the North is still up 29% without them.
The table and both charts now exclude returned orders.

You > which channel sells more?

Hailer >
Online: 55% of orders and 55% of revenue.
Added a channel comparison below the category chart.
```

Hailer resumes your previous conversation on the next start; use `uvx hailer --new` or `/new` for a fresh
thread. `uvx hailer doctor` runs the same startup checks and prints fixes.

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
| `/notebook` | Active notebook, folder, marimo state, URL and launch command. |
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
  writes it to the state file at startup, so the agent's tool server sees the same notebook.
- `/notebook` also lists the notebooks you worked in recently (`Recent:`), most recent first.

```toml
[hailer]
notebook      = "notebooks/analysis.py"   # the default (and first) notebook
notebooks_dir = "notebooks"               # where /notebook new and the agent's notebook_create put files
```

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

Any endpoint that speaks the OpenAI Responses API or Chat Completions works: an internal gateway, Azure
OpenAI, LiteLLM, vLLM, Ollama and so on.

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

Hailer sends the request itself, straight to `base_url`, so what the endpoint sees is small and predictable:
the key as `Authorization: Bearer ...`, your `http_headers`, each `env_http_headers` entry whose variable is
set, your `query_params`, and a body with `model`, `messages` (one system message with Hailer's
instructions, then the conversation), `tools` (Hailer's eleven tools as ordinary functions) and `stream`.
Every request names the model you configured; there are no side requests under other model names. The
endpoint must support function calling. `HTTP_PROXY`, `HTTPS_PROXY` and `NO_PROXY` are honoured, and TLS uses
the operating system's certificate store, so a company root certificate installed on the machine works.

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

## Secrets

`uvx hailer login <provider>` stores the API key in the operating system's credential store through
`keyring` (Windows Credential Manager on Windows) under the service name `hailer`. Hailer reads it when the
agent starts and hands it to the HTTP client inside its own process, which sends it only as the
`Authorization` header of requests to that provider's endpoint. It is never written to `hailer.toml`, the
session file, command history or logs, and never passed on a command line or to another process.
`uvx hailer logout <provider>` removes it.

An environment variable named by `env_key` takes precedence over the credential store, which keeps scripted
and CI use simple. Note that any process running as the same user can read both credential-store entries and
environment variables; the gain is keeping the key out of files and history, not isolation from your own
account.

Every provider needs a key, including the built-in `openai` one. A missing key stops `hailer` at startup
with the `login` command to run.

## Allowed web domains

By default the agent has **no internet access**: it has no shell, and its `fetch_page` tool refuses every
URL. To let it read specific sites for context:

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
result. `/context`, `/status` and the startup panel show the active allowlist. Note that code the agent
runs in the notebook kernel is ordinary Python on your machine: the allowlist governs the agent's own web
tool, not what notebook code can do (see [Security](#security)).

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

## Slash commands

| Command | What it does |
|---|---|
| `/help` | Show the command list. |
| `/status` | Model, provider, credentials source, thread id, token usage, marimo state, web allowlist. |
| `/new` | Start a new conversation thread (context files are re-read). |
| `/model <name>` or `/model <provider>:<name>` | Switch model (and provider); starts a new thread. |
| `/notebook [list \| new <name> [--empty] \| open <name> \| close [name]]` | Show or switch the active notebook (see *Working with several notebooks*). |
| `/context` | List loaded context files, skills, prompts and the web allowlist. |
| `/skill <name> [message]` | Run a turn with a project skill attached. |
| `/prompt <name> [args]` | Send a saved prompt from `.config/hailer/prompts`. |
| `/reload` | Re-read `.config/hailer`; applies from your next message. |
| `/clear` | Clear the screen. |
| `/exit`, `/quit` | Exit Hailer (Ctrl+C at the prompt, or Ctrl+Z then Enter, also exit). |

Ctrl+C while the agent is working cancels that turn, including the request to the model endpoint, and
returns to the prompt. The conversation carries on: your interrupted message is kept and sent together with
the next one. The conversation itself is stored in `.hailer/threads.sqlite` in the workspace, which is how
`hailer` resumes it after a restart; `/new` (or `hailer --new`) discards it and starts another.

Other subcommands: `hailer notebook [--port N] [--no-browser] [--keep-marimo] [--foreground] [--new]` (the
one-command session described in Quick start; marimo runs on the notebooks folder), `hailer exec -c "code"`
(or `hailer exec script.py`, `hailer exec -` for stdin) to run Python in the active notebook's kernel
yourself, `hailer status`, `hailer doctor`,
`hailer login|logout <provider>`, `hailer init [--force]`. Global options: `--verbose`, `--config <path>`,
`--workspace <path>`, `--new`, `--version`.

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

## Logging

Normal runs print only warnings. `uvx hailer --verbose` (or `HAILER_LOG_LEVEL=DEBUG`) logs provider,
session, marimo and tool activity and shows full tracebacks. Log output passes through a redaction filter
that masks bearer tokens and the values of environment variables whose names end in `_KEY`, `_TOKEN`,
`_SECRET` or `_PASSWORD`.

## Tests

```bash
uv run pytest
```

The suite is offline: it needs no API key, no marimo server, no model endpoint and no network. The marimo
protocol is exercised against a local fake server, the agent against a scripted chat model and against a
strict Chat-Completions-only fake gateway on loopback (`tests/fake_gateway.py`, which rejects unknown request
fields and model names the way internal gateways do), and the credential store against an in-memory backend.

`.github/workflows/test.yml` runs the same suite on Ubuntu and Windows with Python 3.12 and 3.13 for every
push to `main` and every pull request.

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
branch and tag atomically. The tag triggers `.github/workflows/release.yml`, which runs the test suite again
on every platform in `test.yml` and builds the sdist and wheel; only when both succeed does it publish to
PyPI through the `pypi` environment (trusted publishing, no token to store) and create the GitHub release
with the files attached.

`--no-push` stops after the local commit and tag, `--version X.Y.Z` releases an exact version (pre-releases
such as `1.2.0rc1` are accepted), and arguments after `--` are passed to pytest.

Repository rulesets restrict this: `main` cannot be force-pushed or deleted and changes to it must come
through a pull request with the test checks green, and `v*` tags can only be created by repository admins,
who also bypass the pull-request rule so the release script can push directly. The `pypi` environment only
deploys from `v*` tags, and `.github/workflows/members-only.yml` closes pull requests opened from forks by
people outside the OpenAfterHours organization. Open your own pull requests from a branch in this repository;
those are always kept.

## Security

**Sent to the configured model endpoint:** your messages; Hailer's system prompt plus everything in
`.config/hailer/context`; the name and description of each skill, and a skill's full body once loaded; every
tool call argument (that is, the Python the agent writes) and the tool results, truncated to
`max_tool_output_chars`; and pages fetched from allowed domains. The instructions name the workspace,
notebooks and data folders by path. With the `openai` provider this goes to OpenAI; with a custom provider,
to the `base_url` you configured, and nowhere else: when older turns are summarised, the same endpoint and
model write the summary.

**Not sent:** your data files or any dataframe, unless code explicitly prints or returns it (the agent is
instructed to inspect schemas, samples and aggregates and to keep outputs compact); API keys or other
secrets (the key stays inside the Hailer process and is masked in logs).

**What the agent can do:** it has Hailer's eleven tools and nothing else: no shell, no file tools, no
tools from other software on your machine. Nothing asks for approval before a tool runs. The tool that
matters is `marimo_execute`: it runs the Python the model writes in your notebook kernel, as you, with
your file and network access. That is what makes the analysis possible, and it is not sandboxed. The
instructions forbid destructive file operations and sending data anywhere, but instructions are not an
enforcement boundary: use an endpoint and model you trust, keep the workspace to data the agent may read,
and press Ctrl+C if a turn goes somewhere you did not intend.

**Tracing:** Hailer's agent library (LangChain) can send full conversation traces to the LangSmith service
when variables such as `LANGSMITH_TRACING` are set in the environment. Hailer switches that off for its own
process at startup, so a variable left over from another project cannot send your prompts and results to a
third party. Set `HAILER_TRACING=1` if you do want the tracing variables in your environment to apply.

**On disk:** the conversation (your messages, the agent's replies, tool calls and their truncated results)
is stored unencrypted in `.hailer/threads.sqlite` inside the workspace until `/new` or `hailer --new`
replaces it. `.hailer/` holds only this local state (the conversation, `session.json`, `notebook.json`,
`marimo.log`) and carries its own `.gitignore` containing `*`, so git ignores the folder in any repository
without an entry in yours; Hailer never edits your `.gitignore`.

## Troubleshooting

`uvx hailer doctor` runs five checks (config, notebook, credentials, marimo, session) and, when a session
exists, confirms that marimo's code-mode API is available in the kernel. With several marimo servers
running it checks the one serving this workspace; having none is only a warning, because `uvx hailer`
starts one.

| Message | Meaning and fix |
|---|---|
| `No marimo server is running for this workspace.` (from `doctor` or `exec`; `status` says `none for this workspace`) | No running server serves this workspace's notebooks folder; servers for other folders are ignored. `uvx hailer` (or `uvx hailer notebook`) starts one and runs the chat in the same terminal; `uvx hailer notebook --foreground` runs only marimo, with Hailer's own Python environment, so the notebook can import Polars, DuckDB and `hailer.periods`. Servers started with `--no-token` register themselves so Hailer finds them; otherwise set `marimo_url` in `hailer.toml`. |
| `Marimo is not running at <url>.` with a hint about `[hailer].marimo_url` | The URL pinned by `marimo_url` or `HAILER_MARIMO_URL` does not answer. Start marimo there, or remove the setting so Hailer starts its own server for this workspace. |
| `Marimo exited early (code N)` or `Marimo did not answer on http://127.0.0.1:2718 within 60 s`, followed by `Log: .hailer\marimo.log` and its last lines | `hailer notebook` could not start marimo. The log tail usually names the cause (port in use by something else, a syntax error in the notebook, marimo not installed in the environment). |
| `the notebook is not open in a browser` followed by `Open http://... in your browser.` | The server is up but has no kernel session. Open the URL; Hailer opens it for you once at startup. The URL ends in `&view-as=present` (app view); `Ctrl+.` in the notebook shows the code. |
| `not found: <path>` for the notebook | The configured notebook (`[hailer].notebook`, or `HAILER_NOTEBOOK`) does not exist. `uvx hailer init` creates it from the starter template (your `hailer.toml` is kept), or fix the path; a deleted *active* notebook is not the cause, because Hailer already falls back to the configured one when the remembered notebook is gone. New notebooks are created from the chat with `/notebook new <name>`. |
| `No notebook named '...' in notebooks.` or `... is outside the notebooks folder.` | `/notebook open` (or the agent's `notebook_open`) only opens marimo notebooks inside `[hailer].notebooks_dir`; the hint lists the available names. Move the file into the folder or point `notebooks_dir` at it. |
| `INTERNAL_MODEL_API_KEY is not set (required by provider 'internal')`, or the same for `OPENAI_API_KEY` and provider `'openai'` | Every provider needs a key. Run `uvx hailer login <provider>` or set the variable in this terminal. Releases up to 0.2 could use a ChatGPT login for the `openai` provider; that is gone, so create an API key. |
| `The model endpoint rejected the API key for provider '...'` | The endpoint returned 401. Re-run `hailer login <provider>`. |
| `Unknown model '...' for provider '...'` | The endpoint does not know `[model].name` (or the name given to `/model`). Its reply is quoted after `The endpoint said:`. |
| `Could not reach the model endpoint at <base_url>` | Check `base_url`, VPN or proxy (`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`), and that the endpoint is running. |
| `The endpoint at <base_url> did not accept the request.` | The endpoint answered 404/405. The hint names the request Hailer sent (`POST <base_url>/responses` or `POST <base_url>/chat/completions`): either `base_url` is wrong, or the endpoint implements the other API, in which case switch `wire_api`. The endpoint's own reply is quoted on the last line, after `The endpoint said:`. |
| `The endpoint at <base_url> rejected the request (HTTP 400)` (or 422) | The endpoint understood the request but refused a field or a value. The line `The endpoint said:` quotes its reply: fix what it names (`stream = false` or `stream_options = false` on the provider, `reasoning_effort = ""` under `[model]` to stop sending reasoning effort). `hailer --verbose` logs the HTTP traffic. |
| `The endpoint at <base_url> refused access (HTTP 403).` | The key was accepted but is not allowed for this model, route or organisation. Check the endpoint's access policy and the provider's `http_headers` / `env_http_headers`. |
| `The endpoint at <base_url> is unavailable (HTTP 429)` (or 5xx) | Rate limit or an outage on the endpoint's side. Retry in a moment. |
| `The conversation no longer fits the model's context window.` | Start again with `/new`, and lower `[model].summarize_after_tokens` so older turns are summarised before the model's limit is reached. |
| `Warning: unknown key [model_providers.x].merge_messages ... is ignored` (also `parallel_tool_calls`, `requires_openai_auth`, `[hailer].codex_home`, `[web].allow_shell_network`) | Settings from releases up to 0.2 that no longer mean anything. The file still loads; delete the keys to silence the warning. |
| `Previous conversation could not be resumed; started a new one.` after upgrading from 0.2 | Conversations from before 0.3 were stored elsewhere and do not carry over. |
| A tool result starting with `ERROR:` inside the conversation | The agent hit a marimo or allowlist problem; the text contains the fix (for example the URL to open). |
| marimo answers 401 or 403 and the hint mentions `HAILER_MARIMO_TOKEN` | The server was started with a token. Export it as `HAILER_MARIMO_TOKEN` (kept in memory only), or restart marimo with `--no-token`. |

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
- `uvx hailer notebook` starts and stops marimo for you; plain `uvx hailer` expects a running server
  and tells you how to start one.
- Every request carries about 10 KB of instructions and 5 KB of tool definitions (measured with an empty
  project context) plus the conversation; an endpoint with a strict request-size limit needs room for that.
- The endpoint must support function calling in the standard OpenAI shape. An endpoint that streams tool
  calls in a non-standard way may lose their arguments; `stream = false` on the provider avoids that.
- After Ctrl+C during a long `marimo_execute`, the turn ends at once but the code keeps running in the
  kernel until it finishes, or until you interrupt or restart the kernel from the notebook.
