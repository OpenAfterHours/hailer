# Hailer

Hailer is a conversational local analytics CLI backed by Codex and a live Marimo notebook. You chat in the
terminal; the agent runs Polars and DuckDB inside the running marimo kernel and puts tables, charts and
summaries in the notebook you have open in your browser. Datasets stay on your machine: the model endpoint
receives your messages, code, and compact results, never the raw files.

## Architecture

```
                   Model endpoint
                (OpenAI or your own
                 Responses API gateway)
                         ▲
                         │
                ┌────────┴────────┐
                │  Codex harness  │   codex app-server, driven through the
Terminal ─────► │  Hailer         │   openai-codex Python SDK; sandboxed shell,
 uv run hailer  │  conversation   │   Hailer's system prompt as base instructions
                └────────┬────────┘
                         │ MCP (stdio)
                ┌────────▼────────┐
                │ hailer-mcp      │   marimo_execute, marimo_status, notebook_cells,
                │ (Hailer tools)  │   notebook_list, notebook_create, notebook_open, notebook_close,
                │                 │   list_periods, load_skill, read_skill_file, fetch_page
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

The CLI starts a Codex app-server through the Python SDK, with the model and provider taken from
`hailer.toml`. Codex is given exactly one tool namespace of Hailer's own: a small stdio MCP server that
talks to the running marimo server over plain HTTP. Code from the agent runs in marimo's *scratchpad*, a
temporary namespace that can read every notebook variable, and durable changes (new cells, edits, runs) go
through marimo's code-mode API so they appear immediately in the browser. The notebook file on disk is
written by marimo itself, never edited behind the kernel's back.

## Requirements

- Windows 11 is the primary target; macOS and Linux work too. No Git Bash, curl, jq or WSL is needed.
- Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
- A web browser. **A kernel session only exists while the notebook is open in a browser tab**; Hailer
  tells you the URL to open when it is not.
- Either an existing Codex login (from the Codex app or `codex` CLI on the same machine) or an API key for
  the provider you configure.
- Pinned dependencies: `marimo==0.24.2` (Hailer drives its private `marimo._code_mode` API, which has no
  stability guarantee) and `openai-codex==0.154.0` (the config override keys and app-server behaviour were
  verified against this version). Upgrade deliberately and re-run the tests.

## Installation

From a checkout of this repository:

```bash
uv sync
```

Released versions are on PyPI:

```bash
uv tool install hailer     # or: pip install hailer
```

Optional sample data (six months of synthetic PRA101-style files, later months gain a column):

```bash
uv run python scripts/make_sample_data.py          # writes data/25-01 pra101.parquet ... 25-06
uv run python scripts/make_sample_data.py --help   # --out, --months, --rows, --start, --seed, --dataset
```

## Quick start

One terminal is enough:

```bash
uv run hailer init        # writes hailer.toml and .config/hailer/{context,skills,prompts} with examples
uv run hailer notebook    # starts marimo, opens the notebook in your browser, and chats right here
```

`hailer notebook` runs the startup checks, starts a token-less marimo server on the **notebooks folder**
in the background (log in `.hailer/marimo.log`), opens the active notebook's URL so the kernel gets a
session, then runs the chat in the same terminal. One server hosts every notebook in the folder, and
marimo's own home page (the server URL without `?file=`) lists them all. When you leave the chat (`/exit`,
Ctrl+Z Enter, or Ctrl+C at the prompt) it stops the marimo server it started. If a marimo server is already
running with a notebook from the folder open, it is reused and left running.

The notebook opens in marimo's **app view**: you see the results, tables and charts the agent produces,
not the code behind them (the URL carries `view-as=present`). To see or edit the code, press `Ctrl+.`
(`Cmd+.` on macOS) in the notebook, or choose *Toggle app view* from the notebook menu; the same shortcut
switches back. It is a normal edit session either way, so the agent works in it exactly the same.

Flags: `--port N` (default 2718; a free port is chosen when it is busy), `--no-browser` (print the URL
instead of opening it), `--keep-marimo` (leave the server running after the chat), `--foreground` (just run
marimo attached to this terminal, no chat), `--new` (start a fresh conversation).

Chat only, when marimo is already running (started by `hailer notebook --keep-marimo` or by
`uv run marimo edit notebooks --no-token` with the notebook open in a browser):

```bash
uv run hailer
```

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

You > load the PRA101 files and show me the periods

Hailer >
Loaded 6 months, 2025-01 to 2025-06 (risk_weight appears from 2025-04).
I added a period table and a period selector to the notebook.

You > compare the latest two periods

Hailer >
RWA rose 5.6% from May to June, mostly in corporate exposures.
Added: month-on-month movement table and a summary.

You > show the movement by exposure class

Hailer >
Corporate +7.9%, retail +1.2%, sovereign flat.
Added an exposure-class breakdown below the comparison.

You > exclude defaults

Hailer >
Done. Corporate still drives the movement; defaults were 3% of the increase.
The breakdown and summary now exclude default_flag rows.

You > create a chart

Hailer >
Added a bar chart of RWA movement by exposure class.

You > summarise the largest drivers

Hailer >
Three corporate counterparties account for two thirds of the increase.
I added a top-contributors table under the chart.
```

Hailer resumes your previous conversation on the next start; use `uv run hailer --new` or `/new` for a fresh
thread. `uv run hailer doctor` runs the same startup checks and prints fixes.

## Working with several notebooks

One analysis, one notebook: the agent can create a fresh notebook or go back to an earlier one, and you
can do the same with slash commands. Everything happens on the one marimo server started on the notebooks
folder, so nothing restarts.

In the chat, just say it:

```text
You > start a new notebook for the Q2 churn review and load the churn files
You > open the pra101 reconciliation notebook we did last week and add a chart by exposure class
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
  `hailer.periods` helpers, `WORKSPACE`, `DATA_DIR`, `period_files`, a welcome cell with the notebook's
  title), so the agent can start analysing straight away. `--empty` gives marimo's plain empty notebook.
- The active notebook is remembered in `.hailer/notebook.json` (git-ignored, next to `session.json`), so
  the next `uv run hailer` or `uv run hailer notebook` resumes where you left off. Delete the file to go
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

All model settings live in `hailer.toml` (repo root, or `.config/hailer/hailer.toml`). `uv run hailer init`
writes a commented starter file.

### OpenAI

```toml
[model]
name = "gpt-5.5"
provider = "openai"
```

Codex reuses the ChatGPT login already on the machine. Alternatively store an API key with
`uv run hailer login openai`, or set `OPENAI_API_KEY` in the terminal.

### A custom or internal endpoint

```toml
[model]
name     = "risk-analyst-v3"          # whatever model id the gateway expects
provider = "internal"

[model_providers.internal]
base_url             = "https://llm.example.internal/v1"
wire_api             = "responses"                # or "chat" for a Chat Completions endpoint
# stream             = true                       # "chat" only: false if the gateway rejects stream = true
env_key              = "INTERNAL_MODEL_API_KEY"   # env var name; value from `hailer login internal` or the shell
requires_openai_auth = false
# name             = "Internal"
# http_headers     = { "X-Team" = "risk-analytics" }
# env_http_headers = { "X-Client-Id" = "INTERNAL_CLIENT_ID" }
# query_params     = { "api-version" = "2025-04-01-preview" }
```

Then:

```bash
uv run hailer login internal     # stores the key in the OS credential store (hidden prompt)
uv run hailer doctor             # config / notebook / credentials / marimo / session, with fixes
uv run hailer
```

Or set `INTERNAL_MODEL_API_KEY` in the terminal instead of logging in; an environment variable always wins
over the credential store. The startup panel and `status` show the provider with its base URL; `doctor` and
`status` show where the key came from (`from keyring` or `from env`), never the value.

**`wire_api` chooses the protocol the endpoint speaks**; `"responses"` must be streaming, `"chat"` streams unless
you turn it off:

| `wire_api` | What Hailer sends | Use it when |
|---|---|---|
| `"responses"` (default) | `POST {base_url}/responses` straight from Codex | The gateway implements the OpenAI Responses API. |
| `"chat"` | `POST {base_url}/chat/completions`, translated by Hailer | The gateway implements Chat Completions only (vLLM, Ollama, LiteLLM, most internal gateways). |

In both modes the gateway must tolerate a `GET {base_url}/models` probe at startup (a 404 is fine). With
`"responses"` it must also accept the `reasoning`, `include` and `client_metadata` request fields.

With `"chat"` Hailer starts a loopback bridge (`127.0.0.1`, random port) for the session, points Codex at it
and translates each request: the system prompt becomes the `system` message, the conversation becomes
`messages` with `tool_calls` / `tool` entries, Codex's function tools become Chat Completions `tools`, and the
streamed `delta.content`, `delta.tool_calls`, `reasoning_content` and final `usage` chunks come back as
Responses events. The bridge forwards the `Authorization` header (from `env_key`), `http_headers`,
`env_http_headers` and `query_params` unchanged and keeps no key of its own; `GET /models` is passed
through. Reasoning effort is sent as `reasoning_effort` when set. The gateway must support function calling.
By default it must also support streaming (`stream: true`; `stream_options.include_usage` is requested for
token counts). If it rejects streamed requests or cannot deliver server-sent events (some internal gateways
and proxies buffer or refuse them), set `stream = false` on the provider: the bridge then sends
`stream: false`, waits for the single JSON reply and hands Codex the same events in one go, so the answer
appears when the turn finishes rather than as it is generated. `stream` applies to `"chat"` only; Codex
always streams the Responses API. The bridge listens on `127.0.0.1` only and is exempted from
`HTTP(S)_PROXY` via `NO_PROXY` in the Codex child environment. Nothing else changes: the startup panel and
`/status` show the provider as `internal (https://..., chat completions)` (`chat completions, no streaming`
with `stream = false`), and endpoint errors name `/chat/completions`.

Hailer keeps the request small and predictable for gateways: its own system prompt replaces Codex's
built-in coding-agent prompt, and web search, multi-agent, plugin and app features are switched off for
the session, so the endpoint sees a handful of function tools plus Hailer's `mcp__hailer` tools.

Several providers can be declared; switch inside a session with `/model <name>` or
`/model <provider>:<name>` (this starts a new thread). One-off overrides: `HAILER_MODEL`,
`HAILER_MODEL_PROVIDER`.

## Secrets

`uv run hailer login <provider>` stores the API key in the operating system's credential store through
`keyring` (Windows Credential Manager on Windows) under the service name `hailer`. At startup Hailer reads
it and injects it **only into the Codex child process environment**, under the variable named by `env_key`.
It is never written to `hailer.toml`, the session file, shell history or logs, and never passed on a command
line. `uv run hailer logout <provider>` removes it.

An environment variable of the same name set in the shell takes precedence, which keeps scripted and CI use
simple. Note that any process running as the same user can read both credential-store entries and
environment variables; the gain is keeping the key out of files and history, not isolation from your own
account.

## Allowed web domains

By default the agent has **no internet access**: the Codex sandbox blocks outbound connections from shell
commands, and Hailer's `fetch_page` tool refuses every URL. To let it read specific sites for context:

```toml
[web]
allowed_domains = ["docs.pola.rs", "duckdb.org", "**.bankofengland.co.uk"]
max_page_bytes = 200000
allow_shell_network = false
```

- Exact hosts match themselves; `*.example.com` matches subdomains only; `**.example.com` matches the apex
  and its subdomains. Matching is case-insensitive and ignores ports. Loopback and private addresses are
  only allowed when listed literally (for example `127.0.0.1`).
- `fetch_page` checks the host before any network I/O and on every redirect, converts HTML to readable
  text, caps the size, and returns the text to the model. A refused URL tells the agent which domains are
  allowed so it can ask you to extend the list.
- `allow_shell_network = true` additionally enables Codex's network proxy for shell commands with the same
  domains (loopback is always added so the marimo server stays reachable). On Windows this path has quirks:
  `curl.exe` fails HTTPS inside the sandbox with a certificate-store error, and the sandbox cannot execute a
  Python interpreter that lives outside the workspace. Prefer `fetch_page`.

Text fetched from an allowed site becomes a tool result and is sent to the model endpoint like any other
result. `/context`, `/status` and the startup panel show the active allowlist.

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
- **prompts/**: `/prompt monthly-pack 2025-06` sends `prompts/monthly-pack.md` with `{{args}}` replaced.
- `/context` lists what is loaded; `/reload` re-reads the folder and applies to the next thread (`/new`).

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
| `/reload` | Re-read `.config/hailer`; applies to the next thread. |
| `/clear` | Clear the screen. |
| `/exit`, `/quit` | Exit Hailer (Ctrl+C at the prompt, or Ctrl+Z then Enter, also exit). |

Ctrl+C while the agent is working interrupts that turn and returns to the prompt.

Other subcommands: `hailer notebook [--port N] [--no-browser] [--keep-marimo] [--foreground] [--new]` (the
one-command session described in Quick start; marimo runs on the notebooks folder), `hailer exec -c "code"`
(or `hailer exec script.py`, `hailer exec -` for stdin) to run Python in the active notebook's kernel
yourself, `hailer status`, `hailer doctor`,
`hailer login|logout <provider>`, `hailer init [--force]`. Global options: `--verbose`, `--config <path>`,
`--workspace <path>`, `--new`, `--version`.

## Data conventions

Monthly files are named `YY-MM <dataset>.parquet`, for example `25-03 pra101.parquet` for March 2025. The
data itself has no period column; Hailer derives it from the filename. Later months may add columns.

`hailer.periods` (imported by the starter notebook, so the agent reuses it):

| Function | Purpose |
|---|---|
| `parse_period(name)` | `"25-03 pra101.parquet"` → `Period(2025, 3)`; `None` if the name does not match. |
| `scan_period_files(data_dir, name=None)` | Sorted `PeriodFile`s for one dataset (or all) in a folder. |
| `load_periods(files, columns=None)` | One Polars DataFrame with a leading `period` column (`YYYY-MM`); schema evolution handled with a relaxed diagonal concat. |
| `scan_periods(files)` | Lazy variant of `load_periods`. |
| `duckdb_periods_view(con, files, view_name="periods")` | Registers a DuckDB view over all files (`union_by_name`) with `period` derived from the filename. |
| `describe_periods(files)` | Compact text: months found, common columns, columns present only in some months. |

A file that cannot be read raises `MalformedParquetError` naming the file. The starter notebook
(`notebooks/analysis.py`) defines `WORKSPACE`, `DATA_DIR` and `period_files` and shows a welcome table.

## Logging

Normal runs print only warnings. `uv run hailer --verbose` (or `HAILER_LOG_LEVEL=DEBUG`) logs provider,
session, marimo and tool activity and shows full tracebacks. Log output passes through a redaction filter
that masks bearer tokens and the values of environment variables whose names end in `_KEY`, `_TOKEN`,
`_SECRET` or `_PASSWORD`.

## Tests

```bash
uv run pytest
```

The suite is offline: it needs no API key, no marimo server, no Codex process and no network. The marimo
protocol is exercised against a local fake server, the Codex SDK against a fake client, and the credential
store against an in-memory backend.

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
`max_tool_output_chars`; pages fetched from allowed domains; and the environment context Codex adds to each
turn (working directory, operating system, shell, date). With the `openai` provider this goes to OpenAI;
with a custom provider, to the `base_url` you configured.

**Not sent:** the parquet files or any dataframe, unless code explicitly prints or returns it (the agent is
instructed to inspect schemas, samples and aggregates and to keep outputs compact); API keys or other
secrets (they travel only as an environment variable of the Codex child process and are masked in logs).

**Sandboxing:** the agent's shell runs in Codex's workspace-write sandbox. It cannot reach the internet
unless `allow_shell_network` is on, can reach loopback (the marimo server), and on Windows cannot execute a
Python interpreter outside the workspace. All Python the agent needs runs in the marimo kernel through the
MCP server, which is the only tool namespace Hailer exposes. Approvals use Codex's `auto_review` mode;
calls to Hailer's own MCP tools are auto-approved on that server.

**Your own Codex configuration:** Codex reads `~/.codex/config.toml`, which for desktop-app users enables
extra MCP servers and plugins (browser, computer use, spreadsheets, ...). Hailer disables those for its
session by name, so an analysis session carries only Hailer's tools. Set `HAILER_CODEX_HOME` to an empty
folder to isolate Hailer completely (you then need an API key via `hailer login` or an environment variable,
because the ChatGPT login lives in the default Codex home).

## Troubleshooting

`uv run hailer doctor` runs the five startup checks (config, notebook, credentials, marimo, session) and,
when a session exists, confirms that marimo's code-mode API is available in the kernel.

| Message | Meaning and fix |
|---|---|
| `Marimo is not running.` then `Start everything in one go: uv run hailer notebook`, `Or start it yourself with: uv run marimo edit notebooks --no-token`, `Then run Hailer again: uv run hailer` | No server answered at the configured or discovered URL. `uv run hailer notebook` starts one on the notebooks folder and runs the chat in the same terminal. Servers started with `--no-token` register themselves so Hailer finds them; otherwise set `marimo_url` in `hailer.toml`. |
| `Marimo exited early (code N)` or `Marimo did not answer on http://127.0.0.1:2718 within 60 s`, followed by `Log: .hailer\marimo.log` and its last lines | `hailer notebook` could not start marimo. The log tail usually names the cause (port in use by something else, a syntax error in the notebook, marimo not installed in the environment). |
| `the notebook is not open in a browser` followed by `Open http://... in your browser.` | The server is up but has no kernel session. Open the URL; Hailer opens it for you once at startup. The URL ends in `&view-as=present` (app view); `Ctrl+.` in the notebook shows the code. |
| `not found: <path>` for the notebook | The configured notebook (`[hailer].notebook`, or `HAILER_NOTEBOOK`) does not exist. Fix the path or restore the file; a deleted *active* notebook is not the cause, because Hailer already falls back to the configured one when the remembered notebook is gone. New notebooks are created from the chat with `/notebook new <name>`. |
| `No notebook named '...' in notebooks.` or `... is outside the notebooks folder.` | `/notebook open` (or the agent's `notebook_open`) only opens marimo notebooks inside `[hailer].notebooks_dir`; the hint lists the available names. Move the file into the folder or point `notebooks_dir` at it. |
| `INTERNAL_MODEL_API_KEY is not set (required by provider 'internal')` | Run `uv run hailer login internal` or set the variable in this terminal. |
| `no OPENAI_API_KEY found; Codex will use its existing ChatGPT login if you have one` | Informational. If the agent then fails to authenticate, `uv run hailer login openai`. |
| `The model endpoint rejected the API key for provider '...'` | The gateway returned 401. Re-run `hailer login <provider>`. |
| `Unknown model '...' for provider '...'` | Fix `[model].name`. |
| `Could not reach the model endpoint at <base_url>` | Check `base_url`, VPN or proxy, and that the endpoint is running. |
| `The endpoint at <base_url> did not accept the request.` with the Responses-API hint | The gateway does not implement `POST /responses`. If it offers Chat Completions, set `wire_api = "chat"` for the provider. |
| `The endpoint at <base_url> did not accept the request.` with the `/chat/completions` hint | `wire_api = "chat"` is set but the gateway rejected `POST /chat/completions` (wrong `base_url`, no function calling, or no streaming: try `stream = false` on the provider). |
| `The Codex runtime refused to start because of a configuration error.` | Usually an interaction with `~/.codex/config.toml`. Run with `--verbose` for the runtime's message, or set `HAILER_CODEX_HOME`. |
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
- `uv run hailer notebook` starts and stops marimo for you; plain `uv run hailer` expects a running server
  and tells you how to start one.
- Windows sandbox quirks listed above apply to the agent's shell; the kernel path is unaffected.
- Custom gateways see roughly 10 KB of instructions and tool schema per turn plus the conversation; strict
  request-size limits or schema validation may need relaxing (see PLAN.md §1a for the measured details).
