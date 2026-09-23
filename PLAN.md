# Hailer: architecture

Hailer lets a user chat with their data: a terminal chat drives an agent that explores local files and writes
tables, charts and summaries into a live marimo notebook in the browser. This file describes the design as it
is now. Module contracts are in [docs/INTERFACES.md](docs/INTERFACES.md), the lessons and regression checks
behind the design in [docs/LEARNINGS.md](docs/LEARNINGS.md), user-facing behaviour on the
[documentation site](docs/site/index.md). Earlier plans, reviews and dated findings are in
[docs/history/](docs/history/PLAN_HISTORY.md).

## What runs where

```
Terminal ── hailer.cli (Typer) ── hailer.chat / chat_ui (composer, slash commands)
               │
agent.py       LangChain create_agent + ChatOpenAI ──HTTPS──► model endpoint (OpenAI or a custom base_url,
               │  API key and conversation stay here            Responses API or Chat Completions)
tools.py       12 tools: marimo_execute, marimo_status, notebook_cells, notebook_check, notebook_list,
               │  notebook_create, notebook_open, notebook_close, list_periods, load_skill, read_skill_file,
               │  fetch_page
               ▼
sandbox.py     MarimoSandbox: the kernel this process started (endpoint, stop, notebooks and data by name)
               │  HTTP + SSE with the kernel's token (marimo_client.py)
               ▼
docker (default)  hailer-kernel-<id>-<suffix> on an --internal network, reached through a forwarder
                  container on 127.0.0.1; /work/notebooks a tmpfs of its own, /work/data the data folder
                  read-only; no network, no host environment, non-root, read-only root, capped memory/CPU/pids
unsafe-local      marimo in Hailer's own Python, as the user; secret-looking variables withheld (opt-in only)
               │
               ▼
Browser UI     the notebook tab; a kernel session exists only while it is open
```

One Python process holds the CLI, the agent and the tools. The model gets those twelve tools and nothing
else: no shell, no file tool, no approval step; `fetch_page` reaches only `[web].allowed_domains`. Code the
model writes runs in marimo's scratchpad, and durable changes go through marimo's private code-mode API
(`marimo._code_mode`), so marimo is pinned (`==0.24.2`) and so is the kernel image's marimo. Code mode
formats the cells the agent creates or edits with the kernel's ruff before they run, and `code_checks.py`
checks the changed cells with ruff and ty on this machine, over a snapshot the kernel returns; the findings
end the tool result (`[checks]` in `hailer.toml` turns each part off).

## The sandbox contract

A started kernel is a `hailer.sandbox.MarimoSandbox`, returned by the runtime's `start` and held in memory by
the chat, the agent and the tools. It is the contract (there is no separate protocol class):

- **Endpoint:** `client()` (the server URL and token), `notebook_url` / `home_url` for the browser (with the
  token) and the model (without it); `stop()` removes everything the start created (a docker kernel copies
  its notebooks back first); `describe()` is the `Kernel:` line.
- **Files by name:** `list_notebooks`, `read_notebook`, `write_notebook` (create or replace), `has_notebook`,
  `list_data`, `data_schema` and `sync_soon()` (ask for a copy back), with names as POSIX paths relative to
  the notebooks or data root (`"sales.py"`, `"q3/review.py"`). Notebook files go through marimo's own file API
  (`/api/files/list_files`, `/file_details`, `/create`, `/update`, `/delete`), which needs no browser session.
- **Paths only inside the runtime layer:** `MarimoClient` turns a name into marimo's `?file=` key and back;
  `hailer.tools` never touches the notebooks or data folders (a test checks the source). The active notebook
  in `.hailer/notebook.json` is a name.

## Notebook sync (docker)

The container's notebooks folder is its own size-capped tmpfs, not a mount of the user's disk.
`hailer.notebook_sync` copies notebooks in at start (after health, before the browser opens) and back after
every turn, on a notebook switch, every 15 s while the chat runs (browser edits) and at stop, before the
containers are removed. One allow-list applies both ways: plain marimo notebooks (`import marimo`,
`marimo.App(`) with safe names, `.py`, at most four folders deep, 5 MiB each and 500 files, no dot or dunder
parts, no tooling files (`conftest.py`, `test_*.py`, `setup.py`, ...) and no names that would shadow an
importable module. Host writes are atomic, never follow links and never delete; a host copy that changed
since the last copy is backed up to `.hailer/notebook-backups/` first. Replies from the container are
untrusted: capped in size and bounded in time, so a hostile kernel cannot hold the exit. The unsafe-local
runtime works on the host folder directly and syncs nothing.

## Lifecycle

- Each `uvx hailer` / `uvx hailer notebook` process starts its own kernel and stops it in a `finally` on every
  way out (`/exit`, EOF, Ctrl+C, errors, `--foreground`). Nothing finds or attaches to a kernel another process
  started; `/exec <code>` runs code in the chat's own kernel.
- Docker objects get per-start names and labels (workspace, role, kernel contract, owner). The owner is a lock
  file the process holds (`.hailer/owner-<id>.lock`), so a start removes only leftovers whose owner is gone,
  and `uvx hailer kernel stop` removes every labelled object of the workspace.
- The kernel's token reaches marimo on stdin (unsafe-local) or through a read-only single-file mount deleted
  once marimo answers (docker); it is never on a command line or in `docker inspect`.
- The kernel image is named by its contract: `ghcr.io/openafterhours/hailer-kernel:marimo<version>-<sha256
  prefix of the Dockerfile, the copied hailer modules and the pinned package versions>`. A CLI-only release
  publishes no image; the release pushes a missing tag before PyPI, and `uvx hailer kernel build` builds it
  locally.
- Docker is the default. Without a usable Docker a start fails closed and names the two ways on: install or
  start Docker, or write `runtime = "unsafe-local"`. The retired `local` value is an error, not an alias.
- Conversations are LangGraph checkpoints in `.hailer/threads.sqlite`; only the latest thread is resumed.

## What leaves the machine

Only to the configured model endpoint: the user's messages, the system prompt with `.config/hailer/context`,
skill bodies when loaded, tool arguments (code) and truncated tool results (which can include data samples),
and pages fetched from allowed domains. Raw data stays local unless code prints it. The API key is resolved in
the Hailer process (environment variable or OS credential store) and never enters the kernel, a file, a
prompt or a log. LangSmith tracing is off unless `HAILER_TRACING` is set.

Environment overrides: `HAILER_CONFIG`, `HAILER_WORKSPACE`, `HAILER_MODEL`, `HAILER_MODEL_PROVIDER`,
`HAILER_KERNEL`, `HAILER_KERNEL_IMAGE`, `HAILER_LOG_LEVEL`, `HAILER_TRACING`. Everything else comes from
`hailer.toml`. `HAILER_DOCKER_TESTS` and `HAILER_TEST_CHROME` steer the opt-in Docker integration test only.

## Risks

- **Private `marimo._code_mode` API:** marimo is pinned in Hailer and in the image contract; the Docker
  integration test runs code mode in a real kernel.
- **A kernel session needs a browser tab:** `marimo_status` detects a missing session; the CLI and the
  notebook tools open the URL and wait.
- **Docker is not always available** (licensing, managed machines): those users either get Docker or accept
  `unsafe-local`, which is labelled in the startup panel, `status`, `/status` and `doctor`. A cloud kernel
  behind the same sandbox contract is the planned answer ([ARCHITECTURE_PROPOSAL.md](docs/ARCHITECTURE_PROPOSAL.md) §6).
- **The host copy of a notebook is up to one turn (or 15 s) behind** the kernel's; files that notebook code
  writes next to the notebooks in Docker are not copied back and are gone when the kernel stops.
- **Cancellation is not rollback:** Ctrl+C stops waiting for a tool call, not the kernel; the next turn tells
  the model the effect is unknown.
- **Strict gateways and LangChain's pace:** request fields are explicit and checked against a strict fake
  gateway; `langchain`, `langchain-openai` and `langgraph-checkpoint-sqlite` are pinned below their next
  major versions.
- **Large outputs:** tool results are capped (head and tail), rich outputs replaced by a placeholder.

## Package layout

```
src/hailer/
├── cli/               Typer app and main (__init__), chat and notebook (chat), kernel pull|build|stop (kernel),
│                      init, login, logout, doctor, status (setup), shared collaborators and checks (common)
├── chat.py, chat_ui.py   ChatController (slash commands, notebook switches), the persistent composer
├── agent.py           build_model, HailerAgent (threads, streamed turns, Ctrl+C), system prompt, error mapping
├── tools.py           HailerTools (the 12 tools) and hailer_tools(config) -> LangChain tools
├── code_checks.py     ruff and ty on the cells the agent writes (and code mode's ruff formatting)
├── sandbox.py         MarimoSandbox, the sandbox contract
├── notebook_sync.py   the allow-listed copy between the docker kernel and the host
├── kernel.py          runtime_for, the unsafe-local runtime, withheld variables
├── kernel_docker.py   DockerRuntime, owner locks, workspace_kernels, stop_workspace_kernels
├── kernel_image.py, kernel_packages.py, docker/Dockerfile, _forward.py   the kernel image and its forwarder
├── marimo_client.py   HTTP/SSE client, session resolution, marimo's file API
├── notebooks.py       notebook names, templates, the active-notebook state
├── config.py, models.py, errors.py, secrets.py, context.py, session.py, web.py, browser.py, log.py, ...
└── periods.py         YY-MM period files and multi-period loaders, also used inside the image
```

## 2026-09-23: the sandbox simplification

Phases 0–4 of [docs/ARCHITECTURE_PROPOSAL.md](docs/ARCHITECTURE_PROPOSAL.md) are implemented on branch
`worktree-sandbox-simplification`: the image is versioned by its content and its token is off the command
line (R5, F7); each process owns one kernel and cross-terminal reuse, `kernel.json`, `marimo_url`,
`--keep-marimo` and `hailer exec` are gone (R2); notebook and data access goes through `MarimoSandbox` and
`PathMap` is gone (R1); the docker kernel's notebooks live in a tmpfs and come back through the allow-listed
sync, which replaced the notebooks mount rules and the planted-file scan (R3); Docker is the default and the
unisolated runtime is `unsafe-local` (R4); `cli.py` became the `hailer.cli` package, the unused environment
overrides went and the history moved to `docs/history/` (R7). The GCP runtime (R6) is not started.

Verified:

- Offline suite on Windows 11, Python 3.13: 1024 passed, 14 skipped (no key, no Docker, loopback only).
- `tests/test_docker_integration.py` with `HAILER_DOCKER_TESTS=strict`: 11 passed on Windows 11 with Docker
  Desktop 29.4.3, including the token absent from `docker inspect`, planted files
  (`.git/`, `conftest.py`, `.vscode/`, a non-notebook `.py`) never reaching the host while notebook edits do,
  and nothing left behind after stop.

Still owed: the Docker integration run on Linux, macOS and `linux/arm64`; a live turn against a paid model on
this branch; a physical Ctrl+C in a Windows console.
