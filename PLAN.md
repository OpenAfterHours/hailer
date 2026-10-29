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
tools.py       15 tools: marimo_execute, marimo_status, notebook_cells, notebook_check, notebook_list,
               │  notebook_create, notebook_open, notebook_close, list_periods, project_files,
               │  project_search, project_read, load_skill, read_skill_file, fetch_page
project.py     the project's own files for the project_* tools: read-only, on this machine, never in a kernel
               ▼
sandbox.py     MarimoSandbox: the kernel this process started (endpoint, stop, notebooks and data by name)
               │  HTTP + SSE with the kernel's token (marimo_client.py)
               ▼
docker (default)  hailer-kernel-<id>-<suffix> on an --internal network, reached through a forwarder
                  container on 127.0.0.1; /work/notebooks a tmpfs of its own, /work/data the data folder and
                  /work/folders/<name> the data folders, read-only; no network, no host environment,
                  non-root, read-only root, capped memory/CPU/pids
unsafe-local      marimo in Hailer's own Python, as the user; secret-looking variables withheld (opt-in only)
               │
               ▼
Browser UI     the notebook tab; a kernel session exists only while it is open
```

One Python process holds the CLI, the agent and the tools. The model gets those fifteen tools and nothing
else: no shell, no approval step, no file tool beyond reading the project's own files when `[project]` is set;
`fetch_page` reaches only `[web].allowed_domains`. Code the
model writes runs in marimo's scratchpad, and durable changes go through marimo's private code-mode API
(`marimo._code_mode`), so marimo is pinned (`==0.24.2`) and so is the kernel image's marimo. Code mode
formats the cells the agent creates or edits with the kernel's ruff before they run, and `code_checks.py`
checks the changed cells with ruff, ty and its own check for values typed into text on this machine, over a
snapshot the kernel returns; the findings end the tool result (`[checks]` in `hailer.toml` turns each part
off).

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

Only to the configured model endpoint: the user's messages, the system prompt with the context folder,
skill bodies when loaded, tool arguments (code) and truncated tool results (which can include data samples and
the project files the agent reads, credential lines withheld), and pages fetched from allowed domains. Raw data stays local unless code prints it. The API key is resolved in
the Hailer process (environment variable or OS credential store) and never enters the kernel, a file, a
prompt or a log. LangSmith tracing is off unless `HAILER_TRACING` is set.

Environment overrides: `HAILER_CONFIG`, `HAILER_WORKSPACE`, `HAILER_MODEL`, `HAILER_MODEL_PROVIDER`,
`HAILER_KERNEL`, `HAILER_KERNEL_IMAGE`, `HAILER_LOG_LEVEL`, `HAILER_TRACING`. Everything else comes from
`hailer.toml` (or `.hailer/hailer.toml` in the project layout) with this machine's `.hailer/local.toml` on top. `HAILER_DOCKER_TESTS` and `HAILER_TEST_CHROME` steer the opt-in Docker integration test only.

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
- **Project files are sent to the model as read:** credential lines are withheld by a heuristic
  (`looks_like_secret`, code-aware), not a guarantee; the folder is opt-in (`[project]`) and shown in the
  startup panel, and git-ignored files, Hailer's state, the data and credential-named files are never listed.

## Package layout

```
src/hailer/
├── cli/               Typer app and main (__init__), chat and notebook (chat), kernel pull|build|stop (kernel),
│                      data list|add|remove (data), init, login, logout, doctor, status (setup), shared
│                      collaborators and checks (common)
├── chat.py, chat_ui.py   ChatController (slash commands, notebook switches), the persistent composer
├── agent.py           build_model, HailerAgent (threads, streamed turns, Ctrl+C), system prompt, error mapping
├── tools.py           HailerTools (the 15 tools) and hailer_tools(config) -> LangChain tools
├── project.py         ProjectFiles: the project's files the project_* tools list, search and read
├── datafolders.py     this machine's [data_folders] in .hailer/local.toml (hailer data add|remove, /data)
├── code_checks.py     ruff, ty and typed-in values on the cells the agent writes (and code mode's ruff formatting)
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

## 2026-10-28: Hailer inside another project

Hailer can live inside an existing project (a uv workspace monorepo included) as a support chat for analysts:
it answers questions about the data and about the product it comes from.

- **Discovery and layout.** Inside a git repository the workspace is the nearest folder with a Hailer config up
  to the repository root, else the uv workspace root, so `uv run hailer` in a member belongs to the whole
  repository. `.hailer/hailer.toml` marks the project layout: notebooks, context, skills and prompts in
  `.hailer/` (committed), state in `.hailer/state/` (ignored); `init` picks it in a folder with `pyproject.toml`
  or `.git` (`--layout`). `.hailer/local.toml` is this machine's settings, merged over the project's.
- **Project files.** `[project]` gives the agent `project_files`, `project_search` and `project_read`: the
  project's code, docs and configuration, read-only, on this machine (`hailer.project`); git's view of the
  files minus Hailer's state, the data and notebooks folders and credential files; credential lines withheld
  (LEARNINGS §17).
- **Data folders.** `[data_folders]` (and `uvx hailer data add <name> <folder>`, `/data`) mount folders from
  anywhere on the machine read-only at `/work/folders/<name>`, as `DATA_FOLDERS["<name>"]` in notebooks, with
  the data folder's mount rules; paths live in each machine's `local.toml` (LEARNINGS §18).

Verified:

- Offline suite on Linux, Python 3.13: 1174 passed, 15 skipped (with main merged in).
- `tests/test_docker_integration.py` with `HAILER_DOCKER_TESTS=strict`: 12 passed on Linux with Docker Engine
  29.9.2 and an image built from this checkout (`scripts/build_kernel_image.py --load`, the pinned base image
  digest pulled through a registry mirror), including the new data folder mount.
- Live, with a real marimo kernel and headless Chromium, in a uv workspace in the project layout: unsafe-local and
  docker kernels both define `DATA_FOLDERS` from a folder `hailer data add` added, `list_periods` and doctor list
  it, and the project tools read the repository.

Still owed: a live turn against a real model with the project tools; the Docker integration run on Windows and
macOS (and a data folder on a mapped drive there); a physical Ctrl+C in a Windows console.
