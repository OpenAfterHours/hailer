# Hailer: implementation plan (history)

> Historical record: `PLAN.md` before its 2026-09-23 rewrite, kept as written. Each section describes the
> design at the time (for example the `local` default runtime, kernels shared between terminals and the
> writable notebooks mount), not the current one, which is in [PLAN.md](../../PLAN.md).

## 2026-09-19: final Docker kernel review and integration

The interrupted review fixes were recovered from `worktree-docker-kernel` and integrated with the current
automatic startup and persistent chat composer. Both chat entry points retain their kernel connection
and reject mismatched kept kernels. The final pass also covers nested repository/workspace controls,
credential folders inside temporary paths, busy-kernel diagnostics, state-write cleanup and retryable
Docker shutdown. The docs below retain the earlier implementation evidence; the current findings and
validation are in [DOCKER_KERNEL_REVIEW.md](DOCKER_KERNEL_REVIEW.md).

## 2026-09-19: the notebook kernel can run in Docker

Where marimo and its kernel run is now a setting, `[kernel] runtime`: `local` (the default, as before) or
`docker`, a Linux container that sees only the notebooks folder (read-write) and the data folder
(read-only), with no network and none of the host's environment. The agent, the conversation and the API
key stay in the Hailer process either way. Built on branch `worktree-docker-kernel`; the design, the spikes
behind it and where the build departs from it are in `docs/DOCKER_KERNEL_PLAN.md`.

Why:

- `marimo_execute` runs the model's Python in the kernel, and a local kernel runs as the user, with the
  user's files, network and environment. The prompt's rules were the only thing between a bad turn and the
  user's files. Docker gives users who have it a boundary that is enforced, without changing how Hailer is
  used.
- The local runtime had avoidable gaps of its own: servers Hailer started had no token, so any local
  program could POST code to `/api/kernel/execute` (it skips marimo's skew check), and the kernel inherited
  every API key in Hailer's environment.

What changed for everyone (the local runtime):

- Servers Hailer starts get a random token, passed with `--token-password-file -` on stdin, never on a
  command line. It is recorded in `.hailer/kernel.json` so chat-only `hailer`, `hailer exec` and a second
  `hailer notebook` can attach. Links Hailer opens or prints carry `access_token`; tool results and the
  system prompt never do.
- The marimo server's environment withholds provider keys, `env_http_headers` variables,
  `HAILER_MARIMO_TOKEN` and secret-looking names; `[kernel] pass_env` lets named ones through.
- `.hailer/marimo.log` is emptied at each start (owner-only on POSIX); log tails Hailer prints mask the token.
- A `Kernel:` line in the startup panel, `status`, `/status` and `doctor`; `uvx hailer kernel stop` stops a
  kept server of either runtime.

What users must do: nothing, to keep working as before. A notebook that reads a secret-looking variable
needs `[kernel] pass_env`. A marimo server started by hand with `--no-token` still works in the local
runtime. A notebook URL without the token (one the agent quoted) shows marimo's sign-in page; `/notebook`
prints the signed-in link. Docker mode is opt-in: `uvx hailer init --kernel docker`, `[kernel] runtime =
"docker"`, `HAILER_KERNEL=docker` or `uvx hailer notebook --kernel docker`.

Verified:

- Live on Windows 11 with marimo 0.24.2 (local runtime): unauthenticated `/api/kernel/execute` and
  `/api/sessions` get 401, `Authorization: Bearer` works, the signed URL opens a session in headless
  Chrome, notebook code sees no `*_KEY` / `*_TOKEN` variable, and a second terminal attaches through
  `kernel.json`.
- Live on Windows 11 with Docker Desktop 29.4.3 (docker runtime): `hailer kernel build`, `doctor`, a start
  in 4.3 s with the image present, a notebook session through the forwarder, `WORKSPACE = /work` and
  `DATA_DIR = /work/data` in the starter notebook, data writes and the network refused, cells saved to the
  host file, and `hailer kernel stop` leaving nothing behind.
- `tests/test_docker_integration.py` (real containers, headless Chrome) passes 7/7 with
  `HAILER_DOCKER_TESTS=strict` on Windows with Docker Desktop 29.4.3 and in WSL Ubuntu. The offline suite:
  698 passed, 9 skipped on Windows; 697 and 10 on WSL Ubuntu. actionlint is clean on both workflows.
- Three adversarial reviews (security, lifecycle, UX) followed the first build; every confirmed finding was
  fixed, and the high ones were re-proven live after the fix (commit d98f5ec).
- 2026-09-19, with the real CLI in throwaway workspaces (the docs pass): `init --kernel docker`, `doctor`
  in both runtimes, `notebook --foreground` in docker mode, reuse refused after `memory` changed
  (`The running kernel was started with other settings (memory 4g, hailer.toml: 2g).`), the chat-only
  warning for the same, the start guard over a running kernel, `kernel stop` for a docker and a local
  kernel, the local-after-docker warning, a refused layout (`notebooks_dir = "."`), the misplaced
  `[model].runtime` warning and a missing image tag (`kernel pull` of an unpublished tag: GHCR answers
  `denied`, Hailer reports it as not published). The image is 195 MB to download and 875 MB on disk
  (amd64).

Findings that shaped the build (beyond the plan's spikes):

- **A record must prove it is still ours.** A port can be reused by another server, and a `--no-token`
  server accepts any token. `answers_with_token` requires `/api/sessions` to refuse the request without
  the token and accept it with it; `live_kernel_state` is the one liveness check every caller uses, and it
  deletes a record only when its process or container is provably gone.
- **Remove by id, never by name.** Container names are fixed per workspace, so an old handle (a chat
  started before a restart) removing by name could destroy a newer kernel. `kernel.json` records the ids
  Docker printed, and a start never removes a *running* kernel container it has no record of.
- **The mounts are part of the boundary.** A notebooks folder equal to the workspace let notebook code
  rewrite `hailer.toml` back to `local`; the layout rules in `config.docker_mount_problems` refuse that and
  the similar cases (home folder, drive root, `.hailer`, the config file, context, skills, prompts).
- **marimo 0.24's edit page ignores `[tool.marimo]` in a project `pyproject.toml` for auto-run;** it reads
  the user configuration, first in its working directory. The image writes `/work/.marimo.toml`
  (`auto_instantiate = true`), so the starter notebook's globals exist when the browser opens it.
- **The chat keeps its server in memory.** `hailer notebook` hands the chat and the tools the server it
  started, so a `kernel.json` rewritten by another terminal cannot redirect them; chat-only `hailer`
  rediscovers on every call and follows a restarted kernel.

Still owed: read time for a large Parquet file through a Windows bind mount compared with a local read; the
`linux/arm64` image (checked by wheel availability only; the first release builds it, and stops before
PyPI if that fails); the first pull from GHCR, and one through a corporate proxy or registry mirror; macOS;
a real mapped network drive. After the first release the GHCR package must be made public once (README,
*Releasing*). Podman is out of scope.

## 2026-09-18: the agent moves from the Codex SDK to LangChain

The agent is now `langchain.agents.create_agent` with `langchain_openai.ChatOpenAI`, running inside the Hailer
process. The OpenAI Codex SDK (`openai-codex`, a bundled native `codex app-server`), the stdio MCP tool server
(`hailer.mcp_server`) and the loopback bridge that translated Responses-API calls to Chat Completions
(`hailer.wire`) are removed, and with them the sandbox, the shell tool, approval and reviewer handling and
every use of `~/.codex`.

Why:

- The SDK ships a 381 MB native runtime (five executables) that has to be launched from a user-writable
  folder, which application control on managed Windows machines commonly blocks. This is the likely cause of
  the "does not work for some users" reports, but no report was seen, so it is not confirmed.
- Codex speaks only the Responses API, which forced the bridge for Chat-Completions-only gateways.
- Codex sends side requests Hailer never asked for, such as the `codex-auto-review` reviewer model on every
  command and tool call, which strict gateways reject.
- The settings Hailer needed were reachable only through private SDK surface, which meant an exact version pin.

Kept: custom endpoints, key storage, all 11 tools, the marimo integration and the slash commands.

What users must do:

- Provide an API key for the built-in `openai` provider as well (`uv run hailer login openai` or
  `OPENAI_API_KEY`). There is no ChatGPT-login fallback any more.
- Nothing about an old `hailer.toml`: it still loads. `[hailer].codex_home`, `requires_openai_auth`,
  `merge_messages`, `parallel_tool_calls` and `[web].allow_shell_network` are reported as unknown keys that are
  ignored, and `HAILER_CODEX_HOME` is no longer read.
- Expect a new conversation: earlier conversations do not carry over.

The lessons for future changes are in [docs/LEARNINGS.md](../LEARNINGS.md), with maintained evidence in
`tests/test_agent.py`, `tests/fake_gateway.py` and `tests/test_tools.py`. The original proposal and
experiments remain in Git history. The rest of this file describes the new design only.

## 2026-09-16: notebooks from the chat

Built, and verified live on the agent of that date (the runs have not been repeated on the LangChain agent).
The user (or the agent) can create new notebooks and reopen old ones without leaving the conversation:

- One marimo server, started on the notebooks **folder** (`marimo edit notebooks --no-token`), hosts every
  notebook; a notebook is opened by URL with an absolute `?file=` key, no restart. Verified on marimo 0.24.2 with
  a websocket probe against both folder and single-file servers before the design was fixed.
- The **active notebook** lives in `.hailer/notebook.json`, shared by the CLI and the agent's tools
  (`hailer.tools`, which re-read it on every call). `hailer.notebooks` owns the state file, notebook listing,
  slugified creation from a starter/empty template and folder-confined path resolution.
- Four tools (`notebook_list`, `notebook_create`, `notebook_open`, `notebook_close`) and the matching
  `/notebook list | new | open | close` slash commands; a `[Hailer] ...` notice rides with the next message after
  a slash-command switch; the system prompt names the folder, not the notebook, and the model learns the active
  notebook from `marimo_status` and those notices.
- Live verification (throwaway workspace, real model turns): "Create a new notebook called e2e beta and add one
  cell that prints hello" → the model called `notebook_create`, `notebook_cells`, `marimo_execute`, the file
  `e2e_beta.py` gained a `hello` cell, a browser tab opened from the tool, and the CLI printed
  `Active notebook is now notebooks/e2e_beta.py.`; "Open the e2e alpha notebook and tell me how many cells it
  has" → `notebook_open` only, reply "I'm now in notebooks/e2e_alpha.py. It has 5 cells." (correct for the
  starter template). `/notebook new`, `/notebook list`, `/notebook close`, `hailer exec` on the active notebook
  and the skew-token `shutdown_session` call were exercised the same way.
- Review fixes folded in: exact-path session matching (no filename fallback), Windows reserved names and a
  64-character cap in slugs, `os.startfile`-first browser opening (a `BROWSER=true` shell had silently opened
  nothing), the switch notice queued before the browser wait, state re-read after interrupted/failed turns,
  `HAILER_NOTEBOOK` persisted to the state file so the tools follow it.

## Status

All milestones M0 to M7 are implemented in this repository (`https://github.com/OpenAfterHours/hailer`); the
agent path (M2, M3) was rebuilt on LangChain on 2026-09-18, and the kernel runtimes (local with a token,
docker) were added on 2026-09-19 (see the section above). The offline test suite (`uv run pytest`) needs no
API key, no marimo and no network beyond loopback. The new agent was verified live on 2026-09-18 with the
real CLI, a running marimo server and a kernel session (§1a has the details): a custom Chat Completions
endpoint, resume after a restart, and one `gpt-5.5` turn over the Responses API with a real key. A physical
Ctrl+C in a console is the one check still owed.

Verified live on 2026-09-15 on Windows 11, against a running marimo server with the notebook open in a
browser. These longer runs predate the LangChain agent and have not been repeated on it:

- Definition-of-done sequence: "create a dataframe with monthly RWA values and show it in the notebook" →
  "turn that into a chart" → "summarise the biggest movement". Cells were created, then edited in place for
  the chart, and the summary quoted the real figures from the sample data.
- Realistic sequence: inspect parquet files → identify periods → compare latest two → biggest month-on-month
  movements → break down by exposure class → plot. Eight cells were added, all ran without errors, and the
  agent applied the example project skill's reconciliation checks on its own.

Investigated against the real installed APIs on Windows 11, Python 3.12/3.13. 2026-09-15: `marimo` 0.24.2,
`polars` 1.44.2, `duckdb` 1.5.5, `typer` 0.27.2, `rich` 15.0.0, `pytest` 9.1.1, `uv` 0.12.1. 2026-09-18:
`langchain` 1.4.1, `langchain-core` 1.6.3, `langchain-openai` 1.6.2, `langgraph` 1.2.11,
`langgraph-checkpoint-sqlite` 3.1.1, `openai` 3.15.0.

## 1. Findings that change the original spec

1. **`marimo-pair` is not a PyPI package.** It is two things:
   - `marimo pair prompt --url ... --codex` (built into marimo >= 0.21): only *prints a prompt string* telling an agent to use a skill.
   - The GitHub repo `marimo-team/marimo-pair`: an Agent Skill (`SKILL.md`, `scripts/execute-code.sh`, `scripts/discover-servers.sh`, `reference/*.md`). The scripts need **bash + curl + jq**. `jq` is not installed on this machine, so the stock scripts would not run here anyway.

2. **The whole "pair" protocol is three plain HTTP calls.** Hailer implements it in pure Python (stdlib `urllib`, `marimo_client.py`), so nothing in Hailer depends on Git Bash or WSL:
   - `GET  {url}/health` (no auth) → `{"status": "healthy"}`
   - `GET  {url}/api/sessions` → `{ "<session_id>": {"filename": ..., "path": ...}, ... }`
   - `POST {url}/api/kernel/execute` with header `Marimo-Session-Id: <id>` and body `{"code": "..."}` → Server-Sent Events: `stdout` / `stderr` (`{"data": "..."}`) then `done` (`{"success": bool, "output": {"mimetype": ..., "data": ...}}`).
   - Optional `Authorization: Bearer <token>` for tokened servers. Servers started with `--no-token` self-register at `%USERPROFILE%\.marimo\servers\<host>_<port>.json` (`server_id, pid, host, port, base_url, started_at, version`). Hailer does not use them: each session talks only to the server it started.

3. **Code executes in a scratchpad**, a temp namespace over the kernel globals. Reads see notebook variables; new top-level bindings are discarded. Durable notebook changes go through the private `marimo._code_mode` API inside the scratchpad:
   ```python
   import marimo._code_mode as cm
   async with cm.get_context() as ctx:
       cid = ctx.create_cell("rwa = ...", hide_code=False, name="rwa_summary")
       ctx.run_cell(cid)
   # also: ctx.edit_cell, delete_cell, move_cell, set_ui_value, packages.add,
   #       ctx.cells[...] (.code/.status/.errors/.output), ctx.graph, ctx.globals
   ```
   This API is explicitly unstable, so Hailer pins the marimo version and keeps all `cm` usage in one place (prompt patterns + helper snippets).

4. **A kernel session only exists while the notebook is open in a browser.** Sessions are created by the websocket connector, not by the server starting. `marimo edit --headless` with no browser tab = no session = nothing to execute against. Hailer must detect "server up, no session" and tell the user to open the URL (or open it for them).

## 1a. Custom (non-OpenAI) endpoints

`hailer.agent.build_model` turns a provider (the built-in `openai` one or a `[model_providers.<id>]` table) into
one `ChatOpenAI` with an explicit `base_url`. Nothing sits between Hailer and the endpoint. The user's side
(§1b: declare the provider, `hailer login <id>`, pick `wire_api`) is unchanged.

- **`base_url` and `use_responses_api` are always explicit**, `https://api.openai.com/v1` included. Checked
  against langchain-openai 1.6.2: left unset, a model named `gpt-5-codex` or `gpt-5.5-pro` is routed to
  `/responses` even on a custom `base_url`, and a stray `LANGSMITH_GATEWAY` environment variable redirects the
  built-in provider's requests.
- **The strict-gateway switches are `stream` and `stream_options` only**, for either wire API.
  `stream = false` → `disable_streaming=True` (sends `stream: false`, reads one JSON reply);
  `stream_options = false` → `stream_usage=False` (the field is omitted; token counts then arrive only if the
  gateway sends usage unasked). `parallel_tool_calls` is never sent, and `[model].reasoning_effort = ""` stops
  sending `reasoning_effort`.
- **Proxies and the HTTP client.** `http_socket_options=()` keeps httpx's detection of `HTTP(S)_PROXY` and the
  system proxy. The agent owns its HTTP client (`openai.DefaultAsyncHttpxClient`): langchain-openai's cached
  client is bound to the first event loop, and a second agent failed with "Event loop is closed".
- **Errors are typed.** `map_exception` switches on `openai.APIStatusError.status_code`, `APITimeoutError` and
  `APIConnectionError` instead of parsing text, and quotes the endpoint's words with secrets masked.

What a bespoke endpoint must provide:

- **One of the two OpenAI wire APIs**, with function tool calls: `wire_api = "responses"` →
  `POST {base_url}/responses`, `wire_api = "chat"` → `POST {base_url}/chat/completions`.
- **Streaming (SSE)**, unless `stream = false`.
- **Bearer-token auth** with the key named by `env_key`; extra headers and query parameters come from
  `http_headers`, `env_http_headers` (sent only when the variable is set) and `query_params`.

What is verified, and what is not:

- The offline test suite drives the real `ChatOpenAI` against a strict Chat-Completions-only fake gateway
  (`tests/fake_gateway.py`, used by `tests/test_agent.py`; standard library, loopback only). Only
  `POST /v1/chat/completions` exists, unknown fields (`stream_options`, `parallel_tool_calls`) and unknown
  model names get a 422, a custom header and an `api-version` query parameter are required, and it can refuse
  `stream: true`. With `stream_options = false`, a one-tool turn is exactly two requests carrying only
  `model`, `messages`, `stream` and `tools`, one system and one user message, the configured model on both.
  `stream = false`, the error hints (one request per 401, 404 or 422, no retry) and an unreachable endpoint
  are covered the same way.
- Live, 2026-09-18, real `hailer` CLI on Windows 11 with a headless marimo 0.24.2 server and a kernel
  session opened through headless Chrome. Against the same fake gateway on a fixed port, the model's
  `marimo_execute("print(1 + 1)")` ran in the real kernel and `2` came back in the answer; the gateway saw
  only `messages`, `model`, `stream` and `tools`, the eleven tools as plain functions, the `api-version`
  query parameter and the custom header. A second run resumed the conversation and sent the earlier turns.
  With `OPENAI_API_KEY` and the built-in provider, `gpt-5.5` over the Responses API called `marimo_status`,
  then `marimo_execute`, and answered `391` for 17 × 23 computed in the kernel (9,223 tokens in, 91 out).
- Ctrl+C is covered by the suite with a real SIGINT to the main thread on Linux (`signal.pthread_kill`) and
  `_thread.interrupt_main()` on Windows. The latter does not wake an `epoll` wait on Linux, so it cannot
  stand in for Ctrl+C there.
- Still owed: a physical Ctrl+C in a Windows console, and a run by one of the users the Codex runtime
  failed for.

### 1b. How a user points Hailer at their own endpoint (user-facing workflow)

1. **Describe the endpoint once in `hailer.toml`** (project root or `.config/hailer/hailer.toml`). Several providers can be defined; `[model].provider` picks the active one.
   ```toml
   [model]
   name     = "risk-analyst-v3"          # whatever model id the gateway expects
   provider = "internal"

   [model_providers.internal]
   base_url = "https://llm.example.internal/v1"
   wire_api = "responses"                  # or "chat" for Chat Completions
   env_key  = "INTERNAL_MODEL_API_KEY"
   # optional:
   # stream           = false              # the endpoint rejects or cannot deliver streaming
   # stream_options   = false              # the endpoint rejects the stream_options field
   # http_headers     = { "X-Team" = "risk-analytics" }
   # env_http_headers = { "X-Client-Id" = "INTERNAL_CLIENT_ID" }
   # query_params     = { "api-version" = "2025-04-01-preview" }   # Azure-style gateways
   ```
2. **Store the secret once with `uv run hailer login internal`.** Hailer prompts with hidden input and saves the key in the OS credential store through `keyring` (Windows Credential Manager, verified working on this machine: `WinVaultKeyring` round-trip OK). Nothing is written to `hailer.toml`, shell history, or any Hailer file. `hailer logout internal` removes it.

   How it reaches the endpoint: `hailer.secrets.resolve_provider_key` reads the key inside the Hailer process and `build_model` hands it to `ChatOpenAI` as `api_key`. It is never written to a file, exported to the user's shell, passed on a command line or logged (redaction filter). Resolution order per provider: (a) the env var named in `env_key`, when set in the shell (for CI/automation), (b) `keyring` entry `service="hailer", username="<provider>:<env_key>"`, (c) neither → a `CredentialsError` suggesting `hailer login <provider>`. The built-in `openai` provider works the same way with `OPENAI_API_KEY` and has no other login. `hailer doctor` and `/status` report where the key came from (`keyring` or `env`), never the value.

   Same-user processes can read Credential Manager entries just as they can read env vars, so this is about not leaving the key in files and history, not about isolating it from the user's own account. `keyring` adds six small pure-Python packages; no bash, no native build.
3. **Run `uvx hailer doctor`.** It reports: config valid, notebook found, where the active provider's key comes from, and whether the kernel runtime can start here (Docker, the image, the folders): the checks normal startup runs too. It starts no kernel and does not call the model endpoint. Failures print a fix, for example:
   ```
   INTERNAL_MODEL_API_KEY is not set (required by provider 'internal')
   Store it once with:

       uv run hailer login internal

   or set the INTERNAL_MODEL_API_KEY environment variable in this terminal.
   ```
4. **Run `uv run hailer`.** The startup panel shows `Model: risk-analyst-v3` and `Provider: internal (https://llm.example.internal/v1)`, with `chat completions` and `no streaming` noted when set.
5. **Switch when needed.** `/model <name>` changes the model and starts a new thread; `/model internal:<name>` or `/model openai:gpt-5.5` changes provider too. One-off overrides: `HAILER_MODEL`, `HAILER_MODEL_PROVIDER`.
6. **If the gateway only speaks Chat Completions**, set `wire_api = "chat"`; no proxy is needed. If it also rejects `stream: true` or the `stream_options` field, set `stream = false` or `stream_options = false`. Hailer turns the symptoms (a 404/405 on the wrong API, a 4xx naming a field) into a hint that names the setting rather than a stack trace.

`/status` inside the chat shows the same provider line plus the key's source, the thread id and token usage, so the user can always confirm which endpoint a session is talking to.

### 1c. Allowed web domains for context

Users can list websites the agent may read for context (regulator pages, internal documentation, library docs). Anything not listed is refused by the tool, not merely discouraged. Config:

```toml
[web]
allowed_domains = [
  "www.bankofengland.co.uk", "**.bankofengland.co.uk",   # ** = apex plus subdomains, * = subdomains only
  "docs.pola.rs", "duckdb.org",
  "confluence.example.internal",
]
max_page_bytes = 200000        # cap per fetch before HTML-to-text conversion
```

The agent's only web tool is `fetch_page(url)` (`hailer.web`), and the allow-list is its one enforcement point: there is no shell and no network proxy layer. With no `[web]` section it refuses every URL and the system prompt tells the model it has no internet access. The allow-list does not cover code the agent runs in the marimo kernel (§7).

`fetch_page` checks the host against `allowed_domains` (private and loopback addresses only when listed literally; a bare `*` is rejected at config validation), re-checks on every redirect hop, fetches with stdlib `urllib`, converts HTML to readable text (stdlib `html.parser`), caps the size, and returns the text to the model. A disallowed host returns a one-line refusal that names the allowed domains so the agent can ask the user to extend the list. Covered offline by `tests/test_web.py`.

What this means for data leaving the machine: text fetched from an allowed site becomes tool output and is sent to the configured model endpoint like any other tool result. The README security section says so, and the startup panel and `/status` show the active allow-list.

Later options (not v1): per-domain auth headers from keyring for internal wikis (`[web.auth."confluence.example.internal"] header = "Authorization" keyring = true`), and a `denied_domains` list where a deny wins.

## 2. Architecture

One Python process holds the CLI, the agent loop and the tools. marimo is the only other process: a child
of Hailer in the local runtime, or a container (plus a forwarder container) in the docker runtime
(`hailer.kernel`, `hailer.kernel_docker`).

```
Terminal  (Typer + Rich)                                   cli.py / session.py
    │  text turns, slash commands
    ▼
agent.py   HailerAgent: create_agent + ChatOpenAI ── HTTP(S) ── model endpoint
    │      one asyncio.Runner; conversations in                (OpenAI or a custom base_url,
    │      .hailer/threads.sqlite                               Responses API or Chat Completions)
    │  in-process calls (a daemon thread per call)
    ▼
tools.py   marimo_execute, marimo_status, notebook_cells, notebook_list,
    │      notebook_create, notebook_open, notebook_close, list_periods,
    │      load_skill, read_skill_file, fetch_page (allow-listed domains only)
    ▼
sandbox.py        (the started kernel: endpoint, stop, notebooks by name, data files; tools know
    │              files only through it)
    ▼
marimo_client.py  (pure-Python HTTP + SSE; Bearer token; notebook names <-> kernel paths; marimo's file API)
    │
    ▼
marimo edit notebooks   (live kernel on the folder; a token since 2026-09-19)
    local:  Hailer's own Python, token on stdin, secrets withheld from its environment
    docker: hailer-kernel-<id> on an internal network, reached through hailer-fwd-<id> on 127.0.0.1;
            /work/notebooks read-write, /work/data read-only, no network
    scratchpad  +  marimo._code_mode  →  Polars / DuckDB / charts
    │
    ▼
Browser UI
```

`hailer.tools`: `HailerTools` holds the 11 tools (`TOOL_NAMES`) as plain methods that return text, errors
included; a method's docstring is the description the model sees. `hailer_tools(config)` wraps them as
LangChain `StructuredTool`s. The model has these tools and nothing else: no shell, no file tool, no approval step.

Why Hailer's own tools instead of the upstream shell scripts:
- No bash/curl/jq; works from `cmd.exe`/PowerShell.
- Typed arguments: multi-line Python arrives intact (cmd.exe has no heredocs; shell quoting was the main Windows failure mode).
- Hailer controls the result shape: truncation of large outputs with head/tail, session resolution by notebook path on every call, clear error strings the model can act on.
- Same client is also exposed as the `/exec <code>` slash command in the chat, for humans and debugging (it runs in the chat's own kernel).

Not used: marimo's hidden `marimo edit --mcp code-mode` flag exposes `list_sessions` / `execute_code` over streamable HTTP at `/mcp/server` (needs `marimo[mcp]`). It is experimental, it requires the user to remember the flag, and the agent has no MCP client.

Agent design rules and the finding behind each ([docs/LEARNINGS.md](../LEARNINGS.md); the docstrings in `hailer.agent` and `hailer.tools`):

- **Sync public methods over one `asyncio.Runner`** that lives as long as the agent. LangGraph's synchronous `stream()` blocks in an untimed wait on Windows and Ctrl+C arrived 5 to 9 s late, after the turn had finished; under asyncio it cancels the task and aborts the HTTP request at once.
- **Blocking tools run on daemon threads.** Each `StructuredTool` has an async path that runs the call on a daemon thread, so Ctrl+C and exit never wait for a kernel call (on the loop's default executor, quitting waited 7 s for a call with 7 s left). The abandoned call finishes in the background and its result is dropped.
- **Conversations are LangGraph checkpoints** in `<workspace>/.hailer/threads.sqlite` (`AsyncSqliteSaver`); `.hailer/session.json` keeps the thread id and counters. An id the store does not know starts a new thread. `new_thread()` (`/new`, `/model`) deletes the previous thread: Hailer only ever resumes the latest.
- **`_settle_history` runs before each turn.** A tool call cut short by Ctrl+C gets a result (endpoints answer 400 to a tool call without one), and a new user message is merged into a trailing unanswered one (strict chat templates refuse two user messages in a row).
- **`SummarizationMiddleware` with an absolute token trigger**: `[model].summarize_after_tokens`, default 100000, 0 = off. The fractional trigger needs a model profile, which no custom deployment name has. The configured model writes the summary, so no request goes out under another model name.
- **The system prompt is sent on every model call**, so `/reload` applies from the next message of the same conversation.
- **LangSmith tracing is forced off** at start-up unless `HAILER_TRACING` is set (§7).
- **Interactive startup enables typing before agent and notebook preparation.** Dependency-only
  imports warm on a daemon worker while the CLI finds or starts the kernel. After server health is
  established, the composer renders immediately; agent setup and browser-session waiting overlap.
  One early submission waits for preparation without blocking editing. HTTP and SQLite resources
  remain on the agent's loop, and startup cancellation settles owned work before cleanup. Separate
  input/dispatch milestones and the offline `scripts/benchmark_startup.py` expose both perceived and
  total startup time; see [docs/LEARNINGS.md](../LEARNINGS.md) §10.

## 3. The `.config` folder: user-supplied context and skills

Layout (project-level, committed alongside the notebook):

```
.config/
└── hailer/
    ├── hailer.toml                 # project config (also accepted at ./hailer.toml; HAILER_CONFIG overrides)
    ├── context/                    # ALWAYS-ON context, loaded every session
    │   ├── 00-team.md              #   sorted by filename, concatenated, ~24 KiB cap with a warning
    │   └── pra101-glossary.md      #   e.g. column meanings, exposure-class rules, house style
    ├── skills/                     # ON-DEMAND skills in the Agent Skills format
    │   └── pra101-reconciliation/
    │       ├── SKILL.md            #   frontmatter: name, description (+ optional allowed-tools)
    │       ├── reference/*.md
    │       └── scripts/*.py
    └── prompts/                    # optional named prompts: /prompt monthly-pack
        └── monthly-pack.md
```

How each part is wired:

- **`context/*.md`** → `context.py` reads them at startup and after `/reload`, and `agent.system_prompt` appends them under a "Project context" heading. `/context` lists what is loaded and its size. Files are sent to the model endpoint every session, so the loader warns if a file looks like it contains a key or token (simple pattern check) and the README says so.
- **`skills/`** → progressive disclosure, Hailer-native so it works on Windows without symlinks:
  1. `context.py` parses each `SKILL.md` frontmatter and puts a compact index (name + description) in the system prompt.
  2. The tools `load_skill(name)` (returns the SKILL.md body and the file list) and `read_skill_file(name, path)` let the model pull in a skill only when the task matches.
  3. `/skill <name> [message]` invokes a skill explicitly: the SKILL.md text goes above the user's message in the same turn, under a `[Hailer]` notice that the system prompt explains.
- **`prompts/`** → `/prompt <name>` sends the file contents as the turn (with `{{args}}` substitution). Cheap and useful for repeatable monthly analyses.
- **`hailer.toml`** (strongly typed via dataclasses + stdlib `tomllib`):
  ```toml
  [hailer]
  notebook    = "notebooks/analysis.py"
  data_dir    = "data"
  context_dir = ".config/hailer/context"
  skills_dir  = ".config/hailer/skills"

  [model]
  name                   = "gpt-5.5"
  provider               = "openai"          # or "internal"
  reasoning_effort       = "medium"          # "" sends none
  summarize_after_tokens = 100000            # 0 = off

  [model_providers.internal]                 # see §1b
  base_url = "https://example.internal/v1"
  wire_api = "responses"                     # or "chat"
  env_key  = "INTERNAL_MODEL_API_KEY"
  ```
  Env overrides: `HAILER_MODEL`, `HAILER_MODEL_PROVIDER`, `HAILER_NOTEBOOK`, `HAILER_NOTEBOOKS_DIR`, `HAILER_DATA_DIR`, `HAILER_MARIMO_URL`, `HAILER_LOG_LEVEL`, `HAILER_CONFIG`, `HAILER_WORKSPACE`, `HAILER_KERNEL` (`[kernel].runtime`; `hailer notebook --kernel` beats it), `HAILER_KERNEL_IMAGE` (`[kernel].image`); `HAILER_MARIMO_TOKEN` and `HAILER_TRACING` exist only as environment variables. The `[kernel]` table (`runtime`, `image`, `memory`, `cpus`, `network`, `pass_env`) is documented in the README, *Isolated kernel (Docker)*. No secret goes in the file: the API key comes from the `env_key` variable or the credential store (§1b), and secret-looking or unknown keys are ignored with a warning.

A global `~/.config/hailer/` layer (same structure, loaded before the project one) is a natural v1.1 addition; not in v1.

## 4. Package layout

```
hailer/
├── pyproject.toml         [project.scripts] hailer = "hailer.cli:main"
├── hailer.toml            example project config
├── README.md, AGENTS.md, .gitignore, PLAN.md, uv.lock
├── docs/                  INTERFACES.md, LEARNINGS.md (guidance and regression checks for future agents),
│                          DOCKER_KERNEL_PLAN.md (the plan behind the 2026-09-19 change)
├── .config/hailer/        example context/, skills/, prompts/ with a short README
├── notebooks/analysis.py  valid marimo notebook: imports (mo, pl, duckdb), paths, welcome cell
├── data/                  .gitkeep, README.md
├── scripts/               make_sample_data.py ("25-01 pra101.parquet" ... for demos/tests), release.py,
│                          build_kernel_image.py (docker buildx for CI and the release)
├── src/hailer/
│   ├── cli.py             Typer app: chat REPL (default), `notebook`, `status`, `doctor`, `login`, `logout`, `init`,
│   │                      `kernel pull|build|stop`
│   ├── sandbox.py         the Sandbox contract and MarimoSandbox (notebook files over marimo's file API)
│   ├── kernel.py          kernel runtimes: the local kernel's environment, LocalRuntime, runtime_for
│   ├── kernel_docker.py   DockerRuntime (the docker CLI through an injectable runner), owner locks, status, kernel stop
│   ├── kernel_image.py    the kernel image: kernel contract tag, build context, build, pull, contract label
│   ├── _forward.py        the asyncio TCP forwarder the docker kernel is reached through
│   ├── docker/Dockerfile  the kernel image (package data)
│   ├── agent.py           build_model (provider → ChatOpenAI), HailerAgent (threads, streamed turns, Ctrl+C),
│   │                      system prompt, error mapping
│   ├── tools.py           HailerTools (the 11 tools in §2) and hailer_tools(config) → LangChain tools
│   ├── marimo_client.py   HTTP/SSE client, session resolution, the marimo launch command
│   ├── notebooks.py       notebook names, active-notebook state (.hailer/notebook.json), resolution, templates
│   ├── browser.py         opens a URL in the user's browser (os.startfile first on Windows)
│   ├── web.py             allow-list matching, fetch_page, HTML to text, truncation
│   ├── secrets.py         API-key resolution (env var, then keyring); storage for login/logout
│   ├── config.py          dataclasses + tomllib + env overrides + validation
│   ├── context.py         context/, skills index, prompts
│   ├── session.py         slash-command parsing, session state (.hailer/session.json)
│   ├── periods.py         YY-MM filename parsing, multi-period Polars/DuckDB loaders
│   ├── models.py          typed internal models (ExecResult, MarimoSession, ProviderConfig, TurnSummary, ...)
│   ├── errors.py          HailerError and its subclasses: a short message plus a hint
│   ├── log.py             logging setup, secret-redaction filter
│   └── prompts/system.md  agent system prompt (package data, loaded with importlib.resources)
└── tests/                 test_<module>.py for agent, tools, cli, config, session, secrets, marimo_client, notebooks,
                           browser, web, context, periods, log, kernel, kernel_docker, kernel_image, forward;
                           test_release, test_build_kernel_image (no key, no marimo, no Docker, loopback only);
                           helpers: fake_gateway.py (the strict Chat-Completions-only gateway of §1a), fake_marimo.py,
                           fake_docker.py, fake_kernel.py; test_docker_integration.py is opt-in (HAILER_DOCKER_TESTS)
```

Dependencies: `marimo` (pinned `==0.24.2`, `_code_mode` is private), `langchain>=1.4,<2`, `langchain-openai>=1.6,<2`, `langgraph-checkpoint-sqlite>=3.1,<4`, `polars[calamine]` (includes `fastexcel` for Excel reading in local and Docker kernels), `duckdb`, `typer`, `rich`, `keyring`; dev: `pytest`. Standard library for TOML, the marimo HTTP/SSE client, page fetching, logging, subprocess.

## 5. Data conventions (`periods.py`)

- `parse_period("25-03 pra101.parquet") -> Period(year=2025, month=3, label="2025-03", stem="pra101")`; invalid names return `None`.
- `scan_period_files(data_dir, name="pra101") -> list[PeriodFile]` sorted by period.
- `load_periods(files) -> pl.DataFrame` using `pl.concat(..., how="diagonal_relaxed")` with a `period` column added per file, so schema evolution (new columns in later months) is tolerated; a malformed file raises `MalformedParquetError` naming the file.
- `duckdb_periods_view(con, files)` registers a view via `read_parquet([...], union_by_name=true, filename=true)` and derives `period` from the filename.
- The notebook imports these (`from hailer.periods import ...`) so the agent reuses them instead of re-deriving the logic each turn.

## 6. Build order (smallest end-to-end proof first)

| Milestone | Deliverable | Proof |
|---|---|---|
| M0 Scaffold | `pyproject.toml`, `uv sync`, package skeleton, notebook, `.gitignore` | `uv run hailer --help` |
| M1 Marimo path | `marimo_client.py` + `/exec` | With the notebook open in a browser: `/exec <cm snippet>` in the chat adds a visible cell |
| M2 Agent path | `tools.py`, `agent.py` (chat model, tool loop, system prompt), hard-wired turn | `> create a simple dataframe and display it in Marimo` produces a notebook change |
| M3 Conversation | REPL, `session.py`, streaming progress, Ctrl+C interrupt, EOF exit, thread resume | Three-turn definition-of-done sequence (dataframe → chart → summary) |
| M4 Config & preflight | `config.py`, `hailer.toml`, env overrides, startup checks with actionable messages, `hailer notebook`, `--verbose` | Each error case in the spec prints a fix, no traceback |
| M5 Data utilities | `periods.py`, sample-data generator, notebook helpers | The realistic PRA101 sequence works |
| M6 Context & skills | `context.py`, `.config/hailer/*`, `/skill`, `/prompt`, `/reload`, `/context`, system prompt | A project skill is listed, loaded on demand, and changes behaviour |
| M7 Tests, README, polish | pytest suite, README with security section, Rich panel | `uv run pytest` green, no API key needed |

Slash commands: `/help /status /new /exit /quit /model /notebook /clear /context /skill /prompt /reload`.

## 7. Risks and how the plan handles them

- **Private `marimo._code_mode` API** → pin marimo; the `cm` patterns live in `prompts/system.md`, the `marimo_execute` description and `marimo_client.py`; the kernel image's contract pins marimo, and the Docker integration test runs code mode in a real kernel.
- **No browser session** → `marimo_status` detects it; the CLI, `notebook_create` and `notebook_open` open the URL (`hailer.browser`) and wait for the session.
- **Session id churn on page refresh** → every call resolves the session by notebook path, never caches ids.
- **LangChain's release pace** → `langchain` and `langchain-openai` are pinned `<2`, `langgraph-checkpoint-sqlite` `<4`; the `build_model` docstring records what to re-check before raising a bound, and the offline fake-gateway tests catch wire regressions on an upgrade.
- **More Python packages**, several with compiled wheels (`orjson`, `ormsgpack`, `xxhash`, `zstandard`, `tiktoken`, `jiter`, `regex`, `uuid-utils`, `sqlite-vec`) → extension modules of the kind Hailer already shipped (`polars`, `duckdb`, `pydantic-core`); nothing installed is a standalone executable.
- **`langsmith` is installed transitively** → tracing is forced off unless `HAILER_TRACING` is set, so a `LANGSMITH_TRACING=true` left over from another project cannot send prompts and tool results to a third party; the explicit `base_url` closes the `LANGSMITH_GATEWAY` variant (§1a).
- **`ChatOpenAI` targets the official OpenAI wire format** → known upstream issues with OpenAI-compatible gateways, none seen in the spike or the tests: tool-call arguments lost when a gateway fragments streamed tool-call chunks unusually (langchain #35514, #35782), where `stream = false` is the workaround; a deployment whose name starts with `o1`, `o3` and so on gets the `developer` role instead of `system`; non-standard `reasoning_content` is ignored.
- **The checkpoint file grows** → LangGraph writes a checkpoint per step and never prunes. Only the latest thread is kept (`/new` and `/model` delete the previous one); one long conversation still grows until then.
- **Long Polars/DuckDB jobs** → `marimo_execute` waits up to 600 s and the client streams SSE. Ctrl+C cancels the turn, not the kernel: the next turn tells the model that the call's effect is unknown.
- **Large outputs into model context** → tool results capped (head/tail with a "truncated" marker), rich HTML/JSON outputs replaced by a placeholder, and the prompt tells the agent to aggregate locally and inspect summaries.
- **What leaves the machine** (README security section): user turns, system prompt + `.config` context, skill bodies when loaded, tool call arguments (code) and truncated results, and pages fetched from allowed domains, all to the configured endpoint only. Raw datasets stay local unless code prints them. With the default local runtime `marimo_execute` runs the model's Python unsandboxed in a kernel that runs as the user (secret-looking variables withheld, the server behind a token), so the prompt's safety rules are instructions, not an enforcement boundary; the docker runtime is the enforced one (two folders, data read-only, no network, no host environment). Secrets are never placed in prompts; logs redact `Authorization` headers and values of any env var named `*_KEY`/`*_TOKEN`/`*_SECRET`/`*_PASSWORD`.
- **Docker is not always available** (Docker Desktop's licence for larger organisations, managed machines that block it, WSL2 or Hyper-V) → `local` stays the default and fully supported; docker is opt-in and fails closed, never falling back to local.
- **Notebooks written in a container are code** that runs on the host if later opened in local mode → a warning on the first local start after a docker kernel used the folder (`.hailer/last-kernel.json`); the README tells users to keep the notebooks folder in git.
- **The kernel image is part of every release** → the release pushes it to GHCR before PyPI, so no released Hailer points at a missing image; `hailer kernel build` removes the dependency on the registry; the image's version label must match Hailer's version.

## 8. Assumptions made (say if any should change)

- The user-facing config folder is `.config/hailer/` as requested; `hailer.toml` is also accepted at the project root.
- Every provider needs an API key, the built-in `openai` one included; `gpt-5.5` is the default model for the `openai` provider.
- The endpoint speaks the official OpenAI wire format, Responses API or Chat Completions, with function tool calls.
- Tools are Hailer's own in-process functions (`hailer.tools`), not the upstream bash scripts and not an MCP server.
- Hailer resumes only the latest conversation of a workspace.
- Git Bash / WSL are not required by anything in Hailer.
- The notebook kernel runs locally unless the user chooses docker; Docker is never required, and the docker
  runtime drives the `docker` CLI (Docker Desktop or Docker Engine), not Podman.
