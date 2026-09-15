# Hailer — implementation plan

Investigated 2026-09-15 against the real installed APIs: `openai-codex` 0.154.0, `marimo` 0.24.2,
`polars` 1.44.2, `duckdb` 1.5.5, `typer` 0.27.2, `rich` 15.0.0, `pytest` 9.1.1, `uv` 0.12.1, Python 3.12/3.13 on Windows 11.

## 1. Findings that change the original spec

1. **`marimo-pair` is not a PyPI package.** It is two things:
   - `marimo pair prompt --url ... --codex` (built into marimo >= 0.21): only *prints a prompt string* telling an agent to use a skill.
   - The GitHub repo `marimo-team/marimo-pair`: an Agent Skill (`SKILL.md`, `scripts/execute-code.sh`, `scripts/discover-servers.sh`, `reference/*.md`). The scripts need **bash + curl + jq**. `jq` is not installed on this machine, so the stock scripts would not run here anyway.

2. **The whole "pair" protocol is three plain HTTP calls.** Hailer will implement it in pure Python (stdlib `urllib`), so nothing in Hailer depends on Git Bash or WSL:
   - `GET  {url}/health` (no auth) → `{"status": "healthy"}`
   - `GET  {url}/api/sessions` → `{ "<session_id>": {"filename": ..., "path": ...}, ... }`
   - `POST {url}/api/kernel/execute` with header `Marimo-Session-Id: <id>` and body `{"code": "..."}` → Server-Sent Events: `stdout` / `stderr` (`{"data": "..."}`) then `done` (`{"success": bool, "output": {"mimetype": ..., "data": ...}}`).
   - Optional `Authorization: Bearer <token>` for tokened servers. Servers started with `--no-token` self-register at `%USERPROFILE%\.marimo\servers\<host>_<port>.json` (`server_id, pid, host, port, base_url, started_at, version`) which Hailer can read for auto-discovery.

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

5. **The Codex Python SDK works on this machine.** `Codex()` launches the bundled `codex.exe` app-server over stdio in ~2.3 s, reuses the existing ChatGPT login in `~/.codex/auth.json`, and lists models `gpt-5.5`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-6-astra`. Key API surface:
   - `Codex(CodexConfig(config_overrides=("key=value", ...), cwd=..., env=...))` — overrides are passed as `codex -c key=value`, exactly the config.toml keys.
   - `codex.thread_start(model=, model_provider=, developer_instructions=, base_instructions=, sandbox=Sandbox.workspace_write, approval_mode=ApprovalMode.deny_all, cwd=, config={...})`, `codex.thread_resume(thread_id)`.
   - `thread.turn(text)` → `TurnHandle.stream()` yields notifications (`item/agentMessage/delta`, `item/started`, `item/completed`, `item/commandExecution/outputDelta`, `item/mcpToolCall/progress`, `turn/completed`); `handle.interrupt()` for Ctrl+C; `handle.run()` → `TurnResult(final_response, items, usage, status, error)`.
   - Turn input can include `SkillInput(name, path)` to explicitly invoke a skill by path.

6. **The provider config in the spec is already Codex's own format.** `[model_providers.internal] base_url / wire_api="responses" / env_key / requires_openai_auth` is native Codex config, so Hailer does not implement any HTTP provider code. Hailer translates `hailer.toml` into `-c` overrides and Codex talks to the endpoint. Verified that `-c` overrides register MCP servers and that a whole-table override **merges** with the user's config rather than replacing it.

7. **This machine's `~/.codex/config.toml` belongs to the Codex desktop app.** It enables MCP servers (`cua_repl`, `node_repl`) and many plugins (browser, computer-use, spreadsheets...). Hailer must not inherit those into an analysis session: it reads the user config with `tomllib` and passes `mcp_servers.<name>.enabled=false` / `plugins."<id>".enabled=false` overrides for everything it did not add. `HAILER_CODEX_HOME` gives full isolation (then auth needs `OPENAI_API_KEY` or a one-off `hailer login`).

8. **The SDK cannot register in-process tools.** Its only server-to-client hook is the approval handler, so Hailer's tools must be an **MCP server** (stdio) or shell commands. Decision: MCP (see §2).

9. **Where Codex finds context on its own** (relevant to the `.config` requirement):
   - Skills: `.agents/skills/` in cwd, each parent up to the repo root, and `~/.agents/skills/` (legacy `.codex/skills` strings also present in the binary).
   - Instructions: `~/.codex/AGENTS.md` (or `AGENTS.override.md`), then `AGENTS.md` in each directory from repo root down to cwd; combined, 32 KiB cap (`project_doc_max_bytes`).

## 1a. Custom (non-OpenAI) endpoints: verified end to end

Tested on 2026-09-15 by pointing the Codex SDK at a fake local server that implements only `POST /v1/responses`, configured purely through `-c` overrides (what Hailer will generate from `hailer.toml`):

```
model="internal-analyst-model"
model_provider="internal"
model_providers.internal.base_url="http://127.0.0.1:8765/v1"
model_providers.internal.wire_api="responses"
model_providers.internal.env_key="INTERNAL_MODEL_API_KEY"
model_providers.internal.requires_openai_auth=false
```

Result: the app-server started with **no ChatGPT/OpenAI login** (`account: None, requires_openai_auth: False`), sent `Authorization: Bearer <value of INTERNAL_MODEL_API_KEY>`, and the turn completed with the fake endpoint's text as `final_response`.

What a bespoke endpoint must provide:

- **The Responses API, streaming.** `POST {base_url}/responses` with `stream: true`, replying with SSE events (`response.created`, `response.output_item.added`, `response.output_text.delta`, `response.output_item.done`, `response.completed` with `usage`). The Codex 0.154 binary contains no Chat Completions client at all, and `wire_api` accepts only `"responses"`. **If the internal endpoint only speaks Chat Completions, a translating proxy (for example LiteLLM) must sit in front of it.** This is the single hard requirement.
- **Tolerate a `GET {base_url}/models?client_version=...` probe** at startup. Returning 404 is fine; Codex carried on.
- **Accept these request fields**: `model`, `instructions`, `input`, `tools` (type `function`), `tool_choice`, `parallel_tool_calls`, `reasoning` (`{"effort": ..., "summary": "auto"}`), `include` (`["reasoning.encrypted_content"]`), `store: false`, `prompt_cache_key`, `client_metadata`. A gateway that rejects unknown fields must be relaxed for `reasoning`, `include` and `client_metadata`.
- **Auth is a bearer token from the env var named in `env_key`**; extra static or env-sourced headers are available via `model_providers.<id>.http_headers` / `env_http_headers` if the gateway needs them.

What Hailer must trim so a strict gateway is not surprised (measured on a "Say hello" turn):

| Configuration | Request size | Tools sent |
|---|---|---|
| Defaults (inherits the desktop app's config) | 106.8 KB | 8 functions incl. `request_plugin_install`, `web_search`, `multi_agent_v1` and 6 `mcp__codex_apps__*` namespaces, plus a `<recommended_plugins>` block |
| `base_instructions` + web search / multi-agent / plugin features off + user plugins disabled | 73.8 KB | still `mcp__node_repl` and `mcp__codex_apps__*` namespaces |
| + `mcp_servers.node_repl.enabled=false`, `features.apps=false`, `features.codex_apps=false`, `features.connectors=false` | **11.3 KB** | `exec_command`, `write_stdin`, `request_user_input`, `view_image` only |

So `agent.py` will always pass: `base_instructions=<Hailer system prompt>` (replaces Codex's built-in "You are a coding agent" prompt), `web_search="disabled"`, `features.web_search_request=false`, `features.multi_agent=false`, `features.multi_agent_v2=false`, `features.plugins=false`, `features.apps=false`, `features.codex_apps=false`, `features.connectors=false`, `features.image_generation=false`, `plugins."<id>".enabled=false` for every plugin in the user's config, and `mcp_servers.<name>.enabled=false` **only for servers defined in the user's `config.toml`** (disabling a plugin-provided server that way creates an incomplete table and the app-server refuses to start with "invalid transport"). Hailer's own MCP server is then the only namespace tool.

### 1b. How a user points Hailer at their own endpoint (user-facing workflow)

1. **Describe the endpoint once in `hailer.toml`** (project root or `.config/hailer/hailer.toml`). Several providers can be defined; `[model].provider` picks the active one.
   ```toml
   [model]
   name     = "risk-analyst-v3"          # whatever model id the gateway expects
   provider = "internal"

   [model_providers.internal]
   base_url             = "https://llm.example.internal/v1"
   wire_api             = "responses"
   env_key              = "INTERNAL_MODEL_API_KEY"
   requires_openai_auth = false
   # optional, passed straight through to Codex:
   # http_headers     = { "X-Team" = "risk-analytics" }
   # env_http_headers = { "X-Client-Id" = "INTERNAL_CLIENT_ID" }
   # query_params     = { "api-version" = "2025-04-01-preview" }   # Azure-style gateways
   ```
2. **Store the secret once with `uv run hailer login internal`.** Hailer prompts with hidden input and saves the key in the OS credential store through `keyring` (Windows Credential Manager, verified working on this machine: `WinVaultKeyring` round-trip OK). Nothing is written to `hailer.toml`, shell history, or any Hailer file. `hailer logout internal` removes it.

   How it reaches Codex: Codex only reads provider keys from the env var named in `env_key`, but the SDK's `CodexConfig(env={...})` sets variables for the **Codex child process only**. So at startup Hailer resolves the key and injects it there; it is never exported to the user's shell, never passed on a command line, and never logged (redaction filter). Resolution order per provider: (a) env var already set in the shell (for CI/automation, keeps the original workflow), (b) `keyring` entry `service="hailer", username="<provider>:<env_key>"`, (c) neither → actionable message suggesting `hailer login <provider>`. The same path works for `provider = "openai"` by injecting `OPENAI_API_KEY`, which avoids Codex's plaintext `~/.codex/auth.json` for API-key users; the existing ChatGPT login still works untouched. `hailer doctor` reports where the key came from (`keyring` or `env`), never the value.

   Same-user processes can read Credential Manager entries just as they can read env vars, so this is about not leaving the key in files and history, not about isolating it from the user's own account. `keyring` adds six small pure-Python packages; no bash, no native build.
3. **Run `uv run hailer doctor`** (also part of normal startup). It reports: config parsed, provider `internal` selected, env var present, `GET {base_url}/models` reachable (404 is acceptable), marimo running, notebook found. Failures print a fix, for example:
   ```
   INTERNAL_MODEL_API_KEY is not set (required by provider "internal").
   Set it in this terminal and run Hailer again:
       set INTERNAL_MODEL_API_KEY=<your key>
   ```
4. **Run `uv run hailer`.** The startup panel shows `Model: risk-analyst-v3` and `Provider: internal (https://llm.example.internal/v1)`. Under the hood Hailer launches the Codex app-server with the provider as `-c` overrides for that session only; it never edits `~/.codex/config.toml`, so the user's Codex desktop/CLI setup is untouched.
5. **Switch when needed.** `/model <name>` changes the model for the next turn; `/model internal:<name>` or `/model openai:gpt-5.5` changes provider too. One-off overrides: `HAILER_MODEL`, `HAILER_MODEL_PROVIDER`. With `provider = "openai"` Hailer instead reuses the existing Codex login or `OPENAI_API_KEY`.
6. **If the gateway only speaks Chat Completions**, run a translating proxy locally and point `base_url` at it (documented in the README with a LiteLLM example). Hailer detects the symptom (404 or a validation error on `POST /responses`) and prints that advice rather than a stack trace.

`hailer status` inside the chat shows the same provider line plus the thread id and token usage, so the user can always confirm which endpoint a session is talking to.

### 1c. Allowed web domains for context (verified enforcement)

Users can list websites the agent may read for context (regulator pages, internal documentation, library docs). Anything not listed is unreachable, not merely discouraged. Config:

```toml
[web]
allowed_domains = [
  "www.bankofengland.co.uk", "**.bankofengland.co.uk",   # ** = apex plus subdomains, * = subdomains only
  "docs.pola.rs", "duckdb.org",
  "confluence.example.internal",
]
max_page_bytes      = 200000   # cap per fetch before HTML-to-text conversion
allow_shell_network = false    # also let shell commands reach the same domains (see below)
```

With no `[web]` section the agent has **no internet access at all**: the default Codex sandbox already blocks outbound connections from shell commands (verified on this machine: `curl https://example.com` fails to connect, while loopback such as marimo is reachable), and Hailer's fetch tool refuses every URL.

Two enforcement layers, both driven by the same list:

1. **Hailer's `fetch_page(url)` MCP tool** (primary path). Runs outside the command sandbox, so TLS and DNS behave normally. It checks the host against `allowed_domains` with Codex's wildcard semantics, fetches with stdlib `urllib`, converts HTML to readable text (stdlib `html.parser`), caps the size, and returns the text to the model. A disallowed host returns a one-line refusal that names the allowed domains so the agent can ask the user to extend the list. Codex's proxy deliberately does not filter MCP tools, so this check is Hailer's responsibility.
2. **Codex's network proxy allowlist for shell commands** (opt-in via `allow_shell_network = true`). Verified working on Windows with the documented schema, passed as `-c` overrides: `features.network_proxy.enabled=true`, `features.network_proxy.domains={"example.com"="allow"}`, `network.enabled=true`, `network.mode="limited"`, `network.domains={...}`, `sandbox_workspace_write.network_access=true`. Result: `http://example.com` returned 200, `https://www.python.org` was blocked with `Network access to "www.python.org" was blocked: domain is not on the allowlist`, and loopback worked only when `127.0.0.1` was listed. Hailer adds `127.0.0.1` and `localhost` automatically so the `hailer exec` shell fallback can still reach marimo. Two Windows-sandbox quirks make this the secondary path: `curl.exe` fails HTTPS inside the sandbox with a schannel credentials error, and the sandbox could not execute a Python interpreter living outside the workspace (`Access is denied` for the uv-managed interpreter under `%APPDATA%`).

What this means for data leaving the machine: text fetched from an allowed site becomes tool output and is sent to the configured model endpoint like any other tool result. The README security section says so, and `hailer status` shows the active allowlist.

Later options (not v1): per-domain auth headers from keyring for internal wikis (`[web.auth."confluence.example.internal"] header = "Authorization" keyring = true`), and a `denied_domains` list mirroring Codex's deny-wins rule.

## 2. Architecture (revised)

```
Terminal  (Typer + Rich)                                   cli.py / session.py
    │  text turns, slash commands
    ▼
agent.py ── openai-codex SDK ── codex.exe app-server ── model endpoint
                                     │                  (OpenAI or internal Responses API,
                                     │ MCP over stdio    configured via -c overrides)
                                     ▼
                              hailer-mcp  (mcp_server.py)
                              tools: marimo_execute, marimo_status,
                                     notebook_cells, list_periods,
                                     load_skill, read_skill_file,
                                     fetch_page (allowlisted domains only)
                                     │
                                     ▼
                              marimo_client.py  (pure-Python HTTP + SSE)
                                     │
                                     ▼
                  marimo edit notebooks/analysis.py --no-token   (live kernel)
                       scratchpad  +  marimo._code_mode  →  Polars / DuckDB / charts
                                     │
                                     ▼
                                 Browser UI
```

Why an MCP server instead of the upstream shell scripts:
- No bash/curl/jq; works from `cmd.exe`/PowerShell.
- Typed arguments: multi-line Python arrives intact (cmd.exe has no heredocs; shell quoting was the main Windows failure mode).
- Hailer controls the result shape: truncation of large outputs with head/tail, session resolution by notebook path on every call, clear error strings the model can act on.
- Same client is also exposed as `uv run hailer exec -c "..."` for humans and debugging.

Alternative kept as a documented option, not the default: marimo's hidden `marimo edit --mcp code-mode` flag exposes `list_sessions` / `execute_code` over streamable HTTP at `/mcp/server` (needs `marimo[mcp]`). Zero Hailer code, but experimental and it requires the user to remember the flag.

Codex session settings: `cwd=<workspace>`, `Sandbox.workspace_write`, `ApprovalMode.deny_all` (no interactive approval prompts in a chat CLI; the agent can still run commands and write inside the workspace), `developer_instructions` = Hailer system prompt + user context (see §3). The Windows sandbox behaviour of the app-server (`windowsSandbox/setupCompleted` notifications exist; user config has `[windows] sandbox = "elevated"`) is verified in milestone M2.

## 3. The `.config` folder: user-supplied context and skills

Layout (project-level, committed alongside the notebook):

```
.config/
└── hailer/
    ├── hailer.toml                 # project config (also accepted at ./hailer.toml; HAILER_CONFIG overrides)
    ├── context/                    # ALWAYS-ON context, loaded every session
    │   ├── 00-team.md              #   sorted by filename, concatenated, ~24 KiB cap with a warning
    │   └── pra101-glossary.md      #   e.g. column meanings, exposure-class rules, house style
    ├── skills/                     # ON-DEMAND skills in the Agent Skills format Codex/Claude use
    │   └── pra101-reconciliation/
    │       ├── SKILL.md            #   frontmatter: name, description (+ optional allowed-tools)
    │       ├── reference/*.md
    │       └── scripts/*.py
    └── prompts/                    # optional named prompts: /prompt monthly-pack
        └── monthly-pack.md
```

How each part is wired:

- **`context/*.md`** → `context.py` reads them at startup and after `/reload`, and appends them under a "Project context" heading in `developer_instructions`. `/context` lists what is loaded and its size. Files are sent to the model endpoint every session, so the loader warns if a file looks like it contains a key or token (simple pattern check) and the README says so.
- **`skills/`** → progressive disclosure, Hailer-native so it works on Windows without symlinks:
  1. `context.py` parses each `SKILL.md` frontmatter and puts a compact index (name + description) in the system prompt.
  2. The MCP server exposes `load_skill(name)` (returns the SKILL.md body and the file list) and `read_skill_file(name, relative_path)`, so the model pulls in a skill only when the task matches.
  3. `/skill <name> [message]` sends `SkillInput(name, path)` with the turn for explicit invocation. If `SkillInput` turns out not to accept arbitrary paths (unverified), the fallback is to prepend the SKILL.md text to the turn.
  4. Users who prefer Codex's own discovery can put skills in `.agents/skills/` instead; documented, not managed by Hailer.
- **`prompts/`** → `/prompt <name>` sends the file contents as the turn (with `{{args}}` substitution). Cheap and useful for repeatable monthly analyses.
- **`hailer.toml`** (strongly typed via dataclasses + stdlib `tomllib`):
  ```toml
  [hailer]
  notebook    = "notebooks/analysis.py"
  data_dir    = "data"
  marimo_url  = "http://127.0.0.1:2718"     # optional; auto-discovered from the registry when omitted
  context_dir = ".config/hailer/context"
  skills_dir  = ".config/hailer/skills"

  [model]
  name             = "gpt-5.5"
  provider         = "openai"                # or "internal"
  reasoning_effort = "medium"

  [model_providers.internal]                 # passed through to Codex verbatim
  base_url             = "https://example.internal/v1"
  wire_api             = "responses"
  env_key              = "INTERNAL_MODEL_API_KEY"
  requires_openai_auth = false
  ```
  Env overrides: `HAILER_MODEL`, `HAILER_MODEL_PROVIDER`, `HAILER_NOTEBOOK`, `HAILER_DATA_DIR`, `HAILER_MARIMO_URL`, `HAILER_LOG_LEVEL`, `HAILER_CODEX_HOME`, `HAILER_CONFIG`. Hailer never reads the secret itself: preflight only checks that `os.environ` *has* the `env_key` name (or that a Codex login exists for the `openai` provider) and Codex reads the value.

A global `~/.config/hailer/` layer (same structure, loaded before the project one) is a natural v1.1 addition; not in v1.

## 4. Package layout

```
hailer/
├── pyproject.toml         [project.scripts] hailer = "hailer.cli:app", hailer-mcp = "hailer.mcp_server:main"
├── hailer.toml            example project config
├── README.md, .gitignore, PLAN.md
├── .config/hailer/        example context/, skills/, prompts/ with a short README
├── notebooks/analysis.py  valid marimo notebook: imports (mo, pl, duckdb), paths, welcome cell
├── data/.gitkeep          (+ scripts/make_sample_data.py generating "25-01 pra101.parquet" ... for demos/tests)
├── src/hailer/
│   ├── cli.py             Typer app: chat REPL (default), `notebook`, `exec`, `status`, `doctor`
│   ├── agent.py           Codex lifecycle: overrides, thread start/resume, streamed turns, interrupt
│   ├── marimo_client.py   HTTP/SSE client, registry discovery, session resolution, typed errors
│   ├── mcp_server.py      stdio MCP server (`mcp` package) exposing the tools in §2
│   ├── config.py          dataclasses + tomllib + env overrides + validation
│   ├── context.py         context/, skills index, prompts
│   ├── session.py         slash-command parsing, session state, thread-id persistence (.hailer/session.json)
│   ├── periods.py         YY-MM filename parsing, multi-period Polars/DuckDB loaders
│   ├── models.py          typed internal models (ExecResult, MarimoSession, TurnSummary, ...)
│   ├── logging.py         logging setup, secret-redaction filter
│   └── prompts/system.md  agent system prompt (package data, loaded with importlib.resources)
└── tests/                 test_config, test_periods, test_session, test_cli, test_marimo_client,
                           test_context, test_agent_overrides  (all mocked; no key, no server)
```

Dependencies: `openai-codex`, `marimo` (pinned, `_code_mode` is private), `mcp`, `polars`, `duckdb`, `typer`, `rich`; dev: `pytest`. Standard library for TOML, HTTP, SSE parsing, logging, subprocess.

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
| M1 Marimo path | `marimo_client.py` + `hailer exec` | With the notebook open in a browser: `uv run hailer exec -c "<cm snippet>"` adds a visible cell |
| M2 Agent path | `mcp_server.py`, `agent.py` (overrides, disable inherited tools, `developer_instructions`), hard-wired turn | `> create a simple dataframe and display it in Marimo` produces a notebook change |
| M3 Conversation | REPL, `session.py`, streaming progress, Ctrl+C interrupt, EOF exit, thread resume | Three-turn definition-of-done sequence (dataframe → chart → summary) |
| M4 Config & preflight | `config.py`, `hailer.toml`, env overrides, startup checks with actionable messages, `hailer notebook`, `--verbose` | Each error case in the spec prints a fix, no traceback |
| M5 Data utilities | `periods.py`, sample-data generator, notebook helpers | The realistic PRA101 sequence works |
| M6 Context & skills | `context.py`, `.config/hailer/*`, `/skill`, `/prompt`, `/reload`, `/context`, system prompt | A project skill is listed, loaded on demand, and changes behaviour |
| M7 Tests, README, polish | pytest suite, README with security section, Rich panel | `uv run pytest` green, no API key needed |

Slash commands: `/help /status /new /exit /quit /model /notebook /clear /context /skill /prompt /reload`.

## 7. Risks and how the plan handles them

- **Private `marimo._code_mode` API** → pin marimo; all `cm` patterns live in `prompts/system.md` and one helper module; `hailer doctor` runs `help(cm)` to detect drift.
- **No browser session** → `marimo_status` detects it; CLI opens the URL with `webbrowser` and waits/prompts.
- **Session id churn on page refresh** → every call resolves the session by notebook path, never caches ids.
- **Inherited desktop-app MCP servers/plugins** → disabled by name via overrides; `HAILER_CODEX_HOME` for full isolation.
- **Windows sandbox / approvals in app-server** → verified in M2; `Sandbox.workspace_write` + `deny_all`; documented fallback.
- **Windows sandbox cannot run interpreters outside the workspace** → observed: `Access is denied` when a shell command invoked the uv-managed Python under `%APPDATA%`, and `python` is not on the sandbox PATH. All Python execution therefore goes through Hailer's MCP server into the marimo kernel (both outside the sandbox); the agent's shell is for light workspace inspection only, and the system prompt says so. M2 checks whether `sandbox_workspace_write.writable_roots` or a workspace-local `.venv` is needed for any shell use of `uv run`.
- **Tool timeouts on big Polars/DuckDB jobs** → `mcp_servers.hailer.tool_timeout_sec=600`; the client streams SSE so nothing blocks silently.
- **Large outputs into model context** → tool results capped (head/tail with a "truncated" marker) and the prompt tells the agent to aggregate locally and inspect summaries.
- **`SkillInput` with arbitrary paths unverified** → text-injection fallback.
- **What leaves the machine** (README security section): user turns, system prompt + `.config` context, skill bodies when loaded, tool call arguments (code) and truncated results, any file the agent reads via its shell, and any `AGENTS.md` Codex discovers. Raw datasets stay local unless code prints them. Secrets are never placed in prompts; logs redact `Authorization` headers and values of any env var named `*_KEY`/`*_TOKEN`.

## 8. Assumptions made (say if any should change)

- The user-facing config folder is `.config/hailer/` as requested; `hailer.toml` is also accepted at the project root.
- The shared `~/.codex` home (existing ChatGPT login) is the default; `gpt-5.5` is the default model for the `openai` provider.
- Tools are exposed to the agent via Hailer's own MCP server, not the upstream bash scripts.
- Git Bash / WSL are not required by anything in Hailer.
