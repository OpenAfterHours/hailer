# Hailer module contracts

This is the contract every module is built against. `PLAN.md` explains *why*; this file says *what* each
module exposes so work can proceed in parallel. Shared types live in `src/hailer/models.py`, errors in
`src/hailer/errors.py`. Do not change those two files without agreement (report a needed change instead).

## Ground rules

- **Windows first.** No bash, curl, jq, or POSIX-only assumptions. Use `pathlib`, `subprocess` with argument
  lists, `os.name`/`sys.platform` checks where needed. Paths in user output use the OS separator.
- **Standard library where practical.** HTTP via `urllib.request`/`http.client`, TOML via `tomllib`, SSE parsing
  by hand, HTML-to-text via `html.parser`. Third-party allowed: `polars`, `duckdb`, `typer`, `rich`, `keyring`,
  `openai_codex`, `mcp`, `marimo` (only in the notebook / `hailer exec` snippets, never imported by the CLI
  process except to locate the executable).
- **Tests are offline.** No API key, no marimo server, no Codex process, no internet. Use fakes: a local
  `http.server` for the marimo protocol, a fake `Codex` object for the SDK, `tmp_path` for files, an in-memory
  keyring backend (`keyring.backends.fail` / a tiny custom backend) for secrets.
- **Secrets never appear** in logs, prompts, tool results, exception messages, or config files. The logging
  redaction filter masks `Authorization` headers and any value whose env var name ends in `_KEY`, `_TOKEN`,
  `_SECRET`, `_PASSWORD`.
- **Errors are actionable.** Raise `HailerError` subclasses with a `hint`. The CLI is the only place that
  prints; library modules never print.
- **Ownership.** Each module has one owner during a wave. Only edit files you own; if you need something from
  another module that is not in this contract, write against the contract and note the gap in your report.
- **Dependencies** are already declared in `pyproject.toml`; do not add any. Report if something is missing.
- **The active notebook is shared state, not config.** `<workspace>/.hailer/notebook.json` (next to
  `session.json`) records which notebook the chat is working in. Exactly one process writes it at a time: the
  CLI between turns (slash commands) and the MCP server during a turn (tool calls). Every reader re-reads it
  before use through `hailer.notebooks.load_active_notebook` and never caches it. `load_active_notebook` reads
  **only the file**: `HAILER_NOTEBOOK` is inherited by the Codex child and therefore by the MCP server, so an
  env-wins rule would make the tool server ignore every switch. Instead the CLI's `_load_config` writes an
  explicit `HAILER_NOTEBOOK` to the file at startup (`save_active_notebook`), so both processes and later sessions
  start on that notebook. Residual race: a tool call still running after Ctrl+C may write the file later; the CLI
  re-reads it after every turn (finished, failed or interrupted) and before `/notebook` and `/status`.

## Runtime facts (verified; see PLAN.md §1, §1a–1c)

- Multi-notebook (verified live on marimo 0.24.2, both `marimo edit <file>` and `marimo edit <dir>`): one
  server opens any *existing* notebook at runtime via `?file=<key>`, no restart needed (in single-file mode the
  session manager registers the path in the workspace allow-list before loading it). Keys resolve against the
  server cwd in single-file mode and against the folder in directory mode; **absolute posix paths work in
  both**, so Hailer always sends `?file=<absolute path>`. `?file=__new__` gives an untitled session
  (`path: null`). A missing file closes the websocket. A second connection to a file that already has a
  session joins it (no duplicate kernel). Sessions outlive the browser tab. Hailer starts marimo on the
  notebooks *folder* so marimo's own home page lists every notebook too.
- Skew protection: every POST except `/api/kernel/execute`, `/ws` and login needs the header
  `Marimo-Server-Token`, whose value is the `data-token` attribute of `<marimo-server-token>` in `GET /`;
  without it marimo answers 401. Needed for `/api/home/{workspace_files,running_notebooks,shutdown_session}`
  and `/api/kernel/{save,rename}`. GETs are exempt.

- marimo protocol: `GET {url}/health`, `GET {url}/api/sessions` → `{sid: {filename, path}}`,
  `POST {url}/api/kernel/execute` with header `Marimo-Session-Id: <sid>`, body `{"code": "..."}`, response
  `text/event-stream` with events `stdout`/`stderr` (`{"data": str}`) and `done`
  (`{"success": bool, "output": {"mimetype": str, "data": str}}`). Optional `Authorization: Bearer <token>`.
  Registry of `--no-token` servers on Windows: `%USERPROFILE%\.marimo\servers\*.json` with keys
  `server_id, pid, host, port, base_url, started_at, version`; on POSIX `$XDG_STATE_HOME/marimo/servers`
  (default `~/.local/state/marimo/servers`). A session exists only while the notebook is open in a browser.
- Durable notebook edits: `import marimo._code_mode as cm; async with cm.get_context() as ctx: ...` with
  `ctx.create_cell(code, *, before, after, hide_code=True, disabled, column, name) -> cell_id`,
  `ctx.edit_cell(target, code=...)`, `ctx.delete_cell`, `ctx.move_cell`, `ctx.run_cell(target)`,
  `ctx.set_ui_value(element, value)`, `ctx.packages.add/remove`, `ctx.cells` (index/name, `.id .code .name
  .status .errors .output`), `ctx.graph`, `ctx.globals`. Top-level `await`/`async with` is supported in the
  scratchpad. New top-level bindings in the scratchpad are discarded; notebook globals are readable by name.
- Codex SDK: `Codex(CodexConfig(config_overrides=tuple[str,...], cwd=str, env=dict))` launches the bundled
  app-server; overrides are `codex -c key=value` (value parsed as TOML). `codex.thread_start(model=,
  model_provider=, base_instructions=, developer_instructions=, sandbox=Sandbox.workspace_write,
  approval_mode=ApprovalMode.auto_review, cwd=, ephemeral=)` only when `uses_codex_reviewer(provider, key_source, account_type)` is true: the built-in `openai` provider with a ChatGPT account, where `account_type` comes from `codex_account_type(codex)` → `codex.account()` (`GetAccountResponse.account.root.type`: "chatgpt" | "apiKey" | "amazonBedrock"; "none" when signed out, which is also what an `OPENAI_API_KEY` in the environment alone reports; `None` when the lookup failed, then the key source decides and "missing" keeps the reviewer). auto_review, not deny_all, because under deny_all/approval_policy "never" Codex rejects every MCP tool call with "MCP tool call requires approval, but approval policy is never", whatever the per-server/per-tool approval settings say (verified live on 2026-09-15). Every other case goes through the raw client, `codex._client.thread_start(reviewer_free_thread_params(kwargs))` / `thread_resume(id, ...)` with `ThreadStartParams(approval_policy="on-request", approvals_reviewer="user", <every other wrapper kwarg, sandbox mapped>)` (unknown kwarg names raise TypeError), wrapped in `openai_codex.api.Thread(client, id)`; a missing `_client` attribute is an AgentError naming the SDK pin. Reason: under auto_review Codex sends every escalation (commands and MCP tool calls) to its reviewer model `codex-auto-review`, which only the ChatGPT backend serves (api.openai.com answers model_not_found for an API key; a strict gateway answers 422), the SDK wrapper offers only deny_all/auto_review, and a thread-level `config={"approvals_reviewer": "user"}` is ignored because the explicit parameter wins (all verified live on 2026-09-16 with openai-codex 0.154.0). `HailerAgent.codex_reviewer` records the choice for the current thread. On such threads Codex asks the client to approve every MCP tool call with the server request `mcpServer/elicitation/request` (`serverName`, `_meta.codex_approval_kind == "mcp_tool_call"`); the SDK's default handler answers `{}` (= "user rejected MCP tool call") and neither `default_tools_approval_mode = "auto"` nor a granular policy stops the request, so `install_approval_handler(codex)` (called in `_ensure_codex`) wraps `codex._client._approval_handler` with `approval_handler(default)`, which answers `{"action": "accept", "content": {}}` for Hailer's own server only (plus `"_meta": {"persist": "session"}` when the request's `_meta.persist` offers it, so Codex asks once per tool per thread; two calls produced one request live, while `content.persist` or a top-level `persist` changed nothing) and delegates everything else to the SDK default (verified live 2026-09-16). `codex.thread_resume(thread_id, ...)`,
  `thread.turn(text) -> TurnHandle` with `.stream()` (notifications: `item/agentMessage/delta`,
  `item/started`, `item/completed`, `item/commandExecution/outputDelta`, `item/mcpToolCall/progress`,
  `turn/completed`), `.interrupt()`, `.run() -> TurnResult(final_response, items, usage, status, error)`.
  `SkillInput(name, path)` may be included in turn input. Errors: `openai_codex.errors.*`.
- Provider config is native Codex config: `model`, `model_provider`, `model_providers.<id>.{base_url,
  wire_api="responses", env_key, requires_openai_auth, name, http_headers, env_http_headers, query_params}`,
  plus Hailer's own `stream=true` (`ProviderConfig.stream`), which is never passed to Codex.
  Codex 0.154 only speaks the Responses API and refuses `wire_api = "chat"`. Hailer accepts `wire_api = "chat"`
  in `hailer.toml` anyway: `agent.HailerAgent._ensure_bridge` starts `wire.ChatBridge` (loopback HTTP server,
  one route per chat provider at `http://127.0.0.1:<port>/<id>`) before the app-server, and
  `build_config_overrides(config, user_cfg, bridge_urls)` hands such providers to Codex as
  `wire_api="responses"` at the bridge URL. The bridge translates `POST /<id>/responses` (Responses request →
  `chat_request_from_responses(body, stream=...)`) into `POST {base_url}/chat/completions` and the reply back
  into Responses SSE events. `chat_request_from_responses` returns `(chat_body, wire.ChatToolMap)`: Codex 0.154
  sends an MCP server's tools as ONE Responses tool `{type: "namespace", name: "mcp__hailer", description,
  tools: [{type: "function", name, description, parameters, strict}, ...]}` (verified live 2026-09-16), which the
  bridge flattens into ordinary function tools under their bare names (`<namespace>__<name>` when a name is
  already taken by a top-level tool, an earlier namespace or an earlier tool of the same namespace; the
  namespace description is not sent, a debug log says so). The map is handed to
  `ChatStreamTranslator(tools=...)`, which emits such calls as `function_call` items carrying `namespace` plus
  the bare `name` (Codex routes MCP calls by that pair; a prefixed name alone is not routed); a reply that names
  `<namespace>__<name>` for a tool advertised under its bare name is mapped the same way. Replayed
  `function_call` input items that carry `namespace` are renamed to the advertised name, as is a namespaced
  `tool_choice`; a `tool_choice` naming a clashing bare name without `namespace` resolves to the top-level tool.
  `chat_completion_upstreams(config)` gives the bridge one `wire.ChatUpstream(base_url,
  stream)` per provider: with `stream=True` (default) the request carries `stream: true` plus
  `stream_options.include_usage` and the chunks are relayed as they arrive; with `stream=False`
  (`stream = false` in `hailer.toml`, for gateways that reject or cannot deliver SSE) it carries `stream: false`,
  `Accept: application/json`, and the single `chat.completion` body is fed to the translator as one chunk, so
  Codex still receives the full event sequence (`ChatStreamTranslator`: `response.created`, `response.output_item.added`,
  `response.output_text.delta`, `response.reasoning_summary_text.delta`, `response.output_item.done` for
  `message` / `function_call` / `custom_tool_call` / `reasoning`, `response.completed` with `usage`,
  `response.failed`). It forwards Codex's request headers and query string upstream, so credentials stay on
  the provider config; `GET /<id>/models` is passed through; `/responses/compact` answers 404 with a body
  naming the bridge, which `map_exception` turns into a dedicated hint. Trust model: the bridge binds
  `127.0.0.1` only, stores no credential, forwards whatever headers its caller supplies, and any local process
  can use it to reach `base_url` (through Hailer's own proxy/CA environment) for the session's lifetime.
  `build_child_env` adds `127.0.0.1,localhost` to `NO_PROXY`/`no_proxy` whenever a chat provider is declared so
  Codex never sends bridge traffic to an `HTTP(S)_PROXY`. `HailerAgent.overrides` is a property derived from
  the current bridge state (no stale pre-bridge copy); a loopback bind failure raises `ProviderError`.
  Secrets reach Codex via the child-process env (`CodexConfig.env`).
- Trimming overrides Hailer always passes: `web_search="disabled"`, `features.web_search_request=false`,
  `features.multi_agent=false`, `features.multi_agent_v2=false`, `features.plugins=false`,
  `features.apps=false`, `features.codex_apps=false`, `features.connectors=false`,
  `features.image_generation=false`, `plugins."<id>".enabled=false` for every `[plugins.*]` id in the user's
  `~/.codex/config.toml`, and `mcp_servers.<name>.enabled=false` **only** for names present in that file's
  `[mcp_servers]` table (never for plugin-provided servers).
- Network overrides when `[web].allow_shell_network = true`: `features.network_proxy.enabled=true`,
  `features.network_proxy.domains={"<d>"="allow",...}`, `network.enabled=true`, `network.mode="limited"`,
  `network.domains={...}`, `sandbox_workspace_write.network_access=true`, always including `127.0.0.1` and
  `localhost` in the domain map. Otherwise pass nothing (default sandbox blocks outbound, allows loopback).
- Hailer's MCP server is registered as `mcp_servers.hailer.command=<sys.executable>`,
  `mcp_servers.hailer.args=["-m","hailer.mcp_server"]`, `mcp_servers.hailer.env={HAILER_CONFIG=...,
  HAILER_WORKSPACE=...}`, `mcp_servers.hailer.tool_timeout_sec=600`, `mcp_servers.hailer.startup_timeout_sec=60`, `mcp_servers.hailer.default_tools_approval_mode="auto"` (without it, MCP tools default to "prompt" approval, which the `never` approval policy auto-rejects and the model reports "tools require approval").
- Windows sandbox facts: shell commands cannot reach the internet by default; loopback works; the sandbox
  could not execute a Python interpreter outside the workspace. All Python execution goes through the MCP
  server → marimo kernel.

## `config.py`  (owner: wave 1 / A)

```python
CONFIG_FILENAMES = ("hailer.toml", ".config/hailer/hailer.toml")
DEFAULT_CONFIG_TEMPLATE: str            # commented hailer.toml written by `hailer init`
def find_workspace(start: Path | None = None) -> Path      # cwd or first parent containing a config file / pyproject.toml; else cwd
def find_config_path(workspace: Path, explicit: Path | None = None, env: Mapping[str, str] = os.environ) -> Path | None
def load_config(workspace: Path | None = None, config_path: Path | None = None, env: Mapping[str, str] = os.environ) -> HailerConfig
def validate(config: HailerConfig) -> list[str]            # human-readable problems, empty if fine
def write_default_config(path: Path, *, overwrite: bool = False) -> None
```
Rules: missing file → defaults (not an error); unparsable → `ConfigError` naming the file and line; unknown
keys are warnings returned by `validate`, not errors. Relative paths resolve against the workspace. Env
overrides: `HAILER_CONFIG`, `HAILER_WORKSPACE`, `HAILER_NOTEBOOK`, `HAILER_DATA_DIR`, `HAILER_MARIMO_URL`,
`HAILER_MARIMO_TOKEN` (value kept in memory only), `HAILER_MODEL`, `HAILER_MODEL_PROVIDER`,
`HAILER_LOG_LEVEL`, `HAILER_CODEX_HOME`, `HAILER_NOTEBOOKS_DIR`. Provider tables `[model_providers.<id>]` map
1:1 to `ProviderConfig`. `[web]` maps to `WebConfig` (`allowed_domains` list, `max_page_bytes`,
`allow_shell_network`). `[hailer]` keys: `notebook`, `notebooks_dir`, `data_dir`, `marimo_url`,
`context_dir`, `skills_dir`, `prompts_dir`, `max_tool_output_chars`, `max_context_bytes`; `[model]` keys:
`name`, `provider`, `reasoning_effort`. `notebooks_dir` (relative to the workspace) is the folder notebooks
are listed from, created in and opened from; it defaults to the configured notebook's folder and is always
filled in by `load_config` (`HailerConfig.notebooks_dir`; use `HailerConfig.notebooks_root`, which falls back
to `notebook.parent` for hand-built configs). `validate` checks: notebook exists, notebook is inside
`notebooks_dir` (fatal, names both keys), `notebooks_dir` exists (warning only), data_dir exists (warning
only), active provider declared (or `openai`), custom provider has `base_url` and `env_key`,
`wire_api in ("responses", "chat")` (`"chat"` requires a `base_url`), `stream = false` only with `wire_api = "chat"`,
domains are well-formed.

Errors added for the notebook feature (`errors.py`): `NotebookExistsError` (create: name taken) and
`NotebookPathError` (outside the notebooks folder, or not a marimo notebook).

## `log.py`  (owner: wave 1 / A)

```python
def setup_logging(level: str | None, *, verbose: bool = False, log_file: Path | None = None) -> logging.Logger  # returns logger "hailer"
class RedactingFilter(logging.Filter)   # masks bearer tokens and values of secret-looking env vars present in os.environ
def get_logger(name: str = "hailer") -> logging.Logger
```
Normal level WARNING to stderr via Rich-free plain handler; `--verbose` → DEBUG.

## `marimo_client.py`  (owner: wave 1 / B)

```python
CM_HELP_CODE = "import marimo._code_mode as cm; help(cm)"
DEFAULT_LAUNCH_DIR = "notebooks"
SERVER_TOKEN_HEADER = "Marimo-Server-Token"
def discover_servers() -> list[MarimoServer]         # registry files; drop entries whose /health does not answer
def find_server(config: HailerConfig) -> MarimoServer | None   # config.marimo_url first, then registry (single live server, else None)
class MarimoClient:
    def __init__(self, base_url: str, token: str | None = None, *, timeout: float = 10.0,
                 notebook: Path | None = None, workspace: Path | None = None, notebooks_dir: Path | None = None) -> None
                 # notebook: the one resolve_session()/execute() default to; notebooks_dir: named in launch hints (else notebook.parent)
    def health(self) -> bool
    def sessions(self) -> list[MarimoSession]
    def resolve_session(self, notebook: Path | None) -> MarimoSession   # match_session() below (exact path only); notebook=None → the single session; 0 → NoSessionError with hint to open the notebook URL; no match → NoSessionError listing the open sessions
    def execute(self, code: str, *, session_id: str | None = None, notebook: Path | None = None,
                on_stdout: Callable[[str], None] | None = None, on_stderr: Callable[[str], None] | None = None,
                timeout: float = 600.0) -> ExecResult   # SSE parse; missing `done` → MarimoExecutionError
    def notebook_url(self, notebook: Path | None = None) -> str
    def server_token(self) -> str                       # data-token of <marimo-server-token> in GET /; cached per instance; MarimoUnavailableError if absent
    def shutdown_session(self, session_id: str) -> None # POST /api/home/shutdown_session {"sessionId": id} + Marimo-Server-Token; 401 → MarimoUnavailableError ("server restarted; retry"), token dropped so the next call refetches it
def match_session(sessions: Sequence[MarimoSession], notebook: Path, workspace: Path | None = None) -> MarimoSession | None
    # the session whose path (or filename when marimo reports no path) is the same file as notebook, compared as
    # normalised absolute paths (case-insensitive on Windows); relative session paths resolve against workspace;
    # untitled sessions never match; NO bare-filename fallback (a/report.py never matches b/report.py); several
    # matches → the last one listed. Used by resolve_session, the CLI (/notebook list, close) and the test fakes.
def notebook_file_key(notebook: Path) -> str          # the ?file= key: absolute, normalised, forward slashes, symlinks/junctions NOT resolved (marimo compares the key against the folder it was started on the same way); works on folder and single-file servers
def open_notebook_url(server: MarimoServer, notebook: Path, workspace: Path | None = None, *, view: str = "app") -> str   # http://127.0.0.1:2718/?file=<quoted absolute posix path>&view-as=present (view="app", default: results only, Ctrl+. toggles the editor); view="edit" omits the parameter; workspace is accepted but unused
def launch_command(notebooks_dir: Path | None, workspace: Path | None = None, *, port: int | None = None) -> list[str]   # ["uv","run","marimo","edit",<folder relative to workspace>,"--no-token", ...]; None → "notebooks"
def notebook_launch_command(config: HailerConfig, *, port: int | None = None) -> list[str]  # launch_command(config.notebooks_root, config.workspace, port=port)
def marimo_server_command(notebooks_dir: Path, workspace: Path, port: int) -> list[str]   # [sys.executable,"-m","marimo","edit",<folder>,"--no-token","--headless","--port",N,"--skip-update-check"]
def launch_hint(notebooks_dir: Path | None, workspace: Path | None = None) -> str   # "Start everything in one go: uv run hailer notebook" first, then the launch_command
def wait_for_health(url: str, timeout: float = 60.0, *, interval: float = 0.5, should_stop=None) -> bool
def wait_for_session(client: MarimoClient, notebook: Path | None, timeout: float = 90.0, *, interval: float = 0.5) -> MarimoSession | None
```
marimo is always started on the notebooks **folder** (never on a single file) so that every notebook,
existing or created later, opens on the same server. Connection refused/timeout → `MarimoUnavailableError`
with the exact `uv run marimo edit <folder> --no-token` command in the hint. HTTP 401/403 →
`MarimoUnavailableError` mentioning `HAILER_MARIMO_TOKEN`.

## `notebooks.py`  (owner: notebook feature / foundation)

Notebook lifecycle shared by the CLI and the MCP server. Standard library only; never imports marimo, typer
or rich. Nothing here talks to the marimo server (that is `marimo_client`); this module owns files and state.

```python
STATE_FILENAME = "notebook.json"           # <workspace>/.hailer/notebook.json: {"active": "<workspace-relative posix path>", "recent": [...]}
RECENT_LIMIT = 10
STARTER_TEMPLATE: str; EMPTY_TEMPLATE: str  # __HAILER_VERSION__ / __HAILER_TITLE__ placeholders, filled by render_template
TEMPLATE_KINDS = ("starter", "empty")
@dataclass(frozen=True) class NotebookInfo: path: Path (absolute); name: str (stem); modified: float; size: int
def state_path(workspace: Path) -> Path
def load_active_notebook(config: HailerConfig) -> Path
    # ONLY the state file (never the environment; see the ground rule): its "active" entry when it resolves inside
    # notebooks_root and exists; otherwise (missing/corrupt file, deleted or foreign path) config.notebook.
def save_active_notebook(config: HailerConfig, notebook: Path) -> None
    # atomic (tmp + replace); stores the workspace-relative posix path; "recent" most-recent-first, unique, max 10; creates .hailer/
def load_recent(config: HailerConfig) -> list[Path]         # existing notebooks only, absolute, most recent first
def notebook_display_name(config: HailerConfig, path: Path) -> str   # workspace-relative posix ("notebooks/q2_churn.py"), absolute posix when outside
def is_marimo_notebook(path: Path) -> bool                 # marimo's rule: .py whose first 1 MB contains "import marimo" and "marimo.App"; errors → False
def list_notebooks(config: HailerConfig) -> list[NotebookInfo]   # recursive under notebooks_root, skips folders starting with "." or "__", sorted by relative path (case-insensitive)
def slugify(name: str) -> str                              # "Q2 Churn (draft).py" → "q2_churn_draft"; must start with a letter (else "nb_" prefix); ValueError when empty
def resolve_notebook(config: HailerConfig, ref: str) -> Path
    # bare name ("q2 churn"), filename, workspace-relative, notebooks_root-relative or absolute; tried as given
    # (workspace then notebooks_root), then "<slug>.py" under notebooks_root, then a case-insensitive unique
    # NotebookInfo.name match. Outside notebooks_root or not a notebook → NotebookPathError; nothing → NotebookNotFoundError
    # (hint lists up to 10 names); several name matches → NotebookNotFoundError listing them.
def marimo_version() -> str                                # importlib.metadata.version("marimo"), fallback "0.24.2"
def render_template(kind: Literal["starter", "empty"], *, title: str, version: str | None = None) -> str
    # starter = the cells of notebooks/analysis.py with the title as heading and WORKSPACE found by walking up to
    # pyproject.toml / hailer.toml; empty = marimo's own empty notebook. Title is made safe for the f-string markdown.
def create_notebook(config: HailerConfig, name: str, *, kind: Literal["starter", "empty"] = "starter") -> Path
    # <notebooks_root>/<slug>.py (folder created); NotebookExistsError when it exists; NotebookPathError when the
    # name has no letters/digits; returns the absolute path. Never touches the state file.
```

## `browser.py`  (owner: notebook feature)

```python
def open_url(url: str) -> bool   # launch the default browser; True when a launcher was started; never raises or prints
```
Shared by the CLI (`_open_browser`) and the MCP server (`HailerTools.open_url` default). Windows: `os.startfile`
first, because `webbrowser` honours a `BROWSER` variable that Git Bash / WSL profiles sometimes set to a non-GUI
command (observed: `BROWSER=true` made `/notebook new` wait 30 s for a tab that never opened); `webbrowser` is the
fallback. POSIX: `webbrowser` picks the launcher, but generic launchers (`xdg-open`, `open`) are spawned with their
stdio discarded, because the MCP server's stdout is the protocol. Module-level `is_windows`, `startfile`,
`webbrowser_open`, `webbrowser_get`, `popen` are injection points for tests only.

## `periods.py`  (owner: wave 1 / B)

```python
PERIOD_RE: re.Pattern            # ^(?P<yy>\d{2})-(?P<mm>\d{2})\s+(?P<stem>.+)$ on the file stem
def parse_period(name: str | Path) -> Period | None
def parse_period_file(path: Path) -> PeriodFile | None
def scan_period_files(data_dir: Path, name: str | None = None, *, suffix: str = ".parquet") -> list[PeriodFile]  # sorted by period
def load_periods(files: Sequence[PeriodFile], *, columns: Sequence[str] | None = None) -> pl.DataFrame  # adds "period" (str YYYY-MM) column, concat how="diagonal_relaxed"
def scan_periods(files: Sequence[PeriodFile]) -> pl.LazyFrame        # lazy variant
def duckdb_periods_view(con: duckdb.DuckDBPyConnection, files: Sequence[PeriodFile], view_name: str = "periods") -> str  # read_parquet([...], union_by_name=true, filename=true) + derived period column; returns view name
def describe_periods(files: Sequence[PeriodFile]) -> str   # compact text: count, first/last period, per-file schema diff summary
```
Century rule: `YY` → `2000 + YY`. A file that fails to read raises `MalformedParquetError` naming the file
(wrap `polars.exceptions.ComputeError`/`OSError`).

## `web.py`  (owner: wave 1 / C)

```python
def host_allowed(host: str, allowed: Sequence[str]) -> bool   # exact; "*.d" subdomains only; "**.d" apex+subdomains; case-insensitive; strips port
def fetch_page(url: str, config: HailerConfig, *, timeout: float = 20.0) -> str   # http/https only; WebAccessDenied otherwise; HTML→text; capped at max_page_bytes then max_tool_output_chars; no redirects to disallowed hosts
def html_to_text(html: str) -> str
```

## `context.py`  (owner: wave 1 / C)

```python
def load_context(config: HailerConfig) -> ContextBundle   # context/*.md sorted, concatenated with "## <filename>" headings, capped at max_context_bytes with a warning; skills index from */SKILL.md frontmatter (name, description); prompts/*.md map
def parse_skill_frontmatter(text: str) -> dict[str, str]
def read_skill(config: HailerConfig, name: str) -> str        # SKILL.md body + list of files under the skill dir
def read_skill_file(config: HailerConfig, name: str, relative: str) -> str   # path-traversal safe
def render_prompt(config: HailerConfig, name: str, args: str) -> str   # replaces {{args}}
def looks_like_secret(text: str) -> bool                       # sk-..., "api_key =", "Bearer ", long hex/base64 tokens
```

## `mcp_server.py`  (owner: wave 1 / C)

stdio MCP server (`mcp` package; inspect the installed version's API before writing). Reads
`HAILER_CONFIG`/`HAILER_WORKSPACE` from env and calls `load_config`. The process outlives many turns, so the
**active notebook is re-read from `.hailer/notebook.json` on every call** (`HailerTools._active_config()` returns
the config with `notebook` replaced by `notebooks.load_active_notebook`); nothing about it is cached. Every kernel
tool acts on the active notebook. Tools (all return plain text, truncated to `config.max_tool_output_chars` with
head/tail and a `[... truncated N chars ...]` marker):

| tool | args | behaviour |
|---|---|---|
| `marimo_execute` | `code: str` | run in the scratchpad against the active notebook's session; returns stdout/output/stderr (rich mimetypes such as text/html or application/json are replaced by a short placeholder, so HTML never reaches the model); errors from `MarimoUnavailableError`/`NoSessionError` become the hint text, not exceptions |
| `marimo_status` | – | server url, version; `active notebook: <workspace-relative> -> session <id> (ready)` or `... has NO session. Ask the user to open <url>`; every session with its notebook and a marker: `(active notebook)`, `(another notebook in the notebooks folder; notebook_open switches to it)`, `(outside the notebooks folder)`, `(unsaved)`; when marimo is down (factory raises, or a configured `marimo_url` answers with `MarimoUnavailableError` on the first request) the text is `marimo: not running` / `active notebook: <name>` / error / hint, so the active notebook is always named |
| `notebook_cells` | `pattern: str = ""` | runs a `cm` snippet listing the active notebook's cells (id, name, first line, status, errors); optional substring filter |
| `notebook_list` | – | `notebooks.list_notebooks`: one line per notebook `  - <name>  modified YYYY-MM-DD HH:MM  <size>  [active] [open]` (`[open]` = has a session; when marimo is unreachable the files are still listed and a trailing line says sessions are unknown); empty folder → `No notebooks in <folder> yet. Create one with notebook_create(name).` |
| `notebook_create` | `name: str, template: str = "starter"` | `notebooks.create_notebook` (+ `save_active_notebook`), then bring-up (below); text: `Created <name> from the <kind> template; it is now the active notebook.` + session line + a reminder of what the template defines; unknown template → `ERROR: unknown template ...`; existing name / unusable name → the `NotebookExistsError` / `NotebookPathError` text |
| `notebook_open` | `notebook: str` | `notebooks.resolve_notebook` (+ `save_active_notebook`), bring-up, then when a session exists the cell listing (`Cells (id, name, status, errors, first line):`); text starts `<name> is now the active notebook.`; unknown / outside-folder references → the `NotebookNotFoundError` / `NotebookPathError` text and the active notebook is unchanged |
| `notebook_close` | `notebook: str = ""` | resolve (empty = active), `client.resolve_session`, `client.shutdown_session(id)`; says the kernel is freed and, for the active notebook, that it stays active but must be reopened; no session → `<name> is not open (no kernel session), so there is nothing to close.`; marimo down → `marimo is not running, so <name> has no kernel session to close.` + error + hint |
| `list_periods` | `name: str = ""` | `describe_periods(scan_period_files(config.data_dir, name or None))` |
| `load_skill` | `name: str` | `read_skill` |
| `read_skill_file` | `name: str, path: str` | `read_skill_file` |
| `fetch_page` | `url: str` | `web.fetch_page`; denial text lists allowed domains |

Bring-up (`HailerTools._bring_up(notebook)`): if the notebook already has a session it is reused and the browser is
not opened; otherwise `open_notebook_url(server, notebook)` is opened through `HailerTools.open_url` (default:
`hailer.browser.open_url`, i.e. `os.startfile` first on Windows and stdio-silenced launchers elsewhere, because
stdout is the MCP transport) and `wait_for_session` polls for up to `HailerTools.session_wait_sec` (default 30 s).
Outcomes are reported as text (`kernel session ready (<id>)`, `no kernel session appeared within N s` + URL,
`Could not open a browser` + URL, `marimo is not running (or not reachable)` + error + hint); the browser never
raises. Every `HailerError` raised while resolving or waiting for the session (a configured `marimo_url` is not
health-checked up front, so a dead server first shows up there) is folded into that last outcome: the notebook
was already created / made active by then, so `notebook_create` / `notebook_open` still begin with `Created ...` /
`... is now the active notebook.` and never read as a failed call. Both hooks are constructor keywords of
`HailerTools` and `build_server` so tests never launch a browser.

`def main() -> None` runs the server; `def build_server(config, *, client_factory=None, open_url=None,
session_wait_sec=30.0)` returns it (for tests; `server.hailer_tools` is the `HailerTools` instance).

## `secrets.py`  (owner: wave 2 / D)

```python
KEYRING_SERVICE = "hailer"
def keyring_username(provider_id: str, env_key: str) -> str        # f"{provider_id}:{env_key}"
def resolve_provider_key(provider: ProviderConfig, env: Mapping[str, str] = os.environ) -> tuple[str | None, str]  # (value, "env"|"keyring"|"missing"); keyring failures → "missing" (log at debug)
def store_provider_key(provider: ProviderConfig, value: str) -> None   # CredentialsError (hint: set the env var) on empty value, missing env_key, or unusable keyring backend
def delete_provider_key(provider: ProviderConfig) -> bool          # False when nothing was stored; CredentialsError when the keyring backend is unusable
```

## `agent.py`  (owner: wave 2 / D)

```python
def read_user_codex_config(codex_home: Path | None = None) -> dict     # tomllib of ~/.codex/config.toml or {}
def build_config_overrides(config: HailerConfig, user_codex_config: dict) -> tuple[str, ...]   # pure; provider + trimming + mcp server + network overrides; values TOML-quoted
def build_child_env(config: HailerConfig, key: str | None, extra_keys: Mapping[str, str] | None = None) -> dict[str, str]   # env for the codex process: provider env_key → key (if any), other declared providers' keys, HAILER_CONFIG/HAILER_WORKSPACE, CODEX_HOME if config.codex_home, HAILER_MARIMO_TOKEN if set (process environment only, never in the -c mcp_servers.hailer.env map; Codex-spawned MCP servers inherit it)
def system_prompt(config: HailerConfig, bundle: ContextBundle) -> str  # prompts/system.md (importlib.resources) + project context + skills index + web allowlist statement + workspace section (workspace, "Notebooks folder: <notebooks_root>", data dir, marimo URL). Invariant: the text does NOT depend on config.notebook, so session.prompt_hash is stable across notebook switches and the CLI's "prompt changed, use /new" warning never fires because of one
class HailerAgent:
    def __init__(self, config: HailerConfig, bundle: ContextBundle, *, codex_factory: Callable[..., Any] | None = None) -> None
    def start(self, *, resume_thread_id: str | None = None) -> str     # returns thread id; resume failure → new thread (log)
    def run_turn(self, text: str, *, on_event: Callable[[AgentEvent], None] | None = None, skill: SkillInfo | None = None, preamble: str | None = None) -> TurnSummary   # stream consumed on a worker thread with a timed queue wait (Ctrl+C works on Windows); on KeyboardInterrupt it interrupts the turn itself, waits up to 10 s for turn/completed, then re-raises KeyboardInterrupt. Input composition: plain string when there is no skill and no preamble; otherwise a list ordered skill → preamble → text ([SkillInput], [TextInput(preamble)], TextInput(text)); a blank preamble counts as none. The CLI uses preamble for its `[Hailer] ...` notices (see cli.py)
    def interrupt(self) -> None      # idempotent; no-op without an active turn; afterwards remaining notifications are collected but not surfaced via on_event
    def new_thread(self) -> str
    def set_model(self, name: str, provider: str | None = None) -> bool   # applies to the next turn; a provider change while running starts a new thread and returns True (the CLI must not start another)
    def close(self) -> None
    thread_id: str | None
```
`codex_factory` defaults to `openai_codex.Codex`; tests inject a fake exposing `thread_start`, `thread_resume`,
`close`, and threads with `turn()` returning a handle with `stream()`/`interrupt()`. Map SDK exceptions to
`AgentError`/`ProviderError`/`CredentialsError` with hints (401 → key rejected; connection refused → base_url;
404/405 or "unknown endpoint" → ProviderError "did not accept the request" whose hint names the protocol the provider uses (`POST .../responses` for `wire_api = "responses"`, `POST .../chat/completions` for `"chat"`) and suggests switching `wire_api`; 403 → ProviderError "refused access (HTTP 403)"; any other 4xx (`unexpected status 4xx ...`) or a validation-style body ("unsupported parameter", "unrecognized request argument", "should match pattern", "schema", ...) → ProviderError "rejected the request" whose hint says what Hailer sent and where to change it (the `stream = false` example only for `wire_api = "chat"`, no `[model_providers]` advice for the built-in openai provider); the wording signals only count when the text looks like a gateway reply (a parsed status, `{"error"`/`{"detail"`, `invalid_request_error`), so the SDK's own "validation error" or Codex's "unsupported operation" fall through to AgentError; 429/5xx, including Codex's `exceeded retry limit, last status: NNN`, → ProviderError "unavailable (HTTP n)"; "model ... not found/does not exist" → ProviderError naming the unknown model only when the text names the configured model, quotes one (`The model 'x' does not exist` → "Unknown model 'x'", with a note when x differs from `[model].name`) or carries `model_not_found`/`"param": "model"`, checked before the endpoint heuristics; "tool ... timed out" → AgentError, never an endpoint failure). Every endpoint-related ProviderError ends its hint with `The endpoint said: <the gateway's own text>` (whitespace collapsed, trimmed to 500 chars, passed through `hailer.log.redact`). Codex's `, url: http://127.0.0.1:<port>/<provider>/responses` suffix is the chat bridge's address and is stripped before matching and quoting (a real endpoint URL is kept); the bridge itself logs every upstream non-2xx reply (URL, status, first 500 bytes) at debug. The raw turn error (`TurnError.message` plus `additional_details` when it adds something) is logged at debug before mapping, and the `error` notification's status line previews `payload.error.message` rather than the object repr. `map_exception(exc, config, *, model=None, provider=None)`.

## `session.py`  (owner: wave 2 / E)

```python
COMMANDS: dict[str, str]     # name -> one-line help, for /help
def parse_command(line: str) -> Command | None      # "/model gpt-5.5" -> Command("model", "gpt-5.5"); "/exit" ; non-slash -> None; unknown slash -> Command(name, args) (CLI reports unknown)
def load_session(workspace: Path) -> SessionState   # .hailer/session.json, tolerant of missing/corrupt
def save_session(workspace: Path, state: SessionState) -> None
def help_text() -> str
```

## `cli.py`  (owner: wave 2 / E)

Typer app; `def main() -> None` is the console entry. Commands: default (no subcommand) → chat; `notebook`
(start marimo on `config.notebooks_root` in the background — `mkdir` it first, `marimo_server_command`, log in
`.hailer/marimo.log` — or reuse a live server; open the active notebook; chat; stop marimo on exit unless
`--keep-marimo`; `--foreground` runs marimo attached without the chat; `_reusable_server` accepts the configured
URL or a live server that has a session for the active notebook or for any notebook under the folder); `exec`
(`-c CODE` | `-` stdin | file; prints result; exit 1 on failure); `status`; `doctor`; `login <provider>`;
`logout <provider>`; `init` (write default config + `.config/hailer` skeleton). Global options `--verbose/-v`,
`--config`, `--workspace`. Startup: `_load_config` (load config, then `config.notebook` := the active notebook
from the state file, or, when `HAILER_NOTEBOOK` is set, that notebook written to the state file) → setup logging →
preflight (config, notebook, credentials, marimo server, session) → Rich panel (`Notebook:` active,
`Notebooks:` folder) → agent start (resume from session file) → REPL. REPL: `You > ` prompt; Ctrl+C during a
turn interrupts it, at the prompt asks once then exits; EOF exits; slash commands
`/help /status /new /exit /quit /model /notebook /clear /context /skill /prompt /reload`.

Notebook switching (`ChatLoop`): `/notebook` shows the active notebook, folder, `Recent:` (from
`notebooks.load_recent`, excluding the active one), marimo state, URL, launch command and `NOTEBOOK_USAGE`;
`/notebook list` marks `active` / `open` (sessions matched with `marimo_client.match_session`); `/notebook new
<name> [--empty]` = `notebooks.create_notebook` then switch; `/notebook open <ref>` = `notebooks.resolve_notebook`
then switch (the active one only re-ensures its session); `/notebook close [ref]` = `client.shutdown_session`.
A switch (`_switch_notebook`) writes the state file, replaces `self.config.notebook`, queues
`_pending_preamble` **before** waiting for the tab, then `_ensure_session` (opens the URL through
`hailer.browser` when there is no session and waits `SWITCH_SESSION_TIMEOUT_SEC` = 30 s), then refines the
notice; Ctrl+C or a marimo error during the wait leaves the notice queued. Exact notice:
`[Hailer] The active notebook is now <workspace-relative> (<detail>). Call notebook_cells before editing.` with
`<detail>` = `created from the starter template` | `created from the empty template` | `reopened`, followed by
`, N cell(s)` when a session exists or `, not open in a browser yet` when none. It is passed once as
`run_turn(..., preamble=...)` with the next message and cleared (also on Ctrl+C; kept if the turn raised before
starting). `_sync_active_notebook()` runs in the `finally` of every turn (finished, failed, interrupted) and,
without opening a browser, at the start of `/notebook` and `/status`: on a change it updates `self.config`,
prints `Active notebook is now <name>.` and (turn end only) opens the URL once when the notebook has no session;
no preamble is queued because the model made the switch itself. `COMMANDS["notebook"]` =
`Show or switch the active notebook. Usage: /notebook [list | new <name> [--empty] | open <name> | close [name]]`.
Progress: a single Rich status line updated from `AgentEvent`s (command names, tool names, streaming text
optional); final answer printed under `Hailer >`. Errors: print message + hint; exit code 1; traceback only
with `--verbose`.

## `prompts/system.md`  (owner: wave 2 / E, reviewed by D)

The agent persona and rules from the spec (see PLAN.md §"Agent system instructions") plus the canonical `cm`
patterns, the scratchpad semantics, the tool list, the "compact outputs only" rule, and the web allowlist
statement. Loaded with `importlib.resources.files("hailer").joinpath("prompts/system.md")`.

## Tests (each owner writes their own; `uv run pytest` must pass offline)

- `test_config.py`, `test_log.py` — A
- `test_marimo_client.py` (fake SSE server on 127.0.0.1), `test_periods.py` (tiny parquet files in tmp_path) — B
- `test_web.py`, `test_context.py`, `test_mcp_server.py` (call tool functions directly with a fake client) — C
- `test_secrets.py`, `test_agent.py` (fake Codex; overrides content; env injection; error mapping) — D
- `test_session.py`, `test_cli.py` (Typer `CliRunner`; slash commands; preflight messages with fakes) — E
