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

## Runtime facts (verified; see PLAN.md §1, §1a–1c)

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
  approval_mode=ApprovalMode.auto_review, cwd=, ephemeral=)` (auto_review, not deny_all: under deny_all/approval_policy "never" Codex rejects every MCP tool call with "MCP tool call requires approval, but approval policy is never", whatever the per-server/per-tool approval settings say; verified live on 2026-09-15), `codex.thread_resume(thread_id, ...)`,
  `thread.turn(text) -> TurnHandle` with `.stream()` (notifications: `item/agentMessage/delta`,
  `item/started`, `item/completed`, `item/commandExecution/outputDelta`, `item/mcpToolCall/progress`,
  `turn/completed`), `.interrupt()`, `.run() -> TurnResult(final_response, items, usage, status, error)`.
  `SkillInput(name, path)` may be included in turn input. Errors: `openai_codex.errors.*`.
- Provider config is native Codex config: `model`, `model_provider`, `model_providers.<id>.{base_url,
  wire_api="responses", env_key, requires_openai_auth, name, http_headers, env_http_headers, query_params}`.
  Codex only speaks the Responses API. Secrets reach Codex via the child-process env (`CodexConfig.env`).
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
`HAILER_LOG_LEVEL`, `HAILER_CODEX_HOME`. Provider tables `[model_providers.<id>]` map 1:1 to
`ProviderConfig`. `[web]` maps to `WebConfig` (`allowed_domains` list, `max_page_bytes`,
`allow_shell_network`). `[hailer]` keys: `notebook`, `data_dir`, `marimo_url`, `context_dir`,
`skills_dir`, `prompts_dir`, `max_tool_output_chars`, `max_context_bytes`; `[model]` keys: `name`,
`provider`, `reasoning_effort`. `validate` checks: notebook exists, data_dir exists (warning only), active
provider declared (or `openai`), custom provider has `base_url` and `env_key`, `wire_api == "responses"`,
domains are well-formed.

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
def discover_servers() -> list[MarimoServer]         # registry files; drop entries whose /health does not answer
def find_server(config: HailerConfig) -> MarimoServer | None   # config.marimo_url first, then registry (single live server, else None)
class MarimoClient:
    def __init__(self, base_url: str, token: str | None = None, *, timeout: float = 10.0) -> None
    def health(self) -> bool
    def sessions(self) -> list[MarimoSession]
    def resolve_session(self, notebook: Path | None) -> MarimoSession   # match by path/filename (normalised, case-insensitive on Windows); 0 → NoSessionError with hint to open the notebook URL; >1 without match → NoSessionError listing them
    def execute(self, code: str, *, session_id: str | None = None, notebook: Path | None = None,
                on_stdout: Callable[[str], None] | None = None, on_stderr: Callable[[str], None] | None = None,
                timeout: float = 600.0) -> ExecResult   # SSE parse; missing `done` → MarimoExecutionError
def open_notebook_url(server: MarimoServer, notebook: Path, workspace: Path, *, view: str = "app") -> str   # URL the user should open; view="app" (default) adds &view-as=present so marimo opens in app view (results only; Ctrl+. toggles the editor): http://127.0.0.1:2718/?file=notebooks/analysis.py&view-as=present; view="edit" omits it
def notebook_launch_command(config: HailerConfig, *, port: int | None = None) -> list[str]  # ["uv","run","marimo","edit",<notebook>,"--no-token", ...]
```
Connection refused/timeout → `MarimoUnavailableError` with the exact `uv run marimo edit ... --no-token`
command in the hint. HTTP 401/403 → `MarimoUnavailableError` mentioning `HAILER_MARIMO_TOKEN`.

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
`HAILER_CONFIG`/`HAILER_WORKSPACE` from env and calls `load_config`. Tools (all return plain text, truncated to
`config.max_tool_output_chars` with head/tail and a `[... truncated N chars ...]` marker):

| tool | args | behaviour |
|---|---|---|
| `marimo_execute` | `code: str` | run in the scratchpad against the configured notebook's session; returns stdout/output/stderr (rich mimetypes such as text/html or application/json are replaced by a short placeholder, so HTML never reaches the model); errors from `MarimoUnavailableError`/`NoSessionError` become the hint text, not exceptions |
| `marimo_status` | – | server url, version, sessions, whether the configured notebook has a session, URL to open if not |
| `notebook_cells` | `pattern: str = ""` | runs a `cm` snippet listing cells (id, name, first line, status, errors); optional substring filter |
| `list_periods` | `name: str = ""` | `describe_periods(scan_period_files(config.data_dir, name or None))` |
| `load_skill` | `name: str` | `read_skill` |
| `read_skill_file` | `name: str, path: str` | `read_skill_file` |
| `fetch_page` | `url: str` | `web.fetch_page`; denial text lists allowed domains |

`def main() -> None` runs the server; `def build_server(config: HailerConfig)` returns it (for tests).

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
def system_prompt(config: HailerConfig, bundle: ContextBundle) -> str  # prompts/system.md (importlib.resources) + project context + skills index + web allowlist statement
class HailerAgent:
    def __init__(self, config: HailerConfig, bundle: ContextBundle, *, codex_factory: Callable[..., Any] | None = None) -> None
    def start(self, *, resume_thread_id: str | None = None) -> str     # returns thread id; resume failure → new thread (log)
    def run_turn(self, text: str, *, on_event: Callable[[AgentEvent], None] | None = None, skill: SkillInfo | None = None) -> TurnSummary   # stream consumed on a worker thread with a timed queue wait (Ctrl+C works on Windows); on KeyboardInterrupt it interrupts the turn itself, waits up to 10 s for turn/completed, then re-raises KeyboardInterrupt
    def interrupt(self) -> None      # idempotent; no-op without an active turn; afterwards remaining notifications are collected but not surfaced via on_event
    def new_thread(self) -> str
    def set_model(self, name: str, provider: str | None = None) -> bool   # applies to the next turn; a provider change while running starts a new thread and returns True (the CLI must not start another)
    def close(self) -> None
    thread_id: str | None
```
`codex_factory` defaults to `openai_codex.Codex`; tests inject a fake exposing `thread_start`, `thread_resume`,
`close`, and threads with `turn()` returning a handle with `stream()`/`interrupt()`. Map SDK exceptions to
`AgentError`/`ProviderError`/`CredentialsError` with hints (401 → key rejected; connection refused → base_url;
404 on `/responses` or schema error → "endpoint must implement the Responses API; run a translating proxy"; "model ... not found/does not exist" → ProviderError naming the unknown model, checked before the endpoint heuristics; "tool ... timed out" → AgentError, never an endpoint failure). `map_exception(exc, config, *, model=None)`.

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
(launch `marimo edit <notebook> --no-token [--port]` in a new console on Windows, foreground elsewhere, and
print the URL); `exec` (`-c CODE` | `-` stdin | file; prints result; exit 1 on failure); `status`; `doctor`;
`login <provider>`; `logout <provider>`; `init` (write default config + `.config/hailer` skeleton). Global
options `--verbose/-v`, `--config`, `--workspace`. Startup: load config → setup logging → preflight (config,
notebook, credentials, marimo server, session) → Rich panel → agent start (resume from session file) →
REPL. REPL: `You > ` prompt; Ctrl+C during a turn interrupts it, at the prompt asks once then exits; EOF exits;
slash commands `/help /status /new /exit /quit /model /notebook /clear /context /skill /prompt /reload`.
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
