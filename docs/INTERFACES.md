# Hailer module contracts

This is the contract every module is built against. `PLAN.md` explains *why* (for the agent:
`docs/LANGCHAIN_MIGRATION.md`); this file says *what* each module exposes so work can proceed in parallel.
Shared types live in `src/hailer/models.py`, errors in `src/hailer/errors.py`. Do not change those two files
without agreement (report a needed change instead).

## Ground rules

- **Windows first.** No bash, curl, jq, or POSIX-only assumptions. Use `pathlib`, `subprocess` with argument
  lists, `os.name`/`sys.platform` checks where needed. Paths in user output use the OS separator.
- **Standard library where practical.** HTTP to marimo and the web via `urllib.request`/`http.client`, TOML
  via `tomllib`, SSE parsing by hand, HTML-to-text via `html.parser`. Third-party allowed: `polars`, `duckdb`,
  `typer`, `rich`, `keyring`, `langchain`, `langchain-openai`, `langgraph-checkpoint-sqlite` (and what they
  install: `langchain_core`, `langgraph`, `openai`, `aiosqlite`; imported only inside functions of `agent.py`
  and `tools.py`, so `hailer --help` stays fast), `marimo` (only in the notebook / `hailer exec` snippets,
  never imported by the CLI process except to locate the executable).
- **Tests are offline.** No API key, no marimo server, no internet. Use fakes: a local `http.server` for the
  marimo protocol, a scripted chat model and the loopback `tests/fake_gateway.py` for the agent, `tmp_path` for
  files, an in-memory keyring backend (`keyring.backends.fail` / a tiny custom backend) for secrets.
- **Secrets never appear** in logs, prompts, tool results, exception messages, or config files. The provider key
  is resolved in-process (`secrets.resolve_provider_key`) and handed to the HTTP client; Hailer never writes it
  to a file or puts it into an environment. The logging redaction filter masks `Authorization` headers, bearer
  tokens, bare `sk-...` keys and any value whose env var name ends in `_KEY`, `_TOKEN`, `_SECRET`, `_PASSWORD`.
- **Errors are actionable.** Raise `HailerError` subclasses with a `hint`. The CLI is the only place that
  prints; library modules never print.
- **Ownership.** Each module has one owner during a wave. Only edit files you own; if you need something from
  another module that is not in this contract, write against the contract and note the gap in your report.
- **Dependencies** are already declared in `pyproject.toml`; do not add any. Report if something is missing.
- **The active notebook is shared state, not config.** `<workspace>/.hailer/notebook.json` (next to
  `session.json`) records which notebook the chat is working in. It has two writers in one process: the CLI
  between turns (slash commands) and the agent's tools during a turn. Every reader re-reads it before use
  through `hailer.notebooks.load_active_notebook` and never caches it. `load_active_notebook` reads **only the
  file**: `HAILER_NOTEBOOK` stays set for the whole session, so an env-wins rule would make the tools ignore
  every switch. Instead the CLI's `_load_config` writes an explicit `HAILER_NOTEBOOK` to the file at startup
  (`save_active_notebook`), so the CLI, the tools and later sessions agree. Residual race: a tool call still
  running after Ctrl+C (it finishes on its daemon thread) may write the file later; the CLI re-reads it after
  every turn (finished, failed or interrupted) and before `/notebook` and `/status`.

## Runtime facts (verified; marimo: PLAN.md §1, LangChain: `docs/LANGCHAIN_MIGRATION.md` §3 and `spikes/langchain/`)

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
- LangChain (checked against langchain 1.4.1, langchain-core 1.6.3, langchain-openai 1.6.2, langgraph 1.2.11,
  langgraph-checkpoint-sqlite 3.1.1, openai 3.15.0 on Windows 11 / Python 3.12):
  - LangGraph's synchronous `stream()` blocks in an untimed `concurrent.futures.wait`; on Windows Ctrl+C
    surfaced 5 to 9 s late. Under `asyncio.Runner().run(graph.astream(...))` it lands within 0.01 s and the
    HTTP request is aborted. `astream` needs `AsyncSqliteSaver`; the sync `SqliteSaver` raises
    `NotImplementedError`.
  - langchain-openai caches its async HTTP client per process, bound to the first event loop: a second agent
    (or a restart of one) fails with "Event loop is closed" unless it passes its own `http_async_client`.
  - With `use_responses_api` unset, model names such as `gpt-5-codex` / `gpt-5.5-pro` are routed to
    `/responses` even on a custom `base_url`; with `base_url` unset, a `LANGSMITH_GATEWAY` env var redirects
    requests. Hailer always passes both explicitly.
  - `stream_usage=False` stops `stream_options` being sent; `parallel_tool_calls` is only sent if passed to
    `bind_tools` (Hailer never does); `disable_streaming=True` sends `stream: false` and the agent's event
    stream still works; `create_agent` sends one system and one user message, and binds the tools when the
    model is first called, not at construction.
  - langchain-openai's default TCP keep-alive transport disables httpx's proxy auto-detection when a system
    proxy is present; `http_socket_options=()` turns it off. `openai` 3.x uses `httpx2`, which uses the OS
    trust store by default.
  - `SummarizationMiddleware` never fires with its default `trigger=None`, and `("fraction", f)` raises
    `ValueError` for a model name without a profile (every custom deployment name), so Hailer uses
    `("tokens", N)`.
  - Ctrl+C while a tool runs leaves an `AIMessage` with `tool_calls` and no `ToolMessage` in the checkpoint;
    endpoints answer 400 to that. LangChain ships no fix (deepagents does); Hailer patches it in
    `HailerAgent._settle_history` with `aupdate_state(..., as_node="tools")`.
  - Endpoint errors are `openai.APIStatusError` with `.status_code` and `.body`; a 404 surfaces as a
    langchain-openai class named `OpenAIModelNotFoundError`, so map on the status code, not the class name.
    A 4xx is not retried.
  - In langchain-core 1.6 `message.text` is a `str` that is also callable, and calling it emits a
    `LangChainDeprecationWarning`, which a live run printed into the user's terminal. `agent._message_text`
    tests `isinstance(text, str)` first. The pytest config ignores `DeprecationWarning`, so
    `test_agent.py::test_a_turn_raises_no_warnings_into_the_users_terminal` runs a turn under
    `warnings.simplefilter("error")`.
  - Simulating Ctrl+C: `_thread.interrupt_main()` works on Windows, but on Linux it only sets CPython's
    pending-signal flag and does not interrupt the event loop's `epoll` wait, so the handler ran when the
    wait ended (30 s late in CI). The tests send a real SIGINT to the main thread there
    (`signal.pthread_kill`), which is what a terminal sends.
- Verified live on 2026-09-18 (real `hailer` CLI, headless marimo 0.24.2 server, kernel session opened through
  headless Chrome): (a) custom Chat Completions endpoint (the strict fake gateway on a fixed port): the model's
  `marimo_execute("print(1 + 1)")` ran in the real kernel and `2` came back; the gateway saw only `messages`,
  `model`, `stream`, `tools`, eleven plain function tools, the `api-version` query parameter and the custom
  header; (b) a second run resumed the conversation ("Resumed conversation (1 turns so far)") and sent the
  earlier turns; (c) built-in provider with `OPENAI_API_KEY`: `gpt-5.5` over the Responses API called
  `marimo_status`, then `marimo_execute`, and answered 391 for 17 x 23 computed in the kernel (9,223 tokens
  in, 91 out). Still owed: a physical Ctrl+C in a Windows console, and a run by one of the users the Codex
  runtime failed for.

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
keys are warnings returned by `validate`, not errors, including the keys removed with the Codex SDK
(`[hailer].codex_home`, `[web].allow_shell_network`, provider `merge_messages`, `parallel_tool_calls`,
`requires_openai_auth`), so an old file still loads. Relative paths resolve against the workspace. Env
overrides: `HAILER_CONFIG`, `HAILER_WORKSPACE`, `HAILER_NOTEBOOK`, `HAILER_DATA_DIR`, `HAILER_MARIMO_URL`,
`HAILER_MARIMO_TOKEN` (value kept in memory only), `HAILER_MODEL`, `HAILER_MODEL_PROVIDER`,
`HAILER_LOG_LEVEL`, `HAILER_NOTEBOOKS_DIR`. Provider tables `[model_providers.<id>]` map 1:1 to
`ProviderConfig`: `base_url`, `wire_api`, `stream`, `stream_options` (both default `true`, valid with either
`wire_api`), `env_key` (a declared `[model_providers.openai]` without one gets `OPENAI_API_KEY`), `name`,
`http_headers`, `env_http_headers`, `query_params`. `[web]` maps to `WebConfig` (`allowed_domains` list,
`max_page_bytes`). `[hailer]` keys: `notebook`, `notebooks_dir`, `data_dir`, `marimo_url`, `context_dir`,
`skills_dir`, `prompts_dir`, `max_tool_output_chars`, `max_context_bytes`, `log_level`; `[model]` keys: `name`,
`provider`, `reasoning_effort` (`""` = send none), `summarize_after_tokens` (`ModelConfig`, default 100000,
0 = off). `notebooks_dir` (relative to the workspace) is the folder notebooks are listed from, created in and
opened from; it defaults to the configured notebook's folder and is always filled in by `load_config`
(`HailerConfig.notebooks_dir`; use `HailerConfig.notebooks_root`, which falls back to `notebook.parent` for
hand-built configs). `validate` checks: notebook exists, notebook is inside `notebooks_dir` (fatal, names both
keys), `notebooks_dir` exists (warning only), data_dir exists (warning only), `summarize_after_tokens >= 0`,
active provider declared (or `openai`), custom provider has `base_url` and `env_key` (a `[model_providers.openai]`
table without `base_url` is the built-in endpoint and needs neither), `wire_api in ("responses", "chat")`,
domains are well-formed.

Errors added for the notebook feature (`errors.py`): `NotebookExistsError` (create: name taken) and
`NotebookPathError` (outside the notebooks folder, or not a marimo notebook).

## `log.py`  (owner: wave 1 / A)

```python
def setup_logging(level: str | None, *, verbose: bool = False, log_file: Path | None = None,
                  env: Mapping[str, str] | None = None, stream=None) -> logging.Logger  # returns logger "hailer"; safe to call again
class RedactingFilter(logging.Filter)   # masks Authorization values, bearer tokens, bare sk-... keys (a keyring-sourced key is not in the environment) and values (>= 8 chars) of secret-looking env vars present in os.environ
def redact(text: str, env: Mapping[str, str] | None = None) -> str   # the same rules for text outside logging; agent.map_exception uses it on everything it quotes
def get_logger(name: str = "hailer") -> logging.Logger
```
Normal level WARNING to stderr via Rich-free plain handler; `--verbose` → DEBUG. The loggers of `openai`,
`langchain*`, `langgraph`, `langsmith`, `httpx(2)`, `httpcore(2)`, `urllib3`, `asyncio` and `aiosqlite` are
held at WARNING unless `--verbose`.

## `marimo_client.py`  (owner: wave 1 / B)

```python
CM_HELP_CODE = "import marimo._code_mode as cm; help(cm)"
SERVER_TOKEN_HEADER = "Marimo-Server-Token"
def discover_servers() -> list[MarimoServer]         # registry files; drop entries whose /health does not answer
def find_server(config: HailerConfig) -> MarimoServer | None   # config.marimo_url first, then registry (single live server, else None)
class MarimoClient:
    def __init__(self, base_url: str, token: str | None = None, *, timeout: float = 10.0,
                 notebook: Path | None = None, workspace: Path | None = None) -> None
                 # notebook: the one resolve_session()/execute() default to
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
def launch_command(*, port: int | None = None) -> list[str]   # ["uvx","hailer","notebook","--foreground", ("--port",N)]: the command a person runs to start marimo on its own
def marimo_server_command(notebooks_dir: Path, workspace: Path, port: int, *, headless: bool = True) -> list[str]
    # [sys.executable,"-m","marimo","edit",<folder>,"--no-token",("--headless"),"--port",N,"--skip-update-check"]; used by
    # `hailer notebook` (background, headless) and `--foreground` (headless only with --no-browser). Hailer's own
    # interpreter, never `uv run`: works under uvx without a project .venv.
def launch_hint() -> str   # "Start everything in one go: uvx hailer notebook", then launch_command() for another terminal, then "uvx hailer"
def wait_for_health(url: str, timeout: float = 60.0, *, interval: float = 0.5, should_stop=None) -> bool
def wait_for_session(client: MarimoClient, notebook: Path | None, timeout: float = 90.0, *, interval: float = 0.5) -> MarimoSession | None
```
marimo is always started on the notebooks **folder** (never on a single file) so that every notebook,
existing or created later, opens on the same server. Connection refused/timeout → `MarimoUnavailableError`
with `launch_hint()` as the hint. HTTP 401/403 →
`MarimoUnavailableError` mentioning `HAILER_MARIMO_TOKEN`.

## `notebooks.py`  (owner: notebook feature / foundation)

Notebook lifecycle shared by the CLI and the agent's tools. Standard library only; never imports marimo, typer
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
def ensure_notebook(path: Path, *, title: str, kind: Literal["starter", "empty"] = "starter") -> bool
    # `hailer init`: write the template to exactly `path` (parents created) unless something exists there; True
    # when it wrote. Never overwrites; OSError when it cannot write.
```

## `browser.py`  (owner: notebook feature)

```python
def open_url(url: str) -> bool   # launch the default browser; True when a launcher was started; never raises or prints
```
Shared by the CLI (`_open_browser`) and the agent's tools (`HailerTools.open_url` default). Windows: `os.startfile`
first, because `webbrowser` honours a `BROWSER` variable that Git Bash / WSL profiles sometimes set to a non-GUI
command (observed: `BROWSER=true` made `/notebook new` wait 30 s for a tab that never opened); `webbrowser` is the
fallback. POSIX: `webbrowser` picks the launcher, but generic launchers (`xdg-open`, `open`) are spawned with their
stdio discarded, because a launcher that inherited the terminal would write into the chat. Module-level
`is_windows`, `startfile`, `webbrowser_open`, `webbrowser_get`, `popen` are injection points for tests only.

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

## `tools.py`  (owner: LangChain migration, 2026-09-18)

The agent's tools, run inside the Hailer process (no server, no transport). They are everything the model has:
no shell, no file tool, all Python runs in the marimo kernel.

```python
TOOL_NAMES: tuple[str, ...]      # the 11 names in the table, in the order they are offered to the model
DEFAULT_SESSION_WAIT_SEC = 30.0
class HailerTools:               # one plain method per tool, each -> str; the method's docstring is the description the model sees
    def __init__(self, config: HailerConfig, client_factory: ClientFactory | None = None, *, open_url: UrlOpener | None = None, session_wait_sec: float = DEFAULT_SESSION_WAIT_SEC) -> None
def hailer_tools(config: HailerConfig, *, client_factory: ClientFactory | None = None, open_url: UrlOpener | None = None, session_wait_sec: float = DEFAULT_SESSION_WAIT_SEC) -> list[Any]
    # LangChain StructuredTool.from_function(func=method, coroutine=_in_daemon_thread(method), name=name, description=inspect.getdoc(method)), in TOOL_NAMES order
    # ClientFactory = Callable[[], tuple[Any, MarimoServer]]; UrlOpener = Callable[[str], bool]
```
`_in_daemon_thread(fn)` is the async path the agent uses: the blocking call runs on a daemon thread
(`hailer-tool-<name>`) and its result comes back through `loop.call_soon_threadsafe`. On the loop's default
executor the thread would be joined when the loop closes and at interpreter exit (measured: quitting after
Ctrl+C waited 7 s for a call with 7 s left). So a cancelled turn and process exit never wait for a kernel
call; the abandoned call ends in the background and its result is dropped.

The **active notebook is re-read from `.hailer/notebook.json` on every call** (`HailerTools._active_config()`
returns the config with `notebook` replaced by `notebooks.load_active_notebook`); nothing about it is cached.
Every kernel tool acts on the active notebook. Nothing is raised into the agent loop: a `HailerError` becomes
`ERROR: <message>` plus the hint, anything else `ERROR: unexpected <Type>: <message>`. Tools (all return plain
text, truncated to `config.max_tool_output_chars` with head/tail and a `[... truncated N chars ...]` marker):

| tool | args | behaviour |
|---|---|---|
| `marimo_execute` | `code: str` | run in the scratchpad against the active notebook's session; returns stdout/output/stderr (rich mimetypes such as text/html or application/json are replaced by a short placeholder, so HTML never reaches the model); a failed run starts `Execution failed.`; `MarimoUnavailableError`/`NoSessionError` become `ERROR:` text with the hint, not exceptions |
| `marimo_status` | none | server url, version; `active notebook: <workspace-relative> -> session <id> (ready)` or `... has NO session. Ask the user to open <url>`; every session with its notebook and a marker: `(active notebook)`, `(another notebook in the notebooks folder; notebook_open switches to it)`, `(outside the notebooks folder)`, `(unsaved)`; when marimo is down (factory raises, or a configured `marimo_url` answers with `MarimoUnavailableError` on the first request) the text is `marimo: not running` / `active notebook: <name>` / error / hint, so the active notebook is always named |
| `notebook_cells` | `pattern: str = ""` | runs `marimo_client.LIST_CELLS_CODE`, a `cm` snippet listing the active notebook's cells (id, name, status, error count, first line); optional case-insensitive substring filter |
| `notebook_list` | none | `notebooks.list_notebooks`: one line per notebook `  - <name>  modified YYYY-MM-DD HH:MM  <size>  [active] [open]` (`[open]` = has a session; when marimo is unreachable the files are still listed and a trailing line says sessions are unknown); empty folder → `No notebooks in <folder> yet. Create one with notebook_create(name).` |
| `notebook_create` | `name: str, template: str = "starter"` | `notebooks.create_notebook` (+ `save_active_notebook`), then bring-up (below); text: `Created <name> from the <kind> template; it is now the active notebook.` + session line + a reminder of what the template defines; unknown template → `ERROR: unknown template ...`; existing name / unusable name → the `NotebookExistsError` / `NotebookPathError` text |
| `notebook_open` | `notebook: str` | `notebooks.resolve_notebook` (+ `save_active_notebook`), bring-up, then when a session exists the cell listing (`Cells (id, name, status, errors, first line):`); text starts `<name> is now the active notebook.`; unknown / outside-folder references → the `NotebookNotFoundError` / `NotebookPathError` text and the active notebook is unchanged |
| `notebook_close` | `notebook: str = ""` | resolve (empty = active), `client.resolve_session`, `client.shutdown_session(id)`; says the kernel is freed and, for the active notebook, that it stays active but must be reopened; no session → `<name> is not open (no kernel session), so there is nothing to close.`; marimo down → `marimo is not running, so <name> has no kernel session to close.` + error + hint |
| `list_periods` | `name: str = ""` | `describe_periods(scan_period_files(config.data_dir, name or None))` |
| `load_skill` | `name: str` | `read_skill` |
| `read_skill_file` | `name: str, path: str` | `read_skill_file` |
| `fetch_page` | `url: str` | `web.fetch_page`; denial text lists allowed domains |

Bring-up (`HailerTools._bring_up(notebook)`): if the notebook already has a session it is reused and the browser is
not opened; otherwise `open_notebook_url(server, notebook)` is opened through `HailerTools.open_url` (default:
`hailer.browser.open_url`, i.e. `os.startfile` first on Windows and stdio-silenced launchers elsewhere, so
nothing writes into the chat) and `wait_for_session` polls for up to `HailerTools.session_wait_sec` (default 30 s).
Outcomes are reported as text (`kernel session ready (<id>)`, `no kernel session appeared within N s` + URL,
`Could not open a browser` + URL, `marimo is not running (or not reachable)` + error + hint); the browser never
raises. Every `HailerError` raised while resolving or waiting for the session (a configured `marimo_url` is not
health-checked up front, so a dead server first shows up there) is folded into that last outcome: the notebook
was already created / made active by then, so `notebook_create` / `notebook_open` still begin with `Created ...` /
`... is now the active notebook.` and never read as a failed call. Both hooks are keywords of `HailerTools`
and `hailer_tools` so tests never launch a browser.

## `secrets.py`  (owner: wave 2 / D)

```python
KEYRING_SERVICE = "hailer"
def keyring_username(provider_id: str, env_key: str) -> str        # f"{provider_id}:{env_key}"
def resolve_provider_key(provider: ProviderConfig, env: Mapping[str, str] = os.environ) -> tuple[str | None, str]  # (value, "env"|"keyring"|"missing"); keyring failures → "missing" (log at debug)
def store_provider_key(provider: ProviderConfig, value: str) -> None   # CredentialsError on empty value, missing env_key, or unusable keyring backend (then the hint is: set the env var)
def delete_provider_key(provider: ProviderConfig) -> bool          # False when nothing was stored; CredentialsError when the keyring backend is unusable
```
Precedence: the `env_key` environment variable, then the OS credential store (`hailer login <provider>`); no
`env_key` → `"missing"`. `"missing"` is fatal for every provider, the built-in `openai` included. The value
goes to the caller only (the agent hands it to the HTTP client in-process) and is never logged.

## `agent.py`  (owner: LangChain migration, 2026-09-18)

A LangChain agent (`langchain.agents.create_agent` + `langchain_openai.ChatOpenAI`) on an OpenAI-compatible
endpoint, all inside the Hailer process. Conversations are LangGraph checkpoints (`AsyncSqliteSaver`) in
`<workspace>/.hailer/threads.sqlite`, so they survive a restart.

```python
OPENAI_BASE_URL = "https://api.openai.com/v1"
THREADS_FILENAME = "threads.sqlite"
SUMMARY_KEEP_MESSAGES = 20                 # recent messages that survive a summarisation
TRACING_OPT_IN_ENV = "HAILER_TRACING"
INTERRUPTED_TOOL_RESULT: str               # the result recorded for a tool call cut short by Ctrl+C
def provider_by_id(config: HailerConfig, provider_id: str | None = None) -> ProviderConfig   # default: the configured one; "openai" needs no declaration (env_key OPENAI_API_KEY); undeclared → ConfigError
def provider_base_url(provider: ProviderConfig) -> str    # base_url or OPENAI_BASE_URL, no trailing slash
def request_headers(provider: ProviderConfig, environ: Mapping[str, str]) -> dict[str, str]   # http_headers + every env_http_headers entry whose variable is set and not empty
def build_model(provider: ProviderConfig, model: str, api_key: str | None, *, reasoning_effort: str | None = None, environ: Mapping[str, str] | None = None, http_async_client: Any = None) -> Any   # a ChatOpenAI, see below
def disable_tracing_unless_opted_in(environ: Any = None) -> bool   # sets LANGSMITH_TRACING, LANGSMITH_TRACING_V2, LANGCHAIN_TRACING and LANGCHAIN_TRACING_V2 to "false" (in os.environ by default) unless HAILER_TRACING is 1/true/yes/on; True when tracing was left alone. Run when the agent creates its event loop, so a variable left over from another project cannot send prompts and tool results to LangSmith
def system_prompt(config: HailerConfig, bundle: ContextBundle) -> str  # prompts/system.md (importlib.resources; a built-in fallback persona when missing or empty) + project context + skills index + web allowlist statement + workspace section (workspace, "Notebooks folder: <notebooks_root>", data dir, marimo URL). Invariant: the text does NOT depend on config.notebook, which changes mid-conversation (the model learns it from marimo_status() and CLI notices). Sent with every model call, never stored in the conversation
def map_exception(exc: BaseException, config: HailerConfig, *, model: str | None = None, provider: str | None = None) -> Exception
class HailerAgent:
    def __init__(self, config: HailerConfig, bundle: ContextBundle, *, model_factory: ModelFactory | None = None, tools: list[Any] | None = None, env: Mapping[str, str] | None = None, threads_path: Path | None = None) -> None
        # undeclared provider → ConfigError. For tests: model_factory(provider, model, key) replaces build_model, tools replaces hailer_tools(config), env replaces os.environ, threads_path replaces <workspace>/.hailer/threads.sqlite
    def start(self, *, resume_thread_id: str | None = None, forget_thread_id: str | None = None) -> str   # the thread id: resume_thread_id when the store has it, else a new one; forget_thread_id (`hailer --new`: the conversation the user chose not to resume) is deleted from the store
    def new_thread(self) -> str        # a new id; the previous thread is deleted from the store (private _forget, shared with start: Hailer only ever resumes the latest)
    def set_model(self, name: str, provider: str | None = None) -> None   # from the next turn; undeclared provider → ConfigError and nothing changes; never starts a thread (the CLI starts exactly one)
    def run_turn(self, text: str, *, on_event: Callable[[AgentEvent], None] | None = None, skill: SkillInfo | None = None, preamble: str | None = None) -> TurnSummary
    def close(self) -> None            # idempotent; start() works again afterwards
    bundle: ContextBundle              # property; assigning it (/reload) applies from the next turn of the same conversation
    model: str; provider_id: str; started: bool    # read-only properties
    thread_id: str | None; key_source: str         # "env" | "keyring" | "missing", refreshed whenever the model is built
```
- `build_model`: one `ChatOpenAI` covers both wire APIs. Always passed: `model`, `api_key`,
  `base_url=provider_base_url(provider)`, `use_responses_api=(wire_api == "responses")`,
  `stream_usage=(stream and stream_options)`, `http_socket_options=()`. When configured: `default_headers`
  (`request_headers`), `default_query` (`query_params`), `disable_streaming=True` (`stream = false`),
  `reasoning_effort`, `http_async_client`. `parallel_tool_calls` is never sent, so a strict gateway sees only
  `model`, `messages`, `tools`, `stream` (and `reasoning_effort` when set). Reasons: see "Runtime facts".
- Lifecycle: the public methods are synchronous. One `asyncio.Runner` lives from the first call until
  `close()` and every call is `runner.run(...)`, which turns Ctrl+C into a cancellation that aborts the HTTP
  request in flight. There is no `interrupt()`: `run_turn` re-raises `KeyboardInterrupt` with the turn already
  cancelled. Other non-Hailer exceptions go through `map_exception` with the model and provider in use. The
  graph is built on first use and rebuilt after `set_model` or a `bundle` assignment: the old HTTP client is
  closed, the key resolved (`"missing"` → `CredentialsError` with the `hailer login <id>` hint, for every
  provider), the model built with an `openai.DefaultAsyncHttpxClient()` the agent owns (honours
  `HTTP(S)_PROXY` / `NO_PROXY` and the OS trust store), then `create_agent(model, tools, system_prompt=...,
  middleware=..., checkpointer=saver)`. With `[model].summarize_after_tokens > 0` the middleware is
  `SummarizationMiddleware(model=model, trigger=("tokens", N), keep=("messages", SUMMARY_KEEP_MESSAGES))`.
- `run_turn` sends ONE user message: the skill notice (`[Hailer] The user attached the skill '<name>' to this
  request. Follow its instructions:` + `context.read_skill`), the preamble (blank counts as none; the CLI's
  `[Hailer] ...` notices, see cli.py), then the text, joined by blank lines. First `_settle_history` makes the
  stored conversation valid: every tool call without a result gets a `ToolMessage(INTERRUPTED_TOOL_RESULT)`,
  and when the last stored message is the user's (the turn died before the model answered) the new text is
  appended to it under the same message id, because strict chat templates refuse two user messages in a row.
  The turn runs as `graph.astream(..., stream_mode=["updates", "messages"])` and emits `AgentEvent("tool_call",
  <name>, {"arguments": <80-char preview>})` and `AgentEvent("message_delta", <text>)` (the `model` node only,
  never the summariser). `AgentEventKind` also has `"status"`; the agent does not emit it today.
  `TurnSummary(final_response, thread_id, duration_ms, tool_calls, input_tokens, output_tokens,
  status="completed")`: `final_response` is the last model message without tool calls; tokens are summed over
  the turn's model calls, `None` when the endpoint reported none.
- `map_exception`: a `HailerError` passes through. `openai.APIStatusError` is mapped on `status_code`, in this
  order: 401 → `CredentialsError` (login hint); 400/404/422 whose body names the model (`model_not_found`,
  `param == "model"`, a pydantic `loc` with `'model'`, or "model" with "not found" / "does not exist" ...) →
  `ProviderError` "Unknown model '<model in use>' for provider '<id>'."; 403 → "refused access (HTTP 403)";
  404/405 → "did not accept the request" (hint: the request sent, `POST <base_url>/responses` or
  `/chat/completions`, and the other `wire_api`); 429/5xx → "is unavailable (HTTP n)"; `context_length` /
  "maximum context" in the body → "no longer fits the model's context window" (hint: `/new` or a lower
  `summarize_after_tokens`); any other status → "rejected the request (HTTP n)" (hint: the request sent, the
  switches still on, `stream = false` / `stream_options = false`, and `reasoning_effort = ""`). Each hint ends
  with `The endpoint said: <parsed body, else the message>` (through `hailer.log.redact`, whitespace
  collapsed, 500 chars). `openai.APITimeoutError` → "did not answer in time" and `openai.APIConnectionError` →
  "Could not reach the model endpoint at <base_url>." (both `ProviderError`; the first hint says retry, the
  second names `base_url`, a running endpoint and `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY`; both point at
  `hailer status`, never at `hailer doctor`, which does not call the model endpoint); anything else →
  `AgentError("The agent run failed.")` with the redacted `Type: message` as hint.

## `session.py`  (owner: wave 2 / E)

```python
SESSION_DIRNAME = ".hailer"; SESSION_FILENAME = "session.json"
COMMANDS: dict[str, str]     # name -> one-line help, for /help
EXIT_COMMANDS = ("exit", "quit")
def parse_command(line: str) -> Command | None      # "/model gpt-5.5" -> Command("model", "gpt-5.5"); "/exit" ; non-slash -> None; unknown slash -> Command(name, args) (CLI reports unknown)
def session_path(workspace: Path) -> Path
def load_session(workspace: Path) -> SessionState   # .hailer/session.json, tolerant of missing/corrupt
def save_session(workspace: Path, state: SessionState) -> None   # atomic (tmp + replace)
def help_text() -> str
```
`SessionState` is `thread_id, model, provider, turns, input_tokens, output_tokens`: the file names the thread
to resume and carries the counters; the conversation itself is in `.hailer/threads.sqlite`. No prompt hash is
stored: the system prompt is sent with every turn, so `/reload` applies from the next message.

## `cli.py`  (owner: wave 2 / E)

Typer app; `def main() -> None` is the console entry. Commands: default (no subcommand) → chat; `notebook`
(start marimo on `config.notebooks_root` in the background: `mkdir` it first, `marimo_server_command`, log in
`.hailer/marimo.log`; or reuse a live server; open the active notebook; chat; stop marimo on exit unless
`--keep-marimo`; `--foreground` runs `marimo_server_command(..., headless=--no-browser)` attached, without the
chat; `_reusable_server` accepts the configured
URL or a live server that has a session for the active notebook or for any notebook under the folder); `exec`
(`-c CODE` | `-` stdin | file; prints result; exit 1 on failure); `status`; `doctor` (the preflight checks and a
code-mode probe; it never calls the model endpoint); `login <provider>`; `logout <provider>`; `init` (write
default config + `.config/hailer` skeleton, then load that config and create the configured notebook with
`notebooks.ensure_notebook` and the notebooks and data folders when missing; nothing but `hailer.toml` is ever
overwritten, and only with `--force`; an unloadable `hailer.toml` skips the notebook step). Global options `--verbose/-v`, `--config`, `--workspace`, `--new`
(do not resume), `--version`. Startup: `_load_config` (load config, then `config.notebook` := the active
notebook from the state file, or, when `HAILER_NOTEBOOK` is set, that notebook written to the state file) →
setup logging → Rich panel (`Notebook:` active, `Notebooks:` folder) → preflight (config, notebook,
credentials, marimo server, session) → agent start → REPL. Agent start (`ChatLoop.start`):
`HailerAgent(config, bundle)`, then `start(resume_thread_id=<thread in session.json>)`, or with `--new`
`start(forget_thread_id=<that thread>)` so the stored conversation is deleted; a different id coming back from
a resume means the thread was gone, the CLI says so and resets the counters. Credentials: a `"missing"` key is
a fatal preflight failure for every provider, the built-in `openai` included; `status` and `/status` show
`<env_key> from env|keyring` or `<env_key> missing (run: uvx hailer login <id>)`. REPL: `You > ` prompt;
Ctrl+C during a turn cancels it (the agent re-raises `KeyboardInterrupt`, the CLI prints `Interrupted.`);
Ctrl+C or EOF at the prompt exits; slash commands `/help /status /new /exit /quit /model /notebook /clear
/context /skill /prompt /reload`. `/new` and `/model` share `_new_thread` (`agent.new_thread()`, counters
reset, session saved): `/model <name>` or `/model <provider>:<name>` refuses an undeclared provider, else
calls `agent.set_model(name, provider)` and starts exactly one new thread. `/reload` re-reads `.config/hailer`
and assigns `agent.bundle`; it applies from the next message.

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
`run_turn(..., preamble=...)` with the next message and cleared (also on Ctrl+C; kept when the turn raised an
error). `_sync_active_notebook()` runs in the `finally` of every turn (finished, failed, interrupted) and,
without opening a browser, at the start of `/notebook` and `/status`: on a change it updates `self.config`,
prints `Active notebook is now <name>.` and (turn end only) opens the URL once when the notebook has no session;
no preamble is queued because the model made the switch itself. `COMMANDS["notebook"]` =
`Show or switch the active notebook. Usage: /notebook [list | new <name> [--empty] | open <name> | close [name]]`.
Progress (`_TurnDisplay`): a single Rich status line updated from `AgentEvent`s: `Thinking...`, `Using: <tool>`
on `tool_call`, `Writing reply...` at the first `message_delta`, the text of a `status` event. Deltas are
never printed (a model can stream commentary next to its tool calls and then the answer, which printed the
answer twice); `finish` prints `final_response` once under `Hailer >`, or `(no reply)`. After a turn the CLI
adds the turn and its tokens to the session file. Errors: print message + hint; exit code 1 when startup
fails, the REPL carries on after a failed turn; traceback only with `--verbose`.

## `prompts/system.md`  (owner: wave 2 / E, reviewed by D)

The agent persona and rules: the tool list (every `TOOL_NAMES` entry appears as `` `name(` ``, checked by
`test_agent.py`) with the statement that there is no shell and no file tool, the one-active-notebook rules, the
`[Hailer]` notice rule (a CLI notice, "not the user's words"), the scratchpad semantics and the canonical `cm`
patterns, the data conventions with the "compact outputs only" rule, safety, and how to reply. The web
allowlist, project context, skills index and workspace are appended by `agent.system_prompt`. Loaded with
`importlib.resources.files("hailer").joinpath("prompts/system.md")`.

## Tests (each owner writes their own; `uv run pytest` must pass offline)

- `test_config.py`, `test_log.py`: A
- `test_marimo_client.py` (fake SSE server on 127.0.0.1), `test_periods.py` (tiny parquet files in tmp_path): B
- `test_web.py`, `test_context.py`: C. `test_notebooks.py`, `test_browser.py`: notebook feature
- `test_tools.py`: LangChain migration. Tool methods called directly with a fake marimo client, plus the
  `hailer_tools` registration: names, schemas, docstring descriptions, in-process `invoke`, the daemon-thread
  `ainvoke` path, and a cancelled call that does not wait for a blocked tool
- `test_secrets.py`: D. `test_agent.py`: LangChain migration. Unit tests drive `HailerAgent` through
  `model_factory=` with `ScriptedModel` (a `BaseChatModel` replaying messages, exceptions or callables) and
  two toy tools: `build_model` kwargs, tracing, system prompt, error mapping, lifecycle (resume after a
  restart, `new_thread`, `set_model`, `close`), turns (events, usage, one user message, `/reload`, failed-turn
  merge, summarisation) and Ctrl+C during a model call and during a tool (a real SIGINT to the main thread
  on POSIX, `_thread.interrupt_main()` on Windows; see "Runtime facts").
  Wire-level tests run the real `ChatOpenAI` against `tests/fake_gateway.py` (`FakeGateway`, a stdlib
  `ThreadingHTTPServer` on 127.0.0.1: only `POST /v1/chat/completions`, 404 elsewhere, 422 for unknown fields
  and model names, required key, `X-Client-Id` header and `api-version` query, optional refusal of `stream:
  true`, every request recorded; `port=0` picks a free port, a fixed one serves manual end-to-end runs) and
  assert the exact request fields, headers and query, and one request per 4xx
- `test_session.py`, `test_cli.py` (Typer `CliRunner`; slash commands; preflight messages with fakes; a
  `FakeAgent` with the `HailerAgent` surface the CLI uses): E
