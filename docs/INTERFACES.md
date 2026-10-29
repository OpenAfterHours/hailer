# Hailer module contracts

This is the contract every module is built against. [PLAN.md](../PLAN.md) describes the architecture and
[LEARNINGS.md](LEARNINGS.md) the lessons behind it (the history, including the Docker kernel's plan and
review, is in [history/](history/PLAN_HISTORY.md)); this file says *what*
each module exposes so work can proceed in parallel.
Shared types live in `src/hailer/models.py`, errors in `src/hailer/errors.py`. Do not change those two files
without agreement (report a needed change instead).

## Ground rules

- **Windows first.** No bash, curl, jq, or POSIX-only assumptions. Use `pathlib`, `subprocess` with argument
  lists, `os.name`/`sys.platform` checks where needed. Paths in user output use the OS separator.
- **Standard library where practical.** HTTP to marimo and the web via `urllib.request`/`http.client`, TOML
  via `tomllib`, SSE parsing by hand, HTML-to-text via `html.parser`. Third-party allowed: `polars[calamine]`
  (including `fastexcel` for Excel reading), `duckdb`,
  `typer`, `rich`, `keyring`, `langchain`, `langchain-openai`, `langgraph-checkpoint-sqlite` (and what they
  install: `langchain_core`, `langgraph`, `openai`, `aiosqlite`; imported only inside functions of `agent.py`
  and `tools.py`, so `hailer --help` stays fast), `marimo` (only in the notebook / `/exec` snippets,
  never imported by the CLI process except to locate the executable). Docker is driven through the `docker`
  CLI with `subprocess` (`hailer.kernel_docker.DockerRunner`), never a Docker SDK; `hailer.kernel_docker` is
  imported only when the docker runtime is asked for.
- **Tests are offline.** No API key, no marimo server, no internet, no Docker. Use fakes: a local
  `http.server` for the marimo protocol (`tests/fake_marimo.py`), a scripted chat model and the loopback
  `tests/fake_gateway.py` for the agent, the stateful scripted `docker` CLI in `tests/fake_docker.py`,
  `tmp_path` for files, an in-memory keyring backend (`keyring.backends.fail` / a tiny custom backend) for
  secrets. The unsafe-local runtime's process layer is faked (`tests/fake_kernel.py`): no test connects to a closed port.
  Docker is the default runtime, so a test that builds a config names its runtime (`KernelConfig(runtime=...)`)
  whenever it starts or describes a kernel. The one exception is `tests/test_docker_integration.py`, skipped
  unless `HAILER_DOCKER_TESTS` is set.
- **Secrets never appear** in logs, prompts, tool results, exception messages, or config files. The provider key
  is resolved in-process (`secrets.resolve_provider_key`) and handed to the HTTP client; Hailer never writes it
  to a file or puts it into an environment. A docker kernel gets none of the host's variables, and the
  unsafe-local marimo server's environment leaves out the key's variable and every other name
  `kernel.withheld_variables` recognises (no setting lets one through). The logging redaction filter masks
  `Authorization` headers, bearer tokens, bare `sk-...` keys and any value whose env var name ends in `_KEY`,
  `_TOKEN`, `_SECRET`, `_PASSWORD`. The marimo server's token (`MarimoServer.token`, kept out of `repr`) lives
  in the memory of the session that started the kernel (and, while it runs, in that session's owner-only
  log, deleted with the kernel); it is never written to a state file or mounted into a container. URLs that
  go to the model (tool results, the system prompt) never carry it, and log tails Hailer prints mask it.
- **Errors are actionable.** Raise `HailerError` subclasses with a `hint`. The CLI is the only place that
  prints; library modules never print.
- **Ownership.** Each module has one owner during a wave. Only edit files you own; if you need something from
  another module that is not in this contract, write against the contract and note the gap in your report.
- **Dependencies** are already declared in `pyproject.toml`; do not add any. Report if something is missing.
- **The active notebook is shared state, not config.** `<workspace>/.hailer/notebook.json` (next to
  `session.json`) records which notebook the chat is working in, as a *name* relative to the notebooks folder.
  It has two writers in one process: the CLI between turns (slash commands) and the agent's tools during a turn.
  Every reader re-reads it before use through `hailer.notebooks.load_active_notebook` and never caches it.
  `load_active_notebook` reads **only the file** (no environment variable names a notebook), so the CLI, the
  tools and later sessions agree. Residual race: a tool call still running after Ctrl+C (it finishes on its daemon thread) may write the
  file later; the CLI re-reads it after every turn (finished, failed or interrupted) and before `/notebook` and
  `/status`.
- **Files belong to the sandbox.** Everything above the runtime layer (the tools, the chat, the CLI's session
  code) knows notebooks by name and data files by name and reaches them only through the
  `hailer.sandbox.MarimoSandbox` the session started; `hailer.tools` never reads or writes the notebooks or data
  folders itself (`test_the_tools_never_touch_the_notebooks_or_data_folders`). Host paths and kernel paths meet
  only inside the runtime layer (`MarimoClient`, `MarimoSandbox`). Workspace setup (`uvx hailer init`, the
  doctor's notebook row) stays host-side.

## Runtime facts

For the original marimo verification, see [PLAN_HISTORY.md](history/PLAN_HISTORY.md) §1. For LangChain lessons and maintained regression checks, see
[LEARNINGS.md](LEARNINGS.md) and `tests/test_agent.py`.

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
  A session exists only while the notebook is open in a browser.
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
- Tokens (marimo 0.24.2, verified live 2026-09-18/19): `marimo edit ... --token-password-file -` reads the
  token from stdin (EOF ends it), so it is never on a command line; `--token-password-file <path>` reads it
  from a file as the CLI starts (the docker runtime mounts one, verified 2026-09-22 with Docker Desktop
  29.4.3: the token is then in neither `docker inspect` nor the process list); `--token-password <t>` takes
  it as an argument (Hailer never uses it). Without the token `GET /api/sessions` and `POST
  /api/kernel/execute` answer 401; `Authorization: Bearer <t>` is accepted by the API, and
  `access_token=<t>` in a page URL signs a browser in (marimo prints that URL in its banner, so the log
  holds the token). `/health` needs no token.
- Docker (Docker Desktop 29.4.3 with the WSL2 engine on Windows 11, and WSL Ubuntu; [DOCKER_KERNEL_PLAN.md](history/DOCKER_KERNEL_PLAN.md)
  §3 has the spike): Docker silently drops `-p` for a container on an `--internal` (or `none`) network, so
  the kernel is reached through a forwarder container published on `127.0.0.1` and joined to both
  networks. Inside the container marimo must bind `--host 0.0.0.0`. The `?file=` key must be the kernel
  path (`/work/notebooks/...`); a Windows host path gets no session, and `/api/sessions` reports kernel
  paths. `--mount type=bind` fails on a missing source instead of creating it (root-owned) as `-v` does. A
  tmpfs home needs `uid=`/`gid=` or marimo logs permission errors for `~/.config`. marimo 0.24's edit page
  takes `runtime.auto_instantiate` from the user configuration (`.marimo.toml` in the working directory),
  not from a project `pyproject.toml`, so the image writes `/work/.marimo.toml`. `docker rm -f` exits 0 for a
  container that is not there (it echoes only what it removed); `docker network rm` exits 1 with "not
  found". GHCR answers `denied` for a tag that is not published or not visible.

## `config.py`  (owner: wave 1 / A)

```python
CONFIG_FILENAMES = ("hailer.toml", ".config/hailer/hailer.toml")
DEFAULT_CONFIG_TEMPLATE: str            # commented hailer.toml written by `hailer init`, with [kernel] runtime = "docker"
def find_workspace(start: Path | None = None) -> Path      # cwd or first parent containing a config file / pyproject.toml; else cwd
def find_config_path(workspace: Path, explicit: Path | None = None, env: Mapping[str, str] = os.environ) -> Path | None
def load_config(workspace: Path | None = None, config_path: Path | None = None, env: Mapping[str, str] = os.environ) -> HailerConfig
def validate(config: HailerConfig) -> list[str]            # human-readable problems, empty if fine
def docker_mount_problems(config: HailerConfig, *, windows: bool | None = None) -> list[str]   # every docker-mode data-folder rule (below)
def is_unc_path(path) -> bool                              # \\server\share\... or \\?\UNC\...
def config_template(kernel: str | None = None) -> str      # DEFAULT_CONFIG_TEMPLATE; kernel="docker"|"unsafe-local" names that runtime in its [kernel] section
def runtime_problem(runtime: str, where: str = "[kernel].runtime") -> str   # why a value is not a runtime; the retired "local" has its own message
def write_default_config(path: Path, *, overwrite: bool = False, kernel: str | None = None) -> None
```
Rules: missing file → defaults (not an error); unparsable → `ConfigError` naming the file and line; unknown
keys are warnings returned by `validate`, not errors, including the keys removed with the Codex SDK
(`[hailer].codex_home`, `[web].allow_shell_network`, provider `merge_messages`, `parallel_tool_calls`,
`requires_openai_auth`), so an old file still loads. Relative paths resolve against the workspace. Env
overrides (the complete list): `HAILER_CONFIG`, `HAILER_WORKSPACE`, `HAILER_MODEL`, `HAILER_MODEL_PROVIDER`,
`HAILER_LOG_LEVEL`, `HAILER_KERNEL`, `HAILER_KERNEL_IMAGE`; `HAILER_TRACING` is read by `hailer.agent` only.
Folders and the notebook come from the file alone (`HAILER_NOTEBOOK`, `HAILER_NOTEBOOKS_DIR` and
`HAILER_DATA_DIR` were removed on 2026-09-23 and are not read). Provider tables `[model_providers.<id>]` map 1:1 to
`ProviderConfig`: `base_url`, `wire_api`, `stream`, `stream_options` (both default `true`, valid with either
`wire_api`), `env_key` (a declared `[model_providers.openai]` without one gets `OPENAI_API_KEY`), `name`,
`http_headers`, `env_http_headers`, `query_params`. `[web]` maps to `WebConfig` (`allowed_domains` list,
`max_page_bytes`). `[hailer]` keys: `notebook`, `notebooks_dir`, `data_dir`, `context_dir`,
`skills_dir`, `prompts_dir`, `max_tool_output_chars`, `max_context_bytes`, `log_level`, `input_history`
(bool, default `true`: the chat's input history persists in `.hailer/input_history`; `false` keeps it in
memory for the session; `HailerConfig.input_history`; the template shows it commented out) (the removed
`marimo_url` and `marimo_token` are reported as unknown keys; their values are never quoted); `[model]` keys: `name`,
`provider`, `reasoning_effort` (`""` = send none), `summarize_after_tokens` (`ModelConfig`, default 100000,
0 = off). `notebooks_dir` (relative to the workspace) is the folder notebooks are listed from, created in and
opened from; it defaults to the configured notebook's folder and is always filled in by `load_config`
(`HailerConfig.notebooks_dir`; use `HailerConfig.notebooks_root`, which falls back to `notebook.parent` for
hand-built configs). `validate` checks: notebook exists, notebook is inside `notebooks_dir` (fatal, names both
keys), `notebooks_dir` exists (warning only), data_dir exists (warning only), `summarize_after_tokens >= 0`,
active provider declared (or `openai`), custom provider has `base_url` and `env_key` (a `[model_providers.openai]`
table without `base_url` is the built-in endpoint and needs neither), `wire_api in ("responses", "chat")`,
domains are well-formed.

`[kernel]` maps to `KernelConfig` (models.py): `runtime` (`"docker"` default | `"unsafe-local"`;
`HAILER_KERNEL` overrides it, then `--kernel` in the CLI; stored lower-cased), `image` (`HAILER_KERNEL_IMAGE`
overrides; `""` = the default, `KernelConfig.effective_image` = `kernel_image.default_image()`,
`ghcr.io/openafterhours/hailer-kernel:marimo<marimo version>-<fingerprint>`),
`memory` (`"4g"`, docker `--memory` format, kept as written), `cpus` (number, `2`), `network` (`false`).
The removed `pass_env` is an unknown key (its value is never quoted). A wrong type is a `ConfigError` that
quotes the value as written (`must be a number, not "2" (str)`). `validate` adds: `runtime` is one of
`VALID_KERNEL_RUNTIMES` (fatal, `runtime_problem`), `memory` matches `<number>[b|k|m|g]` above 0,
`cpus > 0`; and with `runtime = "docker"` every `docker_mount_problems` entry (fatal). "The notebooks folder
is the workspace itself" is a warning in either runtime (no mount rule applies to the notebooks folder).
`runtime = "local"` (the name before 2026-09-22) is fatal with its own
message, the same for `HAILER_KERNEL`, `--kernel` (exit 2) and `runtime_for`: `Invalid [kernel].runtime
"local" (or HAILER_KERNEL): the runtime that runs notebook code on this machine is now called
"unsafe-local", because notebook code then runs as you, with your files and network. Write "unsafe-local"
only if you accept that; otherwise use "docker", the isolated default.` (`Use "unsafe-local"` for
`--kernel`). Other values: `Invalid
[kernel].runtime "podman" (or HAILER_KERNEL); use "docker" (...) or "unsafe-local" (...).` (`Invalid
--kernel "podman"; ...` for the flag). A key from `_KNOWN_KERNEL` found
under `[hailer]` or `[model]` (an uncommented `runtime` whose `[kernel]` line is still commented out) is a
warning naming the fix, not "unknown key". The CLI's `config` row turns warnings into a warning row, so
they show at session start too.

`docker_mount_problems(config, *, windows=None)` is the one place for every docker-mode mount rule. Only
the data folder is mounted (read-only; the kernel's notebooks folder is its own tmpfs), so the rules are
about it, at most one message with its fix, in this order: Windows UNC path ("The data folder
\\server\share\sales is on a network share (UNC path), which Docker cannot mount. ..."; checked first, nothing
touches the share); a drive root; is or contains `Path.home()`, the workspace, `<workspace>/.hailer`,
`config.config_path`, `context_dir`, `skills_dir` or `prompts_dir` (compared resolved and by
`os.path.samefile`); is, contains or sits inside a credential folder (`CREDENTIAL_FOLDERS` under home:
`.config`, `.ssh`, `.aws`, `.azure`, `.gnupg`, `.docker`, `.kube`; on Windows also `%APPDATA%` and
`%LOCALAPPDATA%`; "sits inside" does not apply within `tempfile.gettempdir()`). `validate` runs it for docker
configs; `DockerRuntime.mount_problems` is exactly this, run by `prepare` and `start` on every start path
(`--foreground` never validates, and environment variables can move the folder).

Errors added for the notebook feature (`errors.py`): `NotebookExistsError` (create: name taken) and
`NotebookPathError` (a reference outside the notebooks folder, a name `notebooks.check_notebook_name` refuses,
or a notebook that is not UTF-8 text). For the kernel runtimes: `KernelRuntimeError` (a
runtime cannot start or reach marimo: the process exited, Docker is missing or down, an image problem). Types added to `models.py`: `KernelConfig` (above),
`KERNEL_RUNTIME_DOCKER` / `KERNEL_RUNTIME_UNSAFE_LOCAL` / `VALID_KERNEL_RUNTIMES`, `KERNEL_IMAGE_REPOSITORY`, the
`MarimoServer` fields listed under `marimo_client.py`, and `HailerConfig.kernel`.

`[checks]` maps to `CodeChecksConfig` (models.py, `HailerConfig.checks`): `lint`, `typecheck`, `format`, all
booleans defaulting to `true` (a non-boolean is a `ConfigError`, an unknown key a warning). `lint` and
`typecheck` choose ruff and ty for the checks `marimo_execute` and `notebook_check` run; `format` switches on
code mode's ruff formatting in the kernel (see `code_checks.py`).

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
MAX_NOTEBOOK_BYTES = 5 * 1024 * 1024       # the notebook cap (sandbox re-exports it); replies are sized from it
MAX_REPLY_BYTES = 3 * MAX_NOTEBOOK_BYTES + 1 MiB   # every GET /api/sessions and POST reply: a larger Content-Length, or more data,
                                           # is MarimoUnavailableError ("... more than 16 MiB; Hailer stopped reading.") without reading on
MAX_PAGE_BYTES = 2_000_000                 # GET / (the server token's page), read like every reply: bounded, against the deadline
MAX_EXEC_BYTES = 3 * MAX_NOTEBOOK_BYTES + 1 MiB   # execute(): the most its event stream may add up to (lines read at most that long);
                                           # past it MarimoExecutionError; a non-stream error body is read at most 64 KiB
def request_deadline(seconds) -> ContextManager   # thread-local: every request inside must finish (connect, headers, whole body read
                                           # in 64 KiB read1 chunks) by one monotonic deadline; past it MarimoUnavailableError "(out of time)"
                                           # and nothing more is sent. Kernel code can replace the server, so a reply may be forged or trickled
def answers_with_token(url: str, token: str | None, timeout: float = 1.0) -> bool
    # /health answers AND /api/sessions is refused (401/403) without the token AND accepted (200) with it; a
    # --no-token server accepts anything and never passes, so a start only ever accepts the server it started,
    # even when another one listens on that port. token=None: /health only
class MarimoClient:
    def __init__(self, base_url: str, token: str | None = None, *, timeout: float = 10.0,
                 notebooks_path: str | None = None, native_paths: bool = False,
                 notebook: str | None = None, token_in_links: bool = False,
                 token_cache: dict[str, str] | None = None) -> None   # token_cache: shares the server token (a sandbox passes its own)
                 # notebooks_path: the kernel's path of the notebooks folder; notebooks are names relative to it.
                 # native_paths: the kernel's paths are this machine's (unsafe-local runtime), else POSIX paths in a container.
                 # notebook: the name resolve_session()/execute() default to; token: sent as Authorization: Bearer;
                 # token_in_links: notebook links in error hints carry the token (the CLI sets it, the tools never)
    def file_key(self, name: str) -> str        # kernel_path(notebooks_path, name); ValueError without a notebooks folder
    def name_of(self, path: object) -> str | None   # name_in_folder(notebooks_path, path, native=native_paths)
    def health(self) -> bool
    def sessions(self) -> list[MarimoSession]   # filename/path as marimo reports them (path is absolute: os.path.abspath) + name = name_of(path or filename)
    folds_case: bool                            # property: native_paths on Windows (names ignore case)
    def resolve_session(self, notebook: str | None) -> MarimoSession   # match_session(fold=folds_case) below; notebook=None → the single session; 0 → NoSessionError with hint to open the notebook URL; no match → NoSessionError listing the open sessions
    def execute(self, code: str, *, session_id: str | None = None, notebook: str | None = None,
                on_stdout: Callable[[str], None] | None = None, on_stderr: Callable[[str], None] | None = None,
                timeout: float = 600.0) -> ExecResult   # SSE parse; missing `done` → MarimoExecutionError
    def notebook_url(self, notebook: str | None = None) -> str   # the token only with token_in_links; marimo's home page without a name or notebooks folder
    def server_token(self) -> str                       # data-token of <marimo-server-token> in GET /; cached per instance; MarimoUnavailableError if absent
    def shutdown_session(self, session_id: str) -> None # POST /api/home/shutdown_session {"sessionId": id}
    # marimo's file explorer (marimo 0.24.2, @requires("edit"): the bearer token; plus the Marimo-Server-Token
    # header like every POST but /api/kernel/execute; no browser session needed). Kernel paths, camel-case FileInfo
    # (path, name, isDirectory, isMarimoFile, lastModified, size):
    # marimo confines nothing: it reads, writes and follows links on any path it is given (live probe), so the
    # sandbox checks every path first. list_files/create_file/update_file/delete_file turn an HTTP error into
    # MarimoUnavailableError; file_details raises HTTPError (read_notebook explains it)
    def list_files(self, folder: str) -> list[dict]     # POST /api/files/list_files {path}; a missing folder lists as empty; entries' path is os.path.join(folder, name) on the kernel (mixed separators on Windows)
    def file_details(self, path: str, max_bytes: int | None = None) -> dict   # POST /api/files/file_details {path, maxBytes}: file, contents, isBase64, isTooLarge; HTTPError 500 for a missing file
    def create_file(self, folder: str, name: str, data: bytes) -> dict        # POST /api/files/create, multipart (path, type=file, name, file); parents created; never overwrites (picks <stem>_1<suffix>: info.path says where it went)
    def update_file(self, path: str, contents: str) -> dict                   # POST /api/files/update {path, contents}: an existing file; marimo reloads its open session; written with Path.write_text, so CRLF on a Windows kernel (create stores bytes as given)
    def delete_file(self, path: str) -> dict                                  # POST /api/files/delete {path}
    # A 401 on any server-token POST → MarimoUnavailableError ("server restarted; retry"), the cached token dropped
def match_session(sessions: Sequence[MarimoSession], notebook: str, *, fold: bool = False) -> MarimoSession | None
    # the session whose name is notebook (ignoring case with fold); untitled sessions and files outside the folder never match; NO bare-filename
    # fallback (a/report.py never matches b/report.py); several matches → the last one listed. Used by
    # resolve_session, the CLI (/notebook list) and the test fakes.
def notebook_file_key(notebook: Path) -> str          # a host path as marimo knows it: absolute, normalised, forward slashes, symlinks/junctions NOT resolved; the unsafe-local runtime's notebooks_path is notebook_file_key(notebooks_root)
def kernel_path(folder: str, name: str) -> str        # folder + "/" + name: the ?file= key of a notebook
def name_in_folder(folder: str | None, path: object, *, native: bool) -> str | None
    # a reported kernel path as a name in folder; None when empty, relative or outside. native: both resolved
    # (os.path.realpath: a link out of the folder is outside; Windows returns the on-disk spelling), either
    # separator, case-insensitive on Windows; else POSIX
def open_notebook_url(server: MarimoServer, file_key: str, *, view: str = "app", with_token: bool = False) -> str
    # http://127.0.0.1:2718/?file=<quoted key>&view-as=present (view="app", default: results only, Ctrl+. toggles the
    # editor); view="edit" omits the parameter. with_token=True appends &access_token=<server.token>
    # (only for URLs a person opens); the default is token-free, for anything that reaches the model
def home_url(server: MarimoServer, *, with_token: bool = False) -> str   # marimo's home page (the notebooks folder); ?access_token= only with with_token
def marimo_server_command(notebooks_dir: Path, workspace: Path, port: int) -> list[str]
    # [sys.executable,"-m","marimo","edit",<folder>,"--token-password-file","-","--headless","--port",N,"--skip-update-check"];
    # the caller writes the token to the child's stdin and closes it (hailer.kernel.spawn_marimo / attach_marimo),
    # so it never appears on a command line. Used by LocalRuntime for `hailer notebook` and `--foreground`, always
    # headless (the CLI opens the signed-in page itself). Hailer's own interpreter, never `uv run`: works under
    # uvx without a project .venv.
def launch_hint() -> str   # the kernel this session started stopped or does not answer: end the chat (/exit) and run uvx hailer again
def find_free_port(preferred: int = 2718, host: str = "127.0.0.1") -> int   # preferred when a bind succeeds, else an OS-chosen free port
def wait_for_health(url: str, timeout: float = 60.0, *, token: str | None = None, interval: float = 0.5, should_stop=None) -> bool
    # each round: should_stop() first (the starting process or container has gone → False), then answers_with_token(url, token)
    # (only /health without a token). Two starts can pick the same free port: another session's server there never passes
def wait_for_session(client: MarimoClient, notebook: str | None, timeout: float = 90.0, *, interval: float = 0.5, should_stop: Callable[[], bool] | None = None) -> MarimoSession | None  # None on timeout/cancellation; a request already in flight uses the client's timeout
```
marimo is always started on the notebooks **folder** (never on a single file) so that every notebook,
existing or created later, opens on the same server. Connection refused/timeout → `MarimoUnavailableError`
with `launch_hint()` as the hint. HTTP 401/403 → `MarimoUnavailableError` whose hint says something else may
answer on that port and to start a new session.

A Hailer process only ever talks to the server it started itself; the CLI keeps its sandbox in memory and
hands it to the chat, the agent and its tools; nothing is looked up. `MarimoServer` (models.py) carries
`url`, `pid` (unsafe-local), `token` (`repr=False`), `runtime` (`"docker"` default /`"unsafe-local"`) and `network_access`.
`MarimoSession` carries `session_id`, `filename`, `path` (as marimo reports them) and `name` (the notebook's
name in the notebooks folder, or `None`).

## `sandbox.py`  (sandbox contract, 2026-09-22)

A started kernel, as everything above the runtime layer sees it: the server, stopping, a description and
the files. `MarimoSandbox` is the contract (every runtime runs marimo, so every runtime, a cloud one
included, subclasses it for its lifecycle). Notebooks are names (POSIX paths relative to the notebooks
folder: `"sales.py"`, `"q3/review.py"`), data files names relative to the data folder. Imports
`marimo_client` and `notebooks`; `kernel` imports it. Standard library at import time (`data_schema`
imports Polars when called).

```python
MAX_NOTEBOOK_BYTES = 5 * 1024 * 1024       # the largest notebook read_notebook reads
MAX_NOTEBOOK_DEPTH = 4                     # folder levels a notebook may be below the folder (listing and writing)
MAX_NOTEBOOKS = 500; MAX_NOTEBOOK_FOLDERS = 200   # the listing's bounds
SKIPPED_FOLDERS = {"node_modules", "venv", "site-packages"}   # never listed, like folders starting with "." or "__"
@dataclass(frozen=True) class NotebookFile: name: str; modified: float; size: int   # .stem: the file name without .py
@dataclass(frozen=True) class DataFile: name: str; modified: float; size: int
@dataclass(frozen=True) class SandboxEntry: name: str; folder: bool; modified: float; size: int; marimo: bool   # anything in the notebooks folder
MAX_TREE_ENTRIES = 5000                    # list_tree's bound
def notebook_digest(source: str) -> str   # sha256 with \r\n normalised to \n (update writes the kernel's line endings)
@dataclass class MarimoSandbox:
    server: MarimoServer; notebooks_path: str = ""; data_path: str = ""; data_dir: Path | None = None
    native_paths: bool = False; log_hint: str = ""; ended: str = ""; notice: Callable[[str], None] | None = None
    # notebooks_path / data_path: the folders as notebook code reaches them (/work/notebooks and /work/data in a
    # container); data_dir: the host folder behind data_path; ended: why wait() returned (empty after Ctrl+C);
    # notice: where warning lines go (the chat sets its console's print, called from any thread; default print)
    def notify(self, text: str) -> None                       # through notice; never raises
    def describe(self) -> str                                 # the text after "Kernel:" (describe_runtime of the server)
    def stop(self) -> None                                    # idempotent, never raises (runtimes implement it)
    def log_tail(self, lines: int = 15) -> list[str]
    def wait(self) -> int                                     # --foreground: block until the kernel stops or Ctrl+C (runtimes implement it)
    def client(self, *, notebook: str | None = None, token_in_links: bool = False, timeout: float = 10.0) -> MarimoClient   # knows notebooks by name
    def notebook_url(self, name: str, *, with_token: bool = False) -> str    # app view; NotebookPathError for a bad name
    def home_url(self, *, with_token: bool = False) -> str
    def list_notebooks(self) -> list[NotebookFile]            # marimo .py notebooks, sorted by name (case-insensitive), within the bounds above;
                                                              # (local kernel) never behind a link or junction that leaves the folder
    def list_tree(self) -> list[SandboxEntry]                 # every file and folder, sorted (list_notebooks filters it); folders starting with "." or "__" or in SKIPPED_FOLDERS listed but
                                                              # never looked into; MAX_NOTEBOOK_DEPTH / MAX_NOTEBOOK_FOLDERS / MAX_TREE_ENTRIES
    def has_notebook(self, name: str) -> bool                 # one listing of the parent folder (startup's active-notebook check)
    def read_notebook(self, name: str) -> str                 # for copying out; NotebookNotFoundError; NotebookPathError (not UTF-8, larger than MAX_NOTEBOOK_BYTES, behind a link out of the folder)
    def writable_name(self, name: str) -> str                 # check_notebook_name + at most MAX_NOTEBOOK_DEPTH deep; NotebookPathError;
                                                              # runtimes add rules (DockerKernel: sync_refusal); write_notebook calls it first
    def write_notebook(self, name: str, source: str, *, replace: bool = True) -> NotebookFile
        # create (folders too, at most MAX_NOTEBOOK_DEPTH deep) or replace the text; replace=False → NotebookExistsError
        # for an existing file, which is never touched (a create that loses a race: marimo writes <stem>_1.py, which is
        # deleted again). A Windows kernel: a case variant is the existing file and the result has the listing's
        # spelling. A local kernel: refused when the resolved target leaves the folder (marimo follows links)
    def list_data(self) -> list[DataFile]                     # the files directly in the data folder (not hidden); HailerError "Data directory does not exist: <data_path>"
    def data_schema(self, name: str) -> dict[str, str]        # a Parquet file's columns and types as text; MalformedParquetError; a name with a separator or a leading dot → HailerError
    def sync_soon(self) -> None                               # wake the background copy back (after a turn, on a notebook switch); returns at once.
    # The only sync hook: a no-op here and for the local kernel (its notebooks folder is the host's own); DockerKernel wakes its
    # notebook_sync.NotebookSync (kernel.sync), which DockerRuntime.start and DockerKernel.stop drive.
```
Every name goes through `notebooks.check_notebook_name` before any request (`test_names_are_checked_before_any_request`),
and `test_hailer_never_sends_a_path_outside_the_notebooks_folder` checks every path `fake_marimo.py` was asked about.
A docker kernel's paths cannot be resolved from the host, so the link check covers the unsafe-local runtime only; a docker
kernel's notebooks folder is the container's own, and what comes back passes `notebook_sync.sync_refusal`.

## `notebook_sync.py`  (2026-09-22)

The notebook copies between the workspace's notebooks folder and a sandbox with its own (the docker kernel's
tmpfs). Standard library only; uses only the sandbox's file operations, so a kernel elsewhere fits too.

```python
SYNC_INTERVAL_SEC = 15.0                   # the background pass; tests monkeypatch it (read when start() runs)
SYNC_DEADLINE_SEC = 30.0                   # the whole copy in at start and the last copy back at stop
SYNC_CYCLE_SEC = 10.0; SYNC_CYCLE_BYTES = 10 * MAX_NOTEBOOK_BYTES   # one background pass: time and bytes read
BACKUP_FOLDER = "notebook-backups"; TEMP_PREFIX = ".hailer-sync-"
TOOLING_NAMES = ("conftest.py", "test_*.py", "*_test.py", "setup.py", "noxfile.py", "tasks.py", "fabfile.py", "dodo.py", "conf.py",
                 "manage.py", "gunicorn.conf.py", "ipython_config.py", "jupyter_*_config.py", "sitecustomize.py", "usercustomize.py",
                 "__init__.py", "__main__.py")   # fnmatch patterns on the casefolded file name
COMMON_MODULES = ("hailer", "pandas", "numpy", "scipy", "matplotlib", ...)   # modules scripts commonly import
def module_names() -> frozenset[str]       # cached, casefolded: sys.stdlib_module_names, the kernel image's packages and COMMON_MODULES;
                                           # never what is installed with Hailer, so the rule is the same on every machine
SYNC_LOCK_SEC = SYNC_CYCLE_SEC + 5         # the most a copy waits for another under way (then a warning, no copy)
SYNC_STOP_SEC = 2 * SYNC_LOCK_SEC + SYNC_DEADLINE_SEC   # the most DockerKernel.stop waits for the last copy
SESSION_NEW_FILES = 200; SESSION_BYTES = 100 MiB   # per session: then new names are no longer copied back (one warning)
def sync_refusal(name: str, text: str | None = None) -> str | None
    # THE allow-list, both directions (None: may be copied): no backslash; notebooks.check_notebook_name accepts it (relative,
    # no drive/../empty part, no part starting with "." or "__", Windows-safe); ends in ".py" (that spelling); no part with "~"
    # and a digit (an 8.3 short name); at most MAX_NOTEBOOK_DEPTH folders; no TOOLING_NAMES match; no part (folder, or the file
    # without .py) in module_names(). With text: UTF-8 encodable, at most MAX_NOTEBOOK_BYTES, no NUL, contains "import marimo" and
    # "marimo.App(". Reasons: "not a notebook name", "not a .py file", "more than 4 folders deep", "a file name other tools run",
    # "the name of the Python module <part>" (shown in the warning line), "not UTF-8 text", "larger than 5 MiB", "not text",
    # "not a marimo notebook"
class NotebookSync:                        # NotebookSync(sandbox, folder, workspace, *, clock=time.time)
    # in memory: per name the digest last copied in/out (tells a change here from one in the sandbox) and the listing's
    # (modified, size) last seen (an unchanged notebook is not read again, so a pass's byte budget reaches the changed ones);
    # every copy holds one lock; every request of a copy runs under marimo_client.request_deadline
    def host_notebooks(self) -> tuple[list[str], list[tuple[str, str]]]   # (names to copy in, (name, reason) refused .py files);
                                           # never through links/junctions, never into "."/"__"/SKIPPED_FOLDERS folders; non-.py files ignored
    def sync_in(self) -> list[str]         # within SYNC_DEADLINE_SEC: write_notebook each allowed notebook (at most MAX_NOTEBOOKS); one
                                           # "Not copied into the kernel (only marimo notebooks with plain names are): <names, 5 shown>." line
    def sync_out(self, *, seconds: float = SYNC_CYCLE_SEC, budget: int = SYNC_CYCLE_BYTES) -> list[str]
        # list_tree → allowed, case-unique candidates (at most MAX_NOTEBOOKS); read the changed ones until the deadline or budget
        # ("Warning: copying the notebooks back from the kernel stopped after <s> s or <MiB> MiB; ..."); write here when the digest
        # differs: target checked (_target: folders created, never a link/junction, resolved inside the folder; NotebookPathError),
        # backup first when the file here differs from the last copy (".hailer/notebook-backups/<name>.<YYYYmmdd-HHMMSS>.py" + a
        # warning; no write when the backup fails), then _write_atomically (temp file + os.replace). Never deletes here. "Not copied
        # back from the kernel ..." names each refused entry once (folders as "<name>/"); failures are warnings shown once; never raises.
        # Sets complete: True when the pass got to its end with every notebook it had to copy written here (refused names aside),
        # False otherwise (the kernel did not answer, a write failed, the deadline or budget, the session cap, Ctrl+C, a pass under way)
    complete: bool | None                  # the latest sync_out's outcome (None before the first): what a silent repeat cannot say
    def start(self, interval: float | None = None) -> None   # daemon thread "hailer-notebook-sync": sync_out every interval or on soon()
    def soon(self) -> None
    def close(self, *, final: bool = True) -> list[str]   # stop the thread (a pass ends within SYNC_CYCLE_SEC), then (final)
                                           # sync_out(seconds=SYNC_DEADLINE_SEC) once more: one deadline for the whole last copy
    # Every write here also calls statedir.mark_sandbox_wrote; names in warnings have non-printable characters replaced by "?"
```


## `kernel.py`  (owner: docker kernel, 2026-09-19)

Where the notebook kernel runs, and what every runtime shares. Standard library only; imports
`marimo_client` and `sandbox`, and imports `kernel_docker` lazily (`runtime_for`), so most runs never load the docker code.
Every `uvx hailer` / `uvx hailer notebook` process starts its own kernel and stops it on exit; nothing finds
or attaches to a kernel another process started.

```python
LOCAL_LOG_PREFIX = "marimo-"              # .hailer/marimo-<pid>.log: the unsafe-local runtime's log, one per process, deleted on stop
START_TIMEOUT_SEC = 60.0                  # the start's wait, counted after any image pull
KERNEL_WORKDIR, KERNEL_NOTEBOOKS_DIR, KERNEL_DATA_DIR   # PurePosixPath: /work, /work/notebooks (docker: the kernel's own tmpfs), /work/data (read-only mount)
KERNEL_SECRET_SUFFIXES   # SECRET_ENV_SUFFIXES (_KEY _TOKEN _SECRET _PASSWORD) + _PASSWD _PWD _CREDENTIALS _CONNECTION_STRING APIKEY
KERNEL_SECRET_NAMES = ("PGPASSWORD", "MYSQL_PWD", "PASSWORD", "SECRET", "TOKEN")
LOCAL_PROMPT_NOTES; DOCKER_PROMPT_NOTES; DOCKER_NETWORK_PROMPT_NOTES   # what the model is told about the kernel
UNSAFE_LOCAL_HINT                         # the unsafe-local doctor row's hint: how to isolate notebook code

def new_token() -> str                                     # secrets.token_urlsafe(32)
def local_log_path(workspace, pid=None) -> Path            # .hailer/marimo-<pid>.log (this process by default)
def withheld_variables(config: HailerConfig, environ: Mapping[str, str]) -> list[str]
    # sorted names the unsafe-local server does not get: every provider env_key and OPENAI_API_KEY, every
    # env_http_headers variable, KERNEL_SECRET_NAMES, and names ending in KERNEL_SECRET_SUFFIXES (any case);
    # no setting lets one through. Exact names compare case-insensitively on Windows
def kernel_environment(config, environ) -> dict[str, str]                # environ minus withheld_variables
def describe_runtime(kernel: KernelConfig) -> str   # <contract> is kernel_image.contract_tag()
    # "docker (hailer-kernel <contract>; no network; data read-only)" |
    # "docker (hailer-kernel <contract>; network on: the internet and this machine; data read-only)" |
    # "unsafe-local (runs as you; not isolated)" | '<value> (not a kernel runtime: use "docker" or "unsafe-local")'
def runtime_prompt_notes(kernel: KernelConfig) -> str                     # LOCAL_ / DOCKER_ / DOCKER_NETWORK_PROMPT_NOTES; never touches Docker
def spawn_marimo(cmd, cwd, log_path, *, env=None, stdin_text=None) -> Popen   # background; fresh owner-only log (O_TRUNC); token written to stdin, then closed; own process group on Windows
def attach_marimo(cmd, cwd, *, env=None, stdin_text=None) -> Popen            # --foreground: output in this terminal
def kill_tree(pid); stop_process(proc, timeout=5.0, *, kill_tree=kill_tree)   # never raise
def mask_token(lines, token) -> list[str]                                 # token replaced by "<token>" (local log and docker logs tails)
def log_tail(path, lines=15, *, token=None) -> list[str]                  # the last lines, masked
@dataclass class LocalProcesses: spawn, attach, kill_tree, wait_for_health   # the OS layer; tests replace members

@dataclass class LocalKernel(MarimoSandbox): proc; log_path; procs
    # notebooks_path = notebook_file_key(notebooks_root), data_path = str(data_dir), native_paths = True
    # stop: its process (tree on Windows), then the log (it holds the signed-in URL)
    # wait: "marimo shut itself down." (exit 0) / "marimo stopped (exit code N): it was ended from outside
    # this terminal (uvx hailer kernel stop, Task Manager or kill), or it failed; its own output is above."

class KernelRuntime(Protocol):
    name: str                              # "docker" | "unsafe-local"
    def check(self) -> list[Check]         # rows for doctor and hailer notebook
    def prepare(self, say: Callable[[str], None] | None = None) -> None   # slow steps and warnings before any spinner (docker: pull a missing image); start() runs it when not done
    def start(self, port: int, *, foreground: bool = False) -> MarimoSandbox   # starts a new server on 127.0.0.1:port and waits until it answers with this start's token (its own process/containers checked first); KernelRuntimeError (log tail in the hint) and nothing left running on failure
    def describe(self) -> str                                             # the text after "Kernel:"
class LocalRuntime:                        # runtime = "unsafe-local"; LocalRuntime(config, *, procs=None, environ=None, token_factory=new_token, start_timeout=60.0,
                                           #   confirm=ask_yes_no)
    # (the class keeps its name: it is the runtime on this machine; "unsafe-" is the setting's warning)
    # check: one warning "kernel" row (ok=False, fatal=False): describe() + how many variables are withheld,
    #   hint UNSAFE_LOCAL_HINT. kernel_checks(starting=True) drops it: the startup's yellow "▲ Kernel" row says it instead
    # prepare: when statedir.sandbox_wrote_notebooks, a warning line, then confirm("Run them on this machine anyway? [y/N] "):
    #   True clears the mark; False → KernelRuntimeError("Not started: ... were not approved ..."); None (no terminal) →
    #   KernelRuntimeError("Not started: ... no terminal to ask ...", hint: run uvx hailer notebook --kernel unsafe-local in a
    #   terminal and answer yes, or keep --kernel docker)
    # start: marimo_server_command + kernel_environment + the token on stdin; log .hailer/marimo-<pid>.log (None in
    #   foreground); wait_for_health(token=its token, should_stop=its process ended); "Marimo
    #   exited early (code N)." / "Marimo did not answer on <url> within 60 s." with "Last lines of its log:" and the masked
    #   tail otherwise (the log is deleted). No state file: a Hailer that is killed leaves its token-protected marimo
    #   running, to be ended with Task Manager or kill
def runtime_for(config: HailerConfig, runner=None) -> KernelRuntime   # kernel_docker.DockerRuntime | LocalRuntime; ConfigError (config.runtime_problem) for "local" or an unknown runtime; never falls back
```

## `kernel_docker.py`  (owner: docker kernel, 2026-09-19)

The docker runtime. Imports `kernel` and `kernel_image`; imported only when docker is asked for.

```python
KERNEL_PORT = 2718; IMAGE_UID = IMAGE_GID = 1000; KERNEL_HOME = "/home/analyst"
NOTEBOOKS_TMPFS_SIZE = "512m"               # the kernel's notebooks folder: a tmpfs owned by its user
KERNEL_TOKEN_FILE = PurePosixPath("/run/secrets/hailer-token")   # where the kernel reads its marimo token
LABEL_WORKSPACE = "org.openafterhours.hailer.workspace"; LABEL_CONTRACT = "...contract"; LABEL_ROLE = "...role"
LABEL_OWNER = "...owner"                   # the OwnerLock id of the Hailer that started it; all four on every container and network
OWNER_LOCK_PREFIX = "owner-"; OWNER_LOCK_GRACE_SEC = 10.0
DOCKER_TIMEOUT_SEC = 60.0; DOCKER_PROBE_TIMEOUT_SEC = 20.0
DOCKER_NOT_INSTALLED = "Docker is not installed."; DOCKER_NOT_RUNNING = "Docker is not running."
class DockerRunner(Protocol):              # args never include "docker"; both raise docker_not_installed() when there is no docker
    def run(self, args, *, timeout=None, check=False) -> CompletedProcess[str]     # captured text; check: non-zero exit → KernelRuntimeError quoting docker
    def stream(self, args, *, keep_errors=False) -> CompletedProcess[str]          # output in this terminal (pull, build, logs -f); keep_errors keeps docker's last 20 error lines in .stderr
class SubprocessDockerRunner:              # the docker CLI on PATH (shutil.which), stdin empty, UTF-8
DOCKER_INSTALL_ADVICE                      # where to get Docker Desktop / Docker Engine (docs.docker.com links); also init's next steps
UNSAFE_LOCAL_OPTION                        # the last hint line of every "no usable Docker" error: runtime = "unsafe-local" in
                                           # hailer.toml (or HAILER_KERNEL), "only if you accept that notebook code then runs as you, ..."
def without_unsafe_local_option(hint) -> str   # hint minus that line: `hailer kernel pull|build|stop` manage Docker itself
def docker_not_installed() -> KernelRuntimeError          # hint: DOCKER_INSTALL_ADVICE, then UNSAFE_LOCAL_OPTION
def docker_not_running(detail="") -> KernelRuntimeError   # hint names DOCKER_CONTEXT / `docker context use default` when docker said "context"; ends with UNSAFE_LOCAL_OPTION
# engine_version() on Windows containers: "Docker runs windows containers; ..." with the switch advice and UNSAFE_LOCAL_OPTION
def docker_said(result) -> str             # "docker said: <last 3 lines>" or ""
def workspace_id(workspace) -> str         # 10 hex chars of sha256(normcase(resolved path))
def docker_names(workspace, suffix=None) -> DockerNames # hailer-kernel-<id>-<suffix>, hailer-fwd-..., hailer-net-...; suffix: 6 random hex chars (unique per start)
def mount_arg(source, target, *, readonly=False) -> str   # --mount type=bind,source=...,target=...[,readonly] (CSV-quoted)
def linux_host_user() -> tuple[int, int] | None           # (uid, gid) on a Linux host unless root, so the kernel reads data files only
                                                          # the user may read (0600 exports) through the read-only mount; None elsewhere
def write_token_file(workspace, token) -> Path            # .hailer/kernel-token-<random>/token: folder 0700 on POSIX (on Windows the
                                                          # mode is ignored; the workspace's ACL is inherited), file 0644 (the kernel's
                                                          # uid may differ: root host, rootless Docker); the start deletes it in a finally
def remove_token_folder(folder) -> str | None             # deletes it; a warning line when that fails (quiet when already gone)
@dataclass class OwnerLock: id (16 hex); path (.hailer/owner-<id>.lock); fd
    def release(self) -> None              # unlock, close, delete; idempotent, never raises
def acquire_owner_lock(workspace) -> OwnerLock   # create the file (O_EXCL) and take an exclusive non-blocking OS lock
                                                 # (msvcrt.locking on Windows, fcntl.flock elsewhere), held for the process's lifetime
                                                 # unless released: the OS drops it however the process ends, so a reused pid never matters
def owner_alive(workspace, owner_id) -> bool     # the lock file exists and cannot be locked; False for a missing file, a free lock or a
                                                 # label that is not an owner id (no label: an older Hailer's object)
def sweep_owner_locks(workspace, *, grace=10.0) -> None   # delete the lock files nobody holds, unless younger than grace (being created)
def network_location(path, drive_type=...) -> str | None   # Windows: UNC_PATH ("a network share (UNC path)") or "a mapped network drive (Z:)" (GetDriveTypeW == 4)
def data_path_problems(data_dir, *, windows=None, drive_type=None) -> list[str]
    # warnings with the fix: a mapped network drive; symlinks/junctions leading outside (first 5000 entries).
    # Nothing for a UNC path: that is fatal, said once by config.docker_mount_problems
@dataclass class Removal: removed, gone, failed: list[str]
    # one container (rm -f) or network (network rm) at a time, by id. Gone: docker's object-specific "No such container /
    # No such network / no such object / network <x> not found" (never a bare "not found": a broken docker context says
    # "context not found"), and "removal ... is already in progress" (another terminal removing it). "has active
    # endpoints" is retried (NETWORK_RM_RETRIES x NETWORK_RM_RETRY_SEC). Before anything is reported as failed,
    # `docker inspect --type container|network` checks it is still there
@dataclass class DockerKernel(MarimoSandbox):
    # notebooks_path = "/work/notebooks", data_path = "/work/data", data_dir = the host data folder, native_paths = False
    runner; containers; container_ids; network; network_id; owner (OwnerLock)   # what this session started; removed and inspected by id
    sync: NotebookSync | None              # set by start after the copy in; None again once stop made the last copy
    def sync_soon(self) -> None            # sync.soon() (nothing without one)
    def writable_name(self, name) -> str   # MarimoSandbox's rules, then NotebookPathError ("... would stay in the docker kernel and be
                                           # lost when it stops (<reason>).") for a name notebook_sync.sync_refusal refuses (json.py, conftest.py, ...); else MarimoSandbox's
    def remove(self) -> Removal            # containers, then the network
    def stop(self) -> None                 # sync.close(final=True) on a daemon thread, joined at most SYNC_STOP_SEC (then a warning; Ctrl+C
                                           # there: a warning), then remove (always), then
                                           # release the owner lock; stop_error explains a failure ("... Run uvx hailer kernel stop to retry.")
    copied_back: bool | None               # after stop: sync.complete of the last copy (False when it timed out or Ctrl+C cut it; None: no sync)
    def log_tail(self, lines=15, *, container=None) -> list[str]   # docker logs --tail, token masked
    def wait(self) -> int                  # docker logs -f; then sets ended from docker inspect: removed from outside, out of memory, exit 137, "exited (code N)"
@dataclass(frozen=True) class Labelled: id; name; state ("" for a network); role; owner
    running: bool                          # state in running/restarting/paused/removing
class DockerRuntime:
    # DockerRuntime(config, *, runner=None, token_factory=new_token, start_timeout=60.0, health=None, user=linux_host_user, suffix=None)
    name = "docker"; names: DockerNames (unique per object); image: str (config.kernel.effective_image)
    owner_id: str                          # the current start's OwnerLock id (its owner label)
    def check(self) -> list[Check]         # kernel; docker ("Docker <v> (Linux engine)" or the error); image (missing = non-fatal warning,
                                           # another kernel contract = fatal; "<image> (kernel contract <c>)"); data (path warnings; no row for a UNC folder: the config row fails it).
                                           # No notebooks row: the notebooks folder is never mounted
    def engine_version(self) -> str        # docker version --format "{{.Server.Version}} {{.Server.Os}}"; not installed / not running / non-Linux engine → KernelRuntimeError
    def engine_cpus(self) -> int | None
    def pull(self, say=None) -> str        # hailer kernel pull: always downloads, then checks the contract label; returns it
    def mount_problems(self) -> list[str]  # config.docker_mount_problems(config, windows=os.name == "nt"): the same rules as validate()
    def prepare(self, say=None) -> None    # mounts; engine; the image (pulled with progress when missing: "Downloading the
                                           # kernel image <image> (first use; this can take a few minutes) ..."), its contract label
                                           # (another: "The kernel image <image> has kernel contract <c>; this Hailer needs <c'>."); cpus clamped to engine_cpus with a note
    def network_command(self) -> list[str] # network create --internal + labels
    def kernel_command(self, port: int, token_file: Path) -> list[str]
        # run -d --pull never --name hailer-kernel-<id>-<suffix> (--network hailer-net-<id>-<suffix> | -p 127.0.0.1:<port>:2718 with network=true)
        # --init --read-only --tmpfs /tmp --tmpfs /home/analyst:uid=U,gid=G --tmpfs /work/notebooks:uid=U,gid=G,mode=0700,size=512m
        # [--user U:G -e HOME=/home/analyst on Linux]
        # --cap-drop ALL --security-opt no-new-privileges --pids-limit 256 --memory M --memory-swap M --cpus C -w /work
        # labels (workspace, contract, role, owner)
        # --mount <data_dir>:/work/data,readonly --mount <token_file>:/run/secrets/hailer-token,readonly <image>
        # marimo edit notebooks --host 0.0.0.0 --port 2718 --headless --skip-update-check --token-password-file /run/secrets/hailer-token
        # The token is never an argument or environment variable (docker inspect shows both)
    def forwarder_command(self, port: int) -> list[str]
        # create --pull never --name hailer-fwd-<id>-<suffix> -p 127.0.0.1:<port>:2718 --init --read-only --cap-drop ALL
        # --security-opt no-new-privileges --pids-limit 64 --memory 64m labels <image> python -m hailer._forward hailer-kernel-<id>-<suffix> 2718 2718
    def labelled_containers(self) -> list[Labelled]; def labelled_networks(self) -> list[Labelled]   # docker ps -a / network ls --filter label=<workspace>
    def remove_leftovers(self, *, every: bool = False) -> Removal
        # containers, then networks. every=False (a start): those whose owner is not alive (owner_alive), running or not;
        # a live owner's objects always stay. every=True (kernel stop): everything labelled
    def start(self, port: int, *, foreground: bool = False) -> MarimoSandbox   # a DockerKernel
        # prepare if needed; mounts again; acquire_owner_lock (the DockerKernel holds it; released on failure or stop);
        # sweep_owner_locks; remove_leftovers() (a failure is a printed warning: the new names never clash); mkdir the data folder;
        # network (unless network=true), kernel, forwarder (create, network connect, start); wait_for_health through the
        # forwarder with this start's token (should_stop: one of its containers stopped); the token folder is deleted once that
        # wait ends (success, failure or Ctrl+C); then NotebookSync.sync_in and .start(), before
        # anything can open a notebook (Ctrl+C there stops the kernel). No state file: the returned DockerKernel is the only
        # handle. Any failure or Ctrl+C removes what was created. foreground changes nothing (DockerKernel.wait follows the log)
def workspace_kernels(config, runner=None) -> list[str]
    # hailer status, whatever [kernel] runtime says, when the docker CLI is there: one line per running kernel container of
    # this workspace ("docker <name> (its Hailer session is running)" / "(its Hailer session has ended; the next start removes
    # it)"); "docker: not checked (<error>)" when the engine cannot be asked; [] without Docker. Never raises
@dataclass class StopReport: done: list[str]; failed: list[str]
def stop_workspace_kernels(config, runner=None) -> StopReport
    # hailer kernel stop, whichever session started what: remove_leftovers(every=True) ("Removed <names>."); nothing in .hailer
    # is touched (a session still starting keeps its lock and token folder); local kernels are recorded nowhere; nothing is
    # copied back (a killed session's last changes are lost). Docker not running → a done line saying containers were not checked;
    # not installed → nothing; any other Docker error → failed. Never raises. The CLI prints done and failed (red, exit 1)
```

## `kernel_image.py`  (owner: docker kernel, 2026-09-19; kernel contract 2026-09-22)

An image is defined by its **kernel contract**, not by Hailer's version, so a release that changes nothing in
the image reuses the published one.

```python
MARIMO_VERSION = "0.24.2"                   # equal to the marimo pin in pyproject.toml (checked by a test)
IMAGE_PACKAGES = {"marimo": MARIMO_VERSION, "polars": ..., "fastexcel": ..., "duckdb": ..., "ruff": ..., "altair": ..., "plotly": ...}  # exact versions, one place
IMAGE_MODULES = ("errors.py", "periods.py", "_forward.py")   # the hailer modules the image gets (they import no other hailer module)
IMAGE_INIT: str                             # the image's hailer/__init__.py (a docstring: no __version__)
CONTRACT_LABEL = "org.openafterhours.hailer.kernel-contract"  # must equal contract_tag() before Hailer uses an image
def contract_tag() -> str                   # "marimo<MARIMO_VERSION>-<contract_fingerprint()[:12]>", e.g. "marimo0.24.2-64a6b78f25cf"
def default_image() -> str                  # ghcr.io/openafterhours/hailer-kernel:<contract_tag()> (models.KERNEL_IMAGE_REPOSITORY)
def package_dir() -> Path                   # the installed hailer package (where IMAGE_MODULES come from)
def context_files() -> dict[str, bytes]     # "Dockerfile", "hailer/__init__.py" and hailer/<IMAGE_MODULES>, LF endings
def contract_fingerprint() -> str           # sha256 over context_files() and IMAGE_PACKAGES (line endings do not count)
def build_args() -> dict[str, str]          # <NAME>_VERSION per IMAGE_PACKAGES entry + KERNEL_CONTRACT=contract_tag(); never the host's versions
def prepare_context(dest: Path) -> dict[str, str]   # writes context_files() into dest; returns build_args(); dest must not hold hailer/
def build_command(tag, context, args) -> list[str]  # ["build", "--tag", tag, "--build-arg", ..., context]
def build(tag: str, runner: DockerRunner, *, base_image=None, pip_config=None, pip_cert=None, no_cache=False, no_host_config=False, say=None) -> int
def configured_build_options(*, base_image=None, pip_config=None, pip_cert=None, no_cache=False, no_host_config=False, say=None, dry_run=False)  # context manager yielding Docker options; owns temporary secrets
def pull(image: str, runner: DockerRunner) -> CompletedProcess[str]   # docker pull streamed, errors kept
def image_contract(image: str, runner: DockerRunner) -> str | None   # CONTRACT_LABEL; "" when absent (an image from before contracts); None when not on this machine
```

`src/hailer/docker/Dockerfile` is package data: `python:3.12-slim` pinned by digest, `pip install` of
marimo, Polars, fastexcel, DuckDB, altair and plotly at the build args (no defaults), the image modules
under `/opt/hailer-package` on the venv's path, user `analyst` (uid 1000), `/work/.marimo.toml`
(`[runtime] auto_instantiate = true`) and an empty `/work/hailer.toml`, `WORKDIR /work`, and the contract
label. Any change to an input (even a comment) is a new tag. `scripts/build_kernel_image.py` builds the same context
with `docker buildx` (`--platform`, `--tag` (default `default_image()`), `--push` / `--load`,
`--if-missing` (needs `--push`; `docker buildx imagetools inspect --format "{{json .Manifest}}"`: every tag
published with every `--platform` → build nothing; the registry says none exists ("not found", "manifest
unknown") → build; any other error, a tag missing a platform, or only some tags published → exit 1 without
building, so a published tag is never overwritten), `--context`, `--dry-run`, arguments after `--` passed on) for CI and the release.

`kernel_packages.discover()` reads portable pip settings (environment over merged global/user/site
files), falling back to a single uv default index (environment over system/user/nearest-project
configuration). Returns `PackageSettings(options, cert)` with sensitive fields excluded from repr.
No subprocess or third-party dependency is required. Invalid configuration raises a sanitized
`KernelRuntimeError`; local wheel paths and uv routing that pip cannot preserve require an explicit
configuration. Only selected package settings cross into the build, as temporary BuildKit secrets
outside its context. Explicit `pip_config` or `no_host_config` bypasses discovery; explicit `pip_cert`
overrides the discovered CA. CLI and buildx share this lifetime. Dry-run commands use a placeholder
for generated secrets: rerun the script without `--dry-run` to execute them.

## `_forward.py`  (owner: docker kernel, 2026-09-19)

```python
async def pipe(reader, writer) -> None      # copy until EOF, then half-close
def make_handler(host: str, port: int) -> Handler   # pipes each client to host:port; drops the client when the kernel does not answer
async def start(host, port, listen, *, bind="0.0.0.0") -> asyncio.Server
async def serve(host, port, listen, *, bind="0.0.0.0") -> None
def parse_args(argv) -> tuple[str, int, int]   # SystemExit with the usage otherwise
def main(argv=None) -> int                  # python -m hailer._forward <host> <port> <listen-port>
```
Standard library asyncio TCP forwarding (HTTP, the browser's websocket and the streamed execute calls
alike). It only ever connects to the one address it was given, so notebook code gains nothing by reaching
it.

## `notebooks.py`  (owner: notebook feature / foundation)

Notebook names, the active notebook and templates, shared by the CLI and the agent's tools. Standard library
only; never imports marimo, typer or rich. The files are the sandbox's; the only file this module writes in
the notebooks folder is the starter notebook of workspace setup (`ensure_notebook`).

```python
STATE_FILENAME = "notebook.json"           # <workspace>/.hailer/notebook.json: {"version": 2, "active": "<name>", "recent": ["<name>", ...]}
STATE_VERSION = 2
RECENT_LIMIT = 10
RESERVED_NAMES: frozenset[str]             # Windows device names: con prn aux nul conin$ conout$ com1-9 lpt1-9 com¹²³ lpt¹²³
NAME_PART_MAX_LEN = 255
def same_name(a: str, b: str) -> bool      # names ignore case on Windows
STARTER_TEMPLATE: str; EMPTY_TEMPLATE: str  # __HAILER_VERSION__ / __HAILER_TITLE__ placeholders, filled by render_template
TEMPLATE_KINDS = ("starter", "empty")
def check_notebook_name(name: str) -> str
    # either separator accepted, returned with "/"; NotebookPathError unless: relative (no leading "/", no drive),
    # no "", ".", ".." part, no part starting with "." or "__", none of <>:"|?* or control characters (":" would
    # name an NTFS stream) or invisible formatting characters (Unicode Cc/Cf), no part ending in "." or " " or longer
    # than NAME_PART_MAX_LEN, no RESERVED_NAMES stem in any part, a ".py" file name
def notebook_name(config: HailerConfig, path: Path | str) -> str | None   # a host path (absolute, or workspace-relative) as a name in notebooks_root; path arithmetic only
def default_notebook(config: HailerConfig) -> str                        # config.notebook as a name (its file name when outside the folder)
def state_path(workspace: Path) -> Path
def load_active_notebook(config: HailerConfig) -> str
    # ONLY the state file (never the environment or the notebooks folder; see the ground rule): its valid "active"
    # name, else default_notebook. A file that cannot be read or has another "version" (an older Hailer's host
    # paths) is ignored. Whether the notebook still exists is the sandbox's to say
def load_recent(config: HailerConfig) -> list[str]                 # names, most recent first
def save_active_notebook(config: HailerConfig, name: str) -> None
    # checks the name; atomic (tmp + replace); writes the current format;
    # "recent" most-recent-first, unique, max 10; creates .hailer/ via statedir.ensure_state_dir
def notebook_display_name(config: HailerConfig, path: Path) -> str   # a host path, workspace-relative posix ("notebooks/q2_churn.py"), absolute posix when outside
def slugify(name: str) -> str                              # "Q2 Churn (draft).py" → "q2_churn_draft"; must start with a letter (else "nb_" prefix); ValueError when empty
def reference_folders(config: HailerConfig, *kernel_paths: str) -> tuple[str, ...]   # prefixes a reference may start with: the kernel's folder paths, the host folder, its workspace-relative form
def resolve_notebook(ref: str, names: Sequence[str], *, folders: Sequence[str] = ()) -> str
    # map a reference to one of names (from MarimoSandbox.list_notebooks): a bare name ("q2 churn"), file name, a path inside
    # the folder (either separator, with or without .py), any of those after one of folders. Tried: exact, ignoring
    # case, "<slug>.py", then a unique file-name match in any folder. An absolute path or drive left after the folders
    # or a ".." part → NotebookPathError ("outside the notebooks folder"); nothing → NotebookNotFoundError (hint lists
    # up to 10 names); several matches → NotebookNotFoundError listing them
def marimo_version() -> str                                # importlib.metadata.version("marimo"), fallback "0.24.2"
def render_template(kind: Literal["starter", "empty"], *, title: str, version: str | None = None) -> str
    # starter = the cells of notebooks/analysis.py with the title as heading and WORKSPACE found by walking up to
    # pyproject.toml / hailer.toml; empty = marimo's own empty notebook. Title is made safe for the f-string markdown.
def new_notebook(name: str, *, kind: Literal["starter", "empty"] = "starter") -> tuple[str, str]
    # ("<slug>.py", its source); NotebookPathError when the name has no letters/digits. The caller writes it with
    # MarimoSandbox.write_notebook(..., replace=False)
def ensure_notebook(path: Path, *, title: str, kind: Literal["starter", "empty"] = "starter") -> bool
    # workspace setup (`hailer init`): write the template to exactly `path` (parents created) unless something exists
    # there; True when it wrote. Never overwrites; OSError when it cannot write.
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

An image module (`kernel_image.IMAGE_MODULES`): it imports only `hailer.errors` of Hailer, so it defines its
own types (they lived in `models.py` before the kernel contract).

```python
@dataclass(frozen=True, order=True) class Period: year: int; month: int   # .label "2025-03", .short "25-03"
@dataclass(frozen=True) class PeriodFile: path: Path; period: Period; stem: str   # stem lower-cased, e.g. "sales"
PERIOD_RE: re.Pattern            # ^(?P<yy>\d{2})-(?P<mm>\d{2})\s+(?P<stem>.+)$ on the file stem
DATA_SUFFIXES: tuple[str, ...]   # .csv .tsv .parquet .json .jsonl .ndjson .xlsx .xls .xlsb .arrow .feather .ipc
def list_data_files(data_dir: Path) -> list[Path]  # every data file directly in data_dir, any name, sorted by name; [] when missing
def parse_period(name: str | Path) -> Period | None
def parse_period_file(path: Path) -> PeriodFile | None
def scan_period_files(data_dir: Path, name: str | None = None, *, suffix: str = ".parquet") -> list[PeriodFile]  # sorted by period
def load_periods(files: Sequence[PeriodFile], *, columns: Sequence[str] | None = None) -> pl.DataFrame  # adds "period" (str YYYY-MM) column, concat how="diagonal_relaxed"
def scan_periods(files: Sequence[PeriodFile]) -> pl.LazyFrame        # lazy variant
def duckdb_periods_view(con: duckdb.DuckDBPyConnection, files: Sequence[PeriodFile], view_name: str = "periods") -> str  # read_parquet([...], union_by_name=true, filename=true) + derived period column; returns view name
def describe_periods(files: Sequence[PeriodFile], schema: Callable[[PeriodFile], Mapping[str, Any]] | None = None) -> str
    # compact text: count, first/last period, per-file schema diff summary; schema reads a file's columns (default: the
    # Parquet schema at pf.path; list_periods passes the sandbox's data_schema)
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
TOOL_NAMES: tuple[str, ...]      # the 12 names in the table, in the order they are offered to the model
DEFAULT_SESSION_WAIT_SEC = 30.0
NOTEBOOK_LINK_TIP = "The user can also run /notebook in the chat to get the link."
class HailerTools:               # one plain method per tool, each -> str (a ToolReply); the method's docstring is the description the model sees
    def __init__(self, config: HailerConfig, sandbox: MarimoSandbox | None = None, *, open_url: UrlOpener | None = None, session_wait_sec: float = DEFAULT_SESSION_WAIT_SEC) -> None
    # open_url: the chat's opener (HailerAgent(open_url=...)); with --no-browser it opens nothing and returns False
class ToolReply(str):            # the reply text, equal to it, with how the call went beside it
    status: dict[str, Any]       # {"ok": bool, ...}, see below
def hailer_tools(config: HailerConfig, *, sandbox: MarimoSandbox | None = None, open_url: UrlOpener | None = None, session_wait_sec: float = DEFAULT_SESSION_WAIT_SEC) -> list[Any]
    # LangChain StructuredTool.from_function(func=call, coroutine=_in_daemon_thread(call), name=name, description=inspect.getdoc(method),
    #   response_format="content_and_artifact"), in TOOL_NAMES order; call = _content_and_artifact(method) (functools.wraps keeps the
    #   signature the schema is read from) returns (str(reply), reply.status)
    # UrlOpener = Callable[[str], bool]
```

**Status beside the text.** Every reply is a `ToolReply`, which equals its text (callers comparing text are
unaffected), and LangChain gets the text as the tool message's content (byte for byte, a plain `str`) and
`status` as its artifact, which langchain-openai leaves out of both wire formats (`test_agent.py` checks the
requests). `status` is decided where the tool knows, never from the text (which holds whatever the user's code
printed or a file contained): `_run` sets `{"ok": False, "error": <HailerError subclass name>}` for a
`HailerError`, `{"ok": False, "error": <exception type name>, "unexpected": True}` for any other exception,
and `truncated: True` when the text was cut to `max_tool_output_chars`; `marimo_execute` sets
`{"ok": False, "error": "execution_failed"}` from the kernel's `success` flag, and `{"ok": False, "error":
"cell_error", "raised": [<exception class names>]}` when the code ran but a cell it created or edited raised
(the after snapshot's `Cell.error`, from the kernel's graph; a cell that raised before and was not changed does
not count); `notebook_open`, `notebook_create` and `notebook_close` add `notebook` (the name), `active`
(whether it is the active notebook now) and `session` (whether it has a kernel session after the call); a
failed cell listing or check snapshot is `error: "cells_unreadable"` and an unknown template
`error: "unknown_template"`. A reply with code checks adds `findings` (count), `checked` and `skipped` (`ruff` /
`ty`), and `marimo_execute` adds `cells` whenever it compared snapshots (the changed cells' names, `""` for an
unnamed one; `[]` when none changed). `fetch_page` calls `web.fetch_page` with the output cap off and lets `_run`
apply it (the same `truncate_text`, so the same text), so a cut page has `truncated: True`.
`hailer.tool_outcomes` reads it.

The kernel is `sandbox`, the one this session started (kept in memory; nothing is discovered). Without one
every kernel tool answers `ERROR: marimo is not running: this session has no kernel` with `launch_hint()`.
Clients come from `sandbox.client(notebook=<active>)` (never `token_in_links`). Every file the tools know
comes from the sandbox (names; `list_notebooks`, `write_notebook`, `list_data`, `data_schema`): `hailer.tools`
names neither folder and calls no file-system API. **Tool results never carry the token:** every URL in them
is token-free (`sandbox.notebook_url(name)`), and wherever a tool asks the model to have the user
open a URL (a `NoSessionError`, a bring-up without a session, `marimo_status` without a session) the text
adds `NOTEBOOK_LINK_TIP`. The browser Hailer opens itself gets the signed-in URL.
`_in_daemon_thread(fn)` is the async path the agent uses: the blocking call runs on a daemon thread
(`hailer-tool-<name>`) and its result comes back through `loop.call_soon_threadsafe`. On the loop's default
executor the thread would be joined when the loop closes and at interpreter exit (measured: quitting after
Ctrl+C waited 7 s for a call with 7 s left). So a cancelled turn and process exit never wait for a kernel
call; the abandoned call ends in the background and its result is dropped.

The **active notebook is re-read from `.hailer/notebook.json` on every call** (`HailerTools._active()` =
`notebooks.load_active_notebook`, a name); nothing about it is cached. Notebooks appear in results by name.
Every kernel tool acts on the active notebook. Nothing is raised into the agent loop: a `HailerError` becomes
`ERROR: <message>` plus the hint, anything else `ERROR: unexpected <Type>: <message>`. Tools (all return plain
text, truncated to `config.max_tool_output_chars` with head/tail and a `[... truncated N chars ...]` marker):

| tool | args | behaviour |
|---|---|---|
| `marimo_execute` | `code: str` | run in the scratchpad against the active notebook's session; returns stdout/output/stderr (rich mimetypes such as text/html or application/json are replaced by a short placeholder, so HTML never reaches the model); a failed run starts `Execution failed.`; `MarimoUnavailableError`/`NoSessionError` become `ERROR:` text with the hint, not exceptions. Code that mentions `create_cell` or `edit_cell` (with any `[checks]` part on) is bracketed by two `code_checks.snapshot_code` runs, the first with `enable_format=config.checks.format`; when cells changed and `lint` or `typecheck` is on, the `code_checks.report` for the changed cells follows the result after a blank line, the result shortened so the report survives the output cap. A snapshot that cannot be read skips the checks, never the code |
| `notebook_check` | none | a snapshot of the active notebook, then `code_checks.check` over every non-empty cell and the `report` (`Checks (ruff, ty) on the notebook's N cells: ...`); `[checks]` lint and typecheck both off → says so; unreadable cells → `Could not read the notebook's cells.` + the output |
| `marimo_status` | none | server url; `kernel: <sandbox.describe()>` and `kernel paths: notebooks folder <notebooks_path>, data folder <data_path>; use these in code` (`/work/...` for a docker kernel, host paths for local); `active notebook: <name> -> session <id> (ready)` or `... has NO session. Ask the user to open <url>`; every session with its notebook and a marker: `(active notebook)`, `(another notebook in the notebooks folder; notebook_open switches to it)`, `(outside the notebooks folder)`, `(unsaved)`; when marimo is down (no sandbox, or the session's kernel stopped and the first request raises `MarimoUnavailableError`) the text is `marimo: not running` / `active notebook: <name>` / error / hint, so the active notebook is always named |
| `notebook_cells` | `pattern: str = ""` | runs `marimo_client.LIST_CELLS_CODE`, a `cm` snippet listing the active notebook's cells (id, name, status, error count, first line); optional case-insensitive substring filter |
| `notebook_list` | none | `sandbox.list_notebooks()`: `Notebooks in <notebooks_path> (...)`, then one line per notebook `  - <name>  modified YYYY-MM-DD HH:MM  <size>  [active] [open]` (`[open]` = has a session; when marimo is unreachable for sessions a trailing line says so with the error and hint); empty folder → `No notebooks in <notebooks_path> yet. Create one with notebook_create(name).` |
| `notebook_create` | `name: str, template: str = "starter"` | `notebooks.new_notebook` + `sandbox.write_notebook(..., replace=False)` (+ `save_active_notebook`), then bring-up (below); text: `Created <name> from the <kind> template; it is now the active notebook.` + session line + a reminder of what the template defines; unknown template → `ERROR: unknown template ...`; existing name / unusable name → the `NotebookExistsError` / `NotebookPathError` text |
| `notebook_open` | `notebook: str` | `notebooks.resolve_notebook(ref, <sandbox.list_notebooks() names>, folders=notebooks.reference_folders(config, sandbox.notebooks_path))` (the model sees `/work/notebooks/...` in docker mode and may pass it back) (+ `save_active_notebook`), bring-up, then when a session exists the cell listing (`Cells (id, name, status, errors, first line):`); text starts `<name> is now the active notebook.`; unknown / outside-folder references → the `NotebookNotFoundError` / `NotebookPathError` text and the active notebook is unchanged |
| `notebook_close` | `notebook: str = ""` | resolve like `notebook_open` (empty = active), `client.resolve_session`, `client.shutdown_session(id)`; says the kernel is freed and, for the active notebook, that it stays active but must be reopened; no session → `<name> is not open (no kernel session), so there is nothing to close.`; marimo down → `marimo is not running, so <name> has no kernel session to close.` + error + hint |
| `list_periods` | `name: str = ""` | the `YY-MM <dataset>.parquet` names in `sandbox.list_data()` (optionally one dataset), summarised by `periods.describe_periods(..., schema=<sandbox.data_schema>)` (count, span, common columns, columns in some periods, type changes), plus a line naming the other data files with a `periods.DATA_SUFFIXES` type (at most 20); names the data folder as notebook code reaches it (`data_path`: `/work/data` for docker); a missing folder → `ERROR: Data directory does not exist: <data_path>` |
| `load_skill` | `name: str` | `read_skill` |
| `read_skill_file` | `name: str, path: str` | `read_skill_file` |
| `fetch_page` | `url: str` | `web.fetch_page`; denial text lists allowed domains |

Bring-up (`HailerTools._bring_up(notebook)`): if the notebook already has a session it is reused and the browser is
not opened; otherwise the signed-in `sandbox.notebook_url(name, with_token=True)` is opened (the
outcome text reports the token-free URL) through `HailerTools.open_url` (default:
`hailer.browser.open_url`, i.e. `os.startfile` first on Windows and stdio-silenced launchers elsewhere, so
nothing writes into the chat) and `wait_for_session` polls for up to `HailerTools.session_wait_sec` (default 30 s).
Outcomes are reported as text (`kernel session ready (<id>)`, `no kernel session appeared within N s` + URL,
`Could not open a browser` + URL, `marimo is not running (or not reachable)` + error + hint); the browser never
raises. Every `HailerError` raised while resolving or waiting for the session (a kernel that stopped first shows
up there) is folded into that last outcome: the notebook
was already created / made active by then, so `notebook_create` / `notebook_open` still begin with `Created ...` /
`... is now the active notebook.` and never read as a failed call. Both hooks are keywords of `HailerTools`
and `hailer_tools` so tests never launch a browser.

## `tool_outcomes.py`  (chat terminal, 2026-10-28)

```python
SUMMARY_LIMIT = 80
class Outcome(NamedTuple): ok: bool; summary: str = ""
def tool_outcome(name: str, args: Mapping[str, Any] | None, content: Any, status: Any = None, *, error: bool = False) -> Outcome  # never raises
```
The `ok` and `summary` of the agent's `tool_result` events (standard library only). **`ok` never comes from the
text**: `error=True` (a `status="error"` `ToolMessage`, LangChain could not run the tool) is a failure worded by
LangGraph's template (`invalid arguments`, `unknown tool`, else the exception name or `failed`; the arguments it
quotes never show); otherwise `ok` is `status["ok"]` (the `ToolReply` status, the message's artifact), and
without a status (not a Hailer tool) the call is ok with an empty summary. A failure is worded by its error
class, never its message (which may quote a URL with a token, a path or data): `NoSessionError` → `notebook not
open`, `MarimoUnavailableError` → `marimo not running`, `MarimoExecutionError` → `kernel request failed`,
`NotebookNotFoundError` → `notebook not found`, `NotebookExistsError` → `notebook already exists`,
`NotebookPathError` → `not in the notebooks folder` (`not a usable notebook name` for `notebook_create`),
`WebAccessDenied` → `host not allowed`, `cells_unreadable` → `could not read the cells`, `unknown_template` →
`unknown template`, an unexpected exception → `unexpected <Type>`, a plain `HailerError` by tool (`could not
fetch the page`, `skill not found`, `could not read the file`, `could not read the data folder`, else
`failed`). `marimo_execute` code that failed names the exception from its traceback (else `failed`), and a
call whose created or edited cell raised (`cell_error`) is a failure named by the kernel's exception classes
after the cells: `growth, ZeroDivisionError, checks clean`. Only when no snapshots were compared (`[checks]` all
off) does marimo's own line (`created and ran cell 'Hbol' (growth) (error)`, in code that used
`marimo._code_mode`) fail the call, named from the traceback text (else `cell error`); a `re-ran ... (error)`
line never does.

Summaries of calls that worked are read from the parts of the text Hailer writes itself, and never repeat what
the arguments say (the UI shows those as the step's target): `marimo_execute` the changed cells (`cells` from
the status: `growth_by_region`, `a and b`, `4 cells`; without snapshots, and for `deleted scratch` / `installed
altair` / `re-ran report`, marimo's own lines such as `created and ran cell 'Hbol' (name)`, read only when
the code used `marimo._code_mode`), a frame's `4 rows` (Polars `shape: (4, 3)` at the start of a line, pandas
`[4 rows x 3 columns]`, read from the text before the check report) and the checks from the status (`checks
clean`, `2 findings`, `ty skipped`, `checks skipped`); `notebook_check` `clean` / `N findings` / `checks off` /
`no code`; `notebook_cells` `6 cells, 1 with errors` / `N matching cells` / `no cells` (`N+ cells` when the
reply was truncated); `notebook_list` `3 notebooks, 1 open`; `notebook_create` / `notebook_open` the session
outcome (`already open`, `opened in your browser`, `no session yet`, `browser did not open`, `marimo not
running`; open adds the cell count), led by the notebook's name only when the argument did not already name it
(`q2_churn.py` for "Q2 Churn"); `notebook_close` `closed` / `was not open` (`<name>, closed` without an
argument); `marimo_status` `<name> ready` / `<name> not open` / `not running`; `list_periods` `[dataset, ]6
months, 25-01 to 25-06[; discount from 25-04]` (the dataset only without a name argument) or `no period files,
N other files`; `load_skill` `N files` / `no files` (from the last `Files (read with read_skill_file(...)):`
block, so a SKILL.md quoting it changes nothing); `read_skill_file` `N lines` / `truncated`; `fetch_page`
`12.3 KB` (the size of the text the model got; `, truncated` when the status says the reply was cut or the
`Source:` line says `(first N bytes only)`; `, redirected to <host[:port]>` when the page came from another host) or `not text
(<type>)` when the whole reply is fetch_page's not-text notice. A summary is one line of at most
`SUMMARY_LIMIT` printable characters (`...` when cut; `str.isprintable`, as `steps.single_line`), counts and
names only. An unknown tool or reply format gives `""`.

## `code_checks.py`  (owner: code checks, 2026-09-22)

ruff and ty for the cells the agent writes. Both are Hailer dependencies and run as subprocesses in the
Hailer process's environment, whatever the kernel runtime; nothing raises into the agent loop.

```python
SNAPSHOT_MARKER = "HAILER_CELLS:"
RUFF_SELECT = ("F", "PLE", "B006", "B023", "E722")          # ruff runs --isolated: no project config applies
RUFF_IGNORE = ("F401", "F541", "F704", "F811", "PLE1142")   # imports for later cells, private helpers, top-level await
TY_IGNORE = ("unresolved-import",)                          # the kernel's packages are not the host's (docker)
MAX_REPORTED = 20
def snapshot_code(*, enable_format: bool = False) -> str    # scratchpad code printing SNAPSHOT_MARKER + {"cells": [{id, name, code, error}]}
def parse_snapshot(stdout: str) -> Snapshot | None          # Snapshot(cells: tuple[Cell(id, name, code, error=""), ...]) in notebook order
    # error: the class name of the exception the cell's last run raised, from ctx.graph.cells[id].exception (what code mode's own
    # summary reads; "" when it did not raise or the graph cannot be read); tools.marimo_execute fails a call whose changed cells raised
def changed_cells(before: Snapshot, after: Snapshot) -> list[str]   # new or edited, non-empty, notebook order
def dependency_order(cells) -> list[Cell]                   # marimo's compile_cell + DirectedGraph + topological_sort; unplaceable cells last
def syntax_error(cell: Cell) -> Finding | None              # compile with PyCF_ALLOW_TOP_LEVEL_AWAIT
def build_script(snapshot, *, skip=()) -> tuple[str, list[tuple[str, int] | None]]   # "# cell <id>" + code per cell; line -> (cell id, line)
def check(snapshot, cell_ids=None, *, lint=True, typecheck=True, search_paths=()) -> CheckResult   # findings, tools that ran, problems
def report(result, snapshot, cell_ids, *, changed=True) -> str
```

- The snapshot reads `ctx._document.cells`, never `ctx.cells[...].code`: the latter records a read in code
  mode's staleness tracker, and an `edit_cell` of a cell the user changed in the browser would then go
  through without the agent having seen it. With `enable_format` it first sets
  `ctx._kernel.user_config["save"]["format_on_save"] = True` in the kernel's memory: code mode then formats
  new and changed code with ruff before applying it (marimo's `DefaultFormatter`, the user's
  `formatting.line_length`), so the formatted cell is the one that runs. The kernel needs ruff: Hailer's
  own Python has it (unsafe-local runtime) and the kernel image installs it (`IMAGE_PACKAGES["ruff"]`); a server whose
  Python lacks ruff skips formatting silently (marimo logs it). The flag stays on in that kernel until it
  restarts or the user saves marimo settings (which replaces the kernel's copy), so `format = false` takes
  full effect with the next kernel.
- The notebook is checked as one script in dependency order, computed on the host from every cell's code
  (the kernel's graph only holds cells that have run), so ty carries types across cells. Cells Python
  cannot parse are reported as `python syntax-error` and left out of the script. ty runs first with
  `--python sys.executable`, `--extra-search-path` for each of `search_paths` that is a folder and
  `--exit-zero`. The tools pass no search path: the folder the kernel imports local modules from is the
  sandbox's, not one on this machine (and `unresolved-import` is ignored anyway); ruff reads the script on stdin with `--target-version` of the running Python. ty's
  top-level-async syntax errors are dropped, and a ruff finding ty also made (`F821` /
  `unresolved-reference` at the same place, `invalid-syntax` on the same line) is dropped.
- A tool that is missing, exits with an unexpected code, times out (`TIMEOUT_SEC = 30`) or prints
  unreadable output becomes `CheckResult.problems`, shown as `(Check skipped: <why>.)`.
- Report: `Checks (ruff, ty) on the N changed cells: M findings.` (or `on the notebook's N cells`, or
  `: no findings.`), then per finding `- cell <name> (<id>), line L: <tool> <rule>: <message>` and the source
  line indented, at most `MAX_REPORTED`, then `(and K more)`.

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
def start_dependency_warmup() -> concurrent.futures.Future[None]  # start/reuse daemon imports; no clients, keys or conversation state
async def await_dependency_warmup() -> None  # await from this loop; cancellation does not cancel shared imports
def disable_tracing_unless_opted_in(environ: Any = None) -> bool   # sets LANGSMITH_TRACING, LANGSMITH_TRACING_V2, LANGCHAIN_TRACING and LANGCHAIN_TRACING_V2 to "false" (in os.environ by default) unless HAILER_TRACING is 1/true/yes/on; True when tracing was left alone. Run when the agent creates its event loop, so a variable left over from another project cannot send prompts and tool results to LangSmith
def system_prompt(config: HailerConfig, bundle: ContextBundle, server: MarimoServer | None = None) -> str  # prompts/system.md (importlib.resources; a built-in fallback persona when missing or empty) + project context + skills index + web allowlist statement + workspace section. Invariant: the text does NOT depend on config.notebook, which changes mid-conversation (the model learns it from marimo_status() and CLI notices). Sent with every model call, never stored in the conversation
    # Workspace section (_workspace_section), for config.kernel (the runtime the session's kernel was started with):
    # unsafe-local: workspace, "Notebooks folder: <notebooks_root>", data dir as host paths; docker: "/work (in the kernel; only the two
    # folders below are yours)", "/work/notebooks (writable; only marimo notebooks are copied back to the user)", "/work/data (read-only)", never a
    # host path the code could not reach. Then "- Marimo URL: <server.url>" (never the token; left out without a server) and kernel.runtime_prompt_notes(config.kernel): the
    # package-install rule lives in the unsafe-local (and docker network=true) notes, not in system.md
def map_exception(exc: BaseException, config: HailerConfig, *, model: str | None = None, provider: str | None = None) -> Exception
class HailerAgent:
    def __init__(self, config: HailerConfig, bundle: ContextBundle, *, model_factory: ModelFactory | None = None, tools: list[Any] | None = None, env: Mapping[str, str] | None = None, threads_path: Path | None = None, sandbox: MarimoSandbox | None = None) -> None
        # undeclared provider → ConfigError. sandbox: the kernel this session started; the tools use it (hailer_tools(config, sandbox=sandbox)) and the prompt names its server's URL.
        # For tests: model_factory(provider, model, key) replaces build_model, tools replaces hailer_tools(config), env replaces os.environ, threads_path replaces <workspace>/.hailer/threads.sqlite
    def start(self, *, resume_thread_id: str | None = None, forget_thread_id: str | None = None) -> str   # the thread id: resume_thread_id when the store has it, else a new one; forget_thread_id (`hailer --new`: the conversation the user chose not to resume) is deleted from the store
    def new_thread(self) -> str        # a new id; the previous thread is deleted from the store (private _forget, shared with start: Hailer only ever resumes the latest)
    def set_model(self, name: str, provider: str | None = None) -> None   # from the next turn; undeclared provider → ConfigError and nothing changes; never starts a thread (the CLI starts exactly one)
    def run_turn(self, text: str, *, on_event: Callable[[AgentEvent], None] | None = None, skill: SkillInfo | None = None, preamble: str | None = None) -> TurnSummary
    def close(self) -> None            # idempotent; start() works again afterwards
    async def astart(self, *, resume_thread_id: str | None = None, forget_thread_id: str | None = None) -> str
    async def anew_thread(self) -> str
    async def arun_turn(self, text: str, *, on_event: Callable[[AgentEvent], None] | None = None, skill: SkillInfo | None = None, preamble: str | None = None) -> TurnSummary
    async def aclose(self) -> None      # cancel and await the active turn first; leaves caller's loop running
    bundle: ContextBundle              # property; assigning it (/reload) applies from the next turn of the same conversation
    model: str; provider_id: str; started: bool    # read-only properties
    thread_id: str | None; key_source: str         # "env" | "keyring" | "missing", refreshed whenever the model is built
```
- Dependency preparation imports LangChain, its SQLite saver and the OpenAI resource package on a
  daemon worker after disabling tracing unless opted in. `_ensure_graph` awaits this before creating
  loop-owned resources. The process-wide completion survives cancellation and event-loop closure;
  import failures become `AgentError` without repeating heavy imports on the UI thread.
- `build_model`: one `ChatOpenAI` covers both wire APIs. Always passed: `model`, `api_key`,
  `base_url=provider_base_url(provider)`, `use_responses_api=(wire_api == "responses")`,
  `stream_usage=(stream and stream_options)`, `http_socket_options=()`. When configured: `default_headers`
  (`request_headers`), `default_query` (`query_params`), `disable_streaming=True` (`stream = false`),
  `reasoning_effort`, `http_async_client`. `parallel_tool_calls` is never sent, so a strict gateway sees only
  `model`, `messages`, `tools`, `stream` (and `reasoning_effort` when set). Reasons: see "Runtime facts".
- Lifecycle: async methods use the caller's event loop throughout start, turns, reset and close.
  The synchronous facade owns one `asyncio.Runner`, preserving Ctrl+C/`KeyboardInterrupt` behavior.
  Async cancellation propagates as `CancelledError`; callers cancel and await the active turn before
  `aclose()`. HTTP and SQLite resources are bound to one loop; mixing lifecycles or loops before closing
  raises `AgentError`. Cleanup completes before propagating cancellation. Other non-Hailer exceptions
  go through `map_exception` with the model and provider in use. The
  graph is built on first use and rebuilt after `set_model` or a `bundle` assignment: the old HTTP client is
  closed, the key resolved (`"missing"` → `CredentialsError` with the `hailer login <id>` hint, for every
  provider), the model built with an `openai.DefaultAsyncHttpxClient()` the agent owns (honours
  `HTTP(S)_PROXY` / `NO_PROXY` and the OS trust store), then `create_agent(model, tools, system_prompt=...,
  middleware=..., checkpointer=saver)`. With `[model].summarize_after_tokens > 0` the middleware is
  `SummarizationMiddleware(model=model, trigger=("tokens", N), keep=("messages", SUMMARY_KEEP_MESSAGES))`.
- `run_turn` sends ONE user message: the skill notice (`[Hailer] The user attached the skill '<name>' to this
  request. Follow its instructions:` + `context.read_skill`), the preamble (blank counts as none; the CLI's
  `[Hailer] ...` notices, see `hailer.chat`), then the text, joined by blank lines. First `_settle_history` makes the
  stored conversation valid: every tool call without a result gets a `ToolMessage(INTERRUPTED_TOOL_RESULT)`,
  and when the last stored message is the user's (the turn died before the model answered) the new text is
  appended to it under the same message id, because strict chat templates refuse two user messages in a row.
  The turn runs as `graph.astream(..., stream_mode=["updates", "messages"])`. Events (`AgentEventKind =
  "message_delta" | "tool_call" | "tool_result" | "status"`), all emitted synchronously on the owning loop:
  - `AgentEvent("message_delta", <text>)`: a piece of the reply (the `model` node only, never the summariser).
  - `AgentEvent("tool_call", <name>, {"arguments": <80-char preview>, "args": <dict>, "id": <str>})` for each
    call in a `model` update. `args` is a shallow copy of the call's arguments that `json.dumps` accepts (other
    values as `str`; `{}` when they are not a mapping). `id` is unique in the turn, so the UI matches on it
    alone: the call's tool_call_id, or `<id>#<n>` when the turn already used that id (some gateways number
    calls per reply) and `call#<n>` when the model sent none (`n`: the call's position in the turn).
  - `AgentEvent("tool_result", <name>, {"id": <str>, "ok": <bool>, "summary": <str>, "seconds": <float>})` for
    each `ToolMessage` in a `tools` update that answers a call of this turn, after that call's `tool_call` and
    with its `id`. A result is matched by its tool_call_id and, among the turn's waiting calls sharing one, by
    tool name, then oldest first; calls of one model message may finish in any order (LangChain runs each as
    its own task). A `ToolMessage` for no waiting call emits nothing and is not a step. `seconds` is the wall
    time since the `tool_call` event, rounded to ms. `ok` and `summary` come from
    `tool_outcomes.tool_outcome(name, args, content, message.artifact, error=(message.status == "error"))`.
    A notebook tool's result also has `"notebook": {"notebook": <name>, "active": <bool>, "session": <bool>}`
    (from its reply status: the notebook `notebook_open`/`notebook_create` left active or `notebook_close`
    closed, and whether it has a kernel session), so the chat's footer follows a switch at once.
    A call the turn is cancelled in gets no `tool_result` (the UI marks it cancelled), nor does the
    `INTERRUPTED_TOOL_RESULT` that `_settle_history` writes for it before the next turn (state updates are
    not streamed). The bookkeeping is `_TurnEvents(emit).on_update(update)`, testable without a graph.
  - `"status"`: the agent does not emit it today.
  `TurnSummary(final_response, thread_id, duration_ms, tool_calls, input_tokens, output_tokens,
  status="completed", steps)`: `final_response` is the last model message without tool calls; tokens are summed
  over the turn's model calls, `None` when the endpoint reported none; `tool_calls` names every call the model
  asked for and `steps` counts the turn's `tool_result` events (the calls that ran). None of this changes the
  requests sent.
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

## `statedir.py`

```python
STATE_DIRNAME = ".hailer"; GITIGNORE_NAME = ".gitignore"; GITIGNORE_TEXT   # a comment line, then "*"
def state_dir(workspace: Path) -> Path          # <workspace>/.hailer (path only)
def ensure_state_dir(workspace: Path) -> Path   # mkdir -p, then write .hailer/.gitignore unless one exists (never overwritten; a failed write is ignored)
SANDBOX_MARK_NAME = "sandbox-wrote-notebooks"   # a sandbox copied notebooks here since the user last agreed to run them unsafe-local
def mark_sandbox_wrote(workspace) -> None; def sandbox_wrote_notebooks(workspace) -> bool; def clear_sandbox_mark(workspace) -> None
INPUT_HISTORY_NAME = "input_history"; def input_history_path(workspace) -> Path   # .hailer/input_history (hailer.input_history)
```
The only place `.hailer/` is created: `session.save_session`, `notebooks.save_active_notebook`,
`HailerAgent` (the default `threads.sqlite`), `input_history.InputHistory` and the kernel runtimes' record, log and token folders all go through `ensure_state_dir`, so the
folder ignores itself in any git repository. The user's own `.gitignore` is never touched.

## `input_history.py`

```python
MAX_ENTRIES = 1000; MAX_ENTRY_BYTES = 64 KiB; MAX_READ_BYTES = MAX_ENTRIES * (MAX_ENTRY_BYTES + 64)
LOCK_WAIT_SECONDS = 2.0; EXIT_FLUSH_SECONDS = 0.5
def input_history(path: Path | None) -> History   # ThreadedHistory(InputHistory(path)), or InMemoryHistory() for None (the setting off)
def storable(text: str) -> bool                    # not blank, <= MAX_ENTRY_BYTES, no credential shape (below)
class InputHistory(History):                       # prompt_toolkit History kept in path (FileHistory's format)
    def load_history_strings(self) -> Iterable[str]   # newest first; never raises ([] on any failure)
    def store_string(self, string: str) -> None       # queues it for the writer thread; never checks, waits or raises
    def flush(self, timeout: float = LOCK_WAIT_SECONDS) -> bool   # waits for the queued entries
```
The composer (`ChatUI(history_file=...)`) and plain mode's prompt (`_LineReader.history_file`) use it while
`[hailer].input_history` is on; the controller passes `statedir.input_history_path(workspace)`. The file is
prompt_toolkit's `FileHistory` format (`# <time>` line, then each line of the entry behind `+`), so multi-line
entries round-trip. prompt_toolkit's `ThreadedHistory` reads it on a worker thread when the first render
creates the buffer's content: nothing waits for it, and Up recalls entries as they load. Saving never runs
on the UI loop: `store_string` (called by `Buffer.reset(append_to_history=True)` on Enter) only queues the
entry; one daemon writer thread per process (`_WRITER`) checks it (`storable`, the size first, so a pasted
file costs nothing), waits up to `LOCK_WAIT_SECONDS` for the lock and appends it, in submission order. An
`atexit` handler gives queued entries `EXIT_FLUSH_SECONDS`, then leaves the thread (an entry still waiting
for another chat's lock is lost, never half written). A read first waits for this process's queued entries.
Reading keeps the last `MAX_ENTRIES` and rewrites a longer file, or one holding an entry over
`MAX_ENTRY_BYTES` (written by an older build), through `tempfile.mkstemp` in the same folder and
`os.replace`; a file it could read nothing from is never rewritten. Only a file larger than any Hailer writes
is read from its last `MAX_READ_BYTES`. Not written (kept in the session's memory): an entry over
`MAX_ENTRY_BYTES`, and one `storable` refuses: what `hailer.log.redact` masks (a bearer token or
`Authorization:` value, an `sk-` key, the value of a `*_KEY`/`*_TOKEN`/`*_SECRET`/`*_PASSWORD` environment
variable of 8 characters or more), a URL with a password (`scheme://user:password@`), a GitHub token
(`ghp_`, `gho_`, `ghs_`, `ghu_`, `ghr_`, `github_pat_`) or an AWS access key ID (`AKIA`/`ASIA` and 16
characters). `/exec` entries are written like any other. Appends (`O_APPEND`), reads and the rewrite hold an
exclusive lock on `input_history.lock` (`fcntl.flock`; on Windows `msvcrt.locking`, where only `EACCES` and
`EDEADLOCK` mean "held"): two chats in one workspace interleave whole entries and lose none to a rewrite. A
file system without locks (any other locking error) is used unlocked. Links: the history and lock files are
opened with `O_NOFOLLOW` (where it exists) and `O_NONBLOCK`, checked with `lstat` before and `fstat` after;
anything but a regular file (a symlink a cloned repository ships, a FIFO) switches that history to memory
only, logged once at info level. The files are created `0o600`. Any `OSError` (read-only workspace,
permissions, a folder in the way) or corrupt bytes (decoded with `errors="replace"`; lines without `+` end an
entry) only means less history; it is logged at debug level.

## `session.py`  (owner: wave 2 / E)

```python
SESSION_FILENAME = "session.json"   # in statedir.state_dir(workspace)
COMMANDS: dict[str, str]     # name -> one-line help, for /help
EXIT_COMMANDS = ("exit", "quit")
COMMAND_USAGE: dict[str, tuple[str, str]]   # name -> (argument signature, short description): the composer's /help card; every COMMANDS name but the alias quit
COMMAND_GROUPS: dict[str, tuple[str, ...]]  # the card's groups (Conversation, Notebook, Session), each COMMAND_USAGE name once
NOTEBOOK_SUBCOMMANDS: dict[str, str]         # list/new/open/close -> short description: the command menu's second level
def parse_command(line: str) -> Command | None      # "/model gpt-5.5" -> Command("model", "gpt-5.5"); "/exit" ; non-slash -> None; unknown slash -> Command(name, args) (CLI reports unknown); the name ends at the first whitespace, so "/exec\n<code>" keeps its lines
def session_path(workspace: Path) -> Path
def load_session(workspace: Path) -> SessionState   # .hailer/session.json, tolerant of missing/corrupt
def save_session(workspace: Path, state: SessionState) -> None   # atomic (tmp + replace)
def help_text() -> str
```
`SessionState` is `thread_id, model, provider, turns, input_tokens, output_tokens`: the file names the thread
to resume and carries the counters; the conversation itself is in `.hailer/threads.sqlite`. No prompt hash is
stored: the system prompt is sent with every turn, so `/reload` applies from the next message.

## `cli/`  (owner: wave 2 / E)

The `hailer.cli` package: `__init__.py` holds the Typer `app` (global options, command registration) and
`main()`, the console entry (`hailer = "hailer.cli:main"`). `chat.py` has bare `hailer`, `notebook` and the
session code (`_run_session`, `_start_kernel`, `_run_foreground`, `_wait_for_notebook`, `ChatLoop`);
`kernel.py` the `kernel pull|build|stop` sub-app; `setup.py` `init`, `login`, `logout`, `doctor` and `status`.
`common.py` holds what they share: `CliOptions`, the injectable collaborators (`console_factory`,
`_load_config`, `_runtime_for`, `_make_agent`, `_resolve_key`, `_wait_for_session`, `_open_browser`, ...),
output helpers, the preflight checks (`local_checks`, `kernel_checks`) and the chat's terminal pieces
(`_LineReader`, `_TurnDisplay`). The command modules call a collaborator as `common.<name>(...)` and
`ChatLoop` hands `common` to `ChatController` as its `services`, so a test replaces one with
`monkeypatch.setattr(common, "<name>", ...)` and every caller sees it; a name defined in `chat.py` or
`setup.py` is replaced on that module. The modules import only `common` (and `__init__` imports all four), so
there is no import cycle.

Bare `hailer` and `hailer notebook` both call `_run_session`:
load config, print the header, run local and kernel checks, begin dependency-only warmup, then prepare and
start this session's own kernel through the chosen runtime (`_start_kernel`; nothing is looked for or reused).
Interactive mode shows the editable composer after the kernel's row and before agent preparation and
browser-session waiting run concurrently. Plain/piped mode retains the notebook wait before its prompt.

Startup output is a checklist (`transcript.SessionRow`: icon, a ten-cell label, detail, seconds, the same
columns as a step row; label and detail pass `steps.single_line`, so names, warnings and check texts cannot
add rows or terminal controls; on a terminal the columns line up with the composer's live row and a row that
worked keeps to one line, elsewhere each is one plain line that is never cut; `session_note` lines under a
row lose control characters too, and on the CLI's own console a terminal wraps them under the detail column
while pipes and plain mode get whole lines). In order: the header `◆ Hailer <version>   ·   <workspace>` (`~`
for the home folder, `_home_relative`) and a blank line; each preflight check that did not pass as a row
(`_check_rows`: `▲` warning or `✗` failure, labelled `Config`, `Notebook`, `API key`, `Kernel`, `Docker`,
`Image`, `Data`; the summary and hint lines unchanged, the hints under the detail column; a failure exits 1
before any kernel starts); the runtime's own progress (docker's image download) and the Rich spinner
`Starting marimo[ in Docker] on <url> ...`; then `_kernel_row`: `✓ Kernel  docker · isolated · no network ·
data read-only  1.9s` (`docker · network on · data read-only` with `[kernel] network`), or for unsafe-local a
yellow `▲ Kernel  unsafe-local · runs as you, not isolated`; under it, as muted notes, `Port N is busy; using
M.` and, with `--verbose`, `Marimo is running at <url>  (log: <log hint>)`. The settings panel
(`common._startup_panel`) is `hailer status`'s only: `/status` has every setting, plus a `Kernel log` row (the
sandbox's `log_hint`). The notebook's row comes from `_wait_for_notebook`: `✓ Notebook  <name> · already
open`, or (the browser opened, a session came) `✓ Notebook  <name> · open in your browser  2.2s` and the muted
`APP_VIEW_TIP`, or a warning `▲ Notebook  <name> · no kernel session yet` (the agent reports the notebook state
when asked). Plain mode, which has no live row, starts the wait with `○ Notebook  opening <name> in your
browser` (no link). The agent's rows come from `ChatController._started`: `✓ Agent  <model> · <provider> · <env_key>
from <source>  0.8s` (the source the agent resolved; nothing is looked up then), `↺ Resumed  N earlier turns ·
/new starts fresh`, or `▲ Thread  previous conversation could not be resumed; started a new one`; context
warnings are `▲ Context` rows. In the composer, the notebook wait and the agent start show on the live row
(`ChatController.progress(label, detail)` -> `ChatUI.begin_task`/`end_task`: `Notebook  opening analysis.py
in your browser`, `Agent  starting gpt-5.5`, in the session rows' columns, the latest begun one first, a
running step before both) and commit their rows above it as each finishes; the composer prints a blank line
when preparation ends. Plain mode prints the same rows in order, then `Type /help for commands.`.

The signed-in URL (`sandbox.notebook_url(name, with_token=True)`: it carries the access token) is opened
in the browser and, in the chat, printed only when the user must open it by hand: with `--no-browser` (at
once, before the wait: `○ Notebook  open <name> in your browser with this link (it signs you in):`) or when
no session came in time (the warning row, then the link). It is then a note of its own under the row,
printed with `soft_wrap=True` (written whole: the terminal wraps it and it copies as one) and never as an
OSC 8 hyperlink. The same holds for switches: `ChatLoop(..., open_browser=False)` (`--no-browser`) sets
`ChatController.open_browser`, and then neither `/notebook new|open` nor a switch the model made opens a
browser; each prints its link at once instead. `--foreground` has no chat: it prints its signed-in home URL
(below), the only way into that kernel.
`--kernel docker|unsafe-local` overrides the runtime for `notebook` (`_kernel_choice`: any other value, `local`
included, prints `config.runtime_problem` and exits 2). The kernel is stopped in the `finally` of
`_run_session` (and `_run_foreground`) whatever ends the chat: `_start_kernel` is called inside that guarded
block, and it stops a kernel the runtime returned if Ctrl+C arrives before it hands it over. A docker kernel's
`stop` makes the last notebook copy back before its containers go, so every way out (`/exit`, EOF, Ctrl+C, an
error, the end of `--foreground`) copies the kernel's notebook edits back.

Both plain and interactive chat receive the sandbox the runtime started through `ChatController`,
`HailerAgent` and the tools; nothing reads it back from a file, so `/new`, `/model` and notebook switches keep
it. `ChatController` points `sandbox.notice` at its current console (yellow, thread-safe through the composer's
queue), and asks for a copy back (`sandbox.sync_soon()`, `_copy_notebooks_back`) after every turn (finished,
failed or interrupted) and on a `/notebook` switch (`_switch_notebook`). The prompt gets a token-free URL. Once the kernel answers, `_active_notebook(config, sandbox)` picks the
active notebook (the state file's, else `default_notebook` when `sandbox.has_notebook` says it is gone: one
request, not a listing) and saves it; the notebook's startup row shows that
name. The CLI and the chat take clients from `sandbox.client(notebook=..., token_in_links=True)` and links
from `sandbox.notebook_url(name, with_token=True)` / `sandbox.home_url(with_token=True)`: they are printed or
opened for the user, never sent to the model. `_marimo_state(sandbox, notebook)` is the seam the chat uses.

`notebook --foreground` prepares and starts the runtime, prints its signed-in home URL ("Ctrl+C stops it"),
waits for exit or Ctrl+C, and stops it; only the browser uses that kernel. Every Docker start checks
`docker_mount_problems`, including foreground mode. Pull/build progress runs before any startup spinner.

Exit (`_run_chat_loop`, then `_stop_kernel` in `_run_session`'s `finally`): when the chat ends (`/exit`, EOF,
Ctrl+C while idle; nothing prints `Bye.` any more) the muted `Session ended · 4m 12s · this session: 2 turns,
1.1k tokens` (the time since the command started, then this session's turns and tokens,
`ChatController.session_turns`/`session_tokens`; zero tokens are left out; the footer and `/status` count the
whole conversation, `SessionState`, and say so: `/status` reads `this conversation: N turns`; turns and tokens
count finished turns only: a cancelled or failed turn's usage is not known to the chat and adds neither; the composer ends its transcript with a blank line however it
exits, so the line always follows one); then, under a spinner on a terminal while the kernel stops, one line:
`✓ Notebooks copied back to notebooks/   ✓ Kernel stopped` (docker) or `✓ Kernel stopped` (unsafe-local).
`running.stop()` runs whatever the terminal does: a spinner or line that cannot be written (a closed
terminal) never keeps the kernel, or its last copy back, from stopping. The first item is the last copy's own
record, `DockerKernel.copied_back` (from `NotebookSync.complete`; None without a copy), never the warnings,
which say a repeated problem once: `▲ Notebooks not all copied back to notebooks/` when the last copy did not
finish (a dead kernel, a file that could not be written, the deadline, Ctrl+C). Warnings during the stop
(`sandbox.notice` is pointed at the CLI's console) print as `▲` callouts, text unchanged, e.g. `Warning:
stopped before the notebooks were copied back; ...`. A second Ctrl+C while the kernel is removed prints
`Interrupted while the kernel stopped: it may still be running.` with `Run uvx hailer kernel stop to remove it.`
(docker) or how to end the marimo process (unsafe-local), and exits 1. A stop error stays red (`✗ Could not
finish stopping the Docker kernel: ... Run uvx hailer kernel stop to retry.`) in place of `Kernel stopped`,
and the command exits 1. A session that failed before the chat ran (a fatal check, a kernel that did not
start, an agent that could not start in plain mode) has no `Session ended` line.

Other commands: `status` shows the configuration, credentials, context and a `Kernels:` line
(`kernel_docker.workspace_kernels`, via `_workspace_kernels`: running Docker kernels; "none running for this
workspace (each uvx hailer session starts its own)"); `doctor` is a table of `local_checks` + `kernel_checks` (config, notebook,
credentials, the runtime: Docker, the image, the folders) and never starts, probes or calls a kernel or the
model endpoint (an unsafe-local runtime is a WARN `kernel` row with `UNSAFE_LOCAL_HINT`; exit 0); `init [--force]
[--kernel docker|unsafe-local]` creates the workspace (`hailer.toml` always names a runtime, `docker` without the flag;
`_kernel_next_steps` adds, without `docker` on PATH, `DOCKER_INSTALL_ADVICE` and the unsafe-local opt-in, and for
unsafe-local that it is not isolated); `login` and `logout` manage
provider credentials; `kernel pull`, `kernel build [--tag]` and `kernel stop` manage the image and every kernel
of the workspace. Global options include `--verbose/-v`, `--config`, `--workspace`, `--new`, `--plain` and
`--version`. Human-facing links include the token; tool-result links omit it. The doctor's `notebook` row is
host-side: the active notebook's file in the notebooks folder (`(active)`), else the configured one
(`(configured)`).

Agent start (`ChatLoop.start`):
`HailerAgent(config, bundle)`, then `start(resume_thread_id=<thread in session.json>)`, or with `--new`
`start(forget_thread_id=<that thread>)` so the stored conversation is deleted; a different id coming back from
a resume means the thread was gone, the CLI says so and resets the counters. Credentials: a `"missing"` key is
a fatal preflight failure for every provider, the built-in `openai` included; `status` and `/status` show
`<env_key> from env|keyring` or `<env_key> missing (run: uvx hailer login <id>)`; both also show the `Kernel:` line,
in yellow (`common._kernel_style`) for unsafe-local, as `--foreground` does (the startup's `▲ Kernel` row says it there). Interactive chat is selected when stdin/stdout are terminals, unless `--plain` is set or
`TERM` is `dumb`/`unknown`. `--plain` is a global option and also an option on `notebook`.

`hailer/chat.py:ChatController` holds shared session, slash-command and notebook operations;
`hailer.cli.chat.ChatLoop` binds it to `hailer.cli.common`, its injectable collaborators. Plain mode uses `_LineReader`/`_TurnDisplay`. Explicit plain and
unsupported terminals use `Console.input` without starting prompt_toolkit; their Rich console also
disables live status/cursor controls, including during startup and notebook waits.

`chat_ui.py:ChatUI(console, on_submit, footer, *, pt_input=None, pt_output=None, completions=None, history_file=None)` owns one inline prompt_toolkit
application (no alternate screen, mouse capture or CPR) and a serialized output queue. Top to bottom: the live
row (only while busy or starting), a bounded multiline composer in a rounded frame coloured by state (ANSI cyan
ready, bright black busy or starting, red once startup failed) with a `›` prompt, a two-space continuation
indent, a placeholder that says what Enter does now and key hints in the frame's bottom border that follow the
state, then a one-line footer. `footer()` returns `transcript.FooterData` (notebook, `notebook_open`
True/False/None, kernel runtime, model, turns, tokens, the notebook's token-free URL) and is called on every
redraw, so `ChatController.footer` only reads state it holds (the URL is made once per notebook name;
`notebook_open` is what startup, a switch, `/status`, `/notebook` or `/exec` last learned). The footer shows
`● name` (open), `○ name` (not open) or `name` (unknown), `docker · isolated` or a yellow
`▲ unsafe-local · not isolated`, the model and `N turns · Xk tokens` (`working 3.1s` during a turn); when narrow
it drops tokens, turns, the model and a docker kernel, then shortens the notebook, never the unsafe-local
warning. Widths are terminal cells. Every row but the composer is one window that draws its whole line at the
terminal's width (the frame's sides are margins of the composer's window): rows of `VSplit`s doubled the cost
of a redraw. A redraw ticker runs every 80 ms during a turn; while the chat prepares nothing animates and the
live row's timer redraws once a second (agent imports then run on a worker thread and every redraw takes the
GIL from them: §10, `scripts/benchmark_startup.py`); while idle nothing ticks, and a flash schedules the one
redraw that ends it.
Rich transcript output goes through `run_in_terminal`; log handlers are temporarily routed through the same
queue, retaining redaction. Bracketed paste stays enabled throughout the live composer and is restored on exit.
Enter submits once; Alt+Enter (Esc Enter) and Ctrl+J insert a newline (Ctrl+J never involves Esc, for
terminals that do not send Alt as Meta). Up/Down move through a multiline draft and recall history at its
edges: `history_file` (the controller passes `.hailer/input_history` while `[hailer].input_history` is on;
see `input_history.py`) or, for `None`, this session's only. History is read on a worker thread after the
first render, never before it; entries are saved on a writer thread. As you type,
`ThreadedAutoSuggest(AutoSuggestFromHistory())` (searched off the UI loop) shows the rest of the latest matching
history line in grey (`auto-suggestion`, bright black); → (or Ctrl+E, Ctrl+F) at the end of the draft takes
it (`load_auto_suggest_bindings`), and the border says `→ accept · enter send · alt+enter new line`
(`right accept` in ASCII). A busy submission leaves the draft intact, does
not enqueue a request and flashes `Still working · esc cancels · your draft is kept` in the border for 2.5 s
instead of replacing the activity (a flash ends with the turn or with startup). Esc cancels a running turn
through the same path as Ctrl+C: only during a turn, or while the command menu is open, are the key timeouts
short (`ttimeoutlen` 0.1 s, `timeoutlen` 0.3 s; `_sync_timeouts` after every render) so a lone Esc acts at
once; otherwise prompt_toolkit's own (0.5 s, 1.0 s) stay, Esc alone does nothing and Esc then Enter within
about 1.5 s is still a newline. Events that arrive after a cancel do
not replace `Cancelling`. Ctrl+C cancels active work and awaits cleanup, or exits while idle; Ctrl+D exits only
empty and idle; Ctrl+Z then Enter exits while idle. `/exit` and `/quit` also exit.

The command menu (`completion.py`): a one-line draft that starts with `/` completes while typing
(`SlashCompleter`), only while the cursor is at its end (or only spaces follow it). The completions show in a
window under the frame, above the footer, while there are any: `  › /help       Commands and keys` (the mark
on the chosen item, the first while none is), names in one column, descriptions (`COMMAND_USAGE`'s short
forms) shortened with the glyph set's ellipsis to the width, at most `MENU_ROWS` = 6 rows scrolled to the
chosen one and a muted `N more` line. Styles `palette.pick`, `palette.name`, `palette.name.current` (ANSI cyan
bold), `palette.meta` (bright black): never `menu.*`, which prompt_toolkit's default style paints grey. The
border says `↑↓ choose · tab complete · enter <run|send|complete> · esc close` (`up/down` in ASCII) while it
is open, also during a turn (`completion.enter_action`). Up/Down and Tab/Shift+Tab choose (Tab without a menu
opens it). Enter (`ChatUI._enter`, the same with or without the menu on screen): an item the user chose
(arrows or Tab) is written, and submitted at once when the draft is then one of `completion.RUN_ON_ENTER`
(`/help /status /new /context /reload /clear /exit /notebook list`: no arguments); without a choice,
`SlashCompleter.enter_pick` decides from the draft alone, computed then, so neither typing speed nor a paste
changes it: a draft that already is a command (any case) or a command and a whole argument (exactly as typed:
`first-look` beside `first-look-v2`, `sales.py` beside `Sales.py`), or that nothing completes, is submitted as
typed; any other slash draft is completed to the first item and nothing runs (`/n` Enter gives `/new`, a
second Enter sends it). A command that takes arguments is completed with its space, so its own menu follows,
except one typed in full. Esc closes the menu and restores the typed text; it does not cancel a running turn
while the menu is open (the next Esc does). Arguments: `/notebook` `list|new|open|close`
(`session.NOTEBOOK_SUBCOMMANDS`); after `open`/`close` the notebook names (`active` marks the active one);
`/skill` and `/prompt` names (the skill's description, `saved prompt`); `/model` the current model (`current`)
and `<provider>:` for `openai` and each declared provider. Matching is a case-insensitive prefix, names that
match as typed first; names pass `single_line`. `completions()` returns `models.CompletionData` (`notebooks`,
`active_notebook`, `skills`, `prompts`, `model`, `providers`) and, like `footer`, must only read state the
controller holds: `ChatController.completions` takes the notebook names from `_notebook_names`. A `/notebook`
command's own listing (`_names`, `/notebook list`'s) fills it on that command's worker; otherwise
`_refresh_names_soon` lists in a background `asyncio.to_thread` task nothing awaits (after the startup
notebook wait, after `/notebook new`, at the end of each composer turn): one listing at a time (a request
during one makes it list once more after), never through `_blocking` or its stop flag, so neither
`ready_to_answer` nor a turn's cleanup waits for it. `run_interactive` settles it (`_settle_names`) before
the agent closes and the CLI stops the kernel, and a result that arrives after that is ignored. The completer
never lists notebooks itself; a failing `completions` callable leaves the command names. Every submission
still goes through `parse_command`.

The transcript: a submission is echoed behind `▌`. `settle_steps()` ends the steps still open as `no result,
effect unknown`; `ChatController._aturn` calls it when `arun_turn` raises, so those rows come before the error.
Before `finish` the controller reads the active notebook from the state file (`_peek_active_notebook`, a local
read: no marimo request, no browser), so the footer and the reply's link name a notebook a tool switched to;
`_sync_active_notebook` still reports the switch, and opens it, after the answer. `finish(summary:
TurnSummary)` prints, once, the steps still open (`no result, effect unknown`), the reply behind `●` (`chat_ui.ReplyMarkdown`: left-aligned headings,
tables with only a header rule and their column alignment, code behind a thin bar, ANSI colours only) and a
muted line `N steps · 7.7s · 3.1k in · 612 out` (`steps`, `duration_ms` and the token counts; unknown or zero
parts left out) with `↗ <notebook>` as an OSC 8 hyperlink to the token-free URL (the stats keep their room, a
narrow terminal shortens the link). A cancelled turn ends with a
`▲ Cancelled.` callout, and `The conversation continues; the model is told the last step's effect is unknown.`
when a step was still running. The console stays soft-wrapped (`console_factory`, for plain mode and pipes);
the composer prints with `soft_wrap=False` and `crop=False`, so Rich wraps at words inside gutters and an
indented plain line (an error's hints) keeps its indent, except a string containing `://`, printed
soft-wrapped, and a line of a gutter's text that holds a URL (`✗ Could not fetch <url>`), written whole: a
URL is never cut. Everything from outside (names, errors, warnings, model output) reaches Rich as `Text`
or with `markup=False`, never as markup: a notebook called `q[/].py` prints as itself. Glyphs: `transcript.glyphs_for(console.encoding)` is the Unicode set, or `ASCII` (`| * + x ! >`, `+-|`
frame) when the encoding cannot write it (a cp1252 console); there is no setting for it.

Startup is a separate preparation gate: typing and paste work from the first render, and one early
submission waits until both agent and notebook preparation settle. Repeated Enter cannot submit it
twice. Startup failure retains the pending message and next draft instead of dispatching them. The UI
settles its startup task on shutdown before the controller closes the agent. Notebook waiting uses
its own cooperative stop flag and must finish before the CLI stops a kernel it owns.
`ChatController.startup_failed` records an actual preparation error, so leaving the retained-draft
interface still returns exit code 1; intentional startup cancellation does not set it.

`startup.StartupTimings` records idempotent monotonic milestones in seconds and debug-logs them without
user content. `input_ready` is the first composer render; `ready_to_answer` is when the first message
may be dispatched. `notebook_wait_complete` includes a nonfatal timeout, not proof of cell execution.
The CLI also records `checks_complete`, `kernel_ready` and `checklist_shown` (the rows it prints before the
chat: the kernel's, and in plain mode the notebook's); the controller records
`agent_ready`. `scripts/benchmark_startup.py` measures fresh processes offline, including paste latency
and the longest input-loop pause.

The interactive controller awaits `astart`, `arun_turn`, `anew_thread` and `aclose` on one event loop.
Blocking notebook operations run off the UI loop. Cancellation sets a cooperative stop flag, stops
session polling between requests, and waits for the worker before releasing the busy state. Informational
cell counting has a five-second timeout. User submissions are echoed by the UI; expanded prompts and skill contents are
sent to the agent without a second transcript echo. Expected errors keep the composer usable.

Slash commands: `/help /status /new /exit /quit /model /notebook /exec /clear /context /skill /prompt /reload`.
`/exec <code>` (dedented; the code may start on the next line) runs `code` in the scratchpad of the active notebook's session on this chat's own kernel and
prints stdout, the result and stderr (red), `(the code failed)` or `(no output)`; nothing goes to the model.
Without a session it prints the `NoSessionError` with the signed-in URL; without code, its usage.
`/new` and `/model` each reset counters and persist exactly one new thread. Model changes refuse undeclared
providers. `/reload` assigns a freshly read context bundle for the next turn. Context/model/notebook
information refreshes on each render; startup reports the agent's actual model/provider rather than stale
session metadata. `/clear` affects the display only.

Notebook switching (`ChatLoop`), by name throughout (`ChatController.notebook`): `/notebook` shows the active
notebook, folder, `Recent:` (from `notebooks.load_recent`, excluding the active one and names the sandbox does
not list), marimo state, URL and `NOTEBOOK_USAGE`; `/notebook list` = `sandbox.list_notebooks()` marking
`active` / `open` (sessions matched with `marimo_client.match_session`); `/notebook new <name> [--empty]` =
`notebooks.new_notebook` + `sandbox.write_notebook(..., replace=False)` then switch; `/notebook open <ref>` =
`notebooks.resolve_notebook` over the sandbox's names (`reference_folders`: the kernel's folder, the host folder,
`notebooks/...`) then switch (the active one only re-ensures its session); `/notebook close [ref]` =
`client.shutdown_session`. Without a sandbox, list/new/open say marimo is not running.
A switch (`_switch_notebook`) writes the state file, sets `self.notebook`, queues
`_pending_preamble` **before** waiting for the tab, then `_ensure_session` (opens the URL through
`hailer.browser` when there is no session and waits `SWITCH_SESSION_TIMEOUT_SEC` = 30 s, the live row saying
`Notebook  opening <name> in your browser`, plain mode a `○` row; with `--no-browser` it opens nothing and
prints the `○ Notebook  open <name> in your browser with this link` row and the link at once), then refines
the notice; Ctrl+C or a
marimo error during the wait leaves the notice queued. Whatever happened, the switch ends with one row
(`_notebook_row`): `✓ Notebook  q2.py · active · open in your browser  0.6s` (`· already open` when it had a
session; `· created from the starter|empty template · active ...` for `/notebook new`; `analysis.py · already
the active notebook · ...` for `/notebook open` of the active one), or a warning `▲ Notebook  q2.py · active ·
no kernel session yet` followed by `Open it in your browser with this link:` and the signed-in link on a line
of its own (`Open it with the link above.` after `--no-browser` printed it), or `▲ ... · not open in a browser yet` after Ctrl+C or an
error. `/notebook close` prints `✓ Notebook  <name> · closed · session <id> · its browser tab is disconnected`
and, for the active one, `It stays the active notebook; /notebook open <name> reopens it.` under it. Exact notice:
`[Hailer] The active notebook is now <name> (<detail>). Call notebook_cells before editing.` with
`<detail>` = `created from the starter template` | `created from the empty template` | `reopened`, followed by
`, N cell(s)` when a session exists or `, not open in a browser yet` when none. It is passed once as
`run_turn(..., preamble=...)` with the next message and cleared (also on Ctrl+C; kept when the turn raised an
error). `_sync_active_notebook()` runs in the `finally` of every turn (finished, failed, interrupted) and,
without opening a browser, at the start of `/notebook` and `/status`: on a change it updates `self.notebook`,
prints `✓ Notebook  <name> · now active` and (turn end only) opens the URL once when the notebook has no
session (`· opened in your browser`; `· open in your browser` when it has one), never printing it; with
`--no-browser` it prints `○ Notebook  <name> · now active · not open in a browser` and the link instead.
A notebook tool's own try counts: `notebook_open`/`notebook_create` open the browser through the opener the
chat gives the agent (`ChatController._tool_opener`, via `common._make_agent(..., open_url=...)` and
`HailerAgent(open_url=...)` into `hailer_tools`; with `--no-browser` it opens nothing and returns False, so the
tool asks the model to have the user open the link) and wait; so after a turn whose tool brought the notebook
up, the chat does not open it again: with no session it prints `▲ Notebook  <name> · now active · no kernel
session yet` and the signed-in link once on a line of its own. Only a switch no tool reported (a call cut
short) is opened here. The footer follows a switch when the tool's result arrives: the notebook tools' reply
status carries `notebook`, `active` and `session` (the model's text is unchanged), the agent passes them as
`detail["notebook"]` of the `tool_result` event, and `ChatController._follow_notebook_tool` updates
`notebook`/`notebook_open` without any I/O. After a cancelled turn the switch's row is printed after the
composer's `▲ Cancelled.` callout (`_aturn` collects the rows and schedules them for the loop's next turn,
which comes after the callout is queued);
no preamble is queued because the model made the switch itself. `COMMANDS["notebook"]` =
`Show or switch the active notebook. Usage: /notebook [list | new <name> [--empty] | open <name> | close [name]]`.
Progress (plain `_TurnDisplay`, interactive `ChatUI.on_event`), through `steps.StepTracker`: a `tool_call`
starts a step keyed by `detail["id"]` (unique in a turn; a call without one gets a private key that only the
turn's end closes); the `tool_result` with that id commits one row, `✓|✗ <done verb>  <target · summary>
<seconds>`, with the event's `seconds` (a summary that starts with the target loses that repeat). Several steps
may be open; rows appear in result order; a result for no open step prints nothing. On cancellation every open
step is committed as `▲ <done verb>  <target · cancelled, effect unknown>` (§4: cancelling is not a rollback);
a turn that ends with steps open (an error) commits them as `no result, effect unknown`. A worked step's row
is one line (its detail cut with `…`); a failed or cancelled step's detail wraps in its column. The composer's
live row shows a spinner, the oldest open step's live verb and target in the rows' columns (`+N more` for
parallel ones) and, from one second, a right-aligned timer; between steps it shows the latest activity
(`Thinking`, `Writing the answer` at a `message_delta`, a `status` event's text, the text a slash command gives
`console.status(...)` or `set_activity`, without a trailing `...`). Deltas are never printed (a model can stream
commentary next to its tool calls and then the answer, which printed the answer twice). Plain mode prints the
same rows as whole lines when the results arrive (no live row, spinner, cursor control or hard wrap), then
`Hailer >`, `final_response` once (or `(no reply)`) and `  N steps · 1.5s · 3.1k in · 612 out`; Ctrl+C prints
the cancelled rows, then `Interrupted.`. After a turn the CLI adds the turn and its tokens to the session file.
Errors (`common._print_error`, in both modes and the other commands): a red `✗ <message>` callout, then the hint
lines indented four spaces, both texts unchanged; exit code 1 when startup fails, the REPL carries on after a
failed turn; traceback only with `--verbose`.

Wording: `steps.describe_call(name, args) -> Step(live, done, target)` (`steps.VERBS`: "Running code"/"Ran
code", "Reading the notebook"/"Read the notebook", "Scanning the data"/"Scanned the data", ...;
`marimo_execute` is `<n> lines`, or "Adding a cell", "Editing 2 cells", "Changing cells" for a code-mode
script; the target comes from the arguments: notebook, data name, skill, `skill/path`, the URL's host, the
cells pattern; an unknown tool shows its name). Labels pass `steps.single_line`. `hailer/transcript.py` holds
what both presentations print, Rich only: the glyph sets, `Gutter`, `user_message`, `callout`,
`step_row`/`plain_step_row`, `turn_stats`/`turn_footer`, the cards and `FooterData`. In the composer
(`ChatController.interactive`) slash commands print cards; plain mode keeps its lines and tables. `/status`:
a rounded `Session` card with a mark per row (`●` fine, `▲` missing credentials, unsafe-local or marimo not
running, `○` notebook not open in a browser) holding every row of the plain table (model, provider and
credentials; kernel; marimo; notebook and session; notebooks folder; thread, turns and tokens; web access).
`/help`: the commands by `COMMAND_GROUPS` with their `COMMAND_USAGE` signature and description, then the keys
(`enter send · alt+enter or ctrl+j new line · esc cancel · ↑ history · ctrl+c exit`)
(groups in their own column when wide, above their commands when narrow, descriptions under the commands
when very narrow). `/notebook list`: `Notebook  Modified  Size` with `● open`, `○ not open` and `active`.
`/context`: labelled sections, warnings as `▲` lines. `scripts/chat_ui_demo.py` drives the real composer with
the agent's events (ids, summaries, parallel, failed and cancelled steps) and these cards.

## `prompts/system.md`  (owner: wave 2 / E, reviewed by D)

The agent persona and rules: the tool list (every `TOOL_NAMES` entry appears as `` `name(` ``, checked by
`test_agent.py`) with the statement that there is no shell and no file tool, the one-active-notebook rules, the
`[Hailer]` notice rule (a CLI notice, "not the user's words"), the scratchpad semantics and the canonical `cm`
patterns, formatting and the check findings (fix real mistakes, leave findings a checker cannot judge), the data conventions with the "compact outputs only" rule, safety, and how to reply. The web
allowlist, project context, skills index and workspace (with the kernel runtime's notes, which carry the
package-install rule) are appended by `agent.system_prompt`. Loaded with
`importlib.resources.files("hailer").joinpath("prompts/system.md")`.

## Tests (each owner writes their own; `uv run pytest` must pass offline)

- `test_config.py`, `test_log.py`: A
- `test_marimo_client.py` (fake SSE server on 127.0.0.1; names <-> kernel paths; the file endpoints with their
  tokens), `test_periods.py` (tiny parquet files in tmp_path): B
- `test_sandbox.py`: `MarimoSandbox` over `fake_marimo.py`'s file endpoints for a docker-like and a local-like
  kernel (listing rules and bounds, read limits, create/replace, a create that loses a race never overwrites,
  names refused before any request, no path outside the notebooks folder ever sent, a junction/symlink out of
  the folder never followed, case variants on Windows, both tokens on every request, HTTP errors as
  `HailerError`s, the normalised digest) and the data folder
- `test_web.py`, `test_context.py`: C. `test_notebooks.py`, `test_browser.py`: notebook feature
- `test_code_checks.py`: code checks. The real ruff and ty on small notebooks (types across cells, the
  rules left out for notebooks, syntax errors, deduplication, missing tools), the snapshot code and the report
- `test_tools.py`: LangChain migration. Tool methods called directly against `fake_sandbox.py`'s `FolderSandbox`
  (the contract over a tmp folder, clients faked), plus the `hailer_tools` registration: names, schemas,
  docstring descriptions, in-process `invoke`, the daemon-thread `ainvoke` path, a cancelled call that does not
  wait for a blocked tool, an end-to-end pass over a real `MarimoSandbox` and `fake_marimo.py`, and a source
  check that `hailer.tools` touches neither folder. The `tool_outcomes` tests read replies the tools wrote:
  `CodeModeClient` runs `LIST_CELLS_CODE` and the check snapshot for real against a stand-in code-mode
  context, and `code_mode_says` produces marimo's own change lines (its `_print_summary`), so a format change
  in marimo or in a tool fails there. They also check that each reply's status reaches LangChain as the
  artifact with the content unchanged, and that what the user's code prints (`ERROR: ...`, `Execution
  failed.`, a fake code-mode or check line) or a skill file holds never decides `ok` or shows up
- `test_secrets.py`: D. `test_agent.py`: LangChain migration. Unit tests drive `HailerAgent` through
  `model_factory=` with `ScriptedModel` (a `BaseChatModel` replaying messages, exceptions or callables) and
  two toy tools: `build_model` kwargs, tracing, system prompt, error mapping, lifecycle (resume after a
  restart, `new_thread`, `set_model`, `close`), turns (events, usage, one user message, `/reload`, failed-turn
  merge, summarisation), tool steps (each `tool_call` followed by the `tool_result` with its id, parallel calls
  matched by id as they finish, empty and repeated ids made unique and matched by name, `_TurnEvents` ignoring a
  result for no call of the turn, error results, no result for a cancelled call or for the history repair,
  `TurnSummary.steps`; on the wire, with Hailer's own tools streamed and not, the tool messages sent are the
  reply text exactly and carry no status) and Ctrl+C during a model call and during a tool (a real SIGINT to the main thread
  on POSIX, `_thread.interrupt_main()` on Windows; see "Runtime facts").
  Wire-level tests run the real `ChatOpenAI` against `tests/fake_gateway.py` (`FakeGateway`, a stdlib
  `ThreadingHTTPServer` on 127.0.0.1: only `POST /v1/chat/completions`, 404 elsewhere, 422 for unknown fields
  and model names, required key, `X-Client-Id` header and `api-version` query, optional refusal of `stream:
  true`, every request recorded; `port=0` picks a free port, a fixed one serves manual end-to-end runs; "check
  both" in the message gets two tool calls in one reply) and assert the exact request fields, headers and query,
  one request per 4xx, and the gateway's call ids in the tool events
- `test_session.py`, `test_cli.py` (Typer `CliRunner`; slash commands; preflight messages with fakes; a
  `FakeAgent` with the `HailerAgent` surface the CLI uses): E
- `test_completion.py`: the command menu's completions from `CompletionData` only (commands, a command typed
  in full, each command's arguments, nothing outside a one-line slash draft or before the end of it, a failing
  data source, names that cannot add rows, as-typed names first), what Enter does without a choice
  (`enter_pick`: whole commands and arguments sent, partial ones completed) and the hint, and the rows (mark,
  scroll, `N more`, ellipsis per glyph set, no `menu.*` style). `test_input_history.py`: the history file's
  format and round trip, saving on the writer thread, the cap, oversized entries, a file never rewritten when
  nothing could be read, credential filtering, corrupt files, failures that never raise, owner-only files,
  planted links and FIFOs, a short and a long hold of the lock (the shipped wait), Windows lock errors (a fake
  `msvcrt`), two processes appending while one trims, exit that writes the queue but never waits long, and
  the off switch. The composer's keys for both are in `test_chat_ui.py` (Enter runs only a chosen command,
  typed, burst and pasted drafts, as-typed names beside longer ones, no completion mid-draft, Esc during a
  turn, Enter never waiting for the history file, suggestions off the UI loop); the controller's cached names
  (listed off the UI loop, never awaited by startup or turns, one at a time, settled before shutdown, never by
  the completer) and the history file it hands out in `test_chat.py`
- Kernel runtimes (docker kernel, 2026-09-19), none of which start anything: `test_kernel.py` (the withheld
  variables, with no way through, the unsafe-local sandbox's folders and URLs, prompt notes, `LocalRuntime` with
  `LocalProcesses` faked, and a start that never adopts another server on its port), `test_kernel_docker.py`
  (`DockerRuntime` argument lists and labels, fail-closed checks, owner locks and leftovers, sessions side
  by side, `workspace_kernels`, `stop_workspace_kernels`, folder checks, the subprocess runner) against `fake_docker.py` (`FakeDocker`: a stateful scripted docker CLI with ids, unique names,
  `--filter label=...`, Docker 29's answers for missing objects, scripted failures and timeouts),
  `test_kernel_image.py`, `test_forward.py` (real sockets on 127.0.0.1), `test_build_kernel_image.py` (the
  buildx command; `subprocess.run` replaced). Shared helpers: `fake_kernel.py` (`make_config`, a fake
  process layer), `fake_marimo.py` (a loopback marimo that checks the tokens and, like marimo, serves the file
  endpoints without confining paths, recording each one) and `fake_sandbox.py` (`FolderSandbox`, `with_folder_files` for CLI tests of a real runtime).
- `test_docker_integration.py`: opt-in (`HAILER_DOCKER_TESTS=1` skips with the reason when Docker, the
  image or Chrome is missing, `strict` fails instead; `HAILER_TEST_CHROME` picks the browser,
  `HAILER_KERNEL_IMAGE` the image; it never pulls). One module-scoped real `DockerRuntime` kernel for a
  temporary workspace with sample sales data, a session held open by headless Chrome on a real clock
  until fixture teardown (DevTools captures the diagnostic screenshot), then: code runs as
  non-root in `/work` on the mounted data and the starter notebook ran on open; a code-mode cell is saved in
  the host notebook; notebooks are listed, created, read and replaced through marimo's file API in the
  container; data writes, network and DNS are refused; neither a host secret nor the token is in the
  kernel's environment or `docker inspect`; the containers carry this session's live owner lock and the
  contract label; stopping leaves no container, network or token folder. The *Docker kernel* job in `.github/workflows/test.yml` builds the image for the checkout
  (`scripts/build_kernel_image.py --load`) and runs it with `strict`.
