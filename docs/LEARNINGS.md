# Learnings for agents working on Hailer

Read this before changing providers, the agent loop, tools or conversation persistence. These lessons
come from the September 2026 LangChain migration and were checked against Hailer's implementation at
v0.2.6. They describe the shipped behavior; source and regression tests take precedence over historical
experiments. Dependency-specific findings must be rechecked when upgrading the versions in `uv.lock`.

The original experiments and proposal remain in Git history. Maintained evidence lives in
[test_agent.py](../tests/test_agent.py), [fake_gateway.py](../tests/fake_gateway.py),
[test_tools.py](../tests/test_tools.py) and [test_cli.py](../tests/test_cli.py).
See [INTERFACES.md](INTERFACES.md) for module contracts.

## 1. Keep the agent limited to Hailer's capabilities

Hailer uses `langchain.agents.create_agent` with its own in-process tools. The migration experiments
showed that a larger agent framework could add filesystem tools, delegation, dependencies and request
payloads that the product did not need. Inspect the actual requests and tool schemas before introducing
a framework or middleware. Add capabilities to meet a concrete need, and preserve the small interface
between the CLI, `HailerAgent` and `HailerTools`.

Hailer uses API keys resolved from the environment or OS credential store. The migration deliberately
dropped ChatGPT-login authentication rather than depend on private, experimental SDK interfaces.
Do not reintroduce private authentication APIs as a compatibility shortcut.

## 2. Make provider routing and request fields explicit

In [agent.build_model](../src/hailer/agent.py), always pass `base_url` and `use_responses_api`, including
for the built-in provider. During the migration, inferred defaults routed some model names to Responses
despite a custom gateway URL, and `LANGSMITH_GATEWAY` could redirect an unspecified URL.

Strict gateways reject extra fields and unexpected model names. Preserve `stream_options = false`
(`stream_usage=False`) and `stream = false` (`disable_streaming=True`); do not add `parallel_tool_calls`
unconditionally. Non-streaming replies still need tool events, a final answer and usage when supplied.
Preserve configured headers and query parameters. Summarization must use the configured model too.

Use the loopback gateway tests to inspect paths, model names, headers, query parameters, tool schemas,
request counts and streaming/non-streaming bodies. Constructor assertions alone cannot catch extra
fields introduced later by tool binding or middleware. Missing usage is unknown, not zero.

## 3. Own HTTP clients and event loops together

The async agent lifecycle stays on the caller's event loop; the synchronous facade owns one
`asyncio.Runner`. Keep HTTP and SQLite resources on that loop until closed, and reject mixed lifecycles.
The migration found delayed Windows interrupts with LangGraph's synchronous `stream()`, so turns use
`graph.astream(...)`.

Pass an agent-owned `openai.DefaultAsyncHttpxClient` instead of relying on a process-cached client:
reusing a cached client after its loop closed caused `Event loop is closed` errors. Close old clients
when rebuilding the model and finish cleanup even if `aclose()` is cancelled.

Preserve `http_socket_options=()` in `build_model`: the dependency's default keep-alive transport
bypassed proxy detection in the tested version. Recheck proxy and trust-store behavior when changing
the HTTP stack. Tests cover sequential agents, restarts, model rebuilds and resource ownership.

## 4. Cancellation requires history repair and honest tool results

[HailerAgent._settle_history](../src/hailer/agent.py) repairs interrupted or failed turns before the
next request. Every dangling tool call gets a `ToolMessage` with the matching call ID and an explicit
unknown-effect result. A trailing unanswered user message is merged with the next input using the same
message ID. This avoids missing tool results and consecutive user messages that strict endpoints reject,
while retaining what the user wanted to retry.

[tools._in_daemon_thread](../src/hailer/tools.py) keeps blocking kernel calls off the loop's default
executor, whose threads would otherwise delay shutdown. Cancelling the turn stops waiting; it does
**not** stop or roll back the kernel call. The call may finish and change the notebook later. Check state
before retrying a side effect, and preserve the CLI's notebook-state refresh after failed or interrupted
turns. Test both prompt cancellation and immediate shutdown, then a successful follow-up turn.

## 5. Keep conversation lifecycle decisions explicit

Resume uses SQLite checkpoints plus the saved thread ID. An unknown ID starts a fresh conversation.
`/new`, `hailer --new` and the CLI's `/model` flow delete the conversation they replace: Hailer only
resumes the latest thread. `HailerAgent.set_model()` rebuilds the model; the CLI separately starts the
new conversation. Do not infer the user-facing behavior from the lower-level setter alone.

The original proposal suggested keeping history across provider changes. The shipped CLI starts a new
thread because replaying one wire API's history through another was not verified. `/reload`, in contrast,
updates context for the next turn of the existing conversation.

Use an explicit absolute token trigger for `SummarizationMiddleware`. The default trigger did not run,
and fractional triggers required model profiles that custom deployment names lacked. Keep
`summarize_after_tokens` configurable (0 disables it), and keep summarizer output out of the chat reply.

## 6. Ambient configuration must not silently export data

`disable_tracing_unless_opted_in` disables LangSmith/LangChain tracing unless `HAILER_TRACING` is truthy.
Variables inherited from another project must not silently export prompts or tool results. Preserve
the explicit endpoint URL and the tests for tracing opt-in.

Docker builds discover host pip/uv package settings by default (`--no-host-config` opts out;
`--pip-config` overrides discovery). This is limited to package download configuration, never the
whole host environment or provider credentials. Keep synthesized settings in temporary BuildKit
secrets outside the build context and clean them up on failures and interrupts. Discovery errors
must not include parser input, URLs or chained exceptions containing credentials. uv's multi-index
resolution is not pip's resolution: require an explicit pip configuration when it cannot be
translated faithfully. Evidence: `tests/test_kernel_packages.py`, `tests/test_kernel_image.py`,
`tests/test_build_kernel_image.py` and the kernel-build tests in `tests/test_cli.py`.

Keep credentials out of prompts, logs and errors. Use Hailer's credential resolution and redaction
helpers; do not move keys into subprocess arguments or new plaintext config files. The same holds for
the marimo token: the unsafe-local kernel reads it from stdin, the Docker kernel from a read-only single-file
mount (`--token-password-file`), because `docker inspect` shows a container's arguments and environment.
The file is deleted once marimo answers (it reads it as its CLI starts). On Docker Desktop for Windows
(29.4.3, 2026-09-22) the file then also disappears inside the container; on Linux the mount keeps the
deleted inode readable there, which gives notebook code nothing it lacks. The start deletes the
`.hailer/kernel-token-<random>` folder in a `finally`; nothing sweeps token folders, because several sessions
start in one workspace and a sweep could delete another session's token before its marimo read it. Evidence:
`test_the_token_reaches_marimo_in_a_file_that_is_gone_once_it_started`,
`test_the_token_file_is_removed_when_a_start_fails` and the Docker integration test
`test_the_token_is_not_in_docker_inspect_or_the_kernels_command_lines`.

The kernel image is versioned by its kernel contract (`hailer.kernel_image`), not by Hailer's version, so
a CLI-only release publishes no image. The tag is computed from the image's inputs (a sha256 prefix): a
hand-maintained contract number was rejected in review because updating a stored fingerprint without
bumping the number would let a changed image reuse a published tag. `--if-missing` fails closed on any
registry answer other than "not found" and never overwrites a published tag.

## 7. Report structured errors and test visible warnings

Map endpoint errors using HTTP status and response bodies, not exception class names or guessed prose.
For example, the tested dependency wrapped a 404 in `OpenAIModelNotFoundError`. Keep gateway-specific
details redacted and give hints supported by the actual failure. The strict gateway tests assert one
request per tested 4xx failure; do not turn configuration failures into blind retries.

Check the live text path as well as the final answer. A migration run exposed a deprecation warning
because `message.text` was a string that was also callable; calling it warned. `_message_text` handles
the string first. The suite normally filters deprecation warnings, so preserve
`test_a_turn_raises_no_warnings_into_the_users_terminal`, which treats them as errors for a turn.

## 8. Separate automated evidence from live verification

For agent changes, run the maintained checks from the repository root:

```text
uv run --locked pytest tests/test_agent.py tests/test_tools.py tests/test_cli.py
```

Use `uv run --locked pytest` for the full offline suite. CI runs Windows and Linux on Python 3.12 and
3.13; Docker integration is a separate opt-in check. No real API key is needed for the offline suite.

Signal simulation is platform-specific: these tests use `_thread.interrupt_main()` on Windows and a
real SIGINT to the main thread on POSIX. The Windows mechanism did not wake Linux's event-loop wait.
Do not treat a passing simulated interrupt as proof of physical Ctrl+C behavior in every terminal.

The migration recorded live Responses API, strict chat gateway, marimo execution and resume checks on
2026-09-18, but left physical Ctrl+C in a Windows console unverified. Keep that distinction when reporting
coverage. Changes to transport or cancellation need relevant live checks as well as fakes; record the
OS, dependency versions, scenarios tested and any remaining gaps. Promote useful experimental checks
into the maintained suite so future cleanups do not depend on throwaway prototypes.

## 9. Keep browser time aligned with real kernel startup in integration tests

The v0.2.7 release's Docker integration test failed with `DATA_DIR` undefined while the same commit's
main-branch run passed. Its screenshot showed "kernel not found": Chrome was launched with
`--virtual-time-budget=25000`, which fast-forwards JavaScript timers rather than waiting for the real
marimo kernel. A session appearing in `/api/sessions` does not prove its notebook cells have run.

Keep headless Chrome connected on a real clock until fixture teardown. Capture diagnostic screenshots
from that existing page through DevTools; do not use a screenshot command that can exit during startup.
`test_notebook_code_runs_in_the_container_on_the_mounted_data` checks both browser liveness and the
starter notebook's initialized globals. Preserve the auto-run assertion: manually running the cells
would hide a broken startup. Verify with `HAILER_DOCKER_TESTS=strict` and a freshly built kernel image.

## 10. Input readiness must not wait for agent imports or a browser session

The startup investigation on Windows / Python 3.13.7 found a 2.6–4.5 second pause after the startup
panel: `ChatController.run_interactive` awaited agent initialization before running the composer.
Most of that time was importing LangChain/OpenAI and constructing dependency model classes. Merely
creating an async startup task still blocked the UI loop for seconds. Keep dependency-only imports
on the daemon warmup worker, with tracing disabled first, and create HTTP/SQLite/model resources on
the agent's original event loop. A cancelled wait must neither cancel shared imports nor make loop
shutdown wait for them; failed imports must produce an actionable error without re-importing them
on the UI thread.

Start warmup before the kernel start. Once the server is healthy, render the interactive
composer before preparing the agent and waiting for the notebook in parallel. A user can type and
paste while either is pending; one first submission may wait for preparation and must run at most
once. Preserve both that message and any next draft on failure. Ordinary busy-turn submissions
remain unqueued. Startup shutdown must settle notebook work before stopping the owned kernel and
settle agent startup before closing its resources. Notebook preparation owns a separate stop flag
from other blocking controller work.

Measure `input_ready` at the first render and `ready_to_answer` when dispatch is allowed. A completed
notebook wait is only `notebook_wait_complete`: it may have timed out, and even an existing session
does not prove its cells have run. `scripts/benchmark_startup.py` measures fresh processes with the
real agent and recorded terminal, a dummy credential, forbidden network connections, temporary
state and a simulated notebook wait. It reports input, paste, readiness and UI-loop pause timings;
it does not measure uvx installation, Docker or browser startup. `hailer --verbose` reports actual
session milestones. Keep event-controlled startup tests in `test_chat.py`, `test_chat_ui.py`,
`test_cli.py` and `test_agent_warmup.py` alongside the agent lifecycle checks; do not turn a passing
recorded-terminal test into a claim about physical Windows Ctrl+C. The measurements, live-check
scope and reproduction commands are in [STARTUP_PERFORMANCE.md](STARTUP_PERFORMANCE.md).

## 11. One session, one kernel: own it, never find one

Until 2026-09-22 a session could reuse a kernel another terminal had started (`.hailer/kernel.json`,
marimo's registry of `--no-token` servers, `[hailer].marimo_url`, `--keep-marimo`, `hailer exec`). Proving
that a recorded kernel was still the one it described, and never orphaning or destroying a busy one, took
liveness probes, a start guard, settings-mismatch checks and id bookkeeping, and was the most intricate code
in Hailer. Now each `uvx hailer` / `uvx hailer notebook` process starts its own kernel, keeps it in memory
(the `Sandbox` its runtime's start returns, handed to the chat, the agent and the tools) and stops it in the `finally` of
`hailer.cli.chat._run_session` (`_start_kernel` runs inside the guarded block, so Ctrl+C right after the start cannot
leak the kernel); `/exec` replaced `hailer exec`. Do not add a way to find or attach to another process's
kernel: a restart costs a few seconds, which is cheaper than the lifecycle bugs.

Two sessions can pick the same free port. The start's wait therefore checks its own process (local) or
containers (docker) first and then requires `answers_with_token(url, its own token)`: a bare `/health`
check once let a second session adopt the first one's server, and every call then got 401. Evidence:
`test_a_start_never_takes_another_sessions_server_on_the_same_port_for_its_own` and
`test_wait_for_health_with_a_token_accepts_only_the_server_that_holds_it`.

Docker ownership is a lock, not a pid: each start creates `.hailer/owner-<random>.lock`, holds an OS lock on
it (`msvcrt.locking` / `fcntl.flock`) and labels its containers and network with that id. An owner is alive
while its lock file exists and cannot be locked; the OS drops the lock however the process ends, so a reused
pid (or a WSL/Windows pid namespace) can never make a dead owner look alive. A start removes only objects
whose owner is not alive; `uvx hailer kernel stop` removes every labelled object and nothing in `.hailer`.
An earlier design kept a per-session record for the local (now unsafe-local) runtime and killed orphans by pid; it was dropped:
the `finally` stops the unsafe-local kernel on every normal exit, and a killed session's token-protected marimo is
ended by the user (Task Manager or `kill`). Evidence: `test_owner_locks_say_whether_their_hailer_still_runs`,
`test_a_start_never_removes_a_live_sessions_objects`, `test_start_removes_what_owners_that_are_gone_left_by_id`,
`test_two_sessions_run_side_by_side_and_each_removes_only_its_own`,
`test_ctrl_c_right_after_the_kernel_started_still_stops_it` and the owner-label check in
`tests/test_docker_integration.py`.

## 12. Files belong to the sandbox: names above the runtime, paths only inside it

Until 2026-09-22 the tools listed, created and resolved notebooks by walking the host's notebooks folder,
read the data folder for `list_periods`, stored host paths in `.hailer/notebook.json`, and translated host
paths to container paths (and back) through a `PathMap` that every layer carried. That only worked because the
docker kernel's notebooks folder was a bind mount of the host's; it could not hold for notebooks that live in a
container's own filesystem or in a cloud kernel. Now the started kernel is a `hailer.sandbox.Sandbox`: notebooks
are names relative to its notebooks folder (`"q3/review.py"`), notebook files go through marimo's own file API
(`/api/files/list_files`, `/file_details`, `/create`, `/update`, `/delete`; marimo 0.24.2, bearer token plus the
`Marimo-Server-Token` header, no browser session needed), and data files are names from `list_data()`. The only
place a name becomes a path is the runtime layer (`MarimoClient.file_key` / `name_of`, from the sandbox's
`notebooks_path`), for `?file=` keys and session matching.

Keep it that way: `hailer.tools` must not name the notebooks or data folders or call a file-system API
(`test_the_tools_never_touch_the_notebooks_or_data_folders` checks the source). Every name is checked by
`notebooks.check_notebook_name` before a request (no `..`, drive, leading `/`, dot- or dunder-folder, `:`
stream, trailing dot/space or Windows device name): `test_names_are_checked_before_any_request`. marimo's
`create` never overwrites; it writes `<stem>_1.py` instead, so a create that loses a race deletes that copy and
reports the existing file (`test_a_file_that_appears_during_a_create_is_never_overwritten`).

marimo's file endpoints confine nothing and follow links: a live probe against marimo 0.24.2 wrote
`jn/escape.py` through a junction to outside the notebooks folder, and `Q3/Review.py` replaced `q3/review.py`
on Windows and came back with the variant's spelling. So Hailer checks every path before sending it: a local
kernel's names are compared with links resolved (a link out of the folder is outside; reads, writes and
listings stop there), a Windows kernel's names ignore case and keep the listing's spelling, and
`fake_marimo.py` records every path it was asked about instead of confining them
(`test_a_link_out_of_the_notebooks_folder_is_never_followed`, `test_a_case_variant_is_the_same_notebook_on_windows`,
`test_hailer_never_sends_a_path_outside_the_notebooks_folder`). marimo's `update` writes `\r\n` on Windows
while `create` stores the bytes as given: compare contents with `sandbox.notebook_digest`. `notebook.json`
without a `version` is the old host-path format: entries inside the notebooks folder are converted, everything
else is dropped (`test_an_old_state_file_is_converted_to_names`). Workspace setup (`uvx hailer init`, the
doctor's notebook row) may stay host-side.

## 13. A sandbox writes to the host only through the notebook allow-list

Until 2026-09-22 the docker kernel's `/work/notebooks` was a writable bind mount of the host's notebooks
folder. Notebook code could then plant files that other programs on the host run: git hooks and
`core.fsmonitor`, `.vscode`/`.idea`/`.devcontainer` settings, a `hailer.toml`, and for Python tools
`conftest.py`, `test_*.py` (pytest collects them), `sitecustomize.py`. Defending that took mount-layout
refusals (no `.git` or `hailer.toml` anywhere in the notebooks tree, not the workspace, not inside
`.hailer`), a scan for planted control files at every stop and in `doctor`, and warnings that could not stop
an editor acting while the kernel ran. Now the kernel's notebooks folder is a size-capped tmpfs of its own
(`--tmpfs /work/notebooks:uid=U,gid=G,mode=0700,size=512m`, not a volume: nothing to clean up) and
`hailer.notebook_sync.NotebookSync` copies notebooks through marimo's file API: in at start (after health,
before `_active_notebook` and the browser), back after every turn, on a notebook switch, every 15 s from a
daemon thread, and in `DockerKernel.stop` before the containers are removed, which the `finally` of every
session path reaches (`/exit`, EOF, Ctrl+C, errors, `--foreground`).

Keep these properties; each has a test in `tests/test_notebook_sync.py` unless named:

- **One allow-list for both directions**, `sync_refusal(name, text)`: a name `check_notebook_name` accepts
  (so no dot or dunder part), `.py` exactly, no 8.3 short-name part (`~` and a digit), at most four folders,
  no `TOOLING_NAMES` match (test, task and packaging runners, Sphinx/Django/gunicorn/IPython/Jupyter
  configuration, start-up hooks; one casefolded pattern list), no part named like a module Python would import
  instead (`module_names()`: stdlib, Hailer's site-packages, the image's packages, common ones such as pandas;
  a `json.py` next to a script shadows `json`), UTF-8 without NUL, at most 5 MiB, containing `import marimo`
  and `marimo.App(`; at most 500 files. `module_names()` scans site-packages instead of calling
  `importlib.metadata.packages_distributions()`, which took about 3 s on Windows (it reads every RECORD). Widen it only with a reason a host tool will not run the new kind of file
  (`test_the_allow_list_refuses_everything_else`). Case-insensitive duplicates are copied once.
  `DockerKernel.write_notebook` refuses a name the allow-list refuses, so the agent cannot create a notebook
  that would silently stay in the container (`test_a_docker_kernel_refuses_to_create_a_notebook_that_would_never_come_back`).
- **Host writes** are atomic (`write_atomically`: temp file in the same folder + `os.replace`), stay inside
  the notebooks folder and never pass through a symlink or junction, and never delete anything
  (`test_a_write_never_goes_through_a_link_or_junction`, `test_writes_here_are_atomic_and_leave_nothing_behind`).
- **Conflicts**: the digest (`notebook_digest`, newline-normalised) last copied in or out is remembered per
  name, in memory. A host file that differs from it is saved to `.hailer/notebook-backups/<name>.<time>.py`
  before it is replaced; if the backup fails, nothing is written.
- **Kernel code can replace the marimo server**, so every `/api/files` reply may be forged. Replies are
  capped (`marimo_client.MAX_REPLY_BYTES`, sized from the notebook cap; `Content-Length` checked, then a
  bounded chunked read) and every copy runs under one thread-local monotonic deadline
  (`marimo_client.request_deadline`), checked before each request and between reads, so a trickling server
  cannot hold the exit: 30 s for the copy in and the last copy back, 10 s and 50 MiB read per background
  pass. On the deadline Hailer warns, keeps the host copies and goes on (at stop: to container removal).
  Evidence: `test_a_trickling_reply_ends_at_the_deadline`, `test_a_reply_larger_than_the_cap_is_refused_without_reading_it`,
  `test_the_last_copy_has_one_deadline_whatever_the_server_does`, `test_a_pass_stops_at_its_deadline_or_byte_budget_and_leaves_the_rest`.
- **Failures warn, never raise**, and repeat once per message (a dead kernel does not print every 15 s). The
  chat routes `sandbox.notice` to its own console, which the composer queues from any thread.
- **Ordering**: sync-in runs inside `DockerRuntime.start`, so the chat and the browser never see an empty
  folder, and a Ctrl+C there stops the kernel. This costs one list + create per notebook on loopback before
  the composer renders (LEARNINGS §10); keep it bounded rather than moving it after `_active_notebook`. Stop
  joins the thread (a pass ends within 10 s) before the final copy; Ctrl+C during that copy skips it with a warning and the
  containers are still removed (`test_ctrl_c_during_the_last_copy_still_removes_the_containers`).

Linux UID mapping (`--user <uid>:<gid>`) stays: nothing is written through a mount any more, but the kernel
still reads the data folder as the user, and files only the owner may read (`0600` exports) would otherwise be
unreadable. The notebooks-folder mount rules and the planted-file scan are gone; the data-folder rules
(`config.docker_mount_problems`) remain. The unsafe-local runtime keeps the simple "last run by the isolated
docker kernel" warning (`kernel.docker_wrote_notebooks`): a notebook is code wherever it came from. Evidence at
the Docker level: `test_the_notebooks_folder_is_the_containers_own_and_only_notebooks_come_back` and
`test_stop_copies_the_last_edit_back_and_leaves_no_containers_network_or_token` in
`tests/test_docker_integration.py` (Windows 11, Docker Desktop 29.4.3, 2026-09-22, `HAILER_DOCKER_TESTS=strict`).

## 14. Isolation is the default; running unisolated is a decision the user writes down

Until 2026-09-22 the kernel ran in Hailer's own Python, as the user, unless `[kernel] runtime = "docker"`
was set, and `[kernel].pass_env` could hand secret-looking variables to notebook code. Code the model writes
therefore had the user's files and network by default. Now `docker` is the default (`KernelConfig`,
`MarimoServer`, the `init` template, a file without `[kernel]`), and the unisolated runtime is called
`unsafe-local`. Keep these properties:

- **The old name is refused, not mapped.** `runtime = "local"`, `HAILER_KERNEL=local` and `--kernel local`
  are a fatal config problem (`config.runtime_problem`; `--kernel` exits 2; `runtime_for` raises the same
  text) telling the user to write `unsafe-local` only if they accept that notebook code runs as them, with
  their files and network. Silently mapping `local` would keep people unisolated without a decision, and
  silently mapping it to docker would break their start without saying why. Evidence:
  `test_the_retired_local_runtime_is_fatal_and_says_how_to_opt_in` (test_config) and
  `test_the_retired_local_runtime_is_refused_with_the_way_to_opt_in` (test_cli).
- **No fallback, two ways on.** Without a usable Docker (no CLI, engine down, Windows containers) a start
  fails closed; every such hint ends with `kernel_docker.UNSAFE_LOCAL_OPTION` after the install/start advice
  (`test_docker_that_cannot_run_the_kernel_fails_closed`), except in `uvx hailer kernel ...`, which manages
  Docker itself (`test_kernel_pull`). A bad runtime is one `config` row, and its fix names `HAILER_KERNEL`
  when that set it (`test_a_retired_runtime_from_the_environment_is_one_config_row_with_an_environment_fix`). `init` still writes `docker` on a machine without
  Docker and says how to install it or opt in
  (`test_init_writes_docker_and_says_how_to_get_docker_only_when_it_is_missing`).
- **Visible every time.** The `Kernel:` line is `unsafe-local (runs as you; not isolated)` in yellow in the
  startup panel, `status`, `/status` and `--foreground`, and a WARN row in `doctor`; a start drops that row
  so the panel says it once (`test_kernel_line_of_the_unsafe_local_runtime_is_a_warning`,
  `test_doctor_has_a_kernel_row`).
- **No pass-through.** `[kernel].pass_env` is gone; an old file gets the ordinary unknown-key warning, which
  never quotes the value (`test_pass_env_is_gone_and_reported_as_an_unknown_key_without_its_value`). The
  secret-looking names stay withheld from an unsafe-local kernel.
- **The offline suite never needs Docker.** Because the default now reaches Docker, test configs that start
  or describe a kernel name their runtime (`KernelConfig(runtime="unsafe-local")` in the CLI and agent
  fixtures, `runtime="docker"` against `fake_docker.py`); only `tests/test_docker_integration.py` uses a real
  engine.
