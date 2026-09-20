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

Keep credentials out of prompts, logs and errors. Use Hailer's credential resolution and redaction
helpers; do not move keys into subprocess arguments or new plaintext config files.

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

Start warmup before kernel discovery/start. Once the server is healthy, render the interactive
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
