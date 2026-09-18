# Proposal: replace the Codex SDK with LangChain

Status: **accepted and implemented on 2026-09-18** (branch `worktree-langchain-migration`). The decisions
in section 9 were: go with `create_agent`, remove everything Codex (so the ChatGPT-login path and the shell
are gone), keep custom endpoints and stored API keys. The text below is the proposal as written, kept as
the record of why; line references are to v0.2.2. The evidence is a runnable spike in `spikes/langchain/`
(34/34 checks pass on Windows 11, Python 3.12).

Where the implementation differs from the proposal:

- **`/model` still starts a new thread**, as it always has. Keeping a conversation across providers would
  replay one wire API's message history through the other, which the spike did not cover.
- **`merge_messages` and `parallel_tool_calls` were removed outright** rather than accepted as no-ops. An
  old `hailer.toml` still loads; the keys are reported as unknown and ignored.
- **Interrupted turns keep the user's message.** Instead of dropping it, the next message is merged into
  it (same message id), so strict chat templates never see two user messages in a row and "try again"
  still has something to retry. Interrupted tool calls get a result, as proposed.
- **Blocking tools run on daemon threads** (`hailer.tools._in_daemon_thread`) instead of sending marimo's
  kernel interrupt on cancel. That fixes the slow-exit edge from section 3 without depending on a marimo
  behaviour that could not be checked offline. The kernel call itself still runs to completion.
- **The agent owns its HTTP client** (`openai.DefaultAsyncHttpxClient`), which removes finding 2's
  "one event loop per process" constraint, and passes `http_socket_options=()` because langchain-openai's
  default keep-alive transport switches off httpx's proxy detection when a system proxy is present.
- **`hailer --new` and `/new` delete the conversation they replace**, so the checkpoint file does not grow
  with threads nobody can resume.
- **Measured result:** `src/` and `tests/` lost 4,933 lines and gained 1,666, net −3,267. The proposal
  estimated about −4,100. `agent.py` came out at 724 lines, not the 430 estimated (1,343 before): the
  user-facing error hints, the history repair and the docstrings that explain each LangChain finding take
  more room than guessed. The rest of the difference is the new agent tests and their fake gateway.

Verified live on 2026-09-18 with the real `hailer` CLI (Windows 11, Python 3.13, a headless marimo 0.24.2
server and a kernel session opened through headless Chrome):

- **Custom endpoint, Chat Completions:** against the strict fake gateway on loopback, the model's
  `marimo_execute("print(1 + 1)")` ran in the real kernel and `2` came back in the answer. The gateway saw
  only `messages`, `model`, `stream` and `tools`, the eleven tools as plain functions, the configured model
  on every request, the `api-version` query parameter and the custom header.
- **Resume:** a second `hailer` run printed "Resumed conversation (1 turns so far)", kept the thread id and
  sent the earlier turns to the endpoint.
- **OpenAI, Responses API, real key:** `gpt-5.5` called `marimo_status`, then `marimo_execute`, and
  answered `391` for 17 × 23 computed in the kernel (9,223 tokens in, 91 out).
- The first live run also caught a bug the suite had missed: a LangChain deprecation warning printed into
  the chat (the suite ignores `DeprecationWarning`). Fixed, with a test that fails on any warning in a turn.

Still owed: a physical Ctrl+C in a console. The suite raises SIGINT with `_thread.interrupt_main()`, which
takes the same path inside CPython, but nobody has pressed the key against the new agent yet.

## 1. Recommendation

**Move to LangChain's `create_agent`. Do not adopt deepagents now.**

Hailer does not use Codex as a coding agent. The system prompt tells the model the shell "cannot run
Python" and is for "light inspection" only, and it forbids editing notebook files. What Hailer uses is a
model loop with 11 tools of its own. Codex gives it that loop wrapped in a 381 MB native runtime, a
Responses-API-only wire protocol, an approval system and a sandbox, and most of `agent.py` and all of
`wire.py` exist to work around those four things.

A plain LangChain agent runs the same loop in-process. In the spike it talked directly to a strict
Chat-Completions-only gateway (the case PRs #6, #7 and #8 were about) with no bridge, no child process,
no approval handling and no side requests.

| | Today (Codex SDK) | After (LangChain) |
|---|---|---|
| Processes per session | 3: CLI, `codex app-server` (Rust), Hailer MCP server | 1 |
| Native executables shipped | 5 `.exe` files, 381 MB (`codex.exe` alone is 298 MB) | none |
| Chat Completions gateways | loopback HTTP bridge translating Responses ↔ Chat (`wire.py`, 1,058 lines) | `use_responses_api=False` |
| Requests for a one-tool turn | 2, plus one reviewer request per command and tool call when Codex's reviewer is on | exactly 2 (measured) |
| Source lines (`src/hailer`) | 8,222 | about 6,000 (−27%) |
| Test lines | 7,317 | about 5,500 (−25%) |
| Resolved packages (Windows, 3.12) | 64 | 81 |

The package count goes up by 17. That is the main cost on the dependency side; section 7 lists the others.

## 2. Why the Codex SDK keeps breaking for users

I could not see the reports behind "not working for some users", so this is the list of failure classes
the repository history and the installed package show, not a diagnosis of a specific report.

| Failure class | Evidence | Gone after the move? |
|---|---|---|
| A 298 MB unsigned `codex.exe` (plus four more executables, one of them a sandbox *setup* binary) must be launched from a user-writable venv. Application control and endpoint protection on managed Windows machines commonly block exactly that. | `.venv/Lib/site-packages/codex_cli_bin`: 381 MB | Yes. Nothing is executed; the agent is Python making HTTPS calls. |
| Codex speaks only the Responses API. Gateways that offer only Chat Completions need Hailer's bridge, which re-implements streaming, tool calls, namespaces and usage. | `wire.py`; PRs #6, #7, #8 | Yes. Both wire APIs are one constructor flag. |
| Codex sends requests Hailer did not ask for: the `codex-auto-review` reviewer model on every command and tool call, a second system message, an environment-context user message. Strict gateways reject them (`422 model: String should match pattern`). 0.2.2 works around the reviewer and the extra messages, at the cost of the private-API code in the next row. | `agent.py:762-900`, `merge_messages` | Yes. The spike's gateway saw one system message, one user message, the configured model only. |
| The SDK cannot express the approval settings Hailer needs, so Hailer builds wire params through `codex._client`, swaps a private `_approval_handler`, and answers `mcpServer/elicitation/request` itself. Any SDK upgrade can break this, hence the `==0.154.0` pin. | `agent.py:802-900` | Yes. In-process tools need no approval protocol. |
| The user's own `~/.codex/config.toml` (desktop app plugins, MCP servers) leaks into Hailer and can stop the app-server from starting. | `build_config_overrides`, `HAILER_CODEX_HOME`, README troubleshooting | Yes. |
| Errors arrive as Codex prose; Hailer recovers the HTTP status and the gateway's words with six regexes and four lists of signal phrases. | `agent.py:397-672` | Yes. `openai.APIStatusError` carries `status_code` and `body`. |
| Two child launches (`python -m hailer.mcp_server`, the Codex binary) complicate single-file packaging. | moonlit trial, 2026-09-15 | Partly. The MCP child goes; the marimo child stays. |

## 3. What the spike verified

`spikes/langchain/fake_gateway.py` is deliberately hostile: only `POST /v1/chat/completions` exists, unknown
fields and unknown model names get a 422, a custom header and an `api-version` query parameter are
required, and it can refuse `stream: true`. `lc_agent.py` is a prototype `HailerAgent` with the surface
`cli.py` uses today.

| # | Check | Result |
|---|---|---|
| A | Streaming turn with one tool call: tool runs in-process, `tool_call` event and text deltas reach the callback, only `/chat/completions` is called, no `stream_options` or `parallel_tool_calls` on the wire, configured model on every request, exactly 2 requests, tools arrive as plain functions, one system message, token usage captured | 11/11 |
| B | Gateway that rejects `stream: true`: `disable_streaming=True` sends `stream: false`; the agent's event stream and usage still work | 4/4 |
| C | Conversation resumes from a SQLite file after a restart; unknown thread id falls back to a new thread; `/model` switch keeps the conversation | 3/3 |
| D | Errors are typed: 422 bad model, 401 bad key, 422 rejected field, 404 for Responses on a chat-only gateway, connection refused; each status error costs one request (Codex retries a 422 five times); the agent recovers afterwards | 6/6 |
| E | Ctrl+C during a model call, streaming and not: `KeyboardInterrupt` within 0.01 s, the gateway sees the HTTP stream aborted, the next turn works | 5/5 |
| F | Ctrl+C while a tool runs: prompt interrupt; the checkpoint is left with a tool call that has no result; a 12-line repair step fixes it; the abandoned tool thread finishes on its own | 5/5 |

Four findings shaped the design:

1. **Run the graph under asyncio, behind a sync facade.** LangGraph's synchronous `stream()` blocks in an
   untimed `concurrent.futures.wait`. On Windows Ctrl+C surfaced 5 to 9 seconds late, after the turn had
   finished. This is the same problem the Codex SDK had, which `_pump_events` works around today. With
   `asyncio.Runner().run(graph.astream(...))` the interrupt cancels the task and aborts the HTTP request.
   The pump thread and queue are not needed.
2. **One event loop per session.** `langchain-openai` caches its async HTTP client globally; agents on
   different loops produced `Event loop is closed`. One `asyncio.Runner` owned by `HailerAgent` for the
   life of the CLI avoids it, and `/model` rebuilds the graph on the same loop.
3. **Interrupting a tool leaves a dangling tool call.** Real endpoints answer 400 to an assistant
   `tool_calls` message with no tool result. LangChain has no built-in fix (deepagents has one). The
   repair in `lc_agent.py:_arepair` runs before each turn.
4. **Never leave `use_responses_api` or `base_url` unset.** Checked against langchain-openai 1.6.2: with
   the flag unset, a model named `gpt-5-codex` or `gpt-5.5-pro` is routed to `/responses` even on a custom
   `base_url`; an explicit `False` wins. With `base_url` unset, a stray `LANGSMITH_GATEWAY` environment
   variable sends requests to `gateway.smith.langchain.com`; an explicit `base_url` wins. `build_model`
   therefore always passes both, including `https://api.openai.com/v1` for the built-in provider.

One edge to handle in the implementation: after Ctrl+C during a long `marimo_execute`, `close()` waited
7 s for the abandoned tool thread. Sending marimo's kernel interrupt when a turn is cancelled fixes that
and is better behaviour than today, where the kernel keeps running.

Not covered by the spike: the Responses API path against api.openai.com (needs a real key), a real
console Ctrl+C (the spike raises SIGINT with `_thread.interrupt_main()`, which takes the same CPython
path), and a marimo end-to-end turn. All three are in the verification step of the plan.

## 4. Target design

The seam already exists. `cli.py` uses `HailerAgent.start / new_thread / set_model / interrupt / close /
run_turn(...) -> TurnSummary` plus a few attributes, and every tool is already a plain method on
`HailerTools` that returns text. Both stay; what is behind them changes.

```python
# agent.py, the part that replaces wire.py, the -c override builder and the child environment
def build_model(provider: ProviderConfig, model: str, api_key: str | None, effort: str | None) -> ChatOpenAI:
    kwargs = {
        "model": model,
        "api_key": api_key,
        "base_url": provider.base_url or "https://api.openai.com/v1",   # always explicit (finding 4)
        "use_responses_api": provider.wire_api == "responses",          # always explicit (finding 4)
        "stream_usage": provider.stream and provider.stream_options,
    }
    if headers := resolved_headers(provider):      # http_headers + env_http_headers
        kwargs["default_headers"] = headers
    if provider.query_params:
        kwargs["default_query"] = dict(provider.query_params)
    if not provider.stream:
        kwargs["disable_streaming"] = True
    if effort:
        kwargs["reasoning_effort"] = effort
    return ChatOpenAI(**kwargs)

graph = create_agent(
    build_model(...),
    tools=hailer_tools(config),                    # HailerTools methods, in-process
    system_prompt=system_prompt(config, bundle),   # unchanged function
    middleware=[SummarizationMiddleware(model=same_model, trigger=("tokens", 100_000))],
    checkpointer=AsyncSqliteSaver(conn),           # <workspace>/.hailer/threads.sqlite
)
```

`run_turn` iterates `graph.astream(..., stream_mode=["updates", "messages"])` and maps updates to the
existing `AgentEvent` kinds and `TurnSummary` fields, as in `spikes/langchain/lc_agent.py` (about 40 lines).
The newer `stream_events(version="v3")` API is marked experimental upstream, so the plan does not use it.

The summarisation trigger has to be an absolute token count. The middleware never fires with its default
`trigger=None`, and the `("fraction", f)` form raises `ValueError` for a model name LangChain has no
profile for, which is every custom gateway deployment name (checked). A default of 100,000 tokens with a
`[model]` key to override it covers 128k-context models.

### Configuration mapping

Existing `hailer.toml` files keep working. Keys that lose their meaning are still accepted for one release
and ignored, so nobody's config fails to load.

| `hailer.toml` | Today | After |
|---|---|---|
| `[model].name`, `reasoning_effort` | `-c model=`, `-c model_reasoning_effort=` | `ChatOpenAI(model=, reasoning_effort=)` |
| `base_url` | `model_providers.<id>.base_url`, or the bridge URL | `base_url=` |
| `wire_api = "responses" / "chat"` | Codex native / loopback bridge | `use_responses_api=True / False` |
| `stream = false` | bridge only, chat only | `disable_streaming=True`; works for both wire APIs, so the "chat only" validation goes |
| `stream_options = false` | bridge omits the field | `stream_usage=False` |
| `parallel_tool_calls = false` | bridge omits the field | never sent unless asked for (spike A6): key becomes a no-op |
| `merge_messages` | collapses Codex's two system and two user messages | not needed, Hailer sends one of each (spike A10): no-op |
| `env_key` | key placed in the Codex child environment | read by `secrets.resolve_provider_key`, passed as `api_key=` in-process |
| `http_headers`, `env_http_headers`, `query_params` | attached by Codex | `default_headers=`, `default_query=` |
| `requires_openai_auth` | Codex flag | dropped |
| `[hailer].codex_home`, `HAILER_CODEX_HOME` | isolates `~/.codex` | dropped |
| `[web].allow_shell_network` | Codex network proxy for the shell | dropped with the shell |

### Behaviour that gets simpler

- **`/model <provider>:<name>` keeps the conversation.** A thread is a message list, not a Codex object
  bound to a provider (spike C3). `set_model` no longer starts a thread and the CLI branch for it goes.
- **Prompt and context changes apply on the next turn.** The system prompt is passed on every model call
  instead of being fixed at `thread_start`. The prompt-hash bookkeeping in `session.py` and `cli.py`, and
  the "use /new to apply" warning, can go. `/reload` takes effect immediately.
- **`/skill` needs no `SkillInput`.** The skill's `SKILL.md` text goes in the turn preamble.
- **Secrets stay in the process.** No child environment, no `-c` argv to keep keys out of.
- **TLS through corporate proxies.** The `openai` SDK's HTTP layer (`httpx2`) uses the operating system
  trust store by default, so a root CA installed by a TLS-inspecting proxy is honoured.

## 5. Code that can be removed

Line numbers are for `main` at `2ebb8d4` (v0.2.2).

### Deleted outright

| What | Where | Lines |
|---|---|---|
| Responses ↔ Chat Completions bridge: request translation, SSE translator, tool namespace map, loopback HTTP server | `src/hailer/wire.py` (whole file) | 1,058 |
| Its tests (50 tests) | `tests/test_wire.py` (whole file) | 939 |
| Codex feature trimming, MCP constants, provider passthrough fields | `agent.py:54-95` | 42 |
| TOML rendering for `-c` overrides (`toml_key`, `toml_value`, `_override`) | `agent.py:97-139` | 43 |
| `read_user_codex_config` (parsing `~/.codex/config.toml` to disable the user's plugins and MCP servers) | `agent.py:146-157` | 12 |
| `chat_completion_upstreams` | `agent.py:173-190` | 18 |
| `build_config_overrides` | `agent.py:193-276` | 84 |
| `NO_PROXY` handling for the bridge and `build_child_env` | `agent.py:279-330` | 52 |
| Duck-typed notification helpers, `_pump_events` thread, `_StreamError`, Codex factory | `agent.py:680-755` | 76 |
| Reviewer decision, `codex.account()` lookup, raw `ThreadStartParams` building via `codex._client`, private `_approval_handler` swap, elicitation answers | `agent.py:762-900` | 139 |
| MCP server wrapper: `build_server`, `_setup_logging`, `main`, `_config_from_env`, the `hailer-mcp` script, the `HAILER_CONFIG` / `HAILER_WORKSPACE` / `HAILER_LOG_LEVEL` / `HAILER_MARIMO_TOKEN` environment hand-off | `mcp_server.py:91-100, 544-668` | about 135 |

### Replaced by something much smaller

| What | Where | Today | After |
|---|---|---|---|
| `map_exception`: six regexes and four signal-phrase lists that recover an HTTP status and the gateway's words from Codex prose | `agent.py:397-672` | 276 | about 70, switching on `APIStatusError.status_code`, `APIConnectionError`, `APITimeoutError`. The user-facing hints stay. |
| `HailerAgent`: bridge lifecycle, key plumbing into the child, reviewer choice, two thread-opening paths, pump-thread turn loop | `agent.py:907-1343` | 437 | about 190 (the spike's class is 130) |
| Tool registration: 11 wrapper functions whose only content is a docstring | `mcp_server.py:557-638` | 82 | about 15: move the docstrings onto the `HailerTools` methods and register them with `StructuredTool.from_function`. The module becomes `tools.py`. |

`agent.py` goes from 1,343 lines to about 430. `system_prompt()` and `_login_hint()` are kept unchanged.

### Smaller removals

| What | Where |
|---|---|
| `codex_home` / `HAILER_CODEX_HOME`, `requires_openai_auth`, `allow_shell_network`; the "only applies to `wire_api = "chat"`" validation for `stream`, `merge_messages`, `stream_options`, `parallel_tool_calls` | `config.py:62, 74, 334, 362, 372, 389, 519-533`; the matching fields and the `uses_chat_completions` property in `models.py:112-145, 160, 178` |
| `_SHELL_WRAPPERS` and `_display_command` (strip Codex's shell wrapper), the `command` / `command_output` event kinds, the "Reviewer" status row, "Codex will use its existing ChatGPT login" messages | `cli.py:108-128, 572, 643-647, 865, 907, 1487`; `models.py:229` |
| Prompt-hash bookkeeping and the "use /new to apply" warning | `session.py:68-70, 97-100` and the `prompt_hash` argument of `save_session`; `cli.py:131-135, 682-703` |
| New-thread-on-provider-switch branch | `agent.py:1131-1149`; `cli.py:933-936` |
| `tests/test_agent.py` (93 tests): 34 test Codex plumbing and are deleted (overrides, child env, TOML, reviewer, elicitation, bridge, account lookup); 32 test error-prose mapping and shrink to about a dozen status-code tests; 6 system-prompt tests stay as they are; 21 lifecycle and turn tests are rewritten against a fake chat model | 1,448 lines to about 550 |
| README: Codex login and reviewer (227-240), bridge mechanics (286-317), sandbox and "your own Codex configuration" (490-503), five troubleshooting rows (518-529). PLAN.md and docs/INTERFACES.md sections on Codex overrides and approvals. | docs |

### Dependencies

Removed: `openai-codex==0.154.0` with `openai-codex-cli-bin` (381 MB), and `mcp` with the packages only it
needed (`mcp-types`, `jsonschema`, `jsonschema-specifications`, `referencing`, `rpds-py`, `pyjwt`,
`cryptography`, `cffi`, `pycparser`, `opentelemetry-api`, `sse-starlette`, `pywin32`, `attrs`).

Added: `langchain>=1.4,<2`, `langchain-openai>=1.6,<2`, `langgraph-checkpoint-sqlite>=3.1,<4`, which bring
33 packages. `uv pip compile` resolves the new set together with `marimo==0.24.2` for Windows and
Python 3.12 without conflicts.

### Concepts that disappear

These cost more than their line counts, because each one needed live investigation to get right:
the child process and its JSON-RPC notifications; the loopback bridge and `NO_PROXY`; the second child
process and its environment hand-off; approval policy, reviewer and elicitation; interference from
`~/.codex/config.toml`; error prose parsing; secrets routed through a child environment; the exact-version
pin on a private SDK surface (`codex._client`, `_approval_handler`, `_sandbox_mode`).

## 6. deepagents: not now

`create_deep_agent` is `create_agent` plus a fixed set of middleware. I ran the same one-tool turn through
deepagents 0.7.15 against the fake gateway (`spikes/langchain/deep_probe.py`):

- Every request carried 8 tools Hailer did not define: `ls`, `read_file`, `write_file`, `edit_file`,
  `delete`, `glob`, `grep`, `task`. The tool schema was 10,721 bytes per request; the one tool Hailer
  passed in accounts for 194 of them. The state gained a `files` key.
- The install is 62 packages instead of 47. `langchain-anthropic` and `langchain-google-genai` are hard
  dependencies (PyPI metadata), and they bring the `anthropic` and `google-genai` SDKs.
- It is pre-1.0 and moves quickly: 127 releases since July 2025, 27 of them in the last three months
  (PyPI). Its release notes record breaking changes in 0.5, 0.6 and 0.7.
- The filesystem and sub-agent middleware cannot be switched off in `create_deep_agent`; its own docs
  point to plain `create_agent` plus hand-picked middleware for that. Its skills feature reads skill
  bodies through `read_file`, so it needs the file tools; Hailer's `load_skill` does not.
- Its summarisation default for a model without a profile triggers at 170,000 tokens, which is past the
  window of a 128k model behind a gateway.

The last two points are from reading the 0.7.15 source and docs, not from a test.

| deepagents feature | Does Hailer need it? |
|---|---|
| File tools over a virtual or real filesystem | No. The prompt forbids editing notebook files, and analysis happens in the kernel. These tools rebuild the generic coding agent the prompt says Hailer is not. |
| `task` sub-agents | No. There is one analyst and one active notebook; parallel sub-agents would race on it. |
| `skills=` (SKILL.md folders) | Hailer has this: `context.py`, `load_skill`, `read_skill_file`. |
| `memory=` (AGENTS.md) | Hailer has this: `.config/hailer/context`. |
| Summarisation | In LangChain itself (`SummarizationMiddleware`). |
| Dangling tool-call patch | 12 lines in `HailerAgent`. |

Choosing plain `create_agent` keeps the door open, because deepagents' parts are ordinary middleware
that can be added one at a time. Revisit if Hailer grows long autonomous runs (for example "produce the
whole monthly pack" with dozens of tool calls) where planning and context isolation start to matter.

Two alternatives I considered and rejected. Keeping the MCP server and loading it through
`langchain-mcp-adapters` would keep the child process and its environment hand-off for no benefit. Using
the `openai` SDK directly with a hand-written tool loop has the fewest dependencies, but Hailer supports
two wire APIs with different message, tool-call and streaming shapes, so that route means rewriting much
of what `wire.py` knows; LangChain also gives resume, summarisation and a route to non-OpenAI providers.

## 7. What is lost, and the risks

1. **ChatGPT-login authentication has no supported replacement.** Today the built-in `openai` provider
   works with no API key if the machine has a Codex or ChatGPT login. The public `ChatOpenAI` needs an API
   key. `langchain-openai` 1.6.2 does contain a ChatGPT-OAuth model (`_ChatOpenAICodex`, with
   `langchain_openai.chatgpt_oauth.login_chatgpt()`), but it is underscore-private, labelled experimental
   and unofficial, Responses-only, and keeps its own token file, so users would log in again. I would not
   build a release on it. Gateway users are unaffected. This is the one product decision in the proposal
   (section 9).
2. **The model loses its shell.** The prompt already restricts it to listing files, and `marimo_execute`
   and `list_periods` cover that. The sandbox never was Hailer's real boundary, because `marimo_execute`
   runs arbitrary Python in an unsandboxed kernel; removing the shell makes the attack surface smaller.
   If file listing is missed, a `list_files(pattern)` tool is about 15 lines.
3. **Compaction becomes Hailer's job.** `SummarizationMiddleware` does it. Its side request uses the
   model we pass, so it goes out under the user's configured model name, unlike `codex-auto-review`.
4. **Existing conversations do not carry over.** Codex thread ids mean nothing to the new store. The
   CLI already handles this ("Previous conversation could not be resumed; started a new one.").
5. **More packages (64 to 81), several with compiled wheels** (`orjson`, `ormsgpack`, `xxhash`,
   `zstandard`, `tiktoken`, `jiter`, `regex`, `uuid-utils`, `sqlite-vec`). These are extension modules of
   the kind Hailer already ships (`polars`, `duckdb`, `pydantic-core`), not standalone executables.
6. **`langsmith` is installed transitively.** It sends nothing unless tracing environment variables are
   set. Hailer handles local data, so it should set `LANGSMITH_TRACING=false` at start-up unless the user
   opts in, so that a stray variable cannot ship prompts and tool results to a third party. The explicit
   `base_url` from finding 4 closes the `LANGSMITH_GATEWAY` variant of the same problem.
7. **LangChain moves fast.** The 1.x line has a stability policy; pin `<2` and rely on the offline
   fake-gateway test to catch wire regressions on upgrade.
8. **`ChatOpenAI` targets the official OpenAI wire format only.** Known upstream issues with
   OpenAI-compatible gateways, none of which the spike hit: tool-call arguments lost when a provider
   fragments streamed chunks unusually (langchain #35514, #35782; `stream = false` is the workaround and
   Hailer already has that switch); a deployment whose name starts with `o1`, `o3` and so on gets the
   `developer` role instead of `system`; non-standard `reasoning_content` is ignored, which Hailer does
   not display anyway.
9. **Token counts on strict gateways.** With `stream_options = false`, usage arrives only if the gateway
   puts it on the last chunk unasked (the spike's gateway does). That is the same limitation as today.
10. **The checkpoint file grows.** LangGraph writes a checkpoint per step and never prunes. Hailer only
    ever resumes the latest thread, so `/new` can delete the previous thread's rows (`adelete_thread`).

## 8. Plan

One branch, a hard cut, released as 0.3.0. The project is days old and the agent seam is narrow, so a
period with two backends would double the test surface to protect very little. Each step leaves the test
suite green.

1. **Tools in-process.** Rename `mcp_server.py` to `tools.py`; move the tool descriptions onto
   `HailerTools`; add `hailer_tools(config) -> list[BaseTool]`. Delete the MCP wrapper and the `mcp`
   dependency. Existing `HailerTools` tests carry over.
2. **New `agent.py`.** `build_model` (explicit `base_url` and `use_responses_api`), `HailerAgent` on
   `create_agent` with `AsyncSqliteSaver` and one `asyncio.Runner`, the repair step,
   `SummarizationMiddleware` with a token trigger, status-code error mapping, marimo kernel interrupt on
   cancel, `LANGSMITH_TRACING=false` unless the user opted in, `/new` deleting the old thread's
   checkpoints. Unit tests use LangChain's fake chat models; the spike's fake gateway becomes an offline
   integration test (it is a stdlib HTTP server, so the suite stays offline).
3. **Delete.** `wire.py`, `tests/test_wire.py`, the config keys and CLI rows listed in section 5.
4. **Docs and packaging.** README, PLAN.md, docs/INTERFACES.md, `hailer.toml` template, `pyproject.toml`,
   `uv.lock`, release notes that name the ChatGPT-login change.
5. **Live verification before merge.** A real OpenAI key over the Responses API; one affected
   gateway user on a pre-release build; a real Ctrl+C in a Windows console during a model call and during
   `marimo_execute`; a marimo end-to-end turn (create a cell, read it back); resume after restart.

Estimated size: steps 1 to 4 are two to three focused days; step 5 depends on getting an affected user
to try a build.

## 9. Decisions needed

1. **Is losing the ChatGPT-login path acceptable?** I recommend yes: it is the feature that ties Hailer
   to the Codex runtime, and as far as the history shows, the users who are failing are on API keys and
   gateways. If it must stay, there are two routes. Keeping Codex as an optional backend
   (`pip install hailer[codex]`) behind the same `HailerAgent` interface keeps all of the Codex-bound
   source in section 5 (about 2,400 lines) and its tests, so nothing gets simpler. Trying LangChain's
   private `_ChatOpenAICodex` costs about 30 lines but rests on an unofficial API that can change in any
   patch release.
2. **Drop the shell tool outright**, or add a small `list_files` tool in the same release? I recommend
   dropping it and adding the tool only if someone misses it.
3. **What exactly fails for the affected users?** If it is the executable being blocked, or a gateway
   rejecting Codex's requests, this proposal fixes it. If it is something else (for example marimo or
   keyring), it will not, and I would want to see one report before starting step 1.
