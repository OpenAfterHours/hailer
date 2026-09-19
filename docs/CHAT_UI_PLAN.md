# Persistent terminal chat interface

Implementation plan, 2026-09-19. Steps 1-4 are implemented in the persistent composer change; optional streaming remains a follow-up. See the pull request for validation results and remaining manual terminal checks.

## Intended experience

Replace the interactive `You >` prompt with a persistent composer at the foot of the live interface. Submitted messages, Hailer replies, command results and activity appear above it. Keep ordinary terminal scrollback and text selection.

Illustrative layout (names and counts are examples):

```text
Hailer
Conversation resumed. /new starts fresh.

You
Chart revenue by month and region.

Hailer
Revenue increased steadily through the quarter...

Notebook: analysis.py | Model: risk-analyst-v3 | Context: 2 files
Ready
+------------------------------------------------------------------+
| Ask Hailer about your data...                                     |
+------------------------------------------------------------------+
Enter send | Alt+Enter newline | /help commands
```

During a turn, the activity line changes to `Thinking...`, `Using: marimo_execute`, or `Writing reply...`. The composer stays visible and accepts a draft of the next message. Enter during a running turn preserves the draft and explains that the current turn must finish or be cancelled; it does not submit or queue another request.

Recommended first release:

- Enter submits one message, appends it above the composer, and clears the submitted draft.
- Multiline paste remains one editable message. Alt+Enter inserts a newline; add Shift+Enter only where the terminal distinguishes it reliably.
- The input grows to a small maximum height, then scrolls internally. Up/Down move within multiline input and recall session history at its boundaries.
- Show a compact context strip above the input: active notebook, model, and loaded context-file count. `/context` and `/status` print full details into the conversation. Update the strip after notebook/model switches and `/reload`.
- Preserve existing cancellation and exit behavior: Ctrl+C during work cancels the turn; Ctrl+C while idle exits; Ctrl+D on empty input, Windows Ctrl+Z then Enter, and `/exit` also exit while idle.
- Keep `/clear` as a display operation and `/new` as a conversation reset. Resuming still shows a notice; replaying previous sessions on screen is a separate feature.
- Render completed answers with readable Markdown, lists and code blocks. Keep activity concise; do not dump tool arguments or entire context files into the conversation.

## What exists today

| Location | Current behavior | Implication |
| --- | --- | --- |
| `src/hailer/cli.py`: `_LineReader` | `PromptSession`, bracketed paste, in-memory history; plain input for pipes | Reuse the installed input library and preserve the plain mode. |
| `src/hailer/cli.py`: `_TurnDisplay` | Rich spinner during work; prints `TurnSummary.final_response` once | Replace interactive progress rendering without reintroducing duplicate answers. |
| `src/hailer/cli.py`: `ChatLoop` | Serial read/execute loop; command methods print directly to a console | Separate presentation from session operations so output can appear above a running composer. |
| `src/hailer/agent.py`: `HailerAgent` | Synchronous public methods drive async operations through a persistent `asyncio.Runner` | Expose async lifecycle methods for the interactive UI; avoid nested event loops. |
| `src/hailer/models.py`: `AgentEvent` | `message_delta`, `tool_call`, `status`; no explicit assistant-message boundaries | Existing events are sufficient for activity, but need refinement before visible token streaming. |
| `tests/test_cli.py` | Covers paste, history, terminal modes, cancellation, commands and notebook changes | Extend this coverage and retain its behavior guarantees. |

## Implementation approach

Use the existing `prompt_toolkit` dependency for an inline application, with `full_screen=False` and mouse capture disabled. Keep completed conversation output in the terminal's native scrollback. Rich remains useful for formatting completed content and for ordinary CLI commands.

The library supports [running an application on an existing asyncio loop](https://python-prompt-toolkit.readthedocs.io/en/stable/pages/advanced_topics/asyncio.html). Its [terminal output API](https://python-prompt-toolkit.readthedocs.io/en/stable/pages/reference.html#prompt_toolkit.application.run_in_terminal) can print above the active interface and redraw it afterwards. Use one serialized output path through this API; remove the independent Rich spinner from interactive chat.

Proposed responsibilities:

- `src/hailer/chat_ui.py`: composer layout, keyboard bindings, draft/history state, context/activity display, and serialized transcript output.
- `src/hailer/chat.py`: shared chat controller extracted from `ChatLoop`; owns commands, session counters, context loading, turn state and notebook synchronization. Both interactive and plain modes use these operations.
- `src/hailer/cli.py`: Typer entry points, terminal-mode selection, startup and marimo process ownership.
- `src/hailer/agent.py`: async lifecycle/turn entry points alongside the current synchronous facade.

The interactive session owns one event loop. Add async equivalents of `start`, `new_thread`, `run_turn` and `close`, extracting their existing coroutine bodies. The synchronous facade continues to use its runner for plain mode and existing callers; a given agent instance must use one lifecycle consistently. HTTP and SQLite resources are created, used and closed on that same loop. Python explicitly disallows [calling `Runner.run()` inside another running loop](https://docs.python.org/3/library/asyncio-runner.html#asyncio.Runner.run).

Keep exactly one active command or turn. Model requests run as cancellable async tasks; move blocking notebook discovery/session waits off the UI loop with results applied serially by the controller. Cancellation must await task cleanup, preserve history repair and pending notebook notices, and run active-notebook synchronization even after failure. A cancelled request must not be presented as undoing a tool action that already completed.

Route command results, errors, warnings and verbose logs through the same presentation path while chat is active. Retain the existing logging redaction. No background component should write directly over the composer.

## Delivery steps

1. **Prove the terminal layout.** Build a small fake-agent prototype with the framed composer, context strip and simulated activity/output. Validate inline rendering, native selection/scrollback, terminal resizing, long pasted input and typing while output arrives. Check Windows Terminal/PowerShell, an IDE terminal, and a Unix terminal. Treat stable redraw behavior with the existing CPR-disabled setting as a gate before integrating the agent.

2. **Separate presentation and execution.** Extract shared controller operations and introduce the async agent methods. Keep the existing plain interface working through the same session logic. Add focused lifecycle and cancellation tests, including a successful follow-up turn after cancellation and proper HTTP/SQLite cleanup. Avoid putting the entire synchronous agent into an arbitrary worker thread, where cancellation and resource ownership would become harder to control.

3. **Integrate the persistent composer.** Replace the interactive `_LineReader`/`_TurnDisplay` pair with the new UI. Append user submissions and completed replies above the input; display current activity in the live region. Route every slash command and error through it. Refresh context after `/reload`, `/model`, `/notebook`, and model-initiated notebook switches, including interrupted turns. Preserve a draft while work completes or fails.

4. **Finish compatibility and release.** Add `--plain` to force the existing line-oriented presentation, usable with both default chat and `hailer notebook`. Select plain mode automatically for non-TTY input/output and unsupported terminals. Keep status, doctor, login, exec and foreground notebook behavior intact. Document the keyboard controls and fallback in `README.md`, and update the contracts in `docs/INTERFACES.md`.

5. **Optional follow-up: stream assistant text.** Add explicit message identity and boundaries to agent events so interim commentary and separate model messages are distinguishable. Render an in-progress message above the composer and commit it once when complete. Reconcile it with `final_response` instead of printing the answer again. Validate multiple tool rounds, non-streaming providers, partial failures and cancellation. The persistent composer release does not depend on this extension.

## Validation and acceptance

Extend `tests/test_cli.py`, add `tests/test_chat_ui.py` for the terminal interaction, and extend `tests/test_agent.py` for async lifecycle behavior. Use the existing fake agents/gateway and injected prompt-toolkit input/output; no live API key is required.

Acceptance checks:

- The composer remains visible and responsive during slow turns; transcript output never overwrites or loses the draft.
- Submitted text appears once above the composer and reaches the agent once, including multiline paste and `/prompt` or `/skill` execution.
- Each final answer appears once. Activity accurately describes known events; a tool-call event alone is not shown as proof of successful completion.
- Cancellation stops the active request, leaves the interface usable, and permits a subsequent successful turn. Exiting restores cursor, bracketed-paste and terminal modes, including on exceptions.
- Slash commands, session resume/reset, token counters, notebook switching and marimo cleanup retain their existing behavior.
- Context information updates correctly. Token counts are shown only when available; cumulative usage is not labelled as a context-window percentage.
- Narrow windows, long output, Unicode, resizing and verbose logs do not corrupt the layout. Completed output remains selectable in native scrollback.
- Pipes and `--plain` emit readable linear output without interactive cursor controls.

Update terminal tests intentionally: interactive bracketed-paste mode now stays enabled throughout the live composer, including during a turn, then is disabled on shutdown. Preserve the existing no-mouse/no-alternate-screen guarantees and the no-CPR behavior unless the prototype demonstrates a documented compatibility need.

Run focused CLI/UI/agent tests during implementation, then `uv run --locked pytest`. The existing CI matrix covers Windows and Ubuntu on Python 3.12 and 3.13; supplement it with manual terminal checks because captured output cannot establish real selection, scrollback and redraw quality.

The first release is complete when steps 1-4 and these acceptance checks pass. Full transcript replay, searchable history, queued submissions, expandable tool results and token streaming can follow independently.
