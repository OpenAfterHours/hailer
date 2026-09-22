# Startup responsiveness — 2026-09-20

The pause after the startup panel came from awaiting agent initialization before starting the
composer. Fresh-process agent setup took 2.6–4.5 seconds during investigation, largely in the
LangChain/OpenAI imports and their model classes. Notebook browser-session waiting was another
serial gate before chat appeared. Configuration and context loading took about 4 ms together.

## Implemented behavior

- Start dependency-only imports on a daemon worker before the kernel start.
- After server health, render the composer before agent setup and notebook waiting; those two
  preparations run concurrently. Kernel startup and its validation still precede the composer.
- Accept typing and paste immediately. Keep one submitted first message until preparation finishes,
  dispatch it once, and allow an editable next draft. Normal turns retain the existing busy behavior.
- Preserve failed startup input in the draft/history, report the error, and return exit code 1 when
  the user leaves. Intentional cancellation is not a startup failure.
- Cancel and settle startup work before closing agent resources or stopping an owned kernel.
  Shared import work may finish independently; it never owns an HTTP client or conversation store.

## Measurements

Windows, Python 3.13.7; locked environment: LangChain 1.4.1, langchain-core 1.6.3,
langchain-openai 1.6.2, LangGraph 1.2.11, OpenAI 3.15.0, marimo 0.24.2,
prompt-toolkit 3.0.53. These are local observations, not portable timing guarantees.

Three fresh-process samples from the final offline benchmark:

| Metric | Median | Range |
|---|---:|---:|
| First editable render | 0.807 s | 0.744–0.907 s |
| Ready to dispatch first message | 3.673 s | 3.444–4.475 s |
| First paste accepted | 0.014 s | 0.011–0.016 s |
| Longest loop pause after first render | 0.118 s | 0.109–0.158 s |

The benchmark starts its clock at entry to each fresh child script. It uses the real composer and
agent initialization, a dummy key, temporary state and a simulated 0.75-second notebook wait.
All socket connections are rejected before dependency warmup begins. It does not include uvx
installation/resolution, interpreter launch, a real kernel or browser startup. Input monitoring
starts before preparation and injects a paste on the first render; a regression test injects a
0.25-second UI freeze and requires it to be detected.

A separate live check started a real temporary local marimo server with no browser session:
input became editable **0.176 seconds after the startup panel**, accepted a draft while the notebook
wait was pending, and processed a terminal-input Ctrl+C in **0.075 seconds**. The owned server process
stopped, its state record was removed, and bracketed-paste mode was restored. No model request was
made. This was recorded-terminal input, not a physical keyboard/Windows console test. Browser
launch and Docker were not exercised by this live check.

## Reproduce and maintain

```text
uv run --locked python scripts/benchmark_startup.py --samples 3
uv run --locked pytest
uv run --locked hailer --verbose
```

Actual sessions log separate `checks_complete`, `kernel_ready`, `panel_shown`, `input_ready`,
`agent_ready`, `notebook_wait_complete` and `ready_to_answer` milestones. Notebook wait completion
may mean a nonfatal timeout; a session's existence alone does not prove its cells have initialized.
Cancellation stops notebook polling cooperatively, but an in-flight HTTP request or browser launcher
still has to settle before cleanup.

Regression coverage includes held agent/notebook preparation, typing and bracketed paste, exactly-once
submission, failure retention, exit codes, repeated cancellation, kernel cleanup, cold dependency
resource/network audits and independent event-loop lifetimes. Teammate review found and corrected
the startup exit-code regression and the original benchmark's missed-first-stall problem.
