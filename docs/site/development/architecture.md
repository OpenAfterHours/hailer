# Architecture {#architecture}

```
                   Model endpoint
                (OpenAI, or your own endpoint:
                 Responses API or Chat Completions)
                         ▲
                         │ HTTPS
                ┌────────┴────────┐
                │  Hailer         │   one Python process: a LangChain agent
Terminal ─────► │  conversation   │   (create_agent + ChatOpenAI), Hailer's system
 uvx hailer     │  + tools        │   prompt, the conversation kept in .hailer/
                │                 │
                │  marimo_execute, marimo_status, notebook_cells, notebook_list,
                │  notebook_create, notebook_open, notebook_close,
                │  list_periods, load_skill, read_skill_file, fetch_page
                └────────┬────────┘
                         │ HTTP + SSE  (/api/sessions, /api/kernel/execute),
                         │ with the server's token
                ┌────────▼────────┐
                │ Marimo runtime  │   scratchpad over the kernel globals +
                │ Polars · DuckDB │   marimo._code_mode for durable cells
                │ Python          │
                └────────┬────────┘   local (default): Hailer's own Python, as you
                         │            docker: a Linux container with copies of the
                    Browser UI        notebooks and, read-only, the data folder
```

The CLI runs the agent in its own process, with the model and provider taken from `hailer.toml`. The agent
has exactly the eleven tools above and nothing else: no shell, no file editing. The tools talk to the
running marimo server over plain HTTP. Code from the agent runs in marimo's *scratchpad*, a temporary
namespace that can read every notebook variable, and durable changes (new cells, edits, runs) go through
marimo's code-mode API so they appear immediately in the browser. The notebook file on disk is written by
marimo itself, never edited behind the kernel's back.

Where marimo and its kernel run is the *kernel runtime*, set by `[kernel] runtime`: `local` (the default)
starts marimo in Hailer's own Python, `docker` starts it in a container (see
[Isolated kernel (Docker)](../security/docker.md#isolated-kernel-docker)). The agent, the conversation and the API key stay in
the Hailer process either way. Every session starts its own marimo server, keeps it in memory for the
agent's tools as a *sandbox* (its URL and token, and its files), and stops it when the chat ends; nothing attaches to a server
another session started. A docker kernel keeps its notebooks in a folder of its own: the session copies
the workspace's notebooks in when it starts and copies changed marimo notebooks, and nothing else, back
after every turn, every 15 seconds and when it stops (`hailer.notebook_sync`).

## Design records

The current module contracts are in [INTERFACES.md](https://github.com/OpenAfterHours/hailer/blob/main/docs/INTERFACES.md). Read [LEARNINGS.md](https://github.com/OpenAfterHours/hailer/blob/main/docs/LEARNINGS.md) before changing the agent, providers, tools or conversation lifecycle, and [PLAN.md](https://github.com/OpenAfterHours/hailer/blob/main/PLAN.md) for the architecture and implementation history.
