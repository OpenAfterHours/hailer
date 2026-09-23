# Chat and CLI commands {#slash-commands}

| Command | What it does |
|---|---|
| `/help` | Show the command list. |
| `/status` | Model, provider, credentials source, thread id, token usage, marimo state, kernel runtime, web allowlist. |
| `/new` | Start a new conversation thread (context files are re-read). |
| `/model <name>` or `/model <provider>:<name>` | Switch model (and provider); starts a new thread. |
| `/notebook`, `/notebook list`, `/notebook new <name> [--empty]`, `/notebook open <name>`, `/notebook close [name]` | Show or switch the active notebook (see *Working with several notebooks*). |
| `/exec <code>` | Run Python in the active notebook's kernel and print its output, result and errors. Nothing is sent to the model. The notebook must be open in the browser; a pasted block keeps its lines and is dedented. |
| `/context` | List loaded context files, skills, prompts and the web allowlist. |
| `/skill <name> [message]` | Run a turn with a project skill attached. |
| `/prompt <name> [args]` | Send a saved prompt from `.config/hailer/prompts`. |
| `/reload` | Re-read `.config/hailer`; applies from your next message. |
| `/clear` | Clear the screen. |
| `/exit`, `/quit` | Exit Hailer (Ctrl+C at the prompt, Ctrl+D on an empty line, or Ctrl+Z then Enter, also exit). |

Interactive chat keeps a framed input box below the conversation. Your submitted messages, Hailer's
replies and command results appear above it. The context line shows the active notebook, model and
number of loaded context files; the activity line shows what Hailer is doing.

Enter sends; Alt+Enter inserts a newline. Pasting several lines keeps them in one editable message.
Up and Down move through multiline input and recall earlier messages at its boundaries (history stays
in memory for this session). You can draft your next message while Hailer works; Enter preserves that
draft until the current operation finishes or is cancelled. Requests are not queued.

The interface uses normal terminal scrollback and text selection. It does not capture the mouse or
switch to an alternate screen. Use `uvx hailer --plain` or `uvx hailer notebook --plain` for the line-oriented
`You >` interface. Pipes and terminals reporting `TERM=dumb` or `TERM=unknown` use plain input automatically.

Ctrl+C during a turn cancels the model request and keeps the composer open. A notebook command already
performing a blocking operation finishes its cleanup before another command can start. Cancellation does
not undo tool actions that have already happened. Ctrl+C while idle exits; Ctrl+D on empty input or
Ctrl+Z then Enter also exits. The conversation is stored in `.hailer/threads.sqlite`, which is how
`uvx hailer` resumes it after a restart; `/new` (or `uvx hailer --new`) discards it and starts another.
Completed replies are displayed once, with Markdown formatting; live activity does not print interim
model commentary. `/clear` clears the display without resetting the conversation.

Other subcommands:

| Command | What it does |
|---|---|
| `uvx hailer` | Start this session's own notebook server, open the notebook and chat in the terminal; the server stops when the chat ends. |
| `uvx hailer notebook [--port N] [--no-browser] [--foreground] [--new] [--plain] [--kernel RUNTIME]` | The same session with extra startup options; marimo runs on the notebooks folder. `RUNTIME` is `docker` or `unsafe-local` (notebook code runs as you). |
| `uvx hailer status` | Configuration, credentials and this workspace's running Docker kernels (whenever Docker is installed). |
| `uvx hailer doctor` | The startup checks with fixes: configuration, credentials, Docker, the kernel image and folders. It starts no kernel. |
| `uvx hailer login <provider>`, `uvx hailer logout <provider>` | Store or remove a provider's API key. |
| `uvx hailer init [--force] [--kernel RUNTIME]` | Set up a workspace (see Quick start). `hailer.toml` gets `[kernel] runtime = "docker"`, or the `--kernel` runtime. |
| `uvx hailer kernel pull` | Download the kernel image for this Hailer (or `[kernel].image`). |
| `uvx hailer kernel build [--tag <name>]` | Build the kernel image, discovering host pip/uv mirror settings. `--pip-config FILE` overrides discovery; `--no-host-config` disables it. See [corporate mirrors](../security/docker.md#building-with-corporate-mirrors). |
| `uvx hailer kernel stop` | Remove every Docker container and network of this workspace, whichever session started it, and say what was removed. |

Global options go before the subcommand: `--verbose`, `--config <path>`, `--workspace <path>`, `--new`,
`--plain`, `--version` (for example `uvx hailer --workspace C:\projects\sales kernel stop`).

## Environment variables {#environment-variables}

Everything else comes from `hailer.toml`; a variable wins over the file, and a command-line option over both.

| Variable | Overrides |
|---|---|
| `HAILER_CONFIG` | The config file (`--config`) |
| `HAILER_WORKSPACE` | The workspace folder (`--workspace`) |
| `HAILER_MODEL`, `HAILER_MODEL_PROVIDER` | `[model].name`, `[model].provider` |
| `HAILER_KERNEL` | `[kernel].runtime` (`--kernel` wins over it) |
| `HAILER_KERNEL_IMAGE` | `[kernel].image` |
| `HAILER_LOG_LEVEL` | `[hailer].log_level` |
| `HAILER_TRACING` | Nothing in the file: set it to let LangSmith tracing variables apply (see [Security](../security/data-handling.md#security)) |

The notebook and the notebooks and data folders are set only in `hailer.toml` (`[hailer].notebook`,
`notebooks_dir`, `data_dir`).
