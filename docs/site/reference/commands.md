# Chat and CLI commands {#slash-commands}

| Command | What it does |
|---|---|
| `/help` | Show the command list. |
| `/status` | Model, provider, credentials source, thread id, token usage, marimo state, kernel runtime, web allowlist. |
| `/new` | Start a new conversation thread (context files are re-read). |
| `/model <name>` or `/model <provider>:<name>` | Switch model (and provider); starts a new thread. |
| `/notebook`, `/notebook list`, `/notebook new <name> [--empty]`, `/notebook open <name>`, `/notebook close [name]` | Show or switch the active notebook (see *Working with several notebooks*). |
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
switch to an alternate screen. Use `hailer --plain` or `hailer notebook --plain` for the line-oriented
`You >` interface. Pipes and terminals reporting `TERM=dumb` or `TERM=unknown` use plain input automatically.

Ctrl+C during a turn cancels the model request and keeps the composer open. A notebook command already
performing a blocking operation finishes its cleanup before another command can start. Cancellation does
not undo tool actions that have already happened. Ctrl+C while idle exits; Ctrl+D on empty input or
Ctrl+Z then Enter also exits. The conversation is stored in `.hailer/threads.sqlite`, which is how
`hailer` resumes it after a restart; `/new` (or `hailer --new`) discards it and starts another.
Completed replies are displayed once, with Markdown formatting; live activity does not print interim
model commentary. `/clear` clears the display without resetting the conversation.

Other subcommands:

| Command | What it does |
|---|---|
| `hailer` | Start or reuse this workspace's notebook server, open the notebook and chat in the terminal. |
| `hailer notebook [--port N] [--no-browser] [--keep-marimo] [--foreground] [--new] [--plain] [--kernel RUNTIME]` | The same session with extra startup options; marimo runs on the notebooks folder. `RUNTIME` is `local` or `docker`. |
| `hailer exec -c "code"` (or `hailer exec script.py`, `hailer exec -` for stdin) | Run Python in the active notebook's kernel yourself. |
| `hailer status`, `hailer doctor` | Configuration and state; the startup checks with fixes. |
| `hailer login <provider>`, `hailer logout <provider>` | Store or remove a provider's API key. |
| `hailer init [--force] [--kernel RUNTIME]` | Set up a workspace (see Quick start); `--kernel` writes `[kernel] runtime`. |
| `hailer kernel pull` | Download the kernel image for this Hailer (or `[kernel].image`). |
| `hailer kernel build [--tag <name>]` | Build the kernel image, discovering host pip/uv mirror settings. `--pip-config FILE` overrides discovery; `--no-host-config` disables it. See [corporate mirrors](../security/docker.md#building-with-corporate-mirrors). |
| `hailer kernel stop` | Stop this workspace's kernel (local or docker) and remove its containers and network. |

Global options go before the subcommand: `--verbose`, `--config <path>`, `--workspace <path>`, `--new`,
`--plain`, `--version` (for example `uvx hailer --workspace C:\projects\sales kernel stop`).
