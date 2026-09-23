# Kernel runtimes {#kernel-runtimes}

The kernel runtime decides where the notebook's Python runs. The conversation, model connection and API key
stay in Hailer's own process in either mode.

| | Docker (default) | Unsafe local (opt-in) |
|---|---|---|
| Setup | Docker Desktop or Docker Engine, and the kernel image (downloaded on first start) | Included with Hailer |
| Python execution | A non-root user in a Linux container | Hailer's Python, under your account |
| File access | Its own copy of the notebooks; data read-only | Your account's file access |
| Network access | Disabled by default; configurable | Your account's network access |
| Environment | No host environment variables | Secret-looking variables withheld; everything else passed |
| Notebook outputs sent to the model | Yes | Yes |

## Docker, the default

`uvx hailer init` writes `runtime = "docker"` under `[kernel]` in `hailer.toml`, and a workspace without a
`[kernel]` section uses Docker too:

```bash
uvx hailer
```

Hailer pulls the version-matched kernel image if needed. The kernel works on copies of your notebooks:
Hailer copies them in when it starts and copies changed notebooks back after each turn, every 15 seconds
and when it stops, and only marimo notebooks (see [Notebook copies](docker.md#notebook-copies)). For
configuration, corporate mirrors, the data-folder rules and platform details, follow
[Docker setup and operations](docker.md).

Without a usable Docker (not installed, not running, or Docker Desktop set to Windows containers), a start
stops with two ways on: install or start Docker, or opt into the unsafe local runtime. It never runs
notebook code on your machine by itself.

Docker limits the kernel's access to your machine. It does not stop the model receiving notebook tool outputs,
and a notebook written in Docker is still code: it has your account's access if you later run it with the
unsafe local runtime.

## Unsafe local, by your choice

`unsafe-local` runs marimo in Hailer's own Python, as you. Code the model writes can then read and change
every file your account can and reach the network, including services on your company's network. Choose it
only if you accept that, for example on a machine where Docker is not allowed:

```toml
[kernel]
runtime = "unsafe-local"
```

or for one session, `uvx hailer notebook --kernel unsafe-local`, or at setup
`uvx hailer init --kernel unsafe-local`. The startup panel, `/status` and `uvx hailer status` then show
`Kernel: unsafe-local (runs as you; not isolated)` in a warning colour, and `uvx hailer doctor` shows a
warning row. See [data handling and security](data-handling.md) for what Hailer still withholds.

The runtime was called `local` before; that name is refused with a message, so that nobody keeps the
unisolated runtime without writing `unsafe-local` themselves.

## Check the runtime in use

`/status` inside a conversation, or `uvx hailer status` in a terminal, shows the runtime.
`uvx hailer doctor` checks the configuration, Docker, the kernel image and the data folder without starting a
kernel or calling the model endpoint.

Each session starts its own kernel, so changed settings apply from the next `uvx hailer`. To remove
Docker kernels left behind by a Hailer that was killed (the next start also removes them):

```bash
uvx hailer kernel stop
```

See [notebooks and sessions](../using/notebooks.md) for when Hailer starts and stops a server.
