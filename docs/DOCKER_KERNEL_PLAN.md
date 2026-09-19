# Plan: let users choose whether the notebook kernel runs in Docker

Status: **implemented on 2026-09-19** (branch `worktree-docker-kernel`, phases 0 to 4). The decisions in
section 9 were taken as recommended: `local` stays the default and docker is opt-in; the image is published
to GHCR by the release workflow, with `uvx hailer kernel build` as the fallback; `[kernel].network = true`
is offered, off by default and shown in the `Kernel:` line; servers Hailer starts get a token, while
servers users start themselves with `--no-token` keep working in the local runtime; Podman is out of
scope. The text below is the proposal as written on 2026-09-18, kept as the record of why; line references
are to `main` at v0.2.4 (7e7f746). The evidence is two throwaway spikes run that day on Windows 11 with
Docker Desktop 29.4.3, which drove a containerised marimo with Hailer's own `MarimoClient` (section 3). The
user-facing description is the README's *Isolated kernel (Docker)* section; the module contracts are in
`docs/INTERFACES.md`; the verified findings are in `PLAN.md`.

Where the implementation differs from the plan:

- **Three modules, not one.** `hailer.kernel` holds what both runtimes share (`PathMap`, `kernel.json`,
  the one liveness check `live_kernel_state`, the start guard, `LocalRuntime`, `runtime_for`);
  `hailer.kernel_docker` holds `DockerRuntime` and `hailer kernel stop` and is imported only when docker is
  asked for; `hailer.kernel_image` builds and pulls the image. The forwarder ships in the package as
  `hailer._forward`, as planned.
- **The runtime protocol changed shape.** `prompt_notes()` became a function of the settings in effect
  (`kernel.runtime_prompt_notes`), so building the prompt never touches Docker. A `prepare(say)` step was
  added: the image download (with docker's progress) and any warning print before the start's spinner.
- **Cells run when a notebook opens.** The image writes `/work/.marimo.toml` with
  `auto_instantiate = true`. marimo 0.24's edit page takes that setting from the user configuration in its
  working directory, not from a project `pyproject.toml`, and the user's own marimo configuration is never
  mounted (it can hold API keys).
- **Mount-layout rules (new).** Section 4.4's "never mounted" list assumed the default layout. A notebooks
  folder equal to the workspace let notebook code rewrite `hailer.toml` back to `local`, so docker mode now
  refuses a notebooks or data folder that is, or contains, a drive root, the home folder, the workspace,
  `.hailer`, the config file or the context, skills and prompts folders, or that is, contains or sits inside
  a credential folder (`~/.config`, `~/.ssh`, `~/.aws`, `~/.azure`, `~/.gnupg`, `~/.docker`, `~/.kube`,
  `%APPDATA%`, `%LOCALAPPDATA%`); the notebooks folder may not sit inside Hailer's folders either, and the
  data folder may not sit inside the notebooks folder. UNC paths are an error (mapped network drives stay a
  warning), and `marimo_url` cannot be combined with docker. All of these live in one function,
  `config.docker_mount_problems`, which `validate` reports and every start runs again (`--foreground`, which
  never validates, once started a kernel whose "read-only" data sat in the writable notebooks mount).
- **Files that run code later (new).** Notebook code can write `.git` (hooks, `core.fsmonitor`),
  `.vscode`, `.idea`, `.devcontainer` or `hailer.toml` into the notebooks folder, which git, editors (VS Code
  scans nested repositories) or Hailer would act on later. A notebooks folder that already is a git
  repository, or contains a nested repository or Hailer workspace, is refused in docker mode; when a docker kernel stops (chat end, `--foreground` end,
  `kernel stop`) and in `doctor`, Hailer names any of these throughout the notebooks tree in a loud warning.
  Nothing is removed for the user.
- **`kernel.json` records ids and settings.** Besides names and the image it holds the container and
  network ids Docker printed, and the settings the kernel was started with (network, mounts, memory,
  cpus). Everything is removed and inspected by id, so an old handle cannot remove a newer kernel that
  reuses the names.
- **Leftovers are never removed while running.** Section 4.5 had a start remove containers that fail the
  health and token check and start fresh. A start now refuses instead when a kernel container it has no
  working record of is still running (a probe that got it wrong must not destroy a kernel in use), and
  points at `uvx hailer kernel stop`. It also refuses while the recorded kernel, docker or local, does not
  answer but is not provably gone (containers or process still there: busy, stuck, suspended), and keeps the
  record, which is only deleted once it is provably gone or the cleanup succeeded. Stopped leftovers are
  still cleaned up. `kernel stop` racing another terminal's cleanup treats "removal ... already in
  progress" as done, retries a network whose containers are still detaching, and checks what is really
  left before reporting a failure; only object-specific "not found" answers count as gone (a broken docker
  context is reported as Docker not reachable).
- **Reuse only with the same settings.** `uvx hailer notebook` refuses a kept kernel started with other
  folders, network, image, memory or cpus (`The running kernel was started with other settings (...)`,
  with the stop hint); bare `uvx hailer` follows the same rules after integration with the current chat startup. The network setting the prompt and
  the `Kernel:` line show always comes from the kernel in use (`kernel.attach_runtime`), and asking for
  `local` attaches to a running docker kernel, the more isolated of the two.
- **The chat keeps its server in memory.** `hailer notebook` hands the chat and the agent's tools the
  server it started or reused, so a `kernel.json` rewritten by another terminal cannot redirect them;
  bare `uvx hailer` uses the same startup and retains the server too, in both composer and plain mode.
- **More withheld from the local kernel, and a way back.** Beyond the four suffixes of section 4.7, the
  local server's environment also drops `_PASSWD`, `_PWD`, `_CREDENTIALS`, `_CONNECTION_STRING` and `APIKEY`
  suffixes, the names `PASSWORD`, `SECRET`, `TOKEN`, `PGPASSWORD` and `MYSQL_PWD`, and the provider
  `env_http_headers` variables. `[kernel] pass_env` (new, local only) lets named variables through, and
  `doctor` shows how many are withheld.
- **Notebooks are still code (section 8), now with a warning.** `.hailer/last-kernel.json` (new) records
  which runtime last started a server on the notebooks folder; the first local start after a docker kernel
  warns that the notebooks' code will now run on this machine.
- **Tokens and logs.** Model-visible URLs (tool results, the system prompt) never carry the token; the CLI
  opens and prints signed-in ones, and tools that ask the user to open a URL add "run /notebook for the
  link". `.hailer/marimo.log` is emptied and owner-only (POSIX) at each start, because marimo prints its
  signed-in URL, and the tails Hailer prints mask the token.
- **More run flags.** `--memory-swap` equals `--memory` (no swap on top), `--pull never` on the containers,
  `--cpus` lowered to what the engine has (with a note), the forwarder limited to 64 MB and 64 processes,
  and the engine must run Linux containers (Docker Desktop in Windows-containers mode is refused).
- **`network = true` reaches more than the internet.** It also reaches services on the host
  (`host.docker.internal`) and other containers; the `Kernel:` line says `network on: the internet and this
  machine`, and the prompt tells the model so.
- **Doctor and init.** An image that is not downloaded yet is a warning, not a failure (the start downloads
  it); a `notebooks` row warns about a notebooks folder on a network drive; the `data` row scans for links
  leaving the folder (first 5,000 entries). A `runtime` key that lands under `[model]` or `[hailer]` (the
  `[kernel]` line still commented out) is a warning naming the fix, and `init --kernel` says so when it
  kept an existing `hailer.toml`.
- **`--foreground` says why it ended**: out of memory, removed from outside, or the exit code (the local
  runtime too: `marimo stopped (exit code N): it was ended from outside this terminal ...`).
  `uvx hailer kernel stop` stops a kept server of either runtime, reports what it removed, says when Docker
  is not running, and exits 1 when a removal failed.
- **The agent's tools know the kernel.** `marimo_status` reports the runtime and the kernel paths, and
  `notebook_open` / `notebook_close` accept the `/work/...` paths the model sees.
- **`GPG_KEY` stays in the image** (a public signing-key fingerprint): the integration test plants a secret
  in Hailer's environment and checks that it is absent, instead of scanning for secret-looking names.
- **Building.** `uvx hailer kernel build` runs `docker build` for this machine; the release and CI use
  `scripts/build_kernel_image.py` (`docker buildx`, both platforms for the release) on the same build context
  (`kernel_image.prepare_context`). The integration test also runs on Windows when opted in
  (`HAILER_DOCKER_TESTS`); the CI Docker job is not a required check, but the release waits for it.
- **Measured:** a start takes 4.3 s on Windows 11 with the image present; the amd64 image is 195 MB to
  download and 875 MB on disk. Not yet measured: the first pull from GHCR (nothing is published before the
  first release) and read time for a large Parquet file through a Windows bind mount. The arm64 image is
  checked only by wheel availability until the first release builds it.

Final review, merge checks and remaining release validation are recorded in [DOCKER_KERNEL_REVIEW.md](DOCKER_KERNEL_REVIEW.md).

## 1. Summary

Today `marimo_execute` runs the model's Python in a marimo kernel that is a child of Hailer. It has the
same user, environment variables, files and network as Hailer (README, *Security*). This plan lets the
user choose where that kernel runs:

| | `local` (the default, as today) | `docker` (new, opt-in) |
|---|---|---|
| Where marimo and the kernel run | Hailer's own Python, as the user | A Linux container; Hailer and the API key stay on the host |
| Files notebook code can read | Everything the user's account can | The notebooks folder and the data folder only |
| Files it can change | Everything the user's account can | The notebooks folder only; data is read-only |
| Network | The user's | None |
| API keys | Removed from its environment (Phase 0); the OS keyring is still readable | Not present |
| Needs | Nothing new | Docker Desktop (Windows, macOS) or Docker Engine (Linux) |

- **One setting.** The user chooses with `[kernel] runtime = "local" | "docker"`, and can override it for
  one run with `--kernel` or `HAILER_KERNEL`.
- **Choosing `docker` fails closed.** If Docker is missing or not running, Hailer stops and says how to fix
  it. It never falls back to running notebook code on the host.
- **Hailer always shows which runtime is active.**

**Rollout** is in five phases (section 6). Phase 0 hardens the local runtime and is worth shipping on its
own. Phases 1 to 4 add the Docker runtime behind the setting.

## 2. What the user sees

### Choosing

```toml
[kernel]
runtime = "docker"   # "local" (default): marimo runs in Hailer's own Python, as you, not isolated
                     # "docker": marimo runs in a container; needs Docker Desktop or Docker Engine
# image   = "ghcr.io/openafterhours/hailer-kernel:0.3.0"  # default: the image for this Hailer version
# memory  = "4g"     # container memory limit
# cpus    = 2        # container CPU limit
# network = false    # true lets notebook code reach the internet (package installs, remote files)
```

- **At setup:** `uvx hailer init --kernel docker` writes that section with `runtime = "docker"`. Plain
  `uvx hailer init` writes it commented out; when Docker is detected, the printed next steps say how to
  switch it on.
- **For one run:** `uvx hailer notebook --kernel docker` (or `--kernel local`), or `HAILER_KERNEL=docker`
  in the environment. Precedence follows the rest of `config.py`: flag, then environment, then file, then
  default.
- **Managing the image and containers:**
  - `uvx hailer kernel pull` downloads the image ahead of time. The first `hailer notebook` in docker mode
    also downloads it, with progress.
  - `uvx hailer kernel build` builds the image locally, for machines that cannot reach the registry.
  - `uvx hailer kernel stop` removes containers left behind by `--keep-marimo` or a crash.

### Seeing which runtime is active

The startup panel, `hailer status` and `/status` gain one line:

```
Kernel:     docker (hailer-kernel 0.3.0; no network; data read-only)
Kernel:     local (runs as you; not isolated)
```

`hailer doctor` adds rows for the runtime, Docker and the image (section 4.8).

### What changes in docker mode

- **No internet from notebook code.** `ctx.packages.add()` and DuckDB `INSTALL` do not work. Any package
  beyond marimo, Polars, DuckDB, altair and plotly has to be in the image.
- **Only the notebooks folder keeps files.** Notebook code can also write to `/tmp`, but that is wiped
  when the container stops.
- **The first start downloads the image.** How long start-up takes is measured in Phase 2.
- **Notebooks open as before:** in app view, in the user's own browser.

### When Docker cannot be used (`runtime = "docker"`)

| Situation | What Hailer does |
|---|---|
| `docker` is not on PATH | Stops: "Docker is not installed. Install Docker Desktop, or set [kernel] runtime = "local" to run notebook code on this machine without isolation." |
| Docker is installed but the engine is not running | Stops: "Docker Desktop is not running. Start it and run the command again." |
| The image is missing | Pulls it, with progress. If the pull fails, it stops and names `uvx hailer kernel build` and the `[kernel].image` setting as fixes. |
| The image is for another Hailer version | Stops and names the pull and build commands. A different marimo version would break code mode. |
| The data folder is on a mapped network drive or UNC path | Warns at start and in `doctor`: Docker Desktop usually cannot see it. Copy the data to a local folder, or point `data_dir` at one. |

## 3. What the spikes verified

**Setup** for both spikes on 2026-09-18:
- Windows 11, Docker Desktop 29.4.3 with the WSL2 Linux engine.
- Image `python:3.12-slim` with `marimo==0.24.2`, `polars==1.44.2` and `duckdb==1.5.5`, plus the Hailer
  wheel installed with `--no-deps`.
- Sample data from `scripts/make_sample_data.py`.

| Finding | What it means for the design |
|---|---|
| Docker silently ignores `-p` for a container on `--network none` or on an `--internal` network, so the port is never published. | The kernel can't be offline and reachable at the same time on its own. It runs on an internal network behind a forwarder container. That container is published on `127.0.0.1` and joined to both networks. |
| The first spike used the third-party `alpine/socat` image as the forwarder. The second used a ~40-line asyncio forwarder run from the kernel image itself (`python forward.py <kernel> 2718 2718`). It carried `/health`, the browser's websocket (two notebooks opened, giving two sessions), Hailer's streamed `/api/kernel/execute` calls, and code-mode cell creation. | No third-party image is needed. The forwarder ships inside the `hailer` package, so users trust one image. |
| With that layout, notebook code could not resolve DNS, reach `1.1.1.1`, reach the host (`host.docker.internal`), or run `INSTALL httpfs` in DuckDB. | Network isolation holds. |
| Inside the container, marimo must bind `--host 0.0.0.0`. Token auth with `--token-password` works: requests without the token get 401; the browser passes it as `&access_token=<t>` and the API as `Authorization: Bearer <t>`, which `MarimoClient` already sends. marimo 0.24.2's `edit` command also accepts `--token-password-file` (`-` means stdin). | Every server Hailer starts gets a random token. |
| A `?file=` key with the Windows host path gets no session; `?file=/work/notebooks/analysis.py` works. `/api/sessions` reports `/work/...` paths, so `match_session(<host path>)` returns `None`. | Hailer needs a map between host and kernel paths (section 4.2). Without it, every tool fails with "no session". |
| These flags all work together: `--init --read-only --tmpfs /tmp --tmpfs /home/analyst:uid=1000,gid=1000 --cap-drop ALL --security-opt no-new-privileges --memory 4g --cpus 2 --pids-limit 256`. A tmpfs home without `uid=` is owned by root, and marimo then logs permission errors for `~/.config` and `~/.local/state`. | These are the default run flags. |
| Writes succeed only in `/work/notebooks` (the read-write mount) and in `/tmp`. The `:ro` data mount, `/work` and the Python install all refuse writes. Cells created through code mode are saved to the notebook's `.py` file on the host. | The data folder is protected, and notebooks are saved on the host. |
| No host variables ending in `_KEY` or `_TOKEN` reach the kernel. The base image sets its own `GPG_KEY`, which is a public signing-key fingerprint, not a secret. | A scan for secret-looking variables must ignore `GPG_KEY`, or the image should unset it. |
| The starter template finds `WORKSPACE` by walking up to a `pyproject.toml` or `hailer.toml`. With an empty `/work/hailer.toml` in the image, a notebook in a subfolder (`notebooks/team/nested.py`) resolved `WORKSPACE = /work` and `DATA_DIR = /work/data`, and found all 6 period files. | The image ships that marker file, so notebooks need no changes. |
| `hailer.periods` loaded all six months (2,400 rows) from `/work/data`, and DuckDB read the same files. | The notebook helpers work unchanged. |

**Not verified yet.** Each item is a Phase 2 exit check or a later item:
- Docker Engine on a Linux host with `--user`.
- macOS on Apple Silicon (the arm64 image).
- Podman.
- Read speed for large parquet files through a Windows bind mount.
- Image size and first-pull time.
- Pulling through a corporate proxy or registry mirror.
- Network drives.

## 4. Design

### 4.1 A kernel runtime abstraction

New module `hailer/kernel.py`:

```python
class KernelRuntime(Protocol):
    name: str                                           # "local" | "docker"
    paths: PathMap                                      # host <-> kernel paths (identity for local)

    def check(self) -> list[Check]: ...                 # doctor and startup; docker: CLI, engine, image
    def start(self, port: int) -> RunningKernel: ...    # url, token, where the log is, stop()
    def find_running(self) -> RunningKernel | None: ... # reuse, and chat-only `hailer`
    def describe(self) -> str: ...                      # the "Kernel:" line
    def prompt_notes(self) -> str: ...                  # what the model is told about the kernel
```

- **`LocalRuntime`** wraps what `cli.py` does today:
  - `_marimo_server_command` (cli.py:222) and `_spawn_marimo` (252);
  - `_kill_tree` and `_stop_process` (280-308);
  - the registry cleanup (246).
- **`DockerRuntime`** drives the `docker` CLI through `subprocess`. Docker Desktop and Docker Engine both
  ship the CLI, so there is no Docker SDK dependency. Every call goes through one injectable runner, so
  tests can check the exact argument lists without Docker.
- **`runtime_for(config)`** picks the runtime. These commands call it instead of the helpers above:
  - `hailer notebook` (cli.py:1299-1363), via `_reusable_server` (1218) and `_start_marimo` (1242);
  - `exec` (1366), `status` (1408) and `doctor` (1450).

### 4.2 Path mapping at the HTTP boundary

`PathMap` holds pairs of host folder and kernel folder:

| Host | Kernel (docker) |
|---|---|
| `config.notebooks_root` | `/work/notebooks` |
| `config.data_dir` | `/work/data` |

For `local` it is the identity. `MarimoClient` (marimo_client.py:372) takes the map and translates in
exactly two places:
- **Outgoing:** the `?file=` key becomes the kernel path. This covers `notebook_file_key` (110),
  `open_notebook_url` (125) and `notebook_url` (513).
- **Incoming:** `sessions()` (436) rewrites each session's `path` and `filename` to the host path before
  returning it. Paths outside every mounted folder are left as they are, so they never match.

Everything above the client keeps working in host paths, unchanged:
- `match_session` (331) and `resolve_session` (519);
- in tools.py: `_absolute` (127), `_in_folder` (138), `_session_label` (146) and
  `_open_paths_or_error` (256);
- `cli._has_session_under` (1209).

Translating only at the client keeps the change small, and new "no session" bugs can't creep in one
caller at a time.

Mapping rules:
- Windows host paths compare case-insensitively; kernel paths are POSIX.
- A notebook in a subfolder maps to the same subfolder.
- A host path outside both folders has no kernel path. `notebooks.resolve_notebook` already refuses to
  open notebooks outside the notebooks folder, so this adds no new failure.

### 4.3 What the model is told

`agent.system_prompt` (agent.py:186, with the workspace block at 218-224) gives the model host paths
today, and the model copies them into its code. In docker mode it gives kernel paths and adds
`runtime.prompt_notes()`:

```
## Workspace
- Notebooks folder: /work/notebooks (writable; the only place files persist)
- Data directory: /work/data (read-only)
- The kernel runs in an isolated container: no internet access, no package installs
  (ctx.packages.add is unavailable), and /tmp is wiped when the session ends.
```

- **Package installs:** the line in `prompts/system.md:119` that says to use `ctx.packages.add()` moves
  into the local runtime's notes.
- **`list_periods`** (tools.py:543) runs on the host, and reports the data folder by its kernel path.
- **The CLI's own output** keeps showing host paths to the user.

### 4.4 Container layout

```
host                                    Docker
 hailer ──HTTP──► 127.0.0.1:<port> ──► hailer-fwd-<id>     (default + internal network; runs only the forwarder)
 browser ─────────┘                         │
                                            ▼
                                     hailer-kernel-<id>    (internal network only: no route out)
                                       marimo edit notebooks --host 0.0.0.0 --port 2718 --headless
                                       /work/notebooks  <- notebooks_root  (read-write)
                                       /work/data       <- data_dir        (read-only)
```

- **Both containers use the same image.** The forwarder runs `python -m hailer._forward
  hailer-kernel-<id> 2718 2718`. That module is the spike's forwarder, shipped in the `hailer` package,
  so the image needs nothing extra. The forwarder only ever connects to the kernel, so notebook code
  gains nothing by reaching it.
- **`<id>` is a short hash of the workspace path**, so two workspaces can run side by side. The containers
  and the network carry labels (`org.openafterhours.hailer.workspace` and `.version`), so Hailer finds its
  own and `hailer kernel stop` finds any strays.
- **Run flags** are the verified set from section 3, plus:
  - `--memory` and `--cpus` from the config, `--name`, the labels and `-w /work`;
  - on Linux hosts, `--user <uid>:<gid>` and a home tmpfs owned by that uid, so files marimo saves stay
    owned by the user. Still to verify.
- **Never mounted:** the workspace root, `.hailer/`, `.config/hailer/`, the user's home folder, and the
  Docker socket.
- **The token** is random for each start (`secrets.token_urlsafe`) and passed to marimo with
  `--token-password`.
  - `docker inspect` can show it. But anyone who can reach the Docker engine can already `docker exec`
    into the container, so hiding it gains nothing. `--token-password-file` is available if that changes.
  - Hailer keeps it in `.hailer/kernel.json`, with the port, container names and image, so chat-only
    `uvx hailer` and `hailer exec` can attach. `.hailer/` is git-ignored and not mounted, so notebook code
    can't read that file.
  - The browser URL carries `&access_token=<token>`.
- **With `network = true`**, the kernel runs on Docker's default network and its port is published
  directly, with no forwarder. The `Kernel:` line then says `network on`.

### 4.5 Lifecycle

- **Start** (`hailer notebook`), in order:
  1. Check the CLI and the engine with `docker version --format {{.Server.Version}}`.
  2. Check with `docker image inspect` that the image exists and its version label matches. Pull it
     otherwise.
  3. Create the internal network if it doesn't exist.
  4. Run the kernel container.
  5. Create the forwarder, connect it to the internal network, and start it.
  6. Wait for `/health` through the forwarder, using the existing 60 s timeout, counted after any pull.
  7. Write `.hailer/kernel.json`, then open the notebook URL as today.
- **When start fails,** Hailer shows the last lines of `docker logs hailer-kernel-<id>` instead of the
  tail of `.hailer/marimo.log`.
- **Stop** (end of the chat, without `--keep-marimo`): `docker rm -f` both containers, `docker network
  rm`, and delete `.hailer/kernel.json`. Stopping a process, as `_stop_process` does, isn't enough:
  `docker run -d` leaves no child process to stop.
- **After a crash or `kill`,** the containers keep running. The next `hailer notebook` finds them by
  label. If the health check and token work, it reuses them, as `_reusable_server` does for local servers.
  Otherwise it removes them and starts fresh. `hailer kernel stop` does the same by hand.
- **`--foreground`** starts both containers and follows `docker logs -f` for the kernel. Ctrl+C stops and
  removes them.
- **Discovery:** marimo's server registry (`registry_dir`, marimo_client.py:232) is written inside the
  container, so `find_server` (297) cannot see it. The lookup order becomes:
  1. `marimo_url`;
  2. `.hailer/kernel.json`, if its server passes the health check;
  3. the registry (local servers only).

### 4.6 The image

```dockerfile
FROM python:3.12-slim@sha256:<pinned digest>
RUN pip install --no-cache-dir marimo==0.24.2 polars==<ver> duckdb==<ver> altair==<ver> plotly==<ver>
COPY hailer/ /usr/local/lib/python3.12/site-packages/hailer/
RUN useradd --create-home --uid 1000 analyst && mkdir -p /work && touch /work/hailer.toml
USER analyst
WORKDIR /work
LABEL org.opencontainers.image.version="<hailer version>"
```

- **Contents:**
  - The same marimo pin as `pyproject.toml`, because Hailer drives the private `marimo._code_mode` API.
  - Polars and DuckDB.
  - altair and plotly, because `system.md:97` suggests them.
  - The `hailer` package copied in as source, without its dependencies. The kernel only imports
    `hailer.periods`, and the forwarder. `periods` needs Polars and DuckDB; `errors` and `models` use only
    the standard library. LangChain and keyring must not be in the image.
- **The Dockerfile ships as package data**, so `uvx hailer kernel build` works from an installed Hailer.
  It copies the installed `hailer` package into a temporary build context. That works the same from a
  checkout, uvx or pip, and needs no wheel and no PyPI.
- **Where users get it:** the release workflow builds `ghcr.io/openafterhours/hailer-kernel:<version>`
  for `linux/amd64` and `linux/arm64` (Apple Silicon) and pushes it.
  - `[kernel].image` can point somewhere else, such as a company mirror.
  - `hailer kernel build` covers machines that cannot pull.
- **Version check:** the image's version label must equal `hailer.__version__`. Otherwise Hailer stops
  and names the pull and build commands. A development checkout whose version isn't published uses
  `hailer kernel build`.
- **Optional:** unset `GPG_KEY` in the image (see section 3).

### 4.7 Phase 0: harden the local runtime

This is worth doing whatever happens with Docker, and it is small.

1. **Remove secrets from the kernel's environment.** `_spawn_marimo` (cli.py:252) passes no `env=`, so
   marimo inherits Hailer's whole environment. Drop:
   - every provider's `env_key`, and `HAILER_MARIMO_TOKEN`;
   - names ending in `_KEY`, `_TOKEN`, `_SECRET` or `_PASSWORD`, the same rule `log.py` uses for redaction.

   This does not stop notebook code reading the OS keyring, which is why the `Kernel:` line says "not
   isolated".
2. **Give servers Hailer starts a random token** instead of `--no-token` (marimo_client.py:84).
   - Hailer passes it with `--token-password-file -` on stdin, so it isn't on the command line.
   - This closes a gap: `/api/kernel/execute` skips marimo's server-token check, so today any local
     program can POST code to it, and possibly a DNS-rebinding web page too (not tested).
   - The token is stored in `.hailer/kernel.json`, so chat-only `hailer` and `hailer exec` can attach.
   - Servers users start themselves with `--no-token` keep working through the registry, as today.
3. **Add the `Kernel:` line** to the startup panel, `status` and `/status`.

### 4.8 Doctor, status, init and config

- **New doctor rows:**
  - `kernel`: the runtime, and what it isolates.
  - For docker only:
    - `docker`: CLI found, engine running, and its version.
    - `image`: present, and the version matches.
    - `data`: a warning for a UNC path or mapped network drive on Windows, and for symlinks or junctions
      in the data folder that point outside it, which don't resolve in the container.
  - The existing code-mode check (cli.py:1469-1479) also runs `import hailer.periods` in the kernel.
- **`status`, `/status` and the startup panel** show the `Kernel:` line. The marimo URL line shows the
  forwarder URL.
- **`init`** gets `--kernel docker|local` and stays non-interactive, so it remains scriptable. Without the
  flag, it writes the `[kernel]` section commented out. When Docker is detected, it adds a next step: "To
  isolate notebook code, set [kernel] runtime = "docker" (needs Docker Desktop)."
- **Config:**
  - `_KNOWN_TOP` (config.py:50) gains `kernel`, alongside a new `_KNOWN_KERNEL` set.
  - `load_config` (298) parses the section into a new `KernelConfig` on `HailerConfig` (models.py:156).
  - `validate` (446) checks the values.
  - `DEFAULT_CONFIG_TEMPLATE` (86) documents the section.

## 5. Changes by file

| File | Change | Phase |
|---|---|---|
| `src/hailer/models.py` | `KernelConfig` (runtime, image, memory, cpus, network), and a field for it on `HailerConfig` | 1-2 |
| `src/hailer/config.py` | `[kernel]` parsing, `HAILER_KERNEL`, validation, the template | 1-2 |
| `src/hailer/kernel.py` (new) | `KernelRuntime`, `LocalRuntime`, `DockerRuntime`, `PathMap`, the `.hailer/kernel.json` state | 0-2 |
| `src/hailer/_forward.py` (new) | The TCP forwarder | 2 |
| `src/hailer/docker/Dockerfile` (new, package data) | The kernel image | 2 |
| `src/hailer/marimo_client.py` | The path map in `MarimoClient`; `find_server` reads `.hailer/kernel.json`; a token instead of `--no-token` | 0-1 |
| `src/hailer/cli.py` | `notebook`, `exec`, `status` and `doctor` go through the runtime; `--kernel`; `kernel pull`/`build`/`stop`; `init --kernel`; the `Kernel:` line; removing secrets from the kernel's environment | 0-3 |
| `src/hailer/agent.py` | The workspace block uses kernel paths plus `prompt_notes()` | 2 |
| `src/hailer/prompts/system.md` | The package-install line moves into the local runtime's notes | 2 |
| `src/hailer/tools.py` | `list_periods` reports the kernel data path; nothing else changes, thanks to 4.2 | 2 |
| `pyproject.toml` | Include the Dockerfile as package data | 2 |
| `.github/workflows/release.yml` | Build and push the image before publishing to PyPI | 4 |
| `.github/workflows/test.yml` | A Linux job that builds the image and runs the container checks | 4 |
| `README.md`, `docs/INTERFACES.md`, `PLAN.md` | An *Isolated kernel (Docker)* section, plus updates to Security, Troubleshooting, the module contracts and the verified findings | 3 |

## 6. Phases

Each phase ends with the test suite passing on the whole CI matrix, and a live check.

**Phase 0: harden the local runtime (small).**
- Remove secrets from the kernel's environment, add the random token passed with
  `--token-password-file -`, write `.hailer/kernel.json`, and add the `Kernel:` line.
- Done when:
  - a notebook cell cannot see an `OPENAI_API_KEY` set in the terminal;
  - an unauthenticated POST to `/api/kernel/execute` gets 401;
  - `uvx hailer` in a second terminal attaches to a server started by `uvx hailer notebook --keep-marimo`.

**Phase 1: the runtime interface (medium, no behaviour change).**
- Add `KernelRuntime`, `LocalRuntime` and `PathMap` in `MarimoClient`. `[kernel] runtime` is accepted,
  with only `local` implemented.
- Done when the existing suite passes untouched, and new unit tests cover the path map: Windows and POSIX
  host paths, case, subfolders, and paths outside the mounted folders.

**Phase 2: the Docker runtime (large).**
- Add `DockerRuntime`, the forwarder, the Dockerfile, `hailer kernel build`/`pull`/`stop`, `--kernel`,
  the fail-closed checks, and the prompt changes.
- Done when the live end-to-end check passes on Windows with Docker Desktop in docker mode. That check is
  the uvx wheel recipe plus headless Chrome:
  - a real model turn loads data and creates a cell that is saved in the host file;
  - a cell that tries the network, or writes to the data folder, fails;
  - `--keep-marimo`, a crash followed by a restart, and `hailer kernel stop` leave no stray containers.
- Measure: image size, first-pull time, start-up time, and read time for a large parquet file through the
  mount compared with reading it locally.

**Phase 3: user-facing surface (medium).**
- Doctor rows, `init --kernel`, README sections and troubleshooting rows, and updates to INTERFACES.md and
  PLAN.md.

**Phase 4: distribution (medium).**
- The release workflow builds `linux/amd64` and `linux/arm64` images with buildx and pushes
  `ghcr.io/openafterhours/hailer-kernel:<version>`.
  - That job needs `packages: write`.
  - The GHCR package has to be made public once, in the organisation's package settings.
  - `publish` waits for the image, so no released Hailer points at a missing image.
- A CI job on `ubuntu-latest` builds the image from the tree and runs the container checks with headless
  Chrome. It is Linux-only because GitHub's Windows runners only run Windows containers.

**Later:**
- More read-only folders via `[kernel.mounts]`, mounted at `/work/mounts/<name>`. They can't go under
  `/work/data`, because Docker can't create mount points inside a read-only mount.
- An outbound proxy that allows only `[web].allowed_domains`.
- Podman support.
- Per-project extra packages, built into a derived image.
- Container resource use in `/status`.

## 7. Testing

- **Unit tests, on every platform:**
  - `PathMap`.
  - `DockerRuntime` with a fake runner, checking the exact `docker` argument lists: the internal network,
    no `-p` on the kernel, `127.0.0.1` on the forwarder, `:ro` on data, the hardening flags, and nothing
    else mounted.
  - Each fail-closed case (no CLI, engine down, image missing, version mismatch) prints the documented
    message and never starts a local server.
  - Removing secrets from the kernel's environment, and passing the token through.
  - The prompt text for each runtime.
  - Config parsing and precedence.
- **Client tests:** the existing fake marimo server (tests/test_marimo_client.py) returns `/work/...`
  session paths. This covers session matching, URLs and the tools without Docker.
- **Integration, in the Linux CI job:** start the real containers through `DockerRuntime`, open the
  notebook with headless Chrome, then check through `MarimoClient` that:
  - code runs;
  - a code-mode cell is saved to the host file;
  - network access and writes to the data folder fail;
  - stopping leaves nothing behind.
- **Live, on Windows with Docker Desktop:** the Phase 2 exit check, before each release that changes the
  runtime.

## 8. Risks and costs

- **Docker Desktop isn't always an option.** It is free for personal use and small businesses, but larger
  organisations need a paid subscription (check Docker's current terms). Managed machines often block it,
  WSL2 or Hyper-V. That is why `local` stays the default and fully supported.
- **Data can still leave through the conversation.** The container stops notebook code sending data
  anywhere, but anything the code prints goes back to the model endpoint, as it does today.
- **Notebooks are still code.** The container can write to the notebooks folder. A notebook written in
  docker mode and later opened with `runtime = "local"`, or with `marimo run`, runs on the host. The
  mitigation is keeping the notebooks folder in git, so every change is a reviewable diff, and the docs
  should say so.
- **Performance.** Reading through a Windows bind mount is slower than reading locally. How much slower
  for large parquet files is unmeasured (a Phase 2 exit check).
- **Folders Docker can't see.** Mapped network drives, and symlinks or junctions that point outside a
  mounted folder. Doctor warns, but the only fix is copying the data.
- **One more thing to release.** Every release also builds a multi-arch image, so GHCR becomes part of the
  release. `hailer kernel build` means users don't depend on it.
- **This is not an enforcement tool.** A user can always choose `local`. An organisation that must enforce
  isolation should run Hailer itself inside a managed VM or dev container.

## 9. Decisions needed

1. **The default runtime.**
   - Recommended: `local`, with docker as opt-in.
   - Alternative: `docker` whenever Docker is detected. That silently changes how isolated notebook code is
     depending on whether Docker Desktop happens to be running, and adds a pull to the first start.
   - Alternative: an interactive question in `init`. That breaks scripted `init`.
2. **Where the image comes from.**
   - Recommended: published to GHCR by the release workflow, with `hailer kernel build` as the fallback.
   - Alternative: always build locally on first use. There is no registry to maintain, but the first start
     is slow and needs access to PyPI and Docker Hub.
3. **`[kernel].network = true`.**
   - Recommended: offer it, off by default and shown in the `Kernel:` line, because some users need
     package installs or remote data.
   - Alternative: no escape hatch; users who need the network use `local`.
4. **Phase 0 tokens.** Servers Hailer starts get a token, so they can only be joined through Hailer or the
   URL it prints. Servers users start themselves with `--no-token` still work. Is that acceptable?
5. **Podman.** Recommended: out of scope for the first version. The runner interface leaves room for it
   later.
