# Docker setup and operations {#isolated-kernel-docker}

By default the notebook kernel runs in Hailer's own Python, as you: the code the agent writes can read and
change every file your account can, and reach the network (see [Security](data-handling.md#security)). If Docker is
installed, you can run the kernel in a container instead. The agent, the conversation and the API key
stay on your machine; only marimo and the notebook code move into the container.

| | `local` (the default) | `docker` |
|---|---|---|
| Where marimo and notebook code run | Hailer's own Python, as you | A Linux container, as a non-root user |
| Files notebook code can read | Everything your account can | The notebooks folder and the data folder, nothing else |
| Files it can change | Everything your account can | The notebooks folder only; data is read-only |
| Network | Yours | None, unless `network = true` |
| Your environment variables | All except secret-looking names (see [Security](data-handling.md#security)) | None |
| Needs | Nothing | Docker Desktop (Windows, macOS) or Docker Engine (Linux), and the kernel image |

## Choosing the runtime

```toml
[kernel]
runtime  = "docker"   # "local" (default) or "docker"
# image   = ""        # docker: default ghcr.io/openafterhours/hailer-kernel:<hailer version>
# memory  = "4g"      # docker: memory limit, no swap on top
# cpus    = 2         # docker: CPU limit (lowered to what Docker has, with a note)
# network = false     # docker: true lets notebook code reach the internet, this machine and other containers
# pass_env = []       # local: secret-looking variables notebook code may still read (see Security)
```

- **At setup:** `uvx hailer init --kernel docker` writes a `hailer.toml` with this section switched on.
  Plain `uvx hailer init` writes it commented out, and when Docker is found on the machine its next steps
  say how to switch it on. `init` never rewrites an existing `hailer.toml` without `--force`, and says so
  when `--kernel` could not be applied.
- **In an existing file:** uncomment the `[kernel]` line as well as `runtime`. A `runtime` line that ends up
  under `[model]` or `[hailer]` is ignored with a warning that says so.
- **For one run:** `uvx hailer notebook --kernel docker` (or `--kernel local`), or `HAILER_KERNEL=docker` in
  the environment. The flag wins over the variable, the variable over the file. `HAILER_KERNEL_IMAGE`
  overrides `image`.
- **Which one is in use** shows in the startup panel, `uvx hailer status`, `/status` and `uvx hailer doctor`
  (`X.Y.Z` is Hailer's version):

  ```text
  Kernel:     local (runs as you; not isolated)
  Kernel:     docker (hailer-kernel X.Y.Z; no network; data read-only)
  Kernel:     docker (hailer-kernel X.Y.Z; network on: the internet and this machine; data read-only)
  ```

  A docker kernel Hailer started for this workspace is the one in use whatever the file says: `uvx hailer` attaches to it when its mounted folders match and the line says `docker`.
- **Docker mode fails closed.** Docker missing or not running, Docker Desktop set to Windows containers,
  an image for another Hailer version, or a folder layout it refuses (below): Hailer stops and says how to
  fix it (see [Troubleshooting](../reference/troubleshooting.md#troubleshooting)). It never runs notebook code on this machine instead.

## First run: the kernel image

The first `uvx hailer notebook` in docker mode downloads `ghcr.io/openafterhours/hailer-kernel:<version>`,
where the tag is Hailer's own version (about 200 MB to download, 900 MB on disk), with docker's progress
in the terminal. After that a start takes a few seconds (4.3 s measured on Windows 11 with Docker
Desktop). `uvx hailer kernel pull` downloads it ahead of time.

- `uvx hailer kernel build` builds the image on this machine instead, from the Dockerfile that ships inside
  Hailer, with the marimo, Polars, fastexcel and DuckDB versions this Hailer runs. By default it uses Docker Hub
  (the base image) and PyPI (the packages); [corporate mirrors](docker.md#building-with-corporate-mirrors) work too.
  Use it where the registry is out of reach, or for a Hailer version that has
  no published image (development versions never do). `--tag <name>` builds under another name; Hailer
  then says to set `image` under `[kernel]` to use it.
- `[kernel].image` (or `HAILER_KERNEL_IMAGE`) points Hailer at another copy, such as a company mirror.
- The image's version label must equal Hailer's version, because a different marimo would break the
  notebook API Hailer drives. After upgrading Hailer, the next start downloads the matching image; a copy
  for another version stops the start with the pull and build commands.

## Building with corporate mirrors

You can build from an approved Python image and install the kernel packages from your company's PyPI
mirror. No source checkout is needed. Install Hailer itself through your normal corporate Python
package setup first (for example, `python -m pip install hailer` with your host pip configuration).
`hailer kernel build` automatically discovers the host's pip mirror settings, falling back to uv
when pip has no index settings. The same discovery runs in `scripts.build_kernel_image`. For an
existing mirror setup, you can normally run:

```text
hailer kernel build
```

Hailer reads the standard system, user and virtual-environment pip configuration files, including
`PIP_CONFIG_FILE`, and applies `PIP_*` environment overrides. It copies only index URLs, trusted
hosts, proxy, CA certificate, no-index, HTTP(S) find-links, timeout and retry settings. Within pip
files, `[install]` settings override `[global]`. `PIP_CONFIG_FILE` follows pip's user-file suppression
and null-device rules. See [pip configuration](https://pip.pypa.io/en/stable/topics/configuration/).

If pip has no index settings, Hailer checks uv's system and user configuration and the nearest
`uv.toml` or `[tool.uv]` in `pyproject.toml` from the current directory upwards. `uv.toml` wins
over `pyproject.toml` in the same folder. It also reads `[pip]` / `[tool.uv.pip]`, `UV_CONFIG_FILE`,
`UV_NO_CONFIG`, `UV_DEFAULT_INDEX`, `UV_INDEX_URL` and `UV_NO_INDEX`. A single default `[[index]]`
is supported, including `UV_INDEX_<NAME>_USERNAME` / `UV_INDEX_<NAME>_PASSWORD`. Environment settings
override file settings within each tool; pip index settings take precedence over uv index settings.
This build discovery includes project configuration even when Hailer was launched with `uvx`.

uv's additional or explicit indexes and package-specific source routing cannot be translated to
pip's resolution rules automatically; Hailer reports an error asking for `--pip-config` instead.
Local wheel paths and automatic client-certificate mounts also need explicit build configuration.
Host credential stores, keyring helpers, arbitrary environment variables and installation settings
are not copied. See [uv index resolution](https://docs.astral.sh/uv/concepts/indexes/).

`--pip-config FILE` overrides discovery entirely. `--pip-cert FILE` overrides the discovered CA
bundle (`PIP_CERT` / pip's `cert`, then `REQUESTS_CA_BUNDLE`, then `SSL_CERT_FILE`). The CA file is
mounted at its container path automatically. `--no-host-config` disables discovery while retaining
explicit build options, so a base image's own package configuration can be used.

To select settings explicitly, reuse an existing pip file or create one, for example
`company-pip.ini` (the same format on every platform):

```ini
[global]
index-url = https://packages.company.example/simple
extra-index-url =
```

Then build using your registry's Python image:

```text
hailer kernel build --tag hailer-kernel:company --base-image registry.company.example/python:3.13-slim --pip-config company-pip.ini --no-cache
```

If pip needs your company's CA bundle, append `--pip-cert company-ca.pem`. This accepts a PEM CA
bundle and makes it available to pip during installation. Registry login and certificate trust for
pulling the base image are configured in Docker separately (`docker login registry.company.example`).

Point Hailer at the image you built:

```toml
[kernel]
runtime = "docker"
image = "hailer-kernel:company"
```

- **Base requirements:** a Linux image with `/bin/sh`, standard shell utilities, and Python 3.12 or
  newer available as `python3`, including `venv` and `ensurepip`. Hailer creates its own virtual
  environment; the base does not need Hailer, marimo, pip on PATH, or `useradd`. Installation runs as
  root during the build; the finished image runs as UID/GID 1000 with `/home/analyst` as its home.
  An inherited entrypoint is cleared so Hailer can launch marimo and its forwarder.
- **Packages:** the mirror must provide marimo, Polars and DuckDB at the versions installed alongside
  host Hailer, plus Altair 6.3.0, Plotly 7.1.0 and their dependencies, compatible with the base's Python
  and architecture. Use a base/platform with matching wheels, or provide the build tools needed for
  source distributions. Hailer's own package is copied from the host installation, so it does not need
  to be downloaded again inside Docker. Rebuild after upgrading Hailer.
- **Configuration and authentication:** `--pip-config` and `--pip-cert` are optional. They use
  [BuildKit secret mounts](https://docs.docker.com/build/building/secrets/); their contents are not
  copied into the build context or image. An authenticated index URL can go in the pip file; keep
  that file outside version control. Discovered settings are written to a temporary secret outside
  the build context and removed even when the build fails or is interrupted. Hailer reports that
  discovery was used without printing URLs or credentials. The selected pip file overrides the
  base's environment variables for the supported package settings during installation. Other settings follow
  [pip's configuration rules](https://pip.pypa.io/en/stable/topics/configuration/); the empty
  `extra-index-url` above clears inherited extra indexes. Paths inside the pip file refer to the
  build container, so use `--pip-cert` for a CA file on your host.
- **Rebuilding:** `--no-cache` forces package installation again. Docker does not invalidate cached
  layers when [secret contents change](https://docs.docker.com/build/cache/invalidation/#build-secrets),
  so use it after changing your mirror, credentials or CA bundle. Omit it to reuse cached layers.
- **BuildKit:** use a current Docker Desktop or Docker Engine with BuildKit enabled. The Dockerfile
  uses the builder's bundled frontend, so selecting a private base does not introduce another
  public Dockerfile-frontend image download. The running kernel still has no network by default;
  packages are installed at build time.

From a checkout, `python -m scripts.build_kernel_image` accepts the same build options alongside
`--load`, `--push` and `--dry-run`. Its `--context DIR --dry-run` exports the Dockerfile, Hailer package
and a build command for teams that need to adapt the build further. Secret files stay outside that
context; the command refers to their original paths.

## What notebook code can and cannot do in the container

- **Two folders.** It sees the notebooks folder as `/work/notebooks` (read-write, the only place files
  persist) and the data folder as `/work/data` (read-only). Nothing else of your machine: not the
  workspace, `.hailer/`, `.config/hailer/`, your home folder or the Docker socket. The data files are read
  in place through the mount, never copied. The starter notebook finds `WORKSPACE = /work` and
  `DATA_DIR = /work/data`, the model is told those paths, and notebooks need no changes.
- **No network.** No internet, no DNS, no route to this machine. Installing packages (`ctx.packages.add()`)
  and DuckDB's `INSTALL` fail. The image has marimo, Polars, fastexcel, DuckDB, altair, plotly and ruff (which formats the agent's cells); anything else
  needs an image of your own (see [Known limitations](../reference/limitations.md#known-limitations)).
- **No secrets.** None of your environment variables reach the container, and nothing of Hailer's own
  runs in it but the notebook helpers (`hailer.periods`) and the forwarder: no LangChain, no keyring, no
  API key.
- **Limits.** A non-root user with no Linux capabilities, a read-only root file system, at most 256
  processes, and the memory and CPU limits from `[kernel]` (memory without extra swap). On a Linux host
  the kernel runs with your user and group id, so files marimo saves stay yours.
- **Scratch space.** `/tmp` and the home folder are in memory, shared by every notebook in the container,
  and wiped when the kernel stops.
- **Cells run when a notebook opens**, so the starter notebook's `DATA_DIR`, `data_files` and
  `period_files` exist before anyone runs a cell.

`network = true` puts the kernel on Docker's default network with its port published directly on
`127.0.0.1`. Notebook code can then reach the internet, services on this machine (`host.docker.internal`)
and other containers. Everything else above still holds, and the `Kernel:` line says
`network on: the internet and this machine`.

## Which folders can be mounted

In docker mode Hailer refuses a layout that would hand notebook code Hailer's own files, your
credentials, or a way to run code on this machine later. `uvx hailer doctor` reports it in the `config`
row, and every start stops on it, `--foreground` included (the same rules, in one place):

- Neither the notebooks folder nor the data folder may be, or contain, a drive root, your home folder,
  the workspace folder, `.hailer/`, the config file, or the context, skills and prompts folders.
- Neither may be, contain or sit inside a folder that holds credentials: `~/.config`, `~/.ssh`, `~/.aws`,
  `~/.azure`, `~/.gnupg`, `~/.docker`, `~/.kube`, and on Windows `%APPDATA%` and `%LOCALAPPDATA%` (the
  temporary folder inside it is fine).
- The notebooks folder may not sit inside `.hailer/` or those folders either, and the data folder may not
  sit inside the notebooks folder (it would become writable).
- The notebooks folder may not contain `.git` or `hailer.toml`, including in subfolders: notebook code
  could change repository hooks or another workspace's settings. Keep notebooks in a plain subfolder of
  your repository, such as `notebooks/`, and move nested repositories and workspaces outside that folder.
  Hailer checks names without following symlinks or junctions; an unreadable subtree stops startup.
- The default layout, `notebooks/` and `data/` in the workspace, passes. The data folder can be anywhere
  else on the machine: set `[hailer].data_dir` to an absolute path and it is mounted read-only where it is.
- Windows: a folder on a network share (a UNC path such as `\\server\share\sales`) cannot be mounted: it
  is an error in `doctor`'s `config` row and stops the start, with the same message. A mapped network drive
  gets a warning (`doctor` rows `data` and `notebooks`): Docker Desktop usually cannot see it. Symlinks and
  junctions inside the data folder that point outside it do not resolve in the container; `doctor` lists
  them. Copy such data to a folder on a local disk.
- `[hailer].marimo_url` (or `HAILER_MARIMO_URL`) cannot be combined with docker: Hailer starts and finds its
  own containers.

## Starting, reusing and stopping

- **What runs.** `uvx hailer notebook` creates three things, each labelled with the workspace:
  `hailer-net-<id>`, an internal network with no route out; `hailer-kernel-<id>`, the kernel, on that
  network only; and `hailer-fwd-<id>`, a small forwarder from the same image, published on
  `127.0.0.1:<port>` only, which carries the browser's and Hailer's requests to the kernel (Docker
  publishes no port for a container on an internal network). `<id>` comes from the workspace path, so two
  workspaces can run side by side. With `network = true` there is only the kernel, published directly.
  When the chat ends, Hailer removes the kernel it started. Reused kernels are left running.
- **Reuse.** `--keep-marimo` leaves the kernel running. Both `uvx hailer` and `uvx hailer notebook`
  reuse it only when its mounted folders match. When Docker is configured, image, network, memory and
  CPU settings must also match; otherwise the command asks you to run `uvx hailer kernel stop` first.
  `--kernel local` can attach to a running Docker kernel with matching folders. The chat keeps that
  server's token and path mapping in memory, including across notebook and model switches.
- **`--foreground`** starts the kernel and follows its log in this terminal. Ctrl+C stops and removes it;
  when it ends for another reason, Hailer says why (out of memory, removed from outside, exited).
- **`uvx hailer kernel stop`** stops this workspace's kernel, docker or local, and removes every container
  and network labelled with the workspace, running or not. It prints what it removed, says when Docker is
  not running (so leftovers could not be checked), and exits 1 when a recorded Docker kernel cannot be stopped or a removal failed (a container another
  terminal is removing at the same moment counts as removed, and a network whose containers are still
  detaching is retried for a few seconds). If Docker is unreachable, the recorded Docker kernel is kept
  for retry. An unanswered health check alone never proves those containers are gone.
- **Files the kernel may have planted.** When a docker kernel stops (the chat ends, `--foreground` ends,
  `uvx hailer kernel stop`), Hailer scans the notebooks folder and its subfolders for `.git`, `.vscode`,
  `.idea`, `.devcontainer` and `hailer.toml` and prints a loud `WARNING` naming them: notebook code can
  write there, and git or an editor would run commands from them. Delete them (unless you put them there
  yourself) before you run git in that folder or open it in an editor. `uvx hailer doctor` shows the
  same as a `notebooks` warning. Nothing is removed for you.
- **Leftovers.** If Hailer itself is killed, the containers keep running and `.hailer/kernel.json` still
  records them, so the next `uvx hailer notebook` or `uvx hailer` attaches to them and
  `uvx hailer kernel stop` removes them. A start never removes a running kernel container that Hailer has
  no working record of, because it may still be in use: it stops and points at `uvx hailer kernel stop`.
  A start also refuses while the recorded kernel (docker or local) does not answer but is not provably
  gone (its containers or its process still exist: busy, stuck or suspended), and keeps its record so
  `uvx hailer kernel stop` can still find it. Stopped leftovers are cleaned up by the next start.
- **Logs.** `docker logs hailer-kernel-<id>`. A failed start prints its last lines.

## Platforms

- **Windows 11 with Docker Desktop** (WSL2 engine, Linux containers): tested live and by the integration
  test.
- **Linux**: the integration test passes in WSL Ubuntu, and a CI job runs it on Ubuntu (Docker Engine)
  for every pull request.
- **macOS**: Docker Desktop. The release publishes the image for `linux/amd64` and `linux/arm64` (Apple
  Silicon). The CI integration test runs on amd64; it does not cover arm64.
- **Podman** is not supported.
