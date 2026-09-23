# Docker setup and operations {#isolated-kernel-docker}

By default the notebook kernel runs in a Docker container: the code the agent writes sees its own copy of
your notebooks and, read-only, the data folder, with no network. The agent, the conversation and the API
key stay on your machine; only marimo and the notebook code run in the container. The alternative,
`unsafe-local`, runs the kernel in Hailer's own Python, as you: the code can then read and change every file
your account can, and reach the network (see [Kernel runtimes](runtimes.md#kernel-runtimes)).

| | `docker` (the default) | `unsafe-local` (opt-in) |
|---|---|---|
| Where marimo and notebook code run | A Linux container, as a non-root user | Hailer's own Python, as you |
| Files notebook code can read | Its own copy of your notebooks and the data folder, nothing else | Everything your account can |
| Files it can change | Its copy of the notebooks only; data is read-only, and only marimo notebooks are [copied back](#notebook-copies) | Everything your account can |
| Network | None, unless `network = true` | Yours |
| Your environment variables | None | All except secret-looking names (see [Security](data-handling.md#security)) |
| Needs | [Docker Desktop](https://docs.docker.com/desktop/) (Windows, macOS) or [Docker Engine](https://docs.docker.com/engine/install/) (Linux), running, and the kernel image | Nothing |

## Choosing the runtime

```toml
[kernel]
runtime = "docker"                   # "docker" (the default) or "unsafe-local"
# image   = ""                       # docker: default ghcr.io/openafterhours/hailer-kernel:<kernel contract>
# memory  = "4g"                     # docker: memory limit, no swap on top
# cpus    = 2                        # docker: CPU limit (lowered to what Docker has, with a note)
# network = false                    # docker: true lets notebook code reach the internet, this machine and other containers
```

- **At setup:** `uvx hailer init` writes this section with `runtime = "docker"`, whether or not Docker is on
  the machine; without Docker its next steps say where to get it, or how to opt into `unsafe-local`.
  `uvx hailer init --kernel unsafe-local` writes that runtime instead. `init` never rewrites an existing
  `hailer.toml` without `--force`, and says so when `--kernel` could not be applied.
- **In an existing file:** a `hailer.toml` without a `[kernel]` section uses Docker. A `runtime` line that
  ends up under `[model]` or `[hailer]` is ignored with a warning that says so. `runtime = "local"`, the old
  name of the unsafe runtime, is an error that says to write `"unsafe-local"` only if you accept what it means.
- **For one run:** `uvx hailer notebook --kernel docker` (or `--kernel unsafe-local`), or `HAILER_KERNEL` in
  the environment. The flag wins over the variable, the variable over the file. `HAILER_KERNEL_IMAGE`
  overrides `image`.
- **Which one is in use** shows in the startup panel, `uvx hailer status`, `/status` and `uvx hailer doctor`
  (`marimo0.24.2-64a6b78f25cf` is the kernel contract, below; the `unsafe-local` line is in a warning colour):

  ```text
  Kernel:     docker (hailer-kernel marimo0.24.2-64a6b78f25cf; no network; data read-only)
  Kernel:     docker (hailer-kernel marimo0.24.2-64a6b78f25cf; network on: the internet and this machine; data read-only)
  Kernel:     unsafe-local (runs as you; not isolated)
  ```

  Each session starts its own kernel with the settings in effect when it starts, so the line always
  describes the kernel this chat uses.
- **Docker mode fails closed.** Docker missing or not running, Docker Desktop set to Windows containers,
  an image of another kernel contract, or a data folder it refuses (below): Hailer stops and says how to
  fix it (see [Troubleshooting](../reference/troubleshooting.md#troubleshooting)). It never runs notebook
  code on this machine instead; when Docker itself is the problem, the message names the other way on,
  `runtime = "unsafe-local"`, for those who accept that notebook code then runs as them.

## First run: the kernel image

The first `uvx hailer` (or `uvx hailer notebook`) downloads
`ghcr.io/openafterhours/hailer-kernel:marimo<version>-<fingerprint>` (for example `marimo0.24.2-64a6b78f25cf`; about
200 MB to download, 900 MB on disk), with docker's progress in the terminal. After that a start takes a few seconds (4.3 s measured on Windows 11 with Docker
Desktop). `uvx hailer kernel pull` downloads it ahead of time.

- `uvx hailer kernel build` builds the image on this machine instead, from the Dockerfile that ships inside
  Hailer, with the package versions of the kernel contract. By default it uses Docker Hub
  (the base image) and PyPI (the packages); [corporate mirrors](docker.md#building-with-corporate-mirrors) work too.
  Use it where the registry is out of reach, or for a kernel contract that has
  no published image yet (a development checkout that changed the image). `--tag <name>` builds under another name; Hailer
  then says to set `image` under `[kernel]` to use it.
- `[kernel].image` (or `HAILER_KERNEL_IMAGE`) points Hailer at another copy, such as a company mirror.
- **The kernel contract.** An image is defined by its Dockerfile, the few Hailer modules it holds (the
  notebook helpers and the forwarder) and its exact package versions, with marimo equal to the version
  Hailer drives. The tag names that contract: the marimo version plus the first 12 characters of a
  sha256 over all of it, so any change to the image is a new tag. The image's
  `org.openafterhours.hailer.kernel-contract` label must equal the contract this Hailer needs, because a
  different marimo would break the notebook API Hailer drives. Hailer releases that keep the contract share
  one image, so upgrading Hailer downloads a new image only when the contract changed; a copy of another
  contract stops the start with the pull and build commands.

## Building with corporate mirrors

You can build from an approved Python image and install the kernel packages from your company's PyPI
mirror. No source checkout is needed. Install Hailer itself through your normal corporate Python
package setup first (for example, `python -m pip install hailer` with your host pip configuration).
`uvx hailer kernel build` automatically discovers the host's pip mirror settings, falling back to uv
when pip has no index settings. The same discovery runs in `scripts.build_kernel_image`. For an
existing mirror setup, you can normally run:

```text
uvx hailer kernel build
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
uvx hailer kernel build --tag hailer-kernel:company --base-image registry.company.example/python:3.13-slim --pip-config company-pip.ini --no-cache
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
- **Packages:** the mirror must provide marimo, Polars, fastexcel, DuckDB, Altair and Plotly at the
  kernel contract's versions (`IMAGE_PACKAGES` in `hailer.kernel_image`; `uvx hailer kernel build` prints the main
  ones) and their dependencies, compatible with the base's Python and architecture. Use a base/platform
  with matching wheels, or provide the build tools needed for source distributions. Hailer's image modules
  are copied from the host installation, so Hailer itself is not downloaded inside Docker. Rebuild when an
  upgrade changes the kernel contract (the start says so).
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
`--load`, `--push`, `--if-missing` and `--dry-run`. Its `--context DIR --dry-run` exports the Dockerfile, Hailer modules
and a build command for teams that need to adapt the build further. Secret files stay outside that
context; the command refers to their original paths.

## What notebook code can and cannot do in the container

- **Two folders.** It sees its own notebooks folder as `/work/notebooks` (in memory, at most 512 MB,
  holding [copies](#notebook-copies) of your notebooks) and the data folder as `/work/data` (read-only).
  The data folder is the only folder of your machine it sees: not your notebooks folder, the workspace,
  `.hailer/`, `.config/hailer/`, your home folder or the Docker socket. The data files are read in place
  through the mount, never copied. The starter notebook finds `WORKSPACE = /work` and
  `DATA_DIR = /work/data`, the model is told those paths, and notebooks need no changes.
- **No network.** No internet, no DNS, no route to this machine. Installing packages (`ctx.packages.add()`)
  and DuckDB's `INSTALL` fail. The image has marimo, Polars, fastexcel, DuckDB, altair, plotly and ruff (which formats the agent's cells); anything else
  needs an image of your own (see [Known limitations](../reference/limitations.md#known-limitations)).
- **The browser is not the kernel.** "No network" covers the kernel only. Notebook outputs render in your
  browser, which is online: an HTML, Markdown or image output that notebook code produces can load from, or
  send data to, the internet when the browser shows it. Treat the outputs of code you did not review like
  a web page from an unknown site, and do not open notebooks of untrusted origin while signed in to
  anything sensitive in that browser profile.
- **No secrets.** None of your environment variables reach the container, and nothing of Hailer's own
  runs in it but the notebook helpers (`hailer.periods`) and the forwarder: no LangChain, no keyring, no
  API key.
- **Limits.** A non-root user with no Linux capabilities, a read-only root file system, at most 256
  processes, and the memory and CPU limits from `[kernel]` (memory without extra swap; the notebooks
  folder counts towards it as it fills). On a Linux host the kernel runs with your user and group id, so it
  can read data files only you may read.
- **Scratch space.** `/tmp`, the home folder and the notebooks folder are in memory, shared by every
  notebook in the container, and wiped when the kernel stops. A file notebook code writes (an export, a
  chart) is gone then too, unless it is a notebook; write results to show in a notebook's output instead.
- **Cells run when a notebook opens**, so the starter notebook's `DATA_DIR`, `data_files` and
  `period_files` exist before anyone runs a cell.

`network = true` puts the kernel on Docker's default network with its port published directly on
`127.0.0.1`. Notebook code can then reach the internet, services on this machine (`host.docker.internal`)
and other containers. Everything else above still holds, and the `Kernel:` line says
`network on: the internet and this machine`.

## Notebook copies

A docker kernel never writes to your machine. When it starts, Hailer copies your notebooks into the
kernel's own notebooks folder, before the browser opens one; while it runs, marimo, the agent and the
browser change those copies; and Hailer copies the notebooks that changed back into your notebooks folder:

- after every agent turn and when the chat switches notebooks,
- every 15 seconds while the kernel runs (edits made in the browser),
- once more when the kernel stops, before its container is removed: when the chat ends (`/exit`,
  Ctrl+D, Ctrl+C, an error) and when `--foreground` ends.

**Only marimo notebooks with plain names are copied, in either direction.** A file is copied when all of
these hold; everything else stays where it is, and Hailer names what it left in one line
(`Not copied into the kernel ...`, `Not copied back from the kernel ...`):

- a `.py` file (that spelling) with a relative name, at most four folders deep, and no part of its path
  starting with `.` or `__` (so nothing in `.git/`, `.vscode/`, `.idea/`, `.devcontainer/`,
  `__marimo__/` or `__pycache__/`, and no `__init__.py`);
- not a name other tools run: `conftest.py`, `test_*.py`, `*_test.py`, `setup.py`, `noxfile.py`,
  `tasks.py`, `fabfile.py`, `dodo.py`, `conf.py`, `manage.py`, `gunicorn.conf.py`, `ipython_config.py`,
  `jupyter_*_config.py`, `sitecustomize.py`, `usercustomize.py`, `__main__.py` (any case);
- no folder or file named like a Python module that a script in that folder would import instead: the
  standard library (`json.py`, `csv.py`, `calendar.py`), what Hailer has installed and the kernel's packages
  (`polars.py`, `marimo.py`), and common ones such as `pandas.py` (any case; the warning names the module);
- no part that looks like a Windows short name (`~` and a digit, such as `SALES~1.py`);
- UTF-8 text of at most 5 MB that contains `import marimo` and `marimo.App(`;
- at most 500 notebooks.

**Deadlines.** Whatever runs in the kernel can replace its marimo server, so Hailer trusts none of its
answers: a reply is read in pieces up to about 16 MB (three times the notebook limit), the whole copy at
the start and the last copy at the end each finish within 30 seconds, and a background copy stops after
10 seconds or 50 MB read (the rest follows on the next pass). When a copy is cut short, Hailer says so,
keeps the files on your machine as they were and, at the end of a session, goes on to remove the
container. The end of a session waits at most about a minute for the last copy, then removes the container
anyway. One session copies back at most 200 new notebooks and 100 MB; after that, new names stay in the kernel
(with a warning).

So notebook code cannot plant git hooks or settings, editor or dev-container settings, a `hailer.toml`,
test-runner hooks or Python modules in your notebooks folder: those stay in the container and are gone when
it stops. A notebook itself is still code: open one the kernel wrote only in a runtime you trust with it
(the docker kernel, or `unsafe-local` if you read it first). A copy back is written atomically (a temporary file
replaced into place) inside the notebooks folder only, never through a symlink or junction; a failed copy is
a warning, and the file on your machine stays as it was.

**Conflicts.** Hailer remembers what it last copied for each notebook. If the file on your machine changed
since then (you edited it in another editor while the chat ran) and the kernel's copy changed too, your
version is first saved as `.hailer/notebook-backups/<name>.<date-time>.py`, then replaced, with a warning.
A notebook deleted in the kernel is never deleted on your machine. Edits you make on your machine while a
kernel runs are not copied into it; the next session picks them up.

What is not copied back: files that are not notebooks (see above), and the kernel's changes since the
last copy when Hailer is killed or a second Ctrl+C interrupts the last copy (it says so).

## Which data folder can be mounted

The data folder is the only folder of your machine a docker kernel sees (read-only). Hailer refuses one
that would hand notebook code Hailer's own files or your credentials. `uvx hailer doctor` reports it in the
`config` row, and every start stops on it, `--foreground` included (the same rules, in one place):

- It may not be, or contain, a drive root, your home folder, the workspace folder, `.hailer/`, the config
  file, or the context, skills and prompts folders.
- It may not be, contain or sit inside a folder that holds credentials: `~/.config`, `~/.ssh`, `~/.aws`,
  `~/.azure`, `~/.gnupg`, `~/.docker`, `~/.kube`, and on Windows `%APPDATA%` and `%LOCALAPPDATA%` (the
  temporary folder inside it is fine).
- The default, `data/` in the workspace, passes. The data folder can be anywhere else on the machine: set
  `[hailer].data_dir` to an absolute path and it is mounted read-only where it is.
- Windows: a data folder on a network share (a UNC path such as `\\server\share\sales`) cannot be mounted:
  it is an error in `doctor`'s `config` row and stops the start, with the same message. A mapped network
  drive gets a `data` warning: Docker Desktop usually cannot see it. Symlinks and junctions inside the data
  folder that point outside it do not resolve in the container; `doctor` lists them. Copy such data to a
  folder on a local disk.

The notebooks folder is never mounted, so any notebooks folder works in docker mode, including one inside
a git repository or on a network share: only [copies](#notebook-copies) of allow-listed notebooks are
written there.

## Starting and stopping

- **What runs.** Every `uvx hailer` or `uvx hailer notebook` session creates three things of its own:
  `hailer-net-<id>-<suffix>`, an internal network with no route out; `hailer-kernel-<id>-<suffix>`, the
  kernel, on that network only; and `hailer-fwd-<id>-<suffix>`, a small forwarder from the same image,
  published on `127.0.0.1:<port>` only, which carries the browser's and Hailer's requests to the kernel
  (Docker publishes no port for a container on an internal network). `<id>` comes from the workspace path
  and `<suffix>` is random, so several sessions and several workspaces run side by side. With
  `network = true` there is only the kernel, published directly. When the chat ends, Hailer removes what
  it started, by the ids Docker gave it.
- **Labels and owners.** Every container and network carries labels for the workspace, its role, the
  kernel contract and its owner. The owner is a lock file, `.hailer/owner-<id>.lock`, that the Hailer
  session creates and holds an operating-system lock on for as long as it runs; the operating system drops
  the lock when the process ends, however it ends. A start waits only for the server that answers with its
  own token, so two sessions that pick the same port never take each other's server. Nothing attaches to a
  kernel another session started; the chat keeps its own sandbox (the server's token, and where the
  container sees its notebooks and the data folder) in memory, including across notebook and model switches.
- **`--foreground`** starts the kernel and follows its log in this terminal, without a chat. Ctrl+C stops
  and removes it; when it ends for another reason, Hailer says why (out of memory, removed from outside,
  exited).
- **Leftovers.** If Hailer itself is killed, its containers keep running. The next start in the workspace
  removes every labelled object whose owner is not alive (its lock file is gone or nobody holds it, or it
  has no owner label), running or not, and deletes lock files nobody holds. It never removes anything of a
  session that is still running.
- **`uvx hailer kernel stop`** removes every container and network labelled with the workspace, running
  or not, whichever session started it. It prints what it removed, says
  when Docker is not running (so leftovers could not be checked), and exits 1 when a removal failed (a
  container another terminal is removing at the same moment counts as removed, and a network whose
  containers are still detaching is retried for a few seconds).
- **Notebooks at stop.** The session copies its kernel's changed notebooks back before it removes the
  containers (see [Notebook copies](#notebook-copies)). `uvx hailer kernel stop` and the leftover cleanup
  of a later start remove containers without copying: a killed session's last changes are lost.
- **Logs.** `docker logs hailer-kernel-<id>-<suffix>`. A failed start prints its last lines.

## Platforms

- **Windows 11 with Docker Desktop** (WSL2 engine, Linux containers): tested live and by the integration
  test.
- **Linux**: the integration test passes in WSL Ubuntu, and a CI job runs it on Ubuntu (Docker Engine)
  for every pull request.
- **macOS**: Docker Desktop. The release publishes the image for `linux/amd64` and `linux/arm64` (Apple
  Silicon). The CI integration test runs on amd64; it does not cover arm64.
- **Podman** is not supported.
