# Proposal: a simpler sandbox architecture that can move to the cloud

Status: **proposal, 2026-09-22.** Nothing here is built. It reviews `main` at v0.2.8 (4dc96df) against
two goals: the agent's Python must run in a sandbox it cannot escape, and the project must stay small
enough for one maintainer. It also prepares for the kernel running in GCP (or another cloud) with the
same shape. Section 8 lists the decisions that are yours to make before any of it starts.

## 1. Summary

The core design is right and should be kept: the agent, the conversation and the API key stay in the
Hailer process; the model gets eleven narrow tools and no shell; the only thing that runs the model's
code is a marimo server reached over HTTP with a token. That HTTP seam is already what a cloud kernel
needs.

What makes the project hard to maintain is not the sandbox itself but three desktop-specific choices
around it, none of which survive a move to the cloud:

1. **The notebooks folder is a writable bind mount of the user's disk.** Most of the Docker rules
   (mount-layout refusals, planted `.git`/`.vscode` warnings, symlink and UNC checks, UID mapping) exist
   only to make that safe.
2. **Kernels are shared across terminals and restarts.** `kernel.json`, liveness proofs, removal by id,
   settings-mismatch checks, `--keep-marimo`, registry discovery and `marimo_url` exist to support it.
3. **Two runtimes, and the unsafe one is the default.** The product promise is a sandbox, but `local`
   runs the model's code as the user; keeping both doubles the lifecycle code and tests.

Recommendations, in order:

| # | Change | Why | Effect on size |
|---|---|---|---|
| R1 | Define one **Sandbox contract** that owns *all* file access (notebooks and data), not just start/stop | Tools currently read the host folders directly, which breaks as soon as the kernel is remote | Small addition; the seam for every later step |
| R2 | **Own one kernel per `hailer` process**; drop cross-terminal reuse | Removes the most intricate lifecycle code; cloud reuse works differently anyway | Large removal |
| R3 | **Notebooks live in the sandbox and are synced out** (allow-listed `*.py` only), instead of a writable bind mount | Removes the planted-file and notebook mount-layout machinery; identical to the cloud model | Large removal, small addition |
| R4 | **Docker becomes the default**; `local` becomes an explicit, unsafe developer mode | Matches the product promise; later, cloud is the answer for users without Docker | Medium removal once local is dev-only |
| R5 | **Version the kernel image by its contract**, not by Hailer's version | A CLI-only release no longer rebuilds and republishes a 900 MB multi-arch image | Simpler releases |
| R6 | **GCP: a `CloudRunSandbox`** behind the same contract, after a short spike | Same image, same HTTP seam; the cloud replaces the forwarder, the internal network and the mount rules | New module, no change to agent or tools |
| R7 | Housekeeping: split `cli.py`, move history out of `PLAN.md`, fewer env overrides | Smaller files and docs that describe the present | Neutral to negative |

## 2. The architecture today

```
Terminal ── cli.py / chat.py / chat_ui.py
              │
agent.py      LangChain create_agent + ChatOpenAI ──HTTPS──► model endpoint
              │   (API key and conversation stay here)
tools.py      11 tools ─┬─ notebooks.py, periods.py  ──► HOST notebooks/ and data/ folders (direct)
                        └─ marimo_client.py  ──HTTP+SSE, token──┐
                                                                ▼
kernel runtime   local:  marimo child process, as the user (default, not isolated)
                 docker: hailer-kernel-<id> on an --internal network
                         + hailer-fwd-<id> publishing 127.0.0.1:<port>
                         mounts: notebooks/ read-write, data/ read-only
                              │
                         Browser UI (a kernel session exists only while a tab is open)
```

Size, for reference: about 12,000 lines of source and 15,000 of tests. The kernel runtimes alone are
`kernel.py` (1,036), `kernel_docker.py` (1,314), `kernel_image.py` (220), `kernel_packages.py` (229) and
`_forward.py` (90), plus `config.docker_mount_problems` and its helpers (about 210 lines of `config.py`)
and the kernel commands in `cli.py` (1,710 lines in total).

### What is working and should be kept

- **Secrets never enter the sandbox.** Only marimo, the notebook helpers and the forwarder run there.
- **The model's capabilities are enumerated.** No shell, no file tool, an allow-listed `fetch_page`.
- **The kernel is reached over HTTP with a token.** `MarimoClient` does not care where the server is.
- **Fail closed.** Docker mode never silently falls back to running code on the host.
- **Hardened container flags**: non-root, `--cap-drop ALL`, `no-new-privileges`, read-only root, PID,
  memory and CPU limits, no network by default.
- **Evidence culture**: fakes for the gateway, marimo and Docker; `LEARNINGS.md` records why.

## 3. Findings

**F1. The sandbox is opt-in.** The default runtime executes model-written code as the user, with the
user's files and network. The prompt's rules are the only barrier. Every feature is then built and tested
twice (local and docker), and every doc has to explain both.

**F2. The writable notebooks mount drives most of the Docker complexity.** Because notebook code can write
to a folder on the user's disk, Hailer has to refuse layouts that expose `.git`, `hailer.toml`, the home
folder, credential folders or drive roots (`config.docker_mount_problems`); scan for planted `.git`,
`.vscode`, `.idea`, `.devcontainer` on stop (`kernel_docker.planted_files`); map UIDs on Linux; and reason
about UNC paths, mapped drives and symlinks. `DOCKER_KERNEL_REVIEW.md` still calls the planted-file check
"a mitigation": an editor can act on a new file before the warning appears. None of this exists in a cloud
deployment.

**F3. Cross-terminal kernel reuse is the most intricate code in the project.** Supporting `--keep-marimo`,
a second terminal attaching, `hailer exec` against a running kernel and hand-started `--no-token` servers
led to: `kernel.json` with a liveness proof (`answers_with_token`, `live_kernel_state`, `record_is_gone`),
removal by recorded id rather than name, a start guard that refuses when a kernel is "not provably gone",
settings-mismatch errors, the marimo server registry and `workspace_affinity`. Most findings in the
Docker review were races in this area. The benefit is saving a ~4 s container start and keeping variables
across a chat restart, and the starter notebook's `auto_instantiate` already recomputes state on open.

**F4. The runtime seam leaks.** `notebook_list`, `notebook_create`, `notebook_open` and `list_periods`
read and write the host's `notebooks/` and `data/` directly (`notebooks.py`, `periods.scan_period_files`),
and `.hailer/notebook.json` stores host paths that `PathMap` translates. That works only while the kernel
shares the user's disk. It is the first thing that breaks with a remote kernel.

**F5. A kernel session needs a browser tab.** marimo creates the session from the page's WebSocket, so
the agent cannot run code until the user has the notebook open. That is acceptable for this product (the
user watches the notebook) but in the cloud it means the browser must reach the remote marimo, with
authentication and reconnects (see §6).

**F6. The image is tied to Hailer's version.** The version label must equal Hailer's, so every release
builds and pushes a multi-arch image even when only the CLI changed, and the release pipeline stops
before PyPI if that build fails. What actually has to match is the marimo version (the private
`marimo._code_mode` API) and the in-image helpers (`hailer.periods`, `hailer._forward`).

**F7. The Docker token is on a command line.** The local runtime passes it on stdin, but `kernel_command`
passes `--token-password <token>` to `docker run`, so it appears in `docker inspect` and the container's
process list. Anyone who can run `docker` already controls the container, and notebook code already has
the kernel's powers, so the risk is low, but it is inconsistent and would matter more with a shared host.

**F8. Size of the edges.** `cli.py` is 1,710 lines; `config.py` reads about a dozen `HAILER_*` environment
overrides; `PLAN.md` (529 lines) is mostly dated history mixed with the current design, beside
`DOCKER_KERNEL_PLAN.md` (562), `DOCKER_KERNEL_REVIEW.md`, `INTERFACES.md` (980) and `LEARNINGS.md`.
A maintainer has to read a lot to learn what is true today.

## 4. Target architecture

```
Hailer process (user's machine) — unchanged: CLI, agent, tools, API key, conversation
   │
   │  Sandbox contract (R1): start / stop / endpoint / files
   ▼
┌───────────────────────── one of ─────────────────────────┐
│ DockerSandbox (default)        CloudRunSandbox (later)    │
│  kernel container, no network   Cloud Run service,        │
│  data: read-only bind mount     data: read-only GCS mount │
│  notebooks: container volume    notebooks: GCS prefix     │
│  reached via forwarder          reached via HTTPS + IAM   │
│                                                           │
│ UnsafeLocal (developer mode only, never the default)      │
└───────────────────────────────────────────────────────────┘
   same kernel image, same marimo, same MarimoClient over HTTP
```

### 4.1 The Sandbox contract (R1)

Extend the existing `KernelRuntime` protocol so it owns everything the tools need from the kernel's side,
and so no tool touches a host path:

```python
class Sandbox(Protocol):
    def start(self) -> Endpoint: ...            # url, auth headers, browser URL
    def stop(self) -> None: ...                 # idempotent; removes everything it created
    def status(self) -> SandboxStatus: ...      # for the startup panel, /status, doctor
    def describe(self) -> str: ...              # the "Kernel:" line and prompt notes

    # Files, in sandbox terms ("notebooks/sales.py", "data/25-03 sales.parquet")
    def list_notebooks(self) -> list[NotebookInfo]: ...
    def create_notebook(self, name: str, source: str) -> str: ...
    def list_data(self) -> list[DataFile]: ...
    def sync_out(self, dest: Path) -> list[Path]: ...   # R3: copy notebooks back to the host
```

- `notebooks.py` keeps templates, slugs and name resolution but calls the sandbox for listing and
  creation; `list_periods` parses names from `list_data()`.
- The active notebook is stored as a sandbox-relative name, not a host path, so `PathMap` disappears
  from `notebook.json` and the tools.
- First choice for the file operations: marimo's own file endpoints (the home page's workspace file
  listing and the file explorer's create/read), because they work the same locally and in the cloud.
  This must be checked against marimo 0.24.2 in a spike; the fallback is per-runtime code
  (`docker cp`/`docker exec` for Docker, the GCS API for the cloud).

### 4.2 One owned kernel per process (R2)

`hailer` and `hailer notebook` start a sandbox, use it, and remove it on exit. Nothing attaches to a
kernel it did not start.

- Remove: `--keep-marimo`, `[hailer].marimo_url` / `HAILER_MARIMO_URL`, the marimo server registry
  discovery and `workspace_affinity`, settings-mismatch checks, the "not provably gone" start guard and
  most of `kernel.json`.
- Keep a minimal leftovers record so `hailer kernel stop` (and the next start) can remove containers
  left behind by a killed process: Docker labels already make this a single `docker ps --filter label=`.
- `hailer exec` becomes a slash command inside the running chat (`/exec`), or starts its own
  short-lived sandbox.
- Cost to users: a chat restart re-runs the notebook (about 4 s for the container plus the cells). A
  hand-started marimo server can no longer be used; that was only ever unsandboxed.

### 4.3 Notebooks live in the sandbox (R3)

The container gets a Docker **volume** for `/work/notebooks` instead of a bind mount of the host folder.
Hailer copies the host's notebooks in at start and copies them out after each turn and at exit:

- **Out:** only regular files matching `*.py`, at most N levels deep, no dot-files or dot-folders, size
  capped. Anything else the kernel wrote stays in the volume and is discarded with it. This replaces the
  planted-file scan and the `.git`/`hailer.toml` refusals with an allow-list that cannot be bypassed by
  timing.
- **In:** the same allow-list, so the kernel never sees the workspace's other files.
- **Conflicts:** the sandbox is the source of truth during a session; a host edit to the same notebook
  mid-session is overwritten, with a warning if the host file changed since it was copied in.
- **Data stays a read-only bind mount** in Docker: it is large, read-only removes the write risk, and
  the remaining rules shrink to "not the home folder, a drive root or a credential folder".

Trade-off: the notebook file on disk is no longer live while marimo edits it; it is at most one turn
behind. In return the notebooks folder can be anywhere, including a git repository, and the cloud version
is the same code with a bucket instead of a volume.

### 4.4 Docker by default (R4)

- `hailer init` writes `runtime = "docker"`. With Docker missing, Hailer stops and says how to install it,
  or (once R6 exists) how to use the cloud runtime.
- `local` is renamed `unsafe-local` and only accepted with an explicit flag or setting; it prints a
  warning in the startup panel. It stays useful for developing Hailer and for the offline tests.
- `pass_env`, the withheld-variable logic and local process-tree handling can then be simplified, since
  only developers use that path.

### 4.5 Kernel image versioning (R5)

Tag and label the image with a **kernel contract version**, for example
`hailer-kernel:k3-marimo0.24.2`, where `k3` bumps when `hailer.periods`, `hailer._forward` or the Dockerfile
change. Hailer declares the contract it needs. The release builds the image only when the contract
changes. Consider moving `hailer.periods` into the image as its own small package so the image stops
copying the whole `hailer` package.

### 4.6 Smaller items

- Pass the Docker kernel's token through an environment file or `--token-password-file` on a tmpfs rather
  than on the `docker run` command line (F7).
- Split `cli.py` into `cli/chat.py`, `cli/kernel.py`, `cli/setup.py` (init, login, doctor).
- Keep only the `HAILER_*` overrides that are used (`HAILER_CONFIG`, `HAILER_MODEL`, `HAILER_KERNEL`, the
  log level); the rest can come from `hailer.toml`.
- Move dated sections of `PLAN.md` and the finished Docker plan and review into `docs/history/`; keep
  `PLAN.md` as a short description of the current architecture, and `LEARNINGS.md` as the rules.

## 5. What gets deleted

An estimate, to be confirmed while doing the work; tests shrink in proportion.

| Area | Removed by | Roughly |
|---|---|---|
| `config.docker_mount_problems`, `notebook_control_files`, credential-folder and drive-root rules for notebooks | R3 | 150–200 lines |
| `kernel_docker.planted_files`, `planted_warning`, `last-kernel.json` warnings, UID mapping | R3 | 100 lines |
| `kernel.json` liveness proofs, start guard, settings mismatch, id-vs-name bookkeeping | R2 | 400–600 lines |
| marimo registry discovery, `workspace_affinity`, `find_server`, `marimo_url` | R2 | 200 lines |
| Local-runtime secret withholding and `pass_env` (simplified to a dev-mode warning) | R4 | 100 lines |
| `PathMap` in tools and notebook state | R1 | 100 lines |
| Added: Sandbox file methods, notebook sync with allow-list | R1, R3 | +200–300 lines |

Net: on the order of 1,000 lines of source out of 12,000, and the ones removed are the ones the Docker
review found bugs in.

## 6. The cloud version (GCP)

The same image, the same `MarimoClient`, a new `CloudRunSandbox`. The agent and API key stay on the
user's machine; only the kernel moves.

```
Hailer (laptop) ──HTTPS + Google ID token──► Cloud Run service hailer-kernel-<user>-<session>
Browser ─────────HTTPS (IAP or IAM)────────►   max instances 1, min instances 1 while in use
                                               sandbox service account (no roles but the buckets)
                                               /work/data      ← GCS bucket, read-only volume
                                               /work/notebooks ← GCS prefix (or in-instance + sync_out)
                                               egress: all traffic into a VPC with no route out
```

How the desktop pieces map:

| Desktop | GCP |
|---|---|
| `--internal` network + forwarder container | Cloud Run ingress with authentication; no forwarder |
| `--network none` | Direct VPC egress `all-traffic` into a subnet with no NAT and a deny-all egress firewall rule |
| read-only data bind mount | Cloud Storage volume mount with `readonly = true` |
| notebooks volume + `sync_out` | Cloud Storage prefix (single writer, so last-write-wins is acceptable) |
| memory/CPU/pids flags | Cloud Run instance limits (gen2) |
| marimo token | Cloud Run IAM or IAP in front, plus the marimo token behind it |
| `docker pull` from GHCR | Artifact Registry copy of the same image |

Facts checked on 2026-09-22 in Google's documentation, which shape the design:

- A WebSocket on Cloud Run counts as a request and is **cut at the request timeout, at most 60 minutes**;
  clients must reconnect. Session affinity is **best effort**, so a reconnect can land on another
  instance. With `max-instances = 1` there is no other instance, which is why the design uses one
  service per session. An instance with an open WebSocket is billed as active.
- Cloud Storage **volume mounts can be read-only**; Cloud Storage FUSE is not fully POSIX and has no file
  locking (last write wins), and a written file is staged in memory before upload.
- Direct VPC egress with `all-traffic` sends all outbound traffic through the VPC, where firewall rules
  apply.

Must be proven in a spike before building (the go/no-go list):

1. marimo's editor reconnects after the 60-minute cut **to the same kernel** with `max-instances = 1`,
   and the instance is not recycled while a session is open (with `min-instances = 1`).
2. With egress locked down, notebook code **cannot reach the metadata server or Google APIs** with the
   sandbox service account's token. Cloud Run exposes a metadata server to the container; if notebook
   code can obtain a usable token, the service account must have nothing but read on the data bucket
   and write on its notebooks prefix, and it must not be shared between users.
3. The GCS read-only mount still works when egress is locked down (the mount is set up by the platform,
   but that has to be observed, not assumed).
4. Cold start with the image and the mount stays under ~30 s, since it is paid on every session.
5. The browser can reach the service through IAP (or `gcloud run services proxy` as a first step) and
   Hailer can call it with an ID token from the user's Application Default Credentials.

If (1) fails, the fallback is **GKE Autopilot with GKE Sandbox (gVisor)** and a default-deny
NetworkPolicy: better for long-lived stateful kernels, but a cluster to operate. Start with Cloud Run
because it has nothing to run between sessions.

Keep the first cloud version single-user: the CLI creates and deletes the service with the user's own
Google credentials, no broker service and no database. A multi-user control plane (quotas, shared data,
an organisation's projects) is a separate project; the Sandbox contract is the seam it would plug into.
The same contract fits other clouds (Azure Container Apps, AWS Fargate/ECS) with their own storage mounts.

## 7. Phasing

Each phase ships on its own and leaves the product working.

| Phase | Content | Proof |
|---|---|---|
| 0 | R5 image versioning, token off the command line, `cli.py` split, docs moved to `docs/history/` | Suite green; a CLI-only release publishes no image |
| 1 | R1 Sandbox contract; tools and notebook state use sandbox-relative names; spike marimo's file endpoints | Suite green; tools have no host path left (grep) |
| 2 | R2 owned kernel per process; remove reuse, registry and `marimo_url` | `test_docker_integration.py` strict; `kernel stop` cleans leftovers of a killed process |
| 3 | R3 notebook volume + allow-listed sync; shrink mount rules | Integration test: a planted `.git` in the container never reaches the host |
| 4 | R4 docker default, `unsafe-local` | `init` writes docker; local needs the explicit flag |
| 5 | GCP spike (§6 list), then `CloudRunSandbox` | The spike's five checks recorded in `LEARNINGS.md` style |

## 8. Decisions needed from you

1. **Notebook sync instead of a live bind mount (R3).** Accept that the host copy is up to one turn
   behind, in exchange for removing the planted-file risk and matching the cloud? *Recommended: yes.*
2. **Drop cross-terminal reuse (R2)**, including `--keep-marimo`, `marimo_url` and attaching to
   hand-started marimo servers? *Recommended: yes.*
3. **Docker as the default and local as `unsafe-local` (R4).** This affects users on machines without
   Docker (Docker Desktop licensing, locked-down laptops) until the cloud runtime exists. Alternative:
   keep `local` the default until R6 ships, but still label it unsafe. *Recommended: switch when R6 is
   at least in preview, label it unsafe now.*
4. **Cloud target.** Cloud Run first with GKE Autopilot + gVisor as the fallback, single-user, the CLI
   managing the service directly. *Recommended: yes, gated on the §6 spike.*
