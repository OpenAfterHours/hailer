# Docker kernel: final adversarial review

> Historical record: this describes the design at the time it was written (2026-09-18/19). The current
> architecture is in [PLAN.md](../../PLAN.md).

Reviewed on 2026-09-19 on `worktree-docker-kernel`, including the interrupted second review fixes and
integration with main's automatic kernel startup and persistent chat composer.

## Findings and resolutions

| Finding | Resolution |
|---|---|
| Foreground mode could expose data through the writable notebooks mount. | All Docker start paths use the same mount checks. Data inside notebooks is refused before container creation. |
| Notebook code could modify Hailer's own files or credential directories. | Refuse guarded mounts, including credential subdirectories. The Windows AppData temporary-folder allowance does not exempt SSH or cloud credential folders. |
| Git or editor controls could turn a writable notebooks folder into host execution. | Refuse `.git` and `hailer.toml` throughout the notebooks tree; fail closed on unreadable subtrees. Warn about repository/editor/workspace controls at stop and in doctor, without deleting files or following links. This is a mitigation: host editors can act on newly written files before shutdown. |
| A busy or suspended kernel could lose its record during a refused start. | Both runtimes refuse replacement until the recorded kernel is provably gone. A broken Docker context or empty inspect response is not proof of death. |
| An old chat could stop a newer kernel using the same names. | Remove containers and networks by recorded IDs and delete state only when the token still matches. |
| Concurrent cleanup could report missing objects as failures. | Recognize object-specific missing-object errors, handle removal in progress, retry network detachment, and inspect before reporting a failure. |
| Failed cleanup or an unreachable engine could discard recoverable state. | Keep the Docker record for retry, report the failure, and exit 1. A later `kernel stop` can finish cleanup. |
| A state-file write failure could leave a newly started kernel behind. | Both runtimes stop their new kernel and report the disk/permissions failure. Docker leftovers remain discoverable by workspace labels if cleanup also fails. |
| Status could display the configured runtime while the actual kernel was busy. | Diagnostics retain the recorded identity independently of session health. |
| Main's new composer and automatic startup could lose Docker connection details. | Both chat entry points share runtime startup, check settings before reuse, and pass the server object through synchronous and asynchronous chat, notebook switches, and model/thread changes. |
| User-facing diagnostics and docs differed. | Align foreground/data checks, UNC errors, help text, runtime descriptions, cleanup messages, pass-through secret warnings, and documentation. |

## Validation

- The full offline suite passed on Windows with Python 3.13: **828 passed, 10 skipped** (77 seconds).
  The PR workflow also runs Python 3.12 and
  3.13 on Windows and Linux.
- The strict Docker integration suite passed **7/7** against an image built from the reviewed source. It checks
  automatic cell execution, saved notebook edits, blocked data writes and network access, withheld
  secrets, and removal of containers, network and state.
- Regression tests cover the findings above, plus the composer's retained Docker connection through
  `/new`, `/model` and notebook switching.
- Static undefined-name/import checks pass. The source distribution and wheel build successfully.
- A separate CLI invocation refused foreground mode with data inside the notebooks folder before
  creating kernel state. Wheel inspection confirmed the Dockerfile, forwarder and runtime modules are
  packaged. Documentation checks confirmed existing README sections and local links/anchors are intact.

Reproduce from a checkout:

```powershell
uv sync --locked
uv run --locked pytest
uv run --locked python -m scripts.build_kernel_image --load --platform linux/amd64
$env:HAILER_DOCKER_TESTS = "strict"
uv run --locked pytest tests/test_docker_integration.py -rs
uv build
```

The integration tests use temporary workspaces and a headless browser. They do not call a paid model
endpoint. The earlier scripted-gateway tests exercise the real agent loop; a live paid-model Docker
turn and large-file Windows bind-mount performance measurements remain release validation work.
The multi-architecture release build gates PyPI publication; end-to-end ARM64 execution still needs
verification. On the first release, the maintainer must make the GHCR package public as described in
the README's release instructions.
