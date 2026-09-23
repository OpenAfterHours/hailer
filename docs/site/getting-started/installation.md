# Installation and upgrades {#installation}

`uvx hailer ...` fetches Hailer from PyPI the first time, keeps it in uv's cache with an environment of
its own (marimo, Polars, DuckDB and the rest), and runs it from there. No `pyproject.toml` or `.venv`
is needed in your analysis folder.

To use the latest release, run `uvx hailer@latest`. Use `uvx hailer@X.Y.Z` to select a specific version.

For a permanent `hailer` command:

```bash
uv tool install hailer
```

Then use `hailer` wherever these instructions say `uvx hailer`. Upgrade that installation with
`uv tool upgrade hailer`. You can also use `pip install hailer` in your own Python 3.12+ environment.
See [uv's tools guide](https://docs.astral.sh/uv/guides/tools/) for more installation options.

## Requirements

- Windows 11 is the primary target; macOS and Linux work too.
- Hailer needs Python 3.12 or newer; uv can download it automatically.
- Keep the notebook open in a web browser while using Hailer.
- Every model provider needs an API key (see [Model configuration](../configuration/models.md#model-configuration)).
- **Docker**, installed and running: notebook code runs in a Docker container by default. Install
  [Docker Desktop](https://docs.docker.com/desktop/) on Windows or macOS (on Windows, keep it on Linux
  containers, its default) or [Docker Engine](https://docs.docker.com/engine/install/) on Linux. The first
  start downloads Hailer's kernel image (`uvx hailer kernel pull` does it ahead of time). See
  [Isolated kernel (Docker)](../security/docker.md#isolated-kernel-docker).
- Without Docker, Hailer can run notebook code on your machine, as you, with your files and network, if you
  choose that yourself: `runtime = "unsafe-local"` under `[kernel]` in `hailer.toml`. See
  [Kernel runtimes](../security/runtimes.md#kernel-runtimes) before you do.

For source checkouts and sample data generation, see [Development](../development/contributing.md#development).
