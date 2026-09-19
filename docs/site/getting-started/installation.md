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
- Docker Desktop (Windows, macOS) or Docker Engine (Linux) is needed only for the
  [optional Docker runtime](../security/docker.md#isolated-kernel-docker).

For source checkouts and sample data generation, see [Development](../development/contributing.md#development).
