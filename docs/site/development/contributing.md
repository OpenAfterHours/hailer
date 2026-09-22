# Development and tests {#development}

To work on Hailer itself:

```bash
git clone https://github.com/OpenAfterHours/hailer.git
cd hailer
uv sync --locked
uv run hailer login openai
uv run hailer
```

Use `uv run hailer ...` in the checkout so you run the code you are changing. The repository already
includes `hailer.toml`, a starter notebook and the data folder. Configure another provider in
`hailer.toml` before logging in if needed.

The checkout also has optional sample data: six months of synthetic sales orders, with an extra column
in later months. Generate it before starting Hailer:

```bash
uv run python scripts/make_sample_data.py          # writes data/25-01 sales.parquet ... 25-06
uv run python scripts/make_sample_data.py --help   # --out, --months, --rows, --start, --seed, --dataset
```

Hailer pins `marimo==0.24.2` because it drives the private `marimo._code_mode` API, which has no stability
guarantee. The agent uses `langchain`, `langchain-openai` and `langgraph-checkpoint-sqlite` within their
declared major-version bounds. Upgrade deliberately and re-run the tests.

## Tests

```bash
uv run pytest
```

The suite is offline: it needs no API key, no marimo server, no model endpoint, no network and no Docker.
The marimo protocol is exercised against a local fake server (`tests/fake_marimo.py`, token-checked like a
server Hailer starts), the agent against a scripted chat model and against a strict Chat-Completions-only
fake gateway on loopback (`tests/fake_gateway.py`, which rejects unknown request fields and model names the
way internal gateways do), and the credential store against an in-memory backend. The kernel runtimes are
tested without starting anything: `tests/test_kernel.py` (path map, owners, the local
runtime's record and orphan cleanup, the kernel's environment, the local runtime with its processes faked), `tests/test_kernel_docker.py` (the docker runtime
against `tests/fake_docker.py`, a scripted `docker` CLI that keeps containers and networks with ids,
labels and `--filter`, like Docker 29), `tests/test_kernel_image.py`, `tests/test_forward.py` (the forwarder
on real loopback sockets) and `tests/test_build_kernel_image.py`. `tests/fake_kernel.py` holds the shared
pieces.

`tests/test_docker_integration.py` drives the real docker runtime: it starts a kernel for a temporary
workspace with sample sales data, opens the notebook in headless Chrome, and checks that code runs as a
non-root user in `/work`, a code-mode cell lands in the host notebook, writes to the data folder and the
network are refused, no host secret or the server token is visible, and stopping leaves nothing behind.
It is opt-in and never pulls the image:

```bash
uv run python -m scripts.build_kernel_image --load     # the image for this checkout, into local Docker
HAILER_DOCKER_TESTS=1 uv run pytest tests/test_docker_integration.py -rs
```

(In PowerShell, set the variable first: `$env:HAILER_DOCKER_TESTS = "1"`.)
`HAILER_DOCKER_TESTS=1` skips, with the reason, when Docker, the image or a browser is missing;
`HAILER_DOCKER_TESTS=strict` fails instead. `HAILER_TEST_CHROME` picks the browser (otherwise
`google-chrome` or `chromium` on PATH, or Chrome's usual install folder on Windows and macOS), and
`HAILER_KERNEL_IMAGE` another image.

`.github/workflows/test.yml` runs the suite on Ubuntu and Windows with Python 3.12 and 3.13 for every
push to `main` and every pull request; the repository requires those four checks by name. A fifth job,
*Docker kernel*, builds the image from the checkout on Ubuntu and runs the integration test with
`HAILER_DOCKER_TESTS=strict` (Linux only: GitHub's Windows runners only run Windows containers). It is
not a required check, but the release waits for it.
