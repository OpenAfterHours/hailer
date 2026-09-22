"""Integration test for the docker kernel runtime: real containers, a real browser session.

Opt-in, because it needs Docker, the kernel image and a Chrome or Chromium:

- ``HAILER_DOCKER_TESTS`` unset (the default): every test here is skipped.
- ``HAILER_DOCKER_TESTS=1``: run when the prerequisites are there; skip, with the reason, when
  Docker is missing or not running, the image is missing or for another Hailer, or no browser is found.
- ``HAILER_DOCKER_TESTS=strict``: the same, but a missing prerequisite fails instead of skipping.
  The ``docker`` CI job uses it, so a runner without Docker or Chrome cannot turn it green.

The image is the one this Hailer runs (``ghcr.io/openafterhours/hailer-kernel:<version>``), or
``HAILER_KERNEL_IMAGE``. The test never pulls: build it first with
``uv run python -m scripts.build_kernel_image --load`` (or ``uvx hailer kernel build``). The browser
is ``HAILER_TEST_CHROME`` when set, else ``google-chrome``/``chromium`` on PATH or Chrome's usual
install folder on Windows and macOS.

Run it with ``HAILER_DOCKER_TESTS=1 uv run pytest tests/test_docker_integration.py -rs``.

One module-scoped kernel serves every test: :class:`~hailer.kernel.DockerRuntime` starts it for a
temporary workspace holding sample sales data, headless Chrome stays connected to the notebook,
and the tests talk to it through
the sandbox's :class:`~hailer.marimo_client.MarimoClient` by notebook name, as the agent's tools do. The last test
stops it and checks that nothing is left behind (by the ids Docker gave it and the workspace label).
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from fake_excel import write_workbook
from hailer import kernel_image
from hailer.config import load_config
from hailer.errors import KernelRuntimeError
from hailer.kernel_docker import LABEL_CONTRACT, LABEL_OWNER, LABEL_WORKSPACE, DockerKernel, DockerRuntime, owner_alive
from hailer.marimo_client import (
    MarimoClient,
    answers_with_token,
    build_create_cell_code,
    find_free_port,
    wait_for_session,
)
from hailer.models import ExecResult, HailerConfig
from hailer.notebooks import default_notebook, ensure_notebook

MODE = os.environ.get("HAILER_DOCKER_TESTS", "").strip().lower()
STRICT = MODE == "strict"
ENABLED = STRICT or MODE in ("1", "true", "yes")

pytestmark = pytest.mark.skipif(
    not ENABLED, reason="the Docker integration test is opt-in: set HAILER_DOCKER_TESTS=1 (see the module docstring)"
)

#: A secret in Hailer's own environment that must never reach notebook code.
SECRET_NAME = "FOO_API_KEY"
SECRET_VALUE = "hailer-integration-secret-6f1c0b"
SAMPLE_MONTHS = 2
SAMPLE_ROWS = 50
PREFERRED_PORT = 2791
SESSION_TIMEOUT_SEC = 90.0
SAVE_TIMEOUT_SEC = 30.0
EXEC_TIMEOUT_SEC = 120.0

_ROOT = Path(__file__).resolve().parents[1]
_CHROME_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome")


def _unavailable(reason: str) -> None:
    """Skip (``HAILER_DOCKER_TESTS=1``) or fail (``strict``) because a prerequisite is missing."""
    if STRICT:
        pytest.fail(f"HAILER_DOCKER_TESTS=strict: {reason}", pytrace=False)
    pytest.skip(reason)


def find_chrome() -> str | None:
    """A Chrome or Chromium executable: ``HAILER_TEST_CHROME``, PATH, then the usual install folders."""
    override = os.environ.get("HAILER_TEST_CHROME", "").strip()
    if override:
        return shutil.which(override) or (override if Path(override).is_file() else None)
    for name in _CHROME_NAMES:
        found = shutil.which(name)
        if found:
            return found
    candidates: list[Path] = []
    if os.name == "nt":
        for base in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            if os.environ.get(base):
                candidates.append(Path(os.environ[base]) / "Google" / "Chrome" / "Application" / "chrome.exe")
    elif sys.platform == "darwin":
        candidates += [
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
        ]
    return next((str(path) for path in candidates if path.is_file()), None)


def open_in_browser(chrome: str, url: str, scratch: Path) -> subprocess.Popen:
    """Keep Chrome connected until fixture teardown, using real time for kernel startup.

    A virtual-time budget can fast-forward marimo's WebSocket timers before the real kernel
    is ready (the v0.2.7 release showed "kernel not found" and never initialized DATA_DIR).
    Capture screenshots through DevTools instead of a command-line mode that exits early.
    """
    args = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={scratch / 'chrome-profile'}",
        "--remote-debugging-port=0",
        "--remote-debugging-address=127.0.0.1",
        "--window-size=1280,900",
        url,
    ]
    if sys.platform.startswith("linux"):
        args.insert(1, "--no-sandbox")  # CI runners often cannot give Chrome its sandbox; the page is our own
    return subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def capture_screenshot(scratch: Path) -> None:
    """Best-effort diagnostics from the existing page; never change its clock or reload it."""
    try:
        # websockets is already required by marimo; no browser driver or extra dependency is needed.
        from websockets.sync.client import connect

        port = int((scratch / "chrome-profile" / "DevToolsActivePort").read_text().splitlines()[0])
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}/json/list", timeout=5) as response:
            pages = json.load(response)
        page = next(page for page in pages if page["type"] == "page")
        with connect(page["webSocketDebuggerUrl"], proxy=None, open_timeout=5, close_timeout=1) as socket:
            socket.send(json.dumps({"id": 1, "method": "Page.captureScreenshot", "params": {"format": "png"}}))
            deadline = time.monotonic() + 5
            while True:
                reply = json.loads(socket.recv(timeout=max(0, deadline - time.monotonic())))
                if reply.get("id") == 1:
                    (scratch / "notebook.png").write_bytes(base64.b64decode(reply["result"]["data"]))
                    return
    except Exception as exc:
        # A diagnostic failure must not mask the test failure or prevent container cleanup.
        # Avoid exception text: the page URL contains the marimo token.
        print(f"Could not capture the notebook screenshot ({type(exc).__name__}).", file=sys.stderr)


def _end(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def write_sales_data(folder: Path) -> list[Path]:
    """Monthly sales files from scripts/make_sample_data.py (a small Polars frame when it is absent)."""
    script = _ROOT / "scripts" / "make_sample_data.py"
    if script.is_file():
        spec = importlib.util.spec_from_file_location("hailer_make_sample_data_script", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.write_sample_data(folder, months=SAMPLE_MONTHS, rows=SAMPLE_ROWS)
    import polars as pl

    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for month in range(1, SAMPLE_MONTHS + 1):
        path = folder / f"25-{month:02d} sales.parquet"
        pl.DataFrame({"region": ["North", "South"] * (SAMPLE_ROWS // 2), "revenue": [10.0] * SAMPLE_ROWS}).write_parquet(path)
        written.append(path)
    return written


def _poll(check: Callable[[], bool], timeout: float, interval: float = 0.5) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if check():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)


@dataclass
class Kernel:
    """The running docker kernel the tests share."""

    config: HailerConfig
    runtime: DockerRuntime
    kernel: DockerKernel
    client: MarimoClient
    chrome: subprocess.Popen
    browser_scratch: Path

    @property
    def notebook(self) -> Path:
        """The notebook's file on the host (the notebooks folder is a bind mount)."""
        return self.config.notebook

    def run(self, code: str) -> ExecResult:
        return self.client.execute(code, notebook=default_notebook(self.config), timeout=EXEC_TIMEOUT_SEC)

    def docker(self, *args: str) -> list[str]:
        result = self.runtime.runner.run(list(args), check=True, timeout=60)
        return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]

    def diagnostics(self) -> str:
        capture_screenshot(self.browser_scratch)
        return "docker logs (kernel):\n" + "\n".join(self.kernel.log_tail(30))


def _clean_env() -> dict[str, str]:
    """This process's environment (the secret included) without HAILER_* overrides, except the image."""
    env = {name: value for name, value in os.environ.items() if not name.upper().startswith("HAILER_")}
    if os.environ.get("HAILER_KERNEL_IMAGE"):
        env["HAILER_KERNEL_IMAGE"] = os.environ["HAILER_KERNEL_IMAGE"]
    return env


@pytest.fixture(scope="module")
def docker_kernel(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Kernel]:
    if shutil.which("docker") is None:
        _unavailable("docker is not on PATH")
    chrome = find_chrome()
    if chrome is None:
        _unavailable("no Chrome or Chromium found (install one, or set HAILER_TEST_CHROME to its path)")

    workspace = tmp_path_factory.mktemp("docker-workspace")
    scratch = tmp_path_factory.mktemp("docker-browser")
    (workspace / "hailer.toml").write_text(
        '[hailer]\nnotebook = "notebooks/analysis.py"\ndata_dir = "data"\n\n[kernel]\nruntime = "docker"\n', encoding="utf-8"
    )
    write_sales_data(workspace / "data")
    write_workbook(workspace / "data" / "Regional sales.xlsx")

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(SECRET_NAME, SECRET_VALUE)  # in Hailer's environment, and so in docker's
        config = load_config(workspace, env=_clean_env())
        assert config.kernel.runtime == "docker"
        runtime = DockerRuntime(config)
        try:
            runtime.engine_version()
        except KernelRuntimeError as err:
            _unavailable(f"Docker cannot be used: {err} {err.hint or ''}".strip())
        found = kernel_image.image_contract(runtime.image, runtime.runner)
        build = "uv run python -m scripts.build_kernel_image --load (or uvx hailer kernel build)"
        if found is None:
            _unavailable(f"the kernel image {runtime.image} is not on this machine; build it with {build}")
        if found != kernel_image.contract_tag():
            needed = kernel_image.contract_tag()
            _unavailable(f"{runtime.image} has kernel contract {found or '(no label)'}, this Hailer needs {needed}; rebuild it with {build}")

        ensure_notebook(config.notebook, title="Sales")
        kernel = runtime.start(find_free_port(PREFERRED_PORT))
        assert isinstance(kernel, DockerKernel)
        chrome_proc: subprocess.Popen | None = None
        try:
            name = default_notebook(config)
            client = kernel.client(notebook=name, timeout=30)
            chrome_proc = open_in_browser(chrome, kernel.notebook_url(name, with_token=True), scratch)
            if wait_for_session(client, name, timeout=SESSION_TIMEOUT_SEC) is None:
                tail = "\n".join(kernel.log_tail(30))
                pytest.fail(f"no marimo session for {config.notebook.name} within {SESSION_TIMEOUT_SEC:.0f} s\n{tail}")
            yield Kernel(
                config=config, runtime=runtime, kernel=kernel, client=client,
                chrome=chrome_proc, browser_scratch=scratch,
            )
        finally:
            if chrome_proc is not None:
                if not (scratch / "notebook.png").exists():
                    capture_screenshot(scratch)
                _end(chrome_proc)
            kernel.stop()
            try:
                runtime.remove_leftovers(every=True)
            except KernelRuntimeError:
                pass


def test_notebook_code_runs_in_the_container_on_the_mounted_data(docker_kernel: Kernel):
    assert docker_kernel.chrome.poll() is None, "Chrome must stay connected while the notebook initializes"
    result = docker_kernel.run("import os, platform\nprint(platform.system(), os.getcwd(), os.getuid() != 0)")
    assert result.success, result.stderr + docker_kernel.diagnostics()
    assert result.stdout.split() == ["Linux", "/work", "True"], "Linux, in /work, not as root"

    result = docker_kernel.run(
        "import polars as pl\n"
        "frame = pl.read_parquet('/work/data/25-01 sales.parquet')\n"
        "print(frame.height, round(float(frame['revenue'].sum()), 2) > 0)\n"
    )
    assert result.success, result.stderr
    assert result.stdout.split() == [str(SAMPLE_ROWS), "True"]

    # The starter notebook ran when the browser opened it, without anyone running a cell (the image's
    # /work/.marimo.toml turns marimo's auto_instantiate on): hailer.periods is in the image, and
    # WORKSPACE / DATA_DIR are the container's folders, which is what the agent relies on.
    def started() -> bool:
        return docker_kernel.run("print(DATA_DIR, len(data_files), len(period_files))").stdout.split() == [
            "/work/data", str(SAMPLE_MONTHS + 1), str(SAMPLE_MONTHS),
        ]  # fmt: skip

    assert _poll(started, 60), docker_kernel.run("print(DATA_DIR)").stderr + docker_kernel.diagnostics()
    assert docker_kernel.chrome.poll() is None, "Chrome must remain connected after notebook initialization"


def test_excel_can_be_read_in_the_container_without_installing_packages(docker_kernel: Kernel):
    result = docker_kernel.run(
        "import fastexcel, polars as pl\n"
        "path = '/work/data/Regional sales.xlsx'\n"
        "print(fastexcel.read_excel(path).sheet_names)\n"
        "sales = pl.read_excel(path, sheet_name='Sales', schema_overrides={'customer_id': pl.String})\n"
        "print(sales.height, sales['revenue'].sum(), sales['customer_id'][0])\n"
        "print(pl.read_excel(path, sheet_name='Budget')['target'].sum())\n"
    )
    assert result.success, result.stderr + docker_kernel.diagnostics()
    assert result.stdout.splitlines() == ["['Sales', 'Budget']", "3 200.5 001", "250"]


def test_a_code_mode_cell_is_saved_in_the_host_notebook(docker_kernel: Kernel):
    code = "ci_total = 6 * 7\nci_total"
    result = docker_kernel.run(build_create_cell_code(code, name="ci_total"))
    assert result.success, result.stderr + docker_kernel.diagnostics()

    def saved() -> bool:
        return "ci_total = 6 * 7" in docker_kernel.notebook.read_text(encoding="utf-8")

    assert _poll(saved, SAVE_TIMEOUT_SEC), "the new cell never reached " + str(docker_kernel.notebook)
    assert _poll(lambda: docker_kernel.run("print(ci_total)").stdout.strip() == "42", 30), "the cell ran in the kernel"


def test_the_data_folder_is_read_only(docker_kernel: Kernel):
    target = docker_kernel.config.data_dir / "written-by-the-kernel.txt"
    result = docker_kernel.run(
        "try:\n"
        "    with open('/work/data/written-by-the-kernel.txt', 'w') as handle:\n"
        "        handle.write('x')\n"
        "    print('WRITTEN')\n"
        "except OSError as err:\n"
        "    print('REFUSED', err.errno)\n"
        "import os\n"
        "try:\n"
        "    os.remove('/work/data/25-01 sales.parquet')\n"
        "    print('DELETED')\n"
        "except OSError as err:\n"
        "    print('REFUSED', err.errno)\n"
    )
    assert result.success, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 2 and all(line.startswith("REFUSED") for line in lines), result.stdout
    assert not target.exists()
    assert (docker_kernel.config.data_dir / "25-01 sales.parquet").is_file()


def test_notebook_code_has_no_network(docker_kernel: Kernel):
    result = docker_kernel.run(
        "import socket, urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('http://1.1.1.1', timeout=5)\n"
        "    print('CONNECTED')\n"
        "except Exception as err:\n"
        "    print('NO_NETWORK', type(err).__name__)\n"
        "try:\n"
        "    socket.getaddrinfo('pypi.org', 443)\n"
        "    print('RESOLVED')\n"
        "except OSError as err:\n"
        "    print('NO_DNS', type(err).__name__)\n"
    )
    assert result.success, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 2 and lines[0].startswith("NO_NETWORK") and lines[1].startswith("NO_DNS"), result.stdout


def test_host_secrets_are_not_in_the_kernel(docker_kernel: Kernel):
    assert os.environ.get(SECRET_NAME) == SECRET_VALUE, "set in Hailer's environment by the fixture"
    result = docker_kernel.run("import os\nprint(sorted(os.environ))\nprint(repr(dict(os.environ)))")
    assert result.success, result.stderr
    assert SECRET_NAME not in result.stdout and SECRET_VALUE not in result.stdout
    assert docker_kernel.kernel.server.token not in result.stdout, "the server token is not in the kernel's environment"


def test_the_token_is_not_in_docker_inspect_or_the_kernels_command_lines(docker_kernel: Kernel):
    """F7: marimo reads the token from a mounted file, so neither ``docker inspect`` (Args, Cmd,
    Env) nor a process list shows it; the file on this machine is gone once the kernel answered."""
    token = docker_kernel.kernel.server.token
    kernel_id = docker_kernel.kernel.container_ids[0]
    inspected = docker_kernel.runtime.runner.run(["inspect", kernel_id], timeout=30)
    assert inspected.returncode == 0 and "--token-password-file" in inspected.stdout, inspected.stderr
    assert token not in inspected.stdout, "the token is not in docker inspect"
    workspace = Path(docker_kernel.config.workspace)
    assert not list((workspace / ".hailer").glob("kernel-token-*")), "the token file was deleted after the start"
    result = docker_kernel.run(
        "import pathlib\n"
        "lines = [p.read_bytes().replace(b'\\0', b' ').decode(errors='replace') for p in pathlib.Path('/proc').glob('[0-9]*/cmdline')]\n"
        "print(len(lines))\n"
        "print('\\n'.join(lines))"
    )
    assert result.success, result.stderr
    assert "marimo" in result.stdout and token not in result.stdout, "no process shows the token"


def test_the_containers_carry_this_process_as_their_owner(docker_kernel: Kernel):
    """The labels a later start (or `uvx hailer kernel stop`) uses to tell a live session's kernel
    from one a killed Hailer left behind; the kernel is removed by the ids Docker gave it."""
    kernel = docker_kernel.kernel
    assert len(kernel.container_ids) == 2 and kernel.network_id
    running = docker_kernel.docker("ps", "--no-trunc", "--format", "{{.ID}}")
    assert all(ident in running for ident in kernel.container_ids), running
    labels = json.loads(docker_kernel.docker("inspect", "--format", "{{json .Config.Labels}}", kernel.container_ids[0])[0])
    assert labels[LABEL_OWNER] == kernel.owner.id and owner_alive(docker_kernel.config.workspace, labels[LABEL_OWNER])
    assert labels[LABEL_CONTRACT] == kernel_image.contract_tag()


def test_notebook_files_go_through_marimos_file_api_in_the_container(docker_kernel: Kernel):
    """The sandbox contract against the real marimo in the container: notebooks listed, created,
    read and replaced by names relative to /work/notebooks, through marimo's file endpoints."""
    from hailer.errors import NotebookExistsError
    from hailer.notebooks import render_template

    kernel = docker_kernel.kernel
    assert default_notebook(docker_kernel.config) in [f.name for f in kernel.list_notebooks()]
    source = render_template("empty", title="File API")
    assert kernel.write_notebook("api/check.py", source, replace=False).name == "api/check.py"
    assert (docker_kernel.config.notebooks_root / "api" / "check.py").read_bytes() == source.encode("utf-8")
    assert kernel.read_notebook("api/check.py") == source
    with pytest.raises(NotebookExistsError):
        kernel.write_notebook("api/check.py", source, replace=False)
    kernel.write_notebook("api/check.py", source + "# again\n")
    assert kernel.read_notebook("api/check.py").endswith("# again\n")
    assert "api/check.py" in [f.name for f in kernel.list_notebooks()]


def test_stop_leaves_no_containers_network_or_token(docker_kernel: Kernel):
    workspace = Path(docker_kernel.config.workspace)
    server = docker_kernel.kernel.server
    names = docker_kernel.runtime.names

    docker_kernel.kernel.stop()

    label = f"label={LABEL_WORKSPACE}={docker_kernel.runtime.workspace_label}"
    assert docker_kernel.docker("ps", "-a", "--filter", label, "--format", "{{.Names}}") == []
    assert docker_kernel.docker("network", "ls", "--filter", label, "--format", "{{.Name}}") == []
    everything = docker_kernel.docker("ps", "-a", "--format", "{{.Names}}")
    assert names.kernel not in everything and names.forwarder not in everything
    assert names.network not in docker_kernel.docker("network", "ls", "--format", "{{.Name}}")
    assert not list((workspace / ".hailer").glob("kernel-token-*")), "no token folder left"
    assert not answers_with_token(server.url, server.token)
    assert docker_kernel.notebook.is_file(), "the notebook stays on the host"
