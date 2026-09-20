"""Tests for hailer.kernel_image: the packaged Dockerfile, the build context and build args, and
the docker commands for building and pulling (a fake runner: nothing is run)."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import subprocess
from pathlib import Path

import pytest

from hailer import __version__, kernel_image
from hailer.errors import KernelRuntimeError
from hailer.models import KernelConfig

pytestmark = pytest.mark.usefixtures("isolated_package_settings")


def test_default_image_is_the_published_one_for_this_version():
    assert kernel_image.default_image() == f"ghcr.io/openafterhours/hailer-kernel:{__version__}"
    assert kernel_image.default_image() == KernelConfig().effective_image


def test_build_args_come_from_the_versions_hailer_runs_with():
    args = kernel_image.build_args()
    assert args == {
        "MARIMO_VERSION": importlib.metadata.version("marimo"),
        "POLARS_VERSION": importlib.metadata.version("polars"),
        "FASTEXCEL_VERSION": importlib.metadata.version("fastexcel"),
        "DUCKDB_VERSION": importlib.metadata.version("duckdb"),
        "HAILER_VERSION": __version__,
    }
    assert args["MARIMO_VERSION"] == "0.24.2", "the pin in pyproject.toml: code mode is a private marimo API"


def test_prepare_context_holds_the_dockerfile_and_the_package_without_bytecode(tmp_path):
    ctx = tmp_path / "ctx"
    args = kernel_image.prepare_context(ctx)
    assert args == kernel_image.build_args()
    dockerfile = (ctx / "Dockerfile").read_text(encoding="utf-8")
    assert b"\r" not in (ctx / "Dockerfile").read_bytes(), "LF endings, even from a Windows checkout"
    for name in args:
        assert f"ARG {name}" in dockerfile, f"{name} is declared"
    for module in ("__init__.py", "_forward.py", "periods.py", "errors.py", "models.py"):
        assert (ctx / "hailer" / module).is_file(), module
    assert not list(ctx.rglob("__pycache__")) and not list(ctx.rglob("*.pyc"))


def test_the_dockerfile_pins_the_base_image_and_runs_as_a_plain_user():
    text = (Path(kernel_image.__file__).parent / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG BASE_IMAGE=python:3.12-slim@sha256:" in text and "FROM ${BASE_IMAGE}" in text
    assert "sysconfig.get_path" in text and "python3.12/site-packages" not in text
    assert "python3 -m venv /opt/hailer-venv" in text and "touch /work/hailer.toml" in text
    assert "> /work/.marimo.toml" in text and "auto_instantiate = true" in text, "a notebook's cells run when it opens"
    assert "USER 1000:1000" in text and "HOME=/home/analyst" in text and "WORKDIR /work" in text
    assert "org.opencontainers.image.version=${HAILER_VERSION}" in text
    for package in ("marimo", "polars", "fastexcel", "duckdb", "altair", "plotly"):
        assert f"{package}==${{{package.upper()}_VERSION}}" in text
    instructions = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#")).lower()
    assert "langchain" not in instructions and "keyring" not in instructions, "no agent and no credential store in the image"


def test_build_streams_docker_build_with_the_context_and_its_args():
    class Recorder:
        args: list[str] = []
        context_ready = False

        def stream(self, args, *, keep_errors=False):
            self.args = list(args)
            context = Path(args[-1])
            self.context_ready = (context / "Dockerfile").is_file() and (context / "hailer" / "_forward.py").is_file()
            return subprocess.CompletedProcess(args, 0, None, "")

    recorder = Recorder()
    assert kernel_image.build("hailer-kernel:dev", recorder) == 0
    assert recorder.args[:3] == ["build", "--tag", "hailer-kernel:dev"] and recorder.context_ready
    expected = [part for name, value in kernel_image.build_args().items() for part in ("--build-arg", f"{name}={value}")]
    assert recorder.args[3:-1] == expected
    assert not Path(recorder.args[-1]).exists(), "the temporary context is removed afterwards"
    assert kernel_image.build_command("t", Path("ctx"), {"A": "1"}) == ["build", "--tag", "t", "--build-arg", "A=1", "ctx"]


def test_corporate_build_keeps_secret_contents_out_of_context_and_command(tmp_path):
    config = tmp_path / "pip, company.ini"
    config.write_text("[global]\nindex-url = https://build:private-token@packages.example/simple\n")
    cert = tmp_path / "company ca.pem"
    cert.write_text("private-test-certificate")

    class Recorder:
        def stream(self, args):
            assert "--no-cache" in args
            assert "BASE_IMAGE=registry.example/python:3.13" in args
            secrets = [next(csv.reader([args[i + 1]])) for i, arg in enumerate(args) if arg == "--secret"]
            assert secrets == [
                ["type=file", "id=pip_config", f"src={config.resolve()}"],
                ["type=file", "id=pip_cert", f"src={cert.resolve()}"],
            ]
            assert "private-token" not in str(args) and "private-test-certificate" not in str(args)
            context = Path(args[-1])
            assert {p.name for p in context.iterdir()} == {"Dockerfile", "hailer"}
            for path in context.rglob("*"):
                if path.is_file():
                    assert b"private-token" not in path.read_bytes()
                    assert b"private-test-certificate" not in path.read_bytes()
            return subprocess.CompletedProcess(args, 0)

    assert kernel_image.build(
        "company/hailer:dev", Recorder(), base_image="registry.example/python:3.13", pip_config=config, pip_cert=cert,
        no_cache=True,
    ) == 0


@pytest.mark.parametrize("option", ["pip_config", "pip_cert"])
@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_invalid_secret_file_fails_before_docker(tmp_path, option, kind):
    source = tmp_path / "missing" if kind == "missing" else tmp_path
    with pytest.raises(KernelRuntimeError, match="Cannot read"):
        kernel_image.build("test", None, **{option: source})


def test_empty_base_image_is_rejected():
    with pytest.raises(KernelRuntimeError, match="must not be empty"):
        kernel_image.build("test", None, base_image="  ")


class Runner:
    """Answers ``image inspect`` with ``reply`` (exit code, stdout); records ``pull``."""

    def __init__(self, reply=(0, "{}")):
        self.reply = reply
        self.calls: list[list[str]] = []

    def run(self, args, *, timeout=None, check=False):
        self.calls.append(list(args))
        code, out = self.reply
        return subprocess.CompletedProcess(args, code, out, "Error: No such image" if code else "")

    def stream(self, args, *, keep_errors=False):
        self.calls.append(list(args))
        self.kept = keep_errors
        return subprocess.CompletedProcess(args, 1, None, "denied" if keep_errors else "")


def test_image_version_reads_the_label():
    image = kernel_image.default_image()
    runner = Runner((0, json.dumps({kernel_image.VERSION_LABEL: "0.2.5", "other": "x"}) + "\n"))
    assert kernel_image.image_version(image, runner) == "0.2.5"
    assert runner.calls == [["image", "inspect", "--format", "{{json .Config.Labels}}", image]]
    assert kernel_image.image_version(image, Runner((1, ""))) is None, "not on this machine"
    assert kernel_image.image_version(image, Runner((0, "null\n"))) == "", "an image without labels"
    assert kernel_image.image_version(image, Runner((0, "not json"))) == ""


def test_pull_streams_docker_pull():
    runner = Runner()
    result = kernel_image.pull("registry.example/hailer-kernel:0.2.5", runner)
    assert runner.calls == [["pull", "registry.example/hailer-kernel:0.2.5"]]
    assert runner.kept and (result.returncode, result.stderr) == (1, "denied"), "docker's error is kept to explain the failure"
