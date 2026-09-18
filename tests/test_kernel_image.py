"""Tests for hailer.kernel_image: the packaged Dockerfile, the build context and build args, and
the docker commands for building and pulling (a fake runner: nothing is run)."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
from pathlib import Path

from hailer import __version__, kernel_image
from hailer.models import KernelConfig


def test_default_image_is_the_published_one_for_this_version():
    assert kernel_image.default_image() == f"ghcr.io/openafterhours/hailer-kernel:{__version__}"
    assert kernel_image.default_image() == KernelConfig().effective_image


def test_build_args_come_from_the_versions_hailer_runs_with():
    args = kernel_image.build_args()
    assert args == {
        "MARIMO_VERSION": importlib.metadata.version("marimo"),
        "POLARS_VERSION": importlib.metadata.version("polars"),
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
    assert "COPY hailer/ /usr/local/lib/python3.12/site-packages/hailer/" in text
    assert "useradd --create-home --uid 1000 analyst" in text and "touch /work/hailer.toml" in text
    assert "USER analyst" in text and "WORKDIR /work" in text
    assert "org.opencontainers.image.version=${HAILER_VERSION}" in text
    for package in ("marimo", "polars", "duckdb", "altair", "plotly"):
        assert f"{package}==${{{package.upper()}_VERSION}}" in text
    instructions = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#")).lower()
    assert "langchain" not in instructions and "keyring" not in instructions, "no agent and no credential store in the image"


def test_build_streams_docker_build_with_the_context_and_its_args():
    class Recorder:
        args: list[str] = []
        context_ready = False

        def stream(self, args):
            self.args = list(args)
            context = Path(args[-1])
            self.context_ready = (context / "Dockerfile").is_file() and (context / "hailer" / "_forward.py").is_file()
            return 0

    recorder = Recorder()
    assert kernel_image.build("hailer-kernel:dev", recorder) == 0
    assert recorder.args[:3] == ["build", "--tag", "hailer-kernel:dev"] and recorder.context_ready
    expected = [part for name, value in kernel_image.build_args().items() for part in ("--build-arg", f"{name}={value}")]
    assert recorder.args[3:-1] == expected
    assert not Path(recorder.args[-1]).exists(), "the temporary context is removed afterwards"
    assert kernel_image.build_command("t", Path("ctx"), {"A": "1"}) == ["build", "--tag", "t", "--build-arg", "A=1", "ctx"]


class Runner:
    """Answers ``image inspect`` with ``reply`` (exit code, stdout); records ``pull``."""

    def __init__(self, reply=(0, "{}")):
        self.reply = reply
        self.calls: list[list[str]] = []

    def run(self, args, *, input=None, timeout=None, check=True, capture=True):
        self.calls.append(list(args))
        code, out = self.reply
        return subprocess.CompletedProcess(args, code, out, "Error: No such image" if code else "")

    def stream(self, args):
        self.calls.append(list(args))
        return 0


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
    assert kernel_image.pull("registry.example/hailer-kernel:0.2.5", runner) == 0
    assert runner.calls == [["pull", "registry.example/hailer-kernel:0.2.5"]]
