"""Tests for hailer.kernel_image: the kernel contract (a tag computed from the image's inputs, the
label, the pinned versions), the packaged Dockerfile, the build context and build args, and the
docker commands for building and pulling (a fake runner: nothing is run)."""

from __future__ import annotations

import ast
import csv
import importlib.metadata
import json
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

from hailer import kernel_image
from hailer.errors import KernelRuntimeError
from hailer.models import KernelConfig

pytestmark = pytest.mark.usefixtures("isolated_package_settings")

_ROOT = Path(__file__).resolve().parents[1]


def test_default_image_is_tagged_with_the_contract_computed_from_its_inputs():
    tag = kernel_image.contract_tag()
    assert tag == f"marimo{kernel_image.MARIMO_VERSION}-{kernel_image.contract_fingerprint()[:12]}"
    assert re.fullmatch(r"marimo[0-9.]+-[0-9a-f]{12}", tag)
    assert kernel_image.default_image() == f"ghcr.io/openafterhours/hailer-kernel:{tag}"
    assert kernel_image.default_image() == KernelConfig().effective_image
    assert KernelConfig(image="company/hailer-kernel:1").effective_image == "company/hailer-kernel:1"


def test_any_change_to_an_input_is_a_new_tag_but_line_endings_are_not(tmp_path, monkeypatch):
    """A changed image can never reuse a published tag, and Windows and Linux checkouts agree."""
    base, installed = kernel_image.contract_tag(), kernel_image.package_dir
    package = tmp_path / "hailer"
    package.mkdir()
    for name in kernel_image.IMAGE_MODULES:
        source = (kernel_image.package_dir() / name).read_bytes().replace(b"\r\n", b"\n")
        (package / name).write_bytes(source.replace(b"\n", b"\r\n"))
    monkeypatch.setattr(kernel_image, "package_dir", lambda: package)
    assert kernel_image.contract_tag() == base, "a Windows checkout (CRLF) has the same tag"
    (package / "periods.py").write_bytes((package / "periods.py").read_bytes() + b"# a comment\r\n")
    assert kernel_image.contract_tag() != base, "even a comment in an image module is a new image"
    monkeypatch.setattr(kernel_image, "package_dir", installed)
    assert kernel_image.contract_tag() == base
    monkeypatch.setitem(kernel_image.IMAGE_PACKAGES, "polars", "0.0.1")
    assert kernel_image.contract_tag() != base, "a package version is part of the contract"


def test_the_images_marimo_is_hailers_pinned_marimo():
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert f"marimo=={kernel_image.MARIMO_VERSION}" in pyproject["project"]["dependencies"]
    assert importlib.metadata.version("marimo") == kernel_image.MARIMO_VERSION, "code mode is a private marimo API"
    assert kernel_image.IMAGE_PACKAGES["marimo"] == kernel_image.MARIMO_VERSION


def test_build_args_are_the_contracts_pinned_versions_not_the_hosts(monkeypatch):
    def no_metadata(name):
        raise AssertionError(f"read {name}'s version from this machine")

    monkeypatch.setattr(importlib.metadata, "version", no_metadata)
    args = kernel_image.build_args()
    assert args.pop("KERNEL_CONTRACT") == kernel_image.contract_tag(), "the image's contract label"
    assert sorted(args.values()) == sorted(kernel_image.IMAGE_PACKAGES.values())
    assert all(version.count(".") == 2 for version in kernel_image.IMAGE_PACKAGES.values()), "exact versions"


def test_prepare_context_holds_the_dockerfile_and_only_the_image_modules(tmp_path):
    ctx = tmp_path / "ctx"
    args = kernel_image.prepare_context(ctx)
    assert args == kernel_image.build_args()
    written = {path.relative_to(ctx).as_posix(): path.read_bytes() for path in ctx.rglob("*") if path.is_file()}
    assert written == kernel_image.context_files(), "exactly what the tag covers: no agent, config or credential code"
    assert {Path(path).name for path in written} == {"Dockerfile", "__init__.py", *kernel_image.IMAGE_MODULES}
    assert not any(b"\r" in data for data in written.values()), "LF endings, even from a Windows checkout"
    dockerfile = written["Dockerfile"].decode("utf-8")
    for name in args:
        assert f"ARG {name}\n" in dockerfile, f"{name} is declared without a default (one place for versions)"
    assert b"__version__" not in written["hailer/__init__.py"], "a Hailer release does not change the image"


def test_the_image_modules_import_nothing_else_of_hailer():
    for name in kernel_image.IMAGE_MODULES:
        tree = ast.parse((kernel_image.package_dir() / name).read_text(encoding="utf-8"))
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        hailer = {module for module in imported if module == "hailer" or module.startswith("hailer.")}
        allowed = {f"hailer.{Path(module).stem}" for module in kernel_image.IMAGE_MODULES}
        assert hailer <= allowed, f"{name} imports {sorted(hailer - allowed)}, which the image does not have"


def test_the_dockerfile_pins_the_base_image_and_runs_as_a_plain_user():
    text = (Path(kernel_image.__file__).parent / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG BASE_IMAGE=python:3.12-slim@sha256:" in text and "FROM ${BASE_IMAGE}" in text
    assert "sysconfig.get_path" in text and "python3.12/site-packages" not in text
    assert "python3 -m venv /opt/hailer-venv" in text and "touch /work/hailer.toml" in text
    assert "> /work/.marimo.toml" in text and "auto_instantiate = true" in text, "a notebook's cells run when it opens"
    assert "USER 1000:1000" in text and "HOME=/home/analyst" in text and "WORKDIR /work" in text
    assert f"{kernel_image.CONTRACT_LABEL}=${{KERNEL_CONTRACT}}" in text, "the label Hailer compares"
    for package in kernel_image.IMAGE_PACKAGES:
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


def test_image_contract_reads_the_contract_label():
    image = kernel_image.default_image()
    labels = {kernel_image.CONTRACT_LABEL: "marimo0.24.2-0123456789ab", "org.opencontainers.image.version": "0.2.5"}
    runner = Runner((0, json.dumps(labels) + "\n"))
    assert kernel_image.image_contract(image, runner) == "marimo0.24.2-0123456789ab"
    assert runner.calls == [["image", "inspect", "--format", "{{json .Config.Labels}}", image]]
    assert kernel_image.image_contract(image, Runner((1, ""))) is None, "not on this machine"
    assert kernel_image.image_contract(image, Runner((0, "null\n"))) == "", "an image without labels"
    assert kernel_image.image_contract(image, Runner((0, "not json"))) == ""
    old = Runner((0, json.dumps({"org.opencontainers.image.version": "0.2.8"}) + "\n"))
    assert kernel_image.image_contract(image, old) == "", "an image from before contracts has no contract"


def test_pull_streams_docker_pull():
    runner = Runner()
    result = kernel_image.pull("registry.example/hailer-kernel:0.2.5", runner)
    assert runner.calls == [["pull", "registry.example/hailer-kernel:0.2.5"]]
    assert runner.kept and (result.returncode, result.stderr) == (1, "denied"), "docker's error is kept to explain the failure"
