"""The kernel image the docker runtime runs: which one, and how to build or pull it.

An image is defined by its **kernel contract**, not by Hailer's version: the packaged Dockerfile,
the few hailer modules copied into it (:data:`IMAGE_MODULES` and a minimal ``hailer/__init__.py``)
and the package versions it installs (:data:`IMAGE_PACKAGES`, marimo equal to Hailer's own pin).
:func:`contract_tag` names it from its content, ``marimo<version>-<first 12 hex of the sha256 of
the inputs>``: the default image's tag and the image's :data:`CONTRACT_LABEL`, which Hailer compares
before using an image. Any change to an input (even a comment) is a new tag, which the next release
publishes; a release that changes none of them reuses the published image.

``src/hailer/docker/Dockerfile`` ships inside the package, so ``uvx hailer kernel build`` works
from any installed Hailer (a checkout, uvx or pip) without a wheel or PyPI: :func:`prepare_context`
writes the build context and returns the build args. The release workflow reuses it for the
published multi-arch image, with the same arguments::

    args = prepare_context(Path("ctx"))  # {"MARIMO_VERSION": ..., "KERNEL_CONTRACT": "marimo0.24.2-0123456789ab"}
    # docker buildx build --platform linux/amd64,linux/arm64 --tag <image>
    #     --build-arg MARIMO_VERSION=... (one per entry of args) ctx

Every ``docker`` call goes through a :class:`~hailer.kernel_docker.DockerRunner`, so tests can
check the argument lists without Docker.
"""

from __future__ import annotations

import configparser
import csv
import hashlib
import importlib.resources
import io
import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from hailer import kernel_packages
from hailer.errors import KernelRuntimeError
from hailer.models import KERNEL_IMAGE_REPOSITORY

if TYPE_CHECKING:  # pragma: no cover - annotations only; hailer.kernel_docker imports this module
    from hailer.kernel_docker import DockerRunner

#: marimo inside the image: equal to Hailer's own pin in pyproject.toml (code mode is a private
#: marimo API, so the kernel's marimo must be the one Hailer drives).
MARIMO_VERSION = "0.24.2"
#: What the image installs (distribution -> exact version), in one place: an image is defined by its
#: contract, never by the versions installed on the machine that builds it.
IMAGE_PACKAGES = {
    "marimo": MARIMO_VERSION,
    "polars": "1.44.2",
    "fastexcel": "0.21.0",
    "duckdb": "1.5.5",
    # Code mode formats the cells the agent writes with the kernel's ruff (hailer.code_checks).
    "ruff": "0.16.8",
    "altair": "6.3.0",
    "plotly": "7.1.0",
}
#: The hailer modules the image gets: the notebook helpers, the errors they raise and the forwarder.
IMAGE_MODULES = ("errors.py", "periods.py", "_forward.py")
#: The image's ``hailer/__init__.py``: not the real one, whose version changes every release.
IMAGE_INIT = '"""Hailer in the kernel image: the notebook helpers (hailer.periods) and the forwarder (hailer._forward)."""\n'
#: The image label that must equal :func:`contract_tag` before Hailer uses an image.
CONTRACT_LABEL = "org.openafterhours.hailer.kernel-contract"
_IMAGE_TIMEOUT_SEC = 30.0


def package_dir() -> Path:
    """The folder of the installed ``hailer`` package (where the image modules come from)."""
    return Path(__file__).resolve().parent


def _lf(data: bytes) -> bytes:
    # LF endings whatever the checkout did (git's autocrlf on Windows): the Dockerfile's RUN lines
    # continue with "\", and the fingerprint must not depend on the platform.
    return data.replace(b"\r\n", b"\n")


def context_files() -> dict[str, bytes]:
    """The build context (POSIX path -> contents): ``Dockerfile``, then ``hailer/__init__.py``
    (:data:`IMAGE_INIT`) and :data:`IMAGE_MODULES`, all with LF endings."""
    dockerfile = importlib.resources.files("hailer").joinpath("docker/Dockerfile")
    files = {"Dockerfile": _lf(dockerfile.read_bytes()), "hailer/__init__.py": IMAGE_INIT.encode("utf-8")}
    for name in IMAGE_MODULES:
        files[f"hailer/{name}"] = _lf((package_dir() / name).read_bytes())
    return files


def contract_fingerprint() -> str:
    """The sha256 of the contract's inputs: :func:`context_files` and :data:`IMAGE_PACKAGES`."""
    digest = hashlib.sha256()
    for path, data in sorted(context_files().items()):
        digest.update(f"file {path} {len(data)}\n".encode())
        digest.update(data)
    for name, version in sorted(IMAGE_PACKAGES.items()):
        digest.update(f"package {name}=={version}\n".encode())
    return digest.hexdigest()


def contract_tag() -> str:
    """``marimo<MARIMO_VERSION>-<contract_fingerprint()[:12]>``: the default image's tag and its
    contract label. Computed from the content, so a changed input can never reuse an old tag."""
    return f"marimo{MARIMO_VERSION}-{contract_fingerprint()[:12]}"


def default_image() -> str:
    """``ghcr.io/openafterhours/hailer-kernel:marimo<version>-<fingerprint>``."""
    return f"{KERNEL_IMAGE_REPOSITORY}:{contract_tag()}"


def build_args() -> dict[str, str]:
    """The Dockerfile's build args: ``<NAME>_VERSION`` per image package and ``KERNEL_CONTRACT``
    (:func:`contract_tag`, which becomes the image's contract label)."""
    args = {f"{name.upper()}_VERSION": version for name, version in IMAGE_PACKAGES.items()}
    args["KERNEL_CONTRACT"] = contract_tag()
    return args


def prepare_context(dest: Path) -> dict[str, str]:
    """Fill ``dest`` with :func:`context_files` and return the build args for it.

    ``dest`` is created when missing and must not already hold a ``hailer`` folder.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "hailer").mkdir()
    for path, data in context_files().items():
        (dest / path).write_bytes(data)
    return build_args()


def build_options(
    *, base_image: str | None = None, pip_config: Path | None = None, pip_cert: Path | None = None,
    no_cache: bool = False,
) -> list[str]:
    """Extra Docker build arguments shared by the CLI and the buildx script.

    Files are passed directly as BuildKit secrets, never read into command arguments or copied
    into the context. Docker parses --secret as CSV (a Windows path can contain commas).
    """
    options: list[str] = ["--no-cache"] if no_cache else []
    if base_image is not None:
        if not base_image.strip():
            raise KernelRuntimeError("The base image must not be empty.", hint="Use --base-image REGISTRY/IMAGE:TAG.")
        options += ["--build-arg", f"BASE_IMAGE={base_image}"]
    for name, path, flag in (("pip_config", pip_config, "--pip-config"), ("pip_cert", pip_cert, "--pip-cert")):
        if path is None:
            continue
        try:
            source = Path(path).expanduser().resolve(strict=True)
            # Check readability before starting a build, without reading the contents.
            with source.open("rb"):
                pass
        except OSError as exc:
            raise KernelRuntimeError(
                f"Cannot read {flag} file: {path}", hint="Choose an existing, readable file.",
            ) from exc
        value = io.StringIO()
        csv.writer(value, lineterminator="\n").writerow(["type=file", f"id={name}", f"src={source}"])
        options += ["--secret", value.getvalue().rstrip("\n")]
    return options


def build_command(tag: str, context: Path, args: dict[str, str], *, options: list[str] | None = None) -> list[str]:
    """``docker build`` arguments (without ``docker``) for ``context`` and its build args."""
    cmd = ["build", "--tag", tag]
    for name, value in args.items():
        cmd += ["--build-arg", f"{name}={value}"]
    return cmd + (options or []) + [str(context)]


@contextmanager
def configured_build_options(
    *, base_image: str | None = None, pip_config: Path | None = None, pip_cert: Path | None = None,
    no_cache: bool = False, no_host_config: bool = False, say: Callable[[str], None] | None = None,
    dry_run: bool = False,
) -> Iterator[list[str]]:
    """Keep discovered credentials in a temporary secret outside the build context.

    Explicit pip config disables discovery. A dry run uses a placeholder instead of
    creating a secret whose path would be invalid when the printed command is used.
    """
    settings = kernel_packages.PackageSettings()
    if pip_config is None and not no_host_config:
        settings = kernel_packages.discover()
    options = build_options(
        base_image=base_image, pip_config=pip_config, pip_cert=pip_cert or settings.cert, no_cache=no_cache,
    )
    if not settings.options:
        if settings.cert is not None and say:
            say("Using the host package CA bundle for this build.")
        yield options
        return
    if say:
        say("Using discovered host package settings for this build (credentials are not displayed).")
    if dry_run:
        if say:
            say("<discovered-pip-config> is temporary; rerun this script without --dry-run to build.")
        yield [*options, "--secret", "type=file,id=pip_config,src=<discovered-pip-config>"]
        return
    with tempfile.TemporaryDirectory(prefix="hailer-build-secrets-") as secret_dir:
        path = Path(secret_dir) / "pip.ini"
        parser = configparser.RawConfigParser()
        # pip's command section beats [global], even in an earlier base-image
        # config file. Preserve the host's effective values at install scope.
        parser["install"] = settings.options
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                parser.write(stream)
        except OSError:
            raise KernelRuntimeError("Cannot prepare the package configuration build secret.") from None
        yield [*options, *build_options(pip_config=path)]


def build(
    tag: str, runner: DockerRunner, *, base_image: str | None = None,
    pip_config: Path | None = None, pip_cert: Path | None = None, no_cache: bool = False,
    no_host_config: bool = False, say: Callable[[str], None] | None = None,
) -> int:
    """Build the image as ``tag`` on this machine, docker's output in this terminal; the exit code."""
    with configured_build_options(
        base_image=base_image, pip_config=pip_config, pip_cert=pip_cert, no_cache=no_cache,
        no_host_config=no_host_config, say=say,
    ) as options:
        with tempfile.TemporaryDirectory(prefix="hailer-kernel-") as tmp:
            context = Path(tmp)
            args = prepare_context(context)
            return runner.stream(build_command(tag, context, args, options=options)).returncode


def pull(image: str, runner: DockerRunner) -> subprocess.CompletedProcess[str]:
    """Download ``image``, docker's progress in this terminal. The result's ``returncode`` is
    docker's exit code and its ``stderr`` docker's last error lines (why a pull failed)."""
    return runner.stream(["pull", image], keep_errors=True)


def image_contract(image: str, runner: DockerRunner) -> str | None:
    """The kernel contract ``image`` was built for (its :data:`CONTRACT_LABEL`; ``""`` without the
    label); ``None`` when the image is not on this machine."""
    result = runner.run(["image", "inspect", "--format", "{{json .Config.Labels}}", image], timeout=_IMAGE_TIMEOUT_SEC)
    if result.returncode != 0:
        return None
    try:
        labels = json.loads((result.stdout or "").strip() or "null")
    except ValueError:
        return ""
    if not isinstance(labels, dict):
        return ""
    return str(labels.get(CONTRACT_LABEL) or "")


__all__ = [
    "CONTRACT_LABEL",
    "IMAGE_INIT",
    "IMAGE_MODULES",
    "IMAGE_PACKAGES",
    "MARIMO_VERSION",
    "build",
    "build_args",
    "build_command",
    "build_options",
    "configured_build_options",
    "context_files",
    "contract_fingerprint",
    "contract_tag",
    "default_image",
    "image_contract",
    "package_dir",
    "prepare_context",
    "pull",
]
