"""The kernel image the docker runtime runs: which one, and how to build or pull it.

``src/hailer/docker/Dockerfile`` ships inside the package, so ``uvx hailer kernel build`` works
from any installed Hailer (a checkout, uvx or pip) without a wheel or PyPI:
:func:`prepare_context` copies that Dockerfile and the installed ``hailer`` package into a build
context and returns the build args, taken from the versions this Hailer runs with. The release
workflow reuses it for the published multi-arch image, with the same arguments::

    args = prepare_context(Path("ctx"))  # {"MARIMO_VERSION": ..., "HAILER_VERSION": ...}
    # docker buildx build --platform linux/amd64,linux/arm64 --tag <image>
    #     --build-arg MARIMO_VERSION=... (one per entry of args) ctx

Every ``docker`` call goes through a :class:`~hailer.kernel_docker.DockerRunner`, so tests can
check the argument lists without Docker.
"""

from __future__ import annotations

import csv
import importlib.metadata
import importlib.resources
import io
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from hailer import __version__
from hailer.errors import KernelRuntimeError
from hailer.models import KERNEL_IMAGE_REPOSITORY

if TYPE_CHECKING:  # pragma: no cover - annotations only; hailer.kernel_docker imports this module
    from hailer.kernel_docker import DockerRunner

#: The image label that must equal ``hailer.__version__`` (a different marimo would break code mode).
VERSION_LABEL = "org.opencontainers.image.version"
#: Distributions whose installed versions the image pins (build arg -> distribution name).
PINNED_DISTRIBUTIONS = {"MARIMO_VERSION": "marimo", "POLARS_VERSION": "polars", "DUCKDB_VERSION": "duckdb"}
_SKIPPED = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
_IMAGE_TIMEOUT_SEC = 30.0


def default_image() -> str:
    """``ghcr.io/openafterhours/hailer-kernel:<hailer version>``."""
    return f"{KERNEL_IMAGE_REPOSITORY}:{__version__}"


def package_dir() -> Path:
    """The folder of the installed ``hailer`` package (what the image gets as source)."""
    return Path(__file__).resolve().parent


def build_args() -> dict[str, str]:
    """The Dockerfile's build args: the installed marimo, Polars and DuckDB, and this Hailer."""
    args = {arg: importlib.metadata.version(dist) for arg, dist in PINNED_DISTRIBUTIONS.items()}
    args["HAILER_VERSION"] = __version__
    return args


def prepare_context(dest: Path) -> dict[str, str]:
    """Fill ``dest`` with a build context and return the build args for it.

    ``dest`` gets ``Dockerfile`` and ``hailer/`` (the installed package without ``__pycache__``);
    it is created when missing and must not already hold a ``hailer`` folder.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    dockerfile = importlib.resources.files("hailer").joinpath("docker/Dockerfile")
    # LF endings whatever the checkout did (git's autocrlf on Windows): its RUN lines continue with "\".
    (dest / "Dockerfile").write_bytes(dockerfile.read_bytes().replace(b"\r\n", b"\n"))
    shutil.copytree(package_dir(), dest / "hailer", ignore=_SKIPPED)
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


def build(
    tag: str, runner: DockerRunner, *, base_image: str | None = None,
    pip_config: Path | None = None, pip_cert: Path | None = None, no_cache: bool = False,
) -> int:
    """Build the image as ``tag`` on this machine, docker's output in this terminal; the exit code."""
    options = build_options(base_image=base_image, pip_config=pip_config, pip_cert=pip_cert, no_cache=no_cache)
    with tempfile.TemporaryDirectory(prefix="hailer-kernel-") as tmp:
        context = Path(tmp)
        args = prepare_context(context)
        return runner.stream(build_command(tag, context, args, options=options)).returncode


def pull(image: str, runner: DockerRunner) -> subprocess.CompletedProcess[str]:
    """Download ``image``, docker's progress in this terminal. The result's ``returncode`` is
    docker's exit code and its ``stderr`` docker's last error lines (why a pull failed)."""
    return runner.stream(["pull", image], keep_errors=True)


def image_version(image: str, runner: DockerRunner) -> str | None:
    """The Hailer version ``image`` was built for (``""`` without the label); ``None`` when the image
    is not on this machine."""
    result = runner.run(["image", "inspect", "--format", "{{json .Config.Labels}}", image], timeout=_IMAGE_TIMEOUT_SEC)
    if result.returncode != 0:
        return None
    try:
        labels = json.loads((result.stdout or "").strip() or "null")
    except ValueError:
        return ""
    if not isinstance(labels, dict):
        return ""
    return str(labels.get(VERSION_LABEL) or "")


__all__ = [
    "PINNED_DISTRIBUTIONS",
    "VERSION_LABEL",
    "build",
    "build_args",
    "build_command",
    "build_options",
    "default_image",
    "image_version",
    "package_dir",
    "prepare_context",
    "pull",
]
