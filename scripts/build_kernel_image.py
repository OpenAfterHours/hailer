"""Build the Hailer kernel image with ``docker buildx``: for developers, CI and the release workflow.

Usage (from the repository root)::

    uv run python -m scripts.build_kernel_image --load                 # into this machine's Docker
    uv run python -m scripts.build_kernel_image --push                 # both platforms, to the registry
    uv run python -m scripts.build_kernel_image --dry-run --push       # print the command only
    uv run python -m scripts.build_kernel_image --load --tag hailer-kernel:dev
    uv run python -m scripts.build_kernel_image --push -- --progress plain  # arguments after -- go to buildx

The build context is the one ``uvx hailer kernel build`` uses (``hailer.kernel_image.prepare_context``):
the packaged Dockerfile and the installed ``hailer`` package (this checkout, under ``uv run``) in a
temporary folder, with one ``--build-arg`` per pinned version (marimo, Polars, fastexcel, DuckDB and Hailer's own,
which becomes the image's version label). Users build their own image with ``uvx hailer kernel build``;
this script adds what a release needs: several platforms and ``--push``.

- ``--tag`` (repeatable) defaults to the image this Hailer runs,
  ``ghcr.io/openafterhours/hailer-kernel:<version>``.
- ``--platform`` defaults to ``linux/amd64,linux/arm64``; with ``--load`` it defaults to this machine's
  platform (Docker's classic image store cannot load a multi-platform image).
- ``--push`` uploads (``docker login`` first); ``--load`` puts the image into the local engine; with
  neither the result stays in buildx's build cache (a check that the image builds).
- ``--context DIR`` prepares the context in ``DIR`` and keeps it; the default is a temporary folder.
- ``--dry-run`` prints the command and runs nothing (the context shows as ``<context>`` unless
  ``--context`` is given, which is then prepared so the printed command can be run by hand).
- ``--base-image``, ``--pip-config`` and ``--pip-cert`` work like ``hailer kernel build``:
  select a corporate Python base and mount pip configuration/CA files as build secrets.
- Host pip/uv package settings are discovered unless ``--pip-config`` or ``--no-host-config``
  is supplied. Dry runs show a placeholder for generated secrets; run this script without
  ``--dry-run`` to create those temporary secrets and execute the build.

Needs Docker with the buildx plugin (Docker Desktop and GitHub's Ubuntu runners have it). Building
``linux/arm64`` on an amd64 machine needs QEMU (Docker Desktop has it; CI runs setup-qemu-action).
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from hailer import kernel_image
from hailer.errors import HailerError

DEFAULT_PLATFORMS = ("linux/amd64", "linux/arm64")
CONTEXT_PLACEHOLDER = "<context>"


def say(message: str) -> None:
    print(f"build_kernel_image: {message}", flush=True)


def warn(message: str) -> None:
    print(f"build_kernel_image: {message}", file=sys.stderr, flush=True)


def buildx_command(
    context: Path | str,
    build_args: Mapping[str, str],
    *,
    tags: Sequence[str],
    platforms: Sequence[str] | None = None,
    output: str | None = None,
    extra: Sequence[str] = (),
    docker: str = "docker",
) -> list[str]:
    """The ``docker buildx build`` argument list (``docker`` first).

    ``platforms`` None or empty leaves ``--platform`` out (the builder's own platform). ``output`` is
    ``"push"``, ``"load"`` or None (build only). ``extra`` goes in just before the context.
    """
    if output not in (None, "push", "load"):
        raise ValueError(f"output must be 'push', 'load' or None, not {output!r}")
    if not tags:
        raise ValueError("at least one tag is needed")
    cmd = [docker, "buildx", "build"]
    if platforms:
        cmd += ["--platform", ",".join(platforms)]
    for tag in tags:
        cmd += ["--tag", tag]
    for name, value in build_args.items():
        cmd += ["--build-arg", f"{name}={value}"]
    if output is not None:
        cmd.append(f"--{output}")
    return [*cmd, *extra, str(context)]


def _platforms(value: str) -> list[str]:
    platforms = [item.strip() for item in value.split(",") if item.strip()]
    if not platforms:
        raise argparse.ArgumentTypeError("give at least one platform, e.g. linux/amd64")
    return platforms


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    argv = list(argv)
    extra: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1 :]

    parser = argparse.ArgumentParser(
        prog="python -m scripts.build_kernel_image",
        description="Build the Hailer kernel image with docker buildx (the context `uvx hailer kernel build` uses).",
        epilog="Arguments after -- are passed to docker buildx build.",
    )
    parser.add_argument(
        "--tag",
        dest="tags",
        action="append",
        metavar="IMAGE",
        help=f"image name and tag, repeatable (default: {kernel_image.default_image()})",
    )
    parser.add_argument(
        "--platform",
        dest="platforms",
        type=_platforms,
        metavar="LIST",
        help=f"comma-separated platforms (default: {','.join(DEFAULT_PLATFORMS)}; with --load: this machine's)",
    )
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--push", dest="output", action="store_const", const="push", help="push the image to its registry")
    where.add_argument("--load", dest="output", action="store_const", const="load", help="load the image into the local Docker engine")
    parser.add_argument("--context", type=Path, metavar="DIR", help="prepare the build context here and keep it")
    parser.add_argument("--dry-run", action="store_true", help="print the command and run nothing")
    parser.add_argument("--base-image", metavar="IMAGE", help="Python 3.12+ base image with venv and ensurepip")
    parser.add_argument("--pip-config", type=Path, metavar="FILE", help="pip build secret; overrides host pip/uv discovery")
    parser.add_argument("--pip-cert", type=Path, metavar="FILE", help="pip PEM CA bundle mounted as a build secret")
    parser.add_argument("--no-cache", action="store_true", help="reinstall packages without cached build layers")
    parser.add_argument("--no-host-config", action="store_true", help="disable host pip/uv package configuration discovery")
    args = parser.parse_args(argv)
    args.extra = extra
    if not args.tags:
        args.tags = [kernel_image.default_image()]
    if args.platforms is None:
        args.platforms = [] if args.output == "load" else list(DEFAULT_PLATFORMS)
    return args


def _format(cmd: Sequence[str]) -> str:
    return shlex.join(cmd)


def _prepare(context: Path) -> dict[str, str]:
    if (context / "hailer").exists():
        raise FileExistsError(f"{context} already holds a hailer folder; pick an empty or new folder for --context")
    return kernel_image.prepare_context(context)


def _build(args: argparse.Namespace, context: Path, build_args: Mapping[str, str], docker: str) -> int:
    cmd = buildx_command(
        context, build_args, tags=args.tags, platforms=args.platforms, output=args.output, extra=args.extra, docker=docker
    )
    say("running " + _format(["docker", *cmd[1:]]))
    if args.output is None:
        say("neither --push nor --load: the result stays in the build cache (a check that the image builds)")
    return subprocess.run(cmd, check=False, stdin=subprocess.DEVNULL).returncode


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        with kernel_image.configured_build_options(
            base_image=args.base_image, pip_config=args.pip_config, pip_cert=args.pip_cert, no_cache=args.no_cache,
            no_host_config=args.no_host_config, say=say, dry_run=args.dry_run,
        ) as options:
            args.extra = [*options, *args.extra]
            return _run(args)
    except HailerError as exc:
        warn(f"aborted: {exc}. {exc.hint}")
        return 1


def _run(args: argparse.Namespace) -> int:
    if args.dry_run:
        if args.context is not None:
            try:
                build_args = _prepare(args.context)
            except OSError as exc:
                warn(f"aborted: {exc}")
                return 1
            context: Path | str = args.context
            say(f"prepared the build context in {args.context}")
        else:
            build_args, context = kernel_image.build_args(), CONTEXT_PLACEHOLDER
        cmd = buildx_command(context, build_args, tags=args.tags, platforms=args.platforms, output=args.output, extra=args.extra)
        print(_format(cmd), flush=True)
        return 0

    docker = shutil.which("docker")
    if docker is None:
        warn("aborted: docker is not on PATH (install Docker Desktop or Docker Engine with the buildx plugin)")
        return 1

    try:
        if args.context is not None:
            build_args = _prepare(args.context)
            return _build(args, args.context, build_args, docker)
        with tempfile.TemporaryDirectory(prefix="hailer-kernel-") as tmp:
            build_args = kernel_image.prepare_context(Path(tmp))
            return _build(args, Path(tmp), build_args, docker)
    except OSError as exc:
        warn(f"aborted: {exc}")
        return 1
    except KeyboardInterrupt:
        warn("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
