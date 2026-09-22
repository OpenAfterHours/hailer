"""``uvx hailer kernel``: the Docker kernel's image (``pull``, ``build``) and this workspace's
kernels (``stop``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from hailer.cli import common
from hailer.cli.common import _config_or_exit, _opts, _print_error
from hailer.errors import HailerError
from hailer.models import HailerConfig

kernel_app = typer.Typer(
    no_args_is_help=True,
    help="The Docker kernel: download or build its image; stop this workspace's kernels.",
)
def _for_kernel_command(err: HailerError) -> HailerError:
    """``err`` without the unsafe-local opt-in: ``hailer kernel ...`` manages Docker itself."""
    from hailer.kernel_docker import without_unsafe_local_option

    err.hint = without_unsafe_local_option(err.hint)
    return err


def _docker_runtime(config: HailerConfig) -> Any:
    from hailer.kernel_docker import DockerRuntime

    return DockerRuntime(config, runner=common._docker_runner())


@kernel_app.command("pull")
def kernel_pull(ctx: typer.Context) -> None:
    """Download the kernel image for this Hailer's kernel contract (or \\[kernel].image) ahead of the first start."""
    opts = _opts(ctx)
    console = common.console_factory()
    config = _config_or_exit(console, opts)
    runtime = _docker_runtime(config)
    try:
        contract = runtime.pull(say=lambda text: console.print(text, markup=False))
    except HailerError as err:
        _print_error(console, _for_kernel_command(err), verbose=opts.verbose)
        raise typer.Exit(code=1)
    console.print(f"{runtime.image} is ready (kernel contract {contract}).", markup=False)


@kernel_app.command("build")
def kernel_build(
    ctx: typer.Context,
    tag: str | None = typer.Option(
        None,
        "--tag",
        help="Name and tag for the image (default: \\[kernel].image, else ghcr.io/openafterhours/hailer-kernel:<kernel contract>).",
        show_default=False,
    ),
    base_image: str | None = typer.Option(
        None, "--base-image", help="Python base image, e.g. registry.company/python:3.13 (Python 3.12+ with venv).",
    ),
    pip_config: Path | None = typer.Option(
        None, "--pip-config", help="Pip configuration build secret; overrides automatic host pip/uv discovery.",
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True,
    ),
    pip_cert: Path | None = typer.Option(
        None, "--pip-cert", help="PEM CA bundle for pip during the build; mounted as a build secret.",
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True,
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Reinstall packages without cached build layers (use after changing mirror settings).",
    ),
    no_host_config: bool = typer.Option(
        False, "--no-host-config", help="Disable automatic discovery of host pip/uv package settings.",
    ),
) -> None:
    """Build the kernel image on this machine, for machines that cannot download it."""
    from hailer import kernel_image
    from hailer.errors import KernelRuntimeError

    opts = _opts(ctx)
    console = common.console_factory()
    config = _config_or_exit(console, opts)
    runtime = _docker_runtime(config)
    image = tag or config.kernel.effective_image
    try:
        runtime.engine_version()
        args = kernel_image.build_args()
        console.print(
            f"Building {image} for kernel contract {args['KERNEL_CONTRACT']} (marimo {args['MARIMO_VERSION']}, "
            f"Polars {args['POLARS_VERSION']}, DuckDB {args['DUCKDB_VERSION']}) ...",
            markup=False,
        )
        code = kernel_image.build(
            image, runtime.runner, base_image=base_image, pip_config=pip_config, pip_cert=pip_cert, no_cache=no_cache,
            no_host_config=no_host_config, say=lambda message: console.print(message, markup=False),
        )
        if code != 0:
            raise KernelRuntimeError(
                f"docker build failed (exit code {code}).",
                hint=(
                    "Docker's output is above. Check access to the base image and package index. "
                    "Corporate mirrors: use --base-image and --pip-config; for a private CA, add --pip-cert. "
                    "The build needs BuildKit and a Linux base with Python 3.12+, venv and ensurepip."
                ),
            )
    except HailerError as err:
        _print_error(console, _for_kernel_command(err), verbose=opts.verbose)
        raise typer.Exit(code=1)
    console.print(f"Built {image}.", markup=False)
    if image != config.kernel.effective_image:
        console.print(f'Hailer uses {config.kernel.effective_image}; set image = "{image}" under [kernel] in hailer.toml to use this one.', markup=False)


@kernel_app.command("stop")
def kernel_stop(ctx: typer.Context) -> None:
    """Stop every Docker kernel of this workspace, whichever session started it, and remove its containers and networks."""
    from hailer.kernel_docker import stop_workspace_kernels

    opts = _opts(ctx)
    console = common.console_factory()
    config = _config_or_exit(console, opts)
    try:
        report = stop_workspace_kernels(config, runner=common._docker_runner())
    except HailerError as err:
        _print_error(console, err, verbose=opts.verbose)
        raise typer.Exit(code=1)
    for line in report.done or ([] if report.failed else ["No kernel is running for this workspace."]):
        console.print(line, markup=False)
    for line in report.failed:
        console.print(line, style="red", markup=False)
    if report.failed:
        raise typer.Exit(code=1)
