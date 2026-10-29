"""``uvx hailer data``: the data folders notebooks see (``list``, ``add``, ``remove``).

A data folder is a folder anywhere on this machine; the docker kernel mounts each one read-only at
``/work/folders/<name>``. ``add`` and ``remove`` edit this machine's ``.hailer/local.toml``
(:mod:`hailer.datafolders`), never the project's config; a running chat sees the change at its next
start.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from hailer.cli import common
from hailer.cli.common import _config_or_exit, _opts, _print_error
from hailer.errors import HailerError
from hailer.models import KERNEL_RUNTIME_DOCKER, HailerConfig

data_app = typer.Typer(
    invoke_without_command=True,
    help="The data folders notebooks see: list them, add a folder from anywhere on this machine, remove one.",
)


def folder_lines(config: HailerConfig) -> list[str]:
    """The data folder and each ``[data_folders]`` entry, one line each, with where notebooks find it."""
    from hailer.config import available_data_folders
    from hailer.kernel import KERNEL_DATA_DIR, KERNEL_FOLDERS_DIR

    docker = config.kernel.runtime == KERNEL_RUNTIME_DOCKER
    shown = f" (in the kernel: {KERNEL_DATA_DIR})" if docker else ""
    lines = [f"data      {config.data_dir}{shown}  DATA_DIR"]
    available = available_data_folders(config)
    for folder in config.data_folders:
        if folder.path is None:
            where = "no path on this machine (uvx hailer data add " + folder.name + " <folder>)"
        elif folder not in available:
            where = f"{folder.path} (does not exist: not mounted)"
        else:
            where = str(folder.path) + (f" (in the kernel: {KERNEL_FOLDERS_DIR / folder.name})" if docker else "")
        about = f"  {folder.description}" if folder.description else ""
        lines.append(f"{folder.name:<9} {where}  DATA_FOLDERS[{folder.name!r}]{about}")
    return lines


def _show(console: Console, config: HailerConfig) -> None:
    for line in folder_lines(config):
        console.print(line, markup=False)
    if not config.data_folders:
        console.print("\nNo more data folders. Add one with: uvx hailer data add <name> <folder>", markup=False)


@data_app.callback()
def data(ctx: typer.Context) -> None:
    """The data folders notebooks see (without a command: list them)."""
    if ctx.invoked_subcommand is None:
        data_list(ctx)


@data_app.command("list")
def data_list(ctx: typer.Context) -> None:
    """List the data folder and the data folders, with where notebooks find them."""
    console = common.console_factory()
    _show(console, _config_or_exit(console, _opts(ctx)))


@data_app.command("add")
def data_add(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Its name in notebooks: DATA_FOLDERS['<name>'] (lower-case letters, digits, - and _)."),
    folder: Path = typer.Argument(..., help="The folder, anywhere on this machine.", show_default=False),
    description: str | None = typer.Option(None, "--description", "-d", help="What is in it, for the agent.", show_default=False),
) -> None:
    """Add a folder from anywhere on this machine (or change where one is), read-only in the kernel.

    It is saved in .hailer/local.toml, this machine's settings, so the project's config and other
    people's machines are not changed. Start Hailer again to see it.
    """
    from hailer.datafolders import add_data_folder, reload
    from hailer.kernel import KERNEL_FOLDERS_DIR

    opts = _opts(ctx)
    console = common.console_factory()
    config = _config_or_exit(console, opts)
    try:
        target = add_data_folder(config, name, folder, description=description)
        config = reload(config)
    except HailerError as err:
        _print_error(console, err, verbose=opts.verbose)
        raise typer.Exit(code=1)
    where = f" (read-only in the kernel at {KERNEL_FOLDERS_DIR / name.strip()})" if config.kernel.runtime == KERNEL_RUNTIME_DOCKER else ""
    console.print(f"Data folder {name.strip()!r} is {target}{where}; saved in .hailer/local.toml.", markup=False)
    console.print(
        f"Notebooks find it as DATA_FOLDERS[{name.strip()!r}] from the next start (a running chat: /exit, then start it again).",
        markup=False,
    )


@data_app.command("remove")
def data_remove(ctx: typer.Context, name: str = typer.Argument(..., help="The data folder's name.")) -> None:
    """Remove a data folder from this machine's settings (.hailer/local.toml)."""
    from hailer.datafolders import reload, remove_data_folder

    opts = _opts(ctx)
    console = common.console_factory()
    config = _config_or_exit(console, opts)
    try:
        removed = remove_data_folder(config, name)
        config = reload(config)
    except HailerError as err:
        _print_error(console, err, verbose=opts.verbose)
        raise typer.Exit(code=1)
    still = next((f for f in config.data_folders if f.name == name.strip()), None)
    if removed:
        console.print(f"Removed data folder {name.strip()!r} from .hailer/local.toml.", markup=False)
    else:
        console.print(f"There is no data folder {name.strip()!r} in .hailer/local.toml.", markup=False)
    if still is not None:
        shown = config.config_path.name if config.config_path is not None else "the config"
        console.print(f"It is still declared in {shown}" + (f" (at {still.path})." if still.path else " (without a path on this machine)."), markup=False)
