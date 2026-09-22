"""Hailer command line: the conversational control plane.

``uvx hailer``            chat with the agent: start this session's own marimo kernel, open the
                          notebook, chat, stop the kernel on exit
``uvx hailer notebook``   the same, with options (--port, --no-browser, --foreground, --kernel)
``uvx hailer status``     show configuration, credentials source and this workspace's running kernels
``uvx hailer doctor``     check the configuration, credentials, Docker, the kernel image and folders
``uvx hailer login``      store an API key in the OS credential store
``uvx hailer logout``     remove it
``uvx hailer init``       set up a workspace: hailer.toml, .config/hailer, the starter notebook, data/
``uvx hailer kernel``     the Docker kernel: ``pull`` or ``build`` its image, ``stop`` this workspace's kernels

The commands live in :mod:`hailer.cli.chat` (the chat and ``notebook``), :mod:`hailer.cli.kernel`
and :mod:`hailer.cli.setup` (``init``, ``login``, ``logout``, ``doctor``, ``status``); what they
share, including the collaborators tests replace, is in :mod:`hailer.cli.common`.
"""

from __future__ import annotations

from pathlib import Path

import typer

from hailer import __version__
from hailer.cli import chat, common, kernel, setup
from hailer.cli.common import CliOptions

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    invoke_without_command=True,
    help="Hailer: chat with your data. Explore, analyse and chart local data files in a live marimo notebook.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"hailer {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging and full tracebacks."),
    config: Path | None = typer.Option(None, "--config", help="Path to hailer.toml.", show_default=False),
    workspace: Path | None = typer.Option(None, "--workspace", help="Project directory (default: auto-detect).", show_default=False),
    new: bool = typer.Option(False, "--new", help="Start a new conversation instead of resuming."),
    plain: bool = typer.Option(False, "--plain", help="Use line-oriented chat without the persistent composer."),
    version: bool = typer.Option(False, "--version", callback=_version_callback, is_eager=True, help="Show the version and exit."),
) -> None:
    del version
    ctx.obj = CliOptions(verbose=verbose, config_path=config, workspace=workspace, new_thread=new, plain=plain)
    if ctx.invoked_subcommand is None:
        chat.run_chat(ctx.obj)


app.command()(chat.notebook)
app.command()(setup.status)
app.command()(setup.doctor)
app.add_typer(kernel.kernel_app, name="kernel")
app.command()(setup.login)
app.command()(setup.logout)
app.command()(setup.init)


def main() -> None:
    common._reconfigure_streams()
    app()
