"""Workspace and account commands: ``init``, ``login``, ``logout``, ``doctor`` and ``status``."""

from __future__ import annotations

import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from hailer import notebooks
from hailer.cli import common
from hailer.cli.common import (
    _config_or_exit,
    _kernel_choice,
    _labelled,
    _opts,
    _print_error,
    _startup_panel,
    kernel_checks,
    local_checks,
)
from hailer.errors import CredentialsError, HailerError
from hailer.models import (
    KERNEL_RUNTIME_DOCKER,
    VALID_KERNEL_RUNTIMES,
    HailerConfig,
    ProviderConfig,
)


def _workspace_kernels(config: HailerConfig) -> list[str]:
    """This workspace's running kernels, one line each (``hailer status``)."""
    from hailer.kernel_docker import workspace_kernels

    return workspace_kernels(config, runner=common._docker_runner())


def _docker_on_path() -> bool:
    return shutil.which("docker") is not None


def status(ctx: typer.Context) -> None:
    """Show configuration, credential source, this workspace's running kernels and loaded context."""
    opts = _opts(ctx)
    console = common.console_factory()
    config = _config_or_exit(console, opts)
    _startup_panel(console, config)
    try:
        provider = config.provider
        _value, source = common._resolve_key(provider)
        if source == "missing":
            console.print(f"Credentials: {provider.env_key or 'API key'} missing (run: uvx hailer login {provider.id})", markup=False)
        else:
            console.print(f"Credentials: {provider.env_key or 'API key'} from {source}", markup=False)
    except KeyError:
        console.print(f"Credentials: provider {config.model.provider!r} is not declared", markup=False)
    kernels = _workspace_kernels(config)
    if not kernels:
        console.print("Kernels:     none running for this workspace (each uvx hailer session starts its own)", markup=False)
    for index, line in enumerate(kernels):
        console.print(("Kernels:     " if index == 0 else "             ") + line, markup=False)
    try:
        bundle = common._load_context(config)
    except HailerError as context_err:
        _print_error(console, context_err, verbose=opts.verbose)
        return
    console.print(
        f"Context:     {len(bundle.context_files)} file(s), {len(bundle.skills)} skill(s), {len(bundle.prompts)} prompt(s)",
        markup=False,
    )
    for warning in bundle.warnings:
        console.print(_labelled("warn", "yellow", f"  {warning}"))
    if config.config_path:
        console.print(f"Config:      {config.config_path}", markup=False)
    else:
        console.print("Config:      defaults (no hailer.toml found; run: uvx hailer init)", markup=False)


def doctor(ctx: typer.Context) -> None:
    """Check the configuration, credentials and the kernel runtime (Docker, the image, the folders)
    and print a table with fixes. No kernel is started or probed."""
    opts = _opts(ctx)
    console = common.console_factory()
    config = _config_or_exit(console, opts)
    checks = local_checks(config) + kernel_checks(config)
    table = Table(title="Hailer doctor", show_lines=False)
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Details")
    for check in checks:
        result = "OK" if check.ok else ("FAIL" if check.fatal else "WARN")
        details = check.summary
        if not check.ok and check.hint:
            details += "\n" + check.hint
        table.add_row(Text(check.name), Text(result), Text(details))
    console.print(table)
    if any(not c.ok and c.fatal for c in checks):
        raise typer.Exit(code=1)


def _provider_for(config: HailerConfig, provider_id: str) -> ProviderConfig:
    if provider_id in config.providers:
        return config.providers[provider_id]
    if provider_id == "openai":
        return ProviderConfig(id="openai", env_key="OPENAI_API_KEY")
    known = ", ".join(sorted({"openai", *config.providers}))
    raise CredentialsError(
        f"Unknown provider {provider_id!r}.",
        hint=f"Declared providers: {known}. Add a [model_providers.{provider_id}] table to hailer.toml first.",
    )


def login(
    ctx: typer.Context,
    provider: str = typer.Argument(..., help="Provider id from hailer.toml (or 'openai')."),
) -> None:
    """Store the provider's API key in the OS credential store (never in a file)."""
    opts = _opts(ctx)
    console = common.console_factory()
    config = _config_or_exit(console, opts)
    try:
        prov = _provider_for(config, provider)
        if not prov.env_key:
            raise CredentialsError(
                f"Provider {provider!r} has no env_key.",
                hint=f"Set model_providers.{provider}.env_key in hailer.toml to the environment variable that names the key.",
            )
        value = typer.prompt(f"API key for {provider} ({prov.env_key})", hide_input=True)
        if not value.strip():
            console.print("Nothing stored (empty key).", markup=False)
            raise typer.Exit(code=1)
        common._store_key(prov, value.strip())
    except HailerError as err:
        _print_error(console, err, verbose=opts.verbose)
        raise typer.Exit(code=1)
    console.print(f"Stored {prov.env_key} for provider {provider!r} in the OS credential store.", markup=False)


def logout(
    ctx: typer.Context,
    provider: str = typer.Argument(..., help="Provider id from hailer.toml (or 'openai')."),
) -> None:
    """Remove the provider's API key from the OS credential store."""
    opts = _opts(ctx)
    console = common.console_factory()
    config = _config_or_exit(console, opts)
    try:
        prov = _provider_for(config, provider)
        removed = common._delete_key(prov)
    except HailerError as err:
        _print_error(console, err, verbose=opts.verbose)
        raise typer.Exit(code=1)
    console.print("Removed." if removed else "No stored key found.", markup=False)


def _example_config_dir() -> Path | None:
    candidate = Path(__file__).resolve().parents[3] / ".config" / "hailer"
    return candidate if candidate.is_dir() else None


_PLACEHOLDERS: dict[str, str] = {
    "context/README.md": (
        "# Project context\n\nMarkdown files in this folder are sent to the model endpoint with every "
        "Hailer session. Put glossaries, column meanings and house rules here. Never put secrets here.\n"
    ),
    "skills/README.md": (
        "# Skills\n\nEach sub-folder holds a SKILL.md with `name:` and `description:` frontmatter. "
        "Hailer lists them to the agent and loads one on demand or via /skill <name>.\n"
    ),
    "prompts/README.md": "# Prompts\n\nEach `<name>.md` can be sent with /prompt <name> [args]; `{{args}}` is replaced.\n",
}
INIT_NOTEBOOK_TITLE = "Hailer workspace"


def _init_notebook_and_data(console: Console, workspace: Path, config_path: Path, *, verbose: bool) -> HailerConfig | None:
    """Create the configured notebook (starter template) and the notebooks and data folders when missing.

    Never overwrites a file. Returns the loaded configuration, or ``None`` when hailer.toml does not load.
    """
    from hailer.config import load_config

    try:
        config = load_config(workspace=workspace, config_path=config_path)
    except HailerError as err:
        _print_error(console, err, verbose=verbose)
        console.print("Skipped the notebook and data folder; fix hailer.toml and run uvx hailer init again.", markup=False)
        return None
    shown = notebooks.notebook_display_name(config, config.notebook)
    created: list[str] = []
    try:
        if notebooks.ensure_notebook(config.notebook, title=INIT_NOTEBOOK_TITLE):
            created.append(f"{shown} (starter notebook)")
        for folder in (config.notebooks_root, config.data_dir):
            if not folder.is_dir():
                folder.mkdir(parents=True)
                created.append(notebooks.notebook_display_name(config, folder) + "/")
    except OSError as err:
        console.print(f"Could not create the notebook or data folder: {err}", style="red", markup=False)
        return config
    console.print("Created " + ", ".join(created) if created else f"{shown} already exists.", markup=False)
    return config


def init(
    ctx: typer.Context,
    force: bool = typer.Option(False, "--force", help="Overwrite an existing hailer.toml."),
    kernel: str | None = typer.Option(
        None,
        "--kernel",
        help='Write \\[kernel] runtime = "docker" (the default: an isolated container) or "unsafe-local" (notebook code runs as you) into hailer.toml.',
        show_default=False,
    ),
) -> None:
    """Set up a workspace: hailer.toml, .config/hailer, the starter notebook and the data folder.

    Only hailer.toml is ever overwritten (with --force); anything else that exists is kept, so running
    it again fills in whatever is missing.
    """
    opts = _opts(ctx)
    console = common.console_factory()
    choice = _kernel_choice(console, kernel)
    workspace = opts.workspace or Path.cwd()
    workspace = workspace.resolve()
    config_path = workspace / "hailer.toml"
    if config_path.exists() and not force:
        console.print(f"{config_path} already exists (use --force to overwrite).", markup=False)
    else:
        from hailer.config import write_default_config

        write_default_config(config_path, overwrite=force, kernel=choice)
        console.print(f'Wrote {config_path} ([kernel] runtime = "{choice or KERNEL_RUNTIME_DOCKER}")', markup=False)

    target = workspace / ".config" / "hailer"
    example = _example_config_dir()
    created: list[str] = []
    for sub in ("context", "skills", "prompts"):
        (target / sub).mkdir(parents=True, exist_ok=True)
    if example is not None and example != target:
        for src in example.rglob("*"):
            if src.is_dir():
                continue
            rel = src.relative_to(example)
            dst = target / rel
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            created.append(str(rel))
    else:
        for rel, text in _PLACEHOLDERS.items():
            dst = target / rel
            if not dst.exists():
                dst.write_text(text, encoding="utf-8")
                created.append(rel)
    if created:
        console.print(f"Created {target} with: " + ", ".join(created), markup=False)
    else:
        console.print(f"{target} already set up.", markup=False)

    config = _init_notebook_and_data(console, workspace, config_path, verbose=opts.verbose)
    runtime = config.kernel.runtime if config is not None else (choice or KERNEL_RUNTIME_DOCKER)
    kept = choice is not None and config is not None and runtime != choice
    if kept:
        console.print(
            f'hailer.toml was kept, so [kernel] runtime is still "{runtime}". To change it, set '
            f'runtime = "{choice}" under [kernel] in hailer.toml (add a [kernel] line if there is none), or run '
            f"uvx hailer init --kernel {choice} --force to rewrite the file.",
            markup=False,
        )
    provider = config.model.provider if config is not None else "<provider>"
    data = notebooks.notebook_display_name(config, config.data_dir) if config is not None else "data"
    console.print(
        "\nNext steps:\n"
        "  1. Edit hailer.toml (model, provider, notebook).\n"
        f"  2. Store the API key once:   uvx hailer login {provider}\n"
        f"  3. Put the files you want to analyse in {data}/ (CSV, Parquet, JSON or Excel, any name).\n"
        "  4. Start marimo and chat:     uvx hailer notebook\n"
        '     then ask, for example: "what is in my data?" or "chart revenue by month"',
        markup=False,
    )
    for line in _kernel_next_steps(runtime, kept=kept):
        console.print(line, markup=False)


def _kernel_next_steps(runtime: str, *, kept: bool = False) -> list[str]:
    """What ``init`` adds to the next steps about where notebook code runs: for docker, how to get
    Docker when it is not on this machine (or opt into unsafe-local); for unsafe-local, that it is
    not isolated (nothing more after the "hailer.toml was kept" advice, which says how to switch)."""
    from hailer.config import runtime_problem
    from hailer.kernel_docker import DOCKER_INSTALL_ADVICE

    if runtime not in VALID_KERNEL_RUNTIMES:  # a kept hailer.toml, e.g. the retired "local"
        return ["\n" + runtime_problem(runtime)]
    if runtime == KERNEL_RUNTIME_DOCKER:
        lines = [
            "\nNotebook code runs in a Docker container (no network; the data folder is read-only).",
            "The first uvx hailer notebook downloads the kernel image (uvx hailer kernel pull does it now); where "
            "it cannot be downloaded, uvx hailer kernel build builds it on this machine.",
        ]
        if not _docker_on_path():
            lines += [
                f"Docker was not found on this machine. {DOCKER_INSTALL_ADVICE}, start it, then run uvx hailer doctor.",
                "Or, only if you accept that notebook code then runs as you, with your files and network (not "
                'isolated): set runtime = "unsafe-local" under [kernel] in hailer.toml '
                "(or run uvx hailer init --kernel unsafe-local --force).",
            ]
        return lines
    if kept:
        return []
    return [
        "\nNotebook code runs on this machine as you, with your files and network (unsafe-local: not isolated). "
        'To isolate it, install and start Docker, then set runtime = "docker" under [kernel] in hailer.toml.'
    ]
