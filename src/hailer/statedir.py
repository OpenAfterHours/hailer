"""The workspace's state folder: Hailer's own state, never user content.

It is ``.hailer/``, or ``.hailer/state/`` in the project layout (a workspace whose config is
``.hailer/hailer.toml``: there ``.hailer/`` holds the project's notebooks, skills and context, which
are committed). It holds ``session.json``, ``notebook.json``, ``threads.sqlite``,
``notebook-backups/``, the ``sandbox-wrote-notebooks`` mark, the chat's ``input_history`` (and its
``.lock``) and, while a kernel runs, the
running session's kernel files (a local kernel's record and log, a docker kernel's token folder
while it starts). It changes on every turn. Everything that writes there creates the folder through :func:`ensure_state_dir`, which
also puts a ``.gitignore`` containing ``*`` inside it: git then ignores the folder in any repository,
without an entry in the user's own ``.gitignore`` (which Hailer never touches). Standard library only.
"""

from __future__ import annotations

from pathlib import Path

STATE_DIRNAME = ".hailer"
#: The state folder inside ``.hailer/`` in the project layout.
PROJECT_STATE_DIRNAME = "state"
#: The config file that marks the project layout.
PROJECT_CONFIG_NAME = "hailer.toml"
#: This machine's settings on top of the project's config (data folder paths, a personal
#: provider): ``.hailer/local.toml`` in both layouts, never committed.
LOCAL_CONFIG_NAME = "local.toml"
GITIGNORE_NAME = ".gitignore"
GITIGNORE_TEXT = "# Created by Hailer: this folder is local state, not something to commit.\n*\n"
#: ``.hailer/.gitignore`` in the project layout: what in the project folder stays on this machine.
PROJECT_GITIGNORE_TEXT = (
    "# Created by Hailer: these stay on this machine; everything else in .hailer/ is the project's.\n"
    f"{PROJECT_STATE_DIRNAME}/\n{LOCAL_CONFIG_NAME}\ndata/\n"
    "# State a standalone workspace kept here before it moved to the project layout:\n"
    "/threads.sqlite*\n/session.json\n/notebook.json\n/notebook-backups/\n/owner-*.lock\n/kernel-token-*/\n"
    "/marimo-*.log\n/sandbox-wrote-notebooks\n/input_history*\n"
)


def project_dir(workspace: Path) -> Path:
    """``<workspace>/.hailer``: the state folder, or in the project layout the project folder."""
    return Path(workspace) / STATE_DIRNAME


def is_project_layout(workspace: Path) -> bool:
    """Whether the workspace uses the project layout (its config is ``.hailer/hailer.toml``)."""
    return (project_dir(workspace) / PROJECT_CONFIG_NAME).is_file()


def state_dir(workspace: Path) -> Path:
    """``<workspace>/.hailer``, or ``<workspace>/.hailer/state`` in the project layout (the path
    only; see :func:`ensure_state_dir`)."""
    folder = project_dir(workspace)
    return folder / PROJECT_STATE_DIRNAME if is_project_layout(workspace) else folder


def local_config_path(workspace: Path) -> Path:
    """``<workspace>/.hailer/local.toml``: this machine's settings (the path only)."""
    return project_dir(workspace) / LOCAL_CONFIG_NAME


def ensure_project_gitignore(workspace: Path) -> None:
    """In the project layout, create ``.hailer/.gitignore`` (state, ``local.toml``, ``data/``) when
    missing. An existing one is never changed, except the ``*`` one Hailer wrote while ``.hailer/`` was a
    standalone workspace's state folder (unedited): it would hide the project's files from git. Never raises."""
    if not is_project_layout(workspace):
        return
    path = project_dir(workspace) / GITIGNORE_NAME
    try:
        if path.read_text(encoding="utf-8") == GITIGNORE_TEXT:
            path.write_text(PROJECT_GITIGNORE_TEXT, encoding="utf-8")
        return
    except OSError:  # FileNotFoundError included: create it
        pass
    try:
        with open(path, "x", encoding="utf-8") as handle:
            handle.write(PROJECT_GITIGNORE_TEXT)
    except OSError:
        pass


#: ``input_history`` in the state folder: what the user typed in the chat, recalled with Up in later
#: sessions (:mod:`hailer.input_history`; off with ``[hailer].input_history = false``).
INPUT_HISTORY_NAME = "input_history"


def input_history_path(workspace: Path) -> Path:
    return state_dir(workspace) / INPUT_HISTORY_NAME


def ensure_state_dir(workspace: Path) -> Path:
    """Create the state folder (:func:`state_dir`) when missing, with its ``.gitignore``, and return it.

    An existing ``.gitignore`` there is never overwritten, so a user who edits it keeps the edit. The
    ``.gitignore`` is a convenience: failing to write it never stops Hailer (the folder itself raises
    as ``mkdir`` does).
    """
    return ensure_ignored_dir(state_dir(workspace))


def ensure_ignored_dir(folder: Path) -> Path:
    """Create ``folder`` when missing, with a ``.gitignore`` containing ``*`` (never replacing one)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with open(folder / GITIGNORE_NAME, "x", encoding="utf-8") as handle:  # "x": never replace one
            handle.write(GITIGNORE_TEXT)
    except OSError:  # FileExistsError included: it is already there
        pass
    return folder


#: ``.hailer/sandbox-wrote-notebooks``: a sandbox (the docker kernel) copied notebooks into the
#: workspace since the user last agreed to run them on this machine (see :mod:`hailer.kernel`).
SANDBOX_MARK_NAME = "sandbox-wrote-notebooks"


def mark_sandbox_wrote(workspace: Path) -> None:
    """Note that a sandbox wrote notebooks here (never raises)."""
    try:
        path = ensure_state_dir(workspace) / SANDBOX_MARK_NAME
        if not path.exists():
            path.write_text("A sandbox copied notebooks into this workspace; uvx hailer asks before running them unisolated.\n", encoding="utf-8")
    except OSError:
        pass


def sandbox_wrote_notebooks(workspace: Path) -> bool:
    return (state_dir(workspace) / SANDBOX_MARK_NAME).exists()


def clear_sandbox_mark(workspace: Path) -> None:
    try:
        (state_dir(workspace) / SANDBOX_MARK_NAME).unlink()
    except OSError:
        pass
