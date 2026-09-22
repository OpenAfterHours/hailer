"""The workspace's ``.hailer`` folder: Hailer's own state, never user content.

It holds ``session.json``, ``notebook.json`` and ``threads.sqlite``, and, while a kernel runs, the
running session's kernel files (a local kernel's record and log, a docker kernel's token folder
while it starts). It changes on every turn. Everything that writes there creates the folder through :func:`ensure_state_dir`, which
also puts a ``.gitignore`` containing ``*`` inside it: git then ignores the folder in any repository,
without an entry in the user's own ``.gitignore`` (which Hailer never touches). Standard library only.
"""

from __future__ import annotations

from pathlib import Path

STATE_DIRNAME = ".hailer"
GITIGNORE_NAME = ".gitignore"
GITIGNORE_TEXT = "# Created by Hailer: this folder is local state, not something to commit.\n*\n"


def state_dir(workspace: Path) -> Path:
    """``<workspace>/.hailer`` (the path only; see :func:`ensure_state_dir`)."""
    return Path(workspace) / STATE_DIRNAME


def ensure_state_dir(workspace: Path) -> Path:
    """Create ``<workspace>/.hailer`` when missing, with its ``.gitignore``, and return it.

    An existing ``.gitignore`` there is never overwritten, so a user who edits it keeps the edit. The
    ``.gitignore`` is a convenience: failing to write it never stops Hailer (the folder itself raises
    as ``mkdir`` does).
    """
    folder = state_dir(workspace)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with open(folder / GITIGNORE_NAME, "x", encoding="utf-8") as handle:  # "x": never replace one
            handle.write(GITIGNORE_TEXT)
    except OSError:  # FileExistsError included: it is already there
        pass
    return folder
