"""Notebook lifecycle shared by the CLI and the agent's tools.

Which notebook is *active*, which notebooks exist in the notebooks folder, how a user's
reference ("q2 churn", "q2_churn.py", "notebooks/q2_churn.py") maps to a file, and how a new
notebook is created from a template. Standard library only; this module never imports
marimo, typer or rich.

The active notebook is persisted in ``<workspace>/.hailer/notebook.json`` next to
``session.json``. Two writers touch that file: the CLI (between turns, for slash commands)
and the agent's tools (during a turn), so there is only ever one writer at a time. Readers
re-read it before every use and never cache it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Literal

from hailer.errors import NotebookExistsError, NotebookNotFoundError, NotebookPathError
from hailer.models import HailerConfig

STATE_DIRNAME = ".hailer"
STATE_FILENAME = "notebook.json"
RECENT_LIMIT = 10
FALLBACK_MARIMO_VERSION = "0.24.2"
TemplateKind = Literal["starter", "empty"]
TEMPLATE_KINDS: tuple[str, ...] = ("starter", "empty")

_SCAN_BYTES = 1024 * 1024  # marimo scans up to 1 MB for the notebook markers
_SLUG_RE = re.compile(r"[^a-z0-9]+")
SLUG_MAX_LEN = 64
#: Windows device names; ``con.py`` would be unusable (or write to the console) on older Windows.
_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
)
_VERSION_PLACEHOLDER = "__HAILER_VERSION__"
_TITLE_PLACEHOLDER = "__HAILER_TITLE__"

# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #

EMPTY_TEMPLATE = '''import marimo

__generated_with = "__HAILER_VERSION__"
app = marimo.App()


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
'''
"""marimo 0.24.2's own empty notebook: one empty cell."""

STARTER_TEMPLATE = '''import marimo

__generated_with = "__HAILER_VERSION__"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import duckdb
    from pathlib import Path

    return Path, duckdb, mo, pl


@app.cell
def _():
    # Hailer's period utilities (YY-MM filename convention, schema-tolerant loading).
    from hailer.periods import (
        describe_periods,
        duckdb_periods_view,
        load_periods,
        scan_period_files,
        scan_periods,
    )

    return describe_periods, duckdb_periods_view, load_periods, scan_period_files, scan_periods


@app.cell
def _(Path, mo):
    # Workspace layout: the workspace is the first folder above this notebook that holds
    # pyproject.toml or hailer.toml (falls back to the notebook folder's parent); data lives
    # in <workspace>/data/.
    _notebook_dir = mo.notebook_dir()
    _start = (_notebook_dir if _notebook_dir is not None else Path.cwd()).resolve()
    WORKSPACE = next(
        (
            _candidate
            for _candidate in (_start, *_start.parents)
            if (_candidate / "pyproject.toml").is_file() or (_candidate / "hailer.toml").is_file()
        ),
        _start.parent,
    )
    DATA_DIR = WORKSPACE / "data"
    return DATA_DIR, WORKSPACE


@app.cell
def _(DATA_DIR, scan_period_files):
    period_files = scan_period_files(DATA_DIR)
    return (period_files,)


@app.cell
def _(DATA_DIR, WORKSPACE, describe_periods, mo, period_files, pl):
    # Welcome / status cell. Hailer adds analysis cells below this one.
    _header = mo.md(
        f"""
    # __HAILER_TITLE__

    **Workspace:** `{WORKSPACE}`
    **Data:** `{DATA_DIR}`

    Chat in the terminal (`uv run hailer`); results, tables and charts appear here.
    """
    )
    if period_files:
        _table = mo.ui.table(
            pl.DataFrame(
                {
                    "file": [f.path.name for f in period_files],
                    "dataset": [f.stem for f in period_files],
                    "period": [f.period.label for f in period_files],
                    "size_kb": [round(f.path.stat().st_size / 1024, 1) for f in period_files],
                }
            ),
            selection=None,
            label="Period files",
        )
        _summary = mo.md("```text\\n" + describe_periods(period_files) + "\\n```")
        welcome = mo.vstack([_header, _table, _summary])
    else:
        welcome = mo.vstack(
            [
                _header,
                mo.callout(
                    mo.md(
                        "No period files found. Add files named like `25-01 pra101.parquet` to the data folder, "
                        "or run `uv run python scripts/make_sample_data.py` for a synthetic example."
                    ),
                    kind="info",
                ),
            ]
        )
    welcome
    return (welcome,)


if __name__ == "__main__":
    app.run()
'''
"""The starter notebook: the same cells as notebooks/analysis.py (imports, hailer.periods
helpers, WORKSPACE / DATA_DIR, period_files, welcome cell) with the notebook title as heading."""


@dataclass(frozen=True)
class NotebookInfo:
    """A marimo notebook found in the notebooks folder."""

    path: Path  # absolute
    name: str  # filename without .py
    modified: float  # st_mtime
    size: int  # bytes


# --------------------------------------------------------------------------- #
# Active-notebook state
# --------------------------------------------------------------------------- #


def state_path(workspace: Path) -> Path:
    """``<workspace>/.hailer/notebook.json``."""
    return Path(workspace) / STATE_DIRNAME / STATE_FILENAME


def _read_state(workspace: Path) -> dict:
    try:
        raw = json.loads(state_path(workspace).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except (ValueError, OSError):
        return False
    return True


def _from_state_entry(config: HailerConfig, value: object) -> Path | None:
    """A stored path that still points at an existing notebook inside the notebooks folder."""
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value.strip())
    if not candidate.is_absolute():
        candidate = Path(config.workspace) / candidate
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if not _inside(resolved, config.notebooks_root) or not resolved.is_file():
        return None
    return resolved


def notebook_display_name(config: HailerConfig, path: Path) -> str:
    """The path as shown to people: relative to the workspace with forward slashes."""
    try:
        resolved = Path(path).resolve()
    except OSError:
        resolved = Path(path)
    try:
        return resolved.relative_to(Path(config.workspace).resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_active_notebook(config: HailerConfig) -> Path:
    """The notebook the chat is working in: the state file's ``active`` entry, else ``config.notebook``.

    Only the state file is consulted, never the environment: ``HAILER_NOTEBOOK`` stays set for
    the whole session, so an env-wins rule would make the tools ignore every switch. The CLI
    persists such an override at startup with :func:`save_active_notebook` instead, so the CLI
    and the tools read the same answer. A missing or
    corrupt file, or an entry that no longer names an existing notebook inside the notebooks
    folder, falls back to the configured notebook.
    """
    active = _from_state_entry(config, _read_state(config.workspace).get("active"))
    return active if active is not None else config.notebook


def load_recent(config: HailerConfig) -> list[Path]:
    """Recently active notebooks, most recent first; only files that still exist."""
    out: list[Path] = []
    for value in _read_state(config.workspace).get("recent") or []:
        resolved = _from_state_entry(config, value)
        if resolved is not None and resolved not in out:
            out.append(resolved)
    return out


def save_active_notebook(config: HailerConfig, notebook: Path) -> None:
    """Persist ``notebook`` as the active one (atomic write) and push it onto the recent list."""
    path = state_path(config.workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = notebook_display_name(config, notebook)
    previous = [r for r in (_read_state(config.workspace).get("recent") or []) if isinstance(r, str) and r != key]
    payload = {"active": key, "recent": [key, *previous][:RECENT_LIMIT]}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Finding notebooks
# --------------------------------------------------------------------------- #


def is_marimo_notebook(path: Path) -> bool:
    """marimo's own rule: a ``.py`` file whose first megabyte has ``import marimo`` and ``marimo.App``."""
    file = Path(path)
    if file.suffix.lower() != ".py":
        return False
    try:
        with file.open("rb") as handle:
            head = handle.read(_SCAN_BYTES)
    except OSError:
        return False
    return b"import marimo" in head and b"marimo.App" in head


def list_notebooks(config: HailerConfig) -> list[NotebookInfo]:
    """Every marimo notebook under the notebooks folder, sorted by relative path.

    Folders whose name starts with ``.`` or ``__`` (``__marimo__``, ``__pycache__``) are skipped.
    """
    try:
        root = Path(config.notebooks_root).resolve()
    except OSError:
        return []
    if not root.is_dir():
        return []
    found: list[tuple[str, NotebookInfo]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith((".", "__")))
        for filename in filenames:
            file = Path(dirpath) / filename
            if not is_marimo_notebook(file):
                continue
            try:
                stat = file.stat()
            except OSError:
                continue
            info = NotebookInfo(path=file, name=file.stem, modified=stat.st_mtime, size=stat.st_size)
            found.append((file.relative_to(root).as_posix().lower(), info))
    found.sort(key=lambda item: item[0])
    return [info for _key, info in found]


def slugify(name: str) -> str:
    """``"Q2 Churn (draft).py"`` -> ``"q2_churn_draft"``.

    Lower-case ASCII letters, digits and underscores only. A leading digit or a Windows reserved
    device name (``con``, ``nul``, ``com1`` ...) gets the prefix ``nb_``; the result is at most
    ``SLUG_MAX_LEN`` characters. ``ValueError`` when the name has no ASCII letter or digit.
    """
    text = str(name).strip()
    if text.lower().endswith(".py"):
        text = text[:-3]
    slug = _SLUG_RE.sub("_", text.lower()).strip("_")
    if not slug:
        raise ValueError(f"notebook name {name!r} needs at least one ASCII letter or digit")
    if not slug[0].isalpha() or slug in _RESERVED_NAMES:
        slug = "nb_" + slug
    return slug[:SLUG_MAX_LEN].rstrip("_")


def _available_hint(config: HailerConfig) -> str:
    names = [info.name for info in list_notebooks(config)]
    where = notebook_display_name(config, config.notebooks_root)
    if not names:
        return f"There are no notebooks in {where} yet; create one first."
    shown = ", ".join(names[:10])
    more = f" (and {len(names) - 10} more)" if len(names) > 10 else ""
    return f"Notebooks in {where}: {shown}{more}."


def _looks_like_path(text: str, given: Path) -> bool:
    return given.is_absolute() or "/" in text or "\\" in text or text.startswith("..")


def _existing_file(candidate: Path) -> Path | None:
    """``candidate`` resolved when it is an existing file; ``None`` otherwise (never raises)."""
    try:
        resolved = candidate.resolve()
        return resolved if resolved.is_file() else None
    except (OSError, ValueError, RuntimeError):
        return None


def _checked_notebook(config: HailerConfig, resolved: Path) -> Path:
    if not is_marimo_notebook(resolved):
        raise NotebookPathError(
            f"{notebook_display_name(config, resolved)} is not a marimo notebook.",
            hint="A notebook is a .py file that imports marimo and defines marimo.App.",
        )
    return resolved


def resolve_notebook(config: HailerConfig, ref: str) -> Path:
    """Map a user's reference to an existing notebook inside the notebooks folder.

    Accepted forms: a bare name (``q2 churn``, ``q2_churn``), a filename (``q2_churn.py``), a
    notebooks-folder-relative path (``sub/x.py``, ``sub/x``, with either separator), a
    workspace-relative path (``notebooks/q2_churn.py``) or an absolute path. Candidates inside the
    notebooks folder are tried first (with and without ``.py``), then workspace-relative ones, then
    ``<slug>.py`` under the folder, then a case-insensitive unique match on the notebook name; a
    file outside the folder can therefore never shadow one inside it. Raises ``NotebookPathError``
    when the reference points outside the folder or at a file that is not a marimo notebook, and
    ``NotebookNotFoundError`` when nothing matches.
    """
    text = (ref or "").strip().strip("\"'").strip()
    if not text:
        raise NotebookNotFoundError("No notebook was given.", hint=_available_hint(config))
    # Accept either separator: the agent may send Windows-style paths, and on POSIX a backslash
    # would otherwise be a literal character in the file name, so "sub\x.py" would never resolve.
    text = text.replace("\\", "/")
    root = Path(config.notebooks_root).resolve()
    root_shown = notebook_display_name(config, root)
    given = Path(text).expanduser()
    if not given.is_absolute() and ".." in given.parts:
        # An explicit escape ("../x.py") is refused even when a notebook of that name exists inside
        # the folder; silently mapping it to the inside twin would hide the user's intent.
        raise NotebookPathError(
            f"{text} points outside the notebooks folder.",
            hint=f"Hailer only opens notebooks under {root_shown}; give the name or the path inside it.",
        )
    with_py = text if text.lower().endswith(".py") else f"{text}.py"
    try:
        slug: str | None = slugify(text)
    except ValueError:
        slug = None
    if given.is_absolute():
        candidates: list[Path] = [given]
        path_like: list[Path] = [given]
    else:
        workspace = Path(config.workspace)
        candidates = [root / text, root / with_py, workspace / text, workspace / with_py]
        if slug is not None:
            candidates.append(root / f"{slug}.py")
        # A relative path is read relative to the notebooks folder: "../x.py" escapes it and is
        # refused; "sub/missing" stays inside and is simply not found.
        path_like = [root / text]

    outside: Path | None = None
    for candidate in candidates:
        resolved = _existing_file(candidate)
        if resolved is None:
            continue
        if _inside(resolved, root):
            return _checked_notebook(config, resolved)
        if outside is None:
            outside = resolved

    wanted = {text.lower(), text.lower().removesuffix(".py")}
    if slug is not None:
        wanted.add(slug)
    matches = [info for info in list_notebooks(config) if info.name.lower() in wanted]
    if len(matches) == 1:
        return matches[0].path
    if len(matches) > 1:
        shown = ", ".join(notebook_display_name(config, m.path) for m in matches)
        raise NotebookNotFoundError(
            f"Several notebooks match {text!r}: {shown}.",
            hint="Give the path instead, for example the first one listed.",
        )
    if outside is not None:
        raise NotebookPathError(
            f"{outside.as_posix()} is outside the notebooks folder.",
            hint=f"Hailer only opens notebooks under {root_shown}. Move the file there or set [hailer].notebooks_dir.",
        )
    if _looks_like_path(text, given):
        for candidate in path_like:
            try:
                resolved = candidate.resolve()
            except (OSError, ValueError, RuntimeError):
                continue
            if not _inside(resolved, root):
                raise NotebookPathError(
                    f"{resolved.as_posix()} is outside the notebooks folder.",
                    hint=f"Hailer only opens notebooks under {root_shown}.",
                )
    raise NotebookNotFoundError(f"No notebook named {text!r} in {root_shown}.", hint=_available_hint(config))


# --------------------------------------------------------------------------- #
# Creating notebooks
# --------------------------------------------------------------------------- #


def marimo_version() -> str:
    """The installed marimo version for ``__generated_with`` (from package metadata, no import)."""
    try:
        return metadata.version("marimo")
    except metadata.PackageNotFoundError:
        return FALLBACK_MARIMO_VERSION


def _safe_title(title: str) -> str:
    """A title that is safe inside the template's f-string markdown block."""
    text = " ".join(str(title).split())
    text = text.replace("\\", "/").replace('"', "'")
    return text.replace("{", "{{").replace("}", "}}") or "Notebook"


def render_template(kind: TemplateKind, *, title: str, version: str | None = None) -> str:
    """The notebook source for ``kind`` with the title and marimo version filled in."""
    if kind not in TEMPLATE_KINDS:
        raise ValueError(f"kind must be one of {', '.join(TEMPLATE_KINDS)}, not {kind!r}")
    text = STARTER_TEMPLATE if kind == "starter" else EMPTY_TEMPLATE
    return text.replace(_VERSION_PLACEHOLDER, version or marimo_version()).replace(_TITLE_PLACEHOLDER, _safe_title(title))


def create_notebook(config: HailerConfig, name: str, *, kind: TemplateKind = "starter") -> Path:
    """Write ``<notebooks_root>/<slug>.py`` from the template and return its absolute path.

    Raises ``NotebookExistsError`` when the file exists and ``NotebookPathError`` when the name
    yields no usable filename. Does not touch the active-notebook state.
    """
    title = " ".join(str(name).split())
    try:
        slug = slugify(title)
    except ValueError as err:
        raise NotebookPathError(
            f"{name!r} is not a usable notebook name.",
            hint="The name needs at least one ASCII letter or digit, for example 'q2 churn' (saved as q2_churn.py).",
        ) from err
    root = Path(config.notebooks_root)
    root_shown = notebook_display_name(config, root)
    target = root / f"{slug}.py"
    exists_error = NotebookExistsError(
        f"A notebook named {slug}.py already exists in {root_shown}.",
        hint="Pick another name, or open the existing notebook instead.",
    )
    if target.exists():
        raise exists_error
    try:
        root.mkdir(parents=True, exist_ok=True)
        _write_new_file(target, render_template(kind, title=title))
    except FileExistsError as err:  # appeared between the check and the write: never overwrite
        raise exists_error from err
    except OSError as err:
        raise NotebookPathError(
            f"Could not create {slug}.py in {root_shown}: {err.strerror or err}.",
            hint="Try a shorter or simpler name, and check that the notebooks folder is writable.",
        ) from err
    return target.resolve()


def _write_new_file(path: Path, text: str) -> None:
    """Create ``path`` with ``text``; ``FileExistsError`` when it already exists (no overwrite)."""
    with open(path, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


__all__ = [
    "EMPTY_TEMPLATE",
    "RECENT_LIMIT",
    "SLUG_MAX_LEN",
    "STARTER_TEMPLATE",
    "STATE_FILENAME",
    "TEMPLATE_KINDS",
    "NotebookInfo",
    "create_notebook",
    "is_marimo_notebook",
    "list_notebooks",
    "load_active_notebook",
    "load_recent",
    "marimo_version",
    "notebook_display_name",
    "render_template",
    "resolve_notebook",
    "save_active_notebook",
    "slugify",
    "state_path",
]
