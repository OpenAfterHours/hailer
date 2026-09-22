"""Notebook names, the active notebook, and templates, shared by the CLI and the agent's tools.

A notebook is known by its *name*, a POSIX path relative to the notebooks folder (``"sales.py"``,
``"q3/review.py"``): :func:`check_notebook_name` says what a name may be, :func:`resolve_notebook`
maps a user's reference ("q2 churn", "q2_churn.py", "notebooks/q2_churn.py") to one of the names
the sandbox listed, and :func:`new_notebook` renders a template for a new one. The files
themselves are the sandbox's (:mod:`hailer.sandbox`); nothing here reads the notebooks folder
except workspace setup (:func:`ensure_notebook`). Standard library only; this module never
imports marimo, typer or rich.

The active notebook is persisted in ``<workspace>/.hailer/notebook.json`` next to
``session.json``, as names. Two writers touch that file: the CLI (between turns, for slash
commands) and the agent's tools (during a turn), so there is only ever one writer at a time.
Readers re-read it before every use and never cache it.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Literal

from hailer.errors import NotebookNotFoundError, NotebookPathError
from hailer.marimo_client import name_in_folder, notebook_file_key
from hailer.models import HailerConfig
from hailer.statedir import ensure_state_dir, state_dir

STATE_FILENAME = "notebook.json"
#: ``notebook.json`` holds names from this version on; a file without it holds host paths.
STATE_VERSION = 2
RECENT_LIMIT = 10
FALLBACK_MARIMO_VERSION = "0.24.2"
TemplateKind = Literal["starter", "empty"]
TEMPLATE_KINDS: tuple[str, ...] = ("starter", "empty")

_SLUG_RE = re.compile(r"[^a-z0-9]+")
SLUG_MAX_LEN = 64
#: Windows device names, with the superscript-digit ports and the console buffers Windows also
#: reserves; ``con.py`` would be unusable (or write to the console) on older Windows.
RESERVED_NAMES = frozenset(
    {
        "con", "prn", "aux", "nul", "conin$", "conout$",
        *(f"{port}{digit}" for port in ("com", "lpt") for digit in (*"123456789", "¹", "²", "³")),
    }
)  # fmt: skip
#: The longest part (folder or file name) of a notebook name; most file systems refuse longer.
NAME_PART_MAX_LEN = 255
#: Characters no part of a notebook name may hold: control characters and what Windows refuses
#: (``:`` would also name an NTFS data stream).
_BAD_NAME_CHARS = re.compile(r'[\x00-\x1f<>:"|?*]')
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
    # Hailer's data helpers: list_data_files finds every data file; the others handle monthly
    # files named "YY-MM <dataset>.parquet" (schema-tolerant loading with a period column).
    from hailer.periods import (
        describe_periods,
        duckdb_periods_view,
        list_data_files,
        load_periods,
        scan_period_files,
        scan_periods,
    )

    return (
        describe_periods,
        duckdb_periods_view,
        list_data_files,
        load_periods,
        scan_period_files,
        scan_periods,
    )


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
def _(DATA_DIR, list_data_files, scan_period_files):
    data_files = list_data_files(DATA_DIR)  # CSV, Parquet, JSON, Excel ... with any name
    period_files = scan_period_files(DATA_DIR)  # the monthly "YY-MM <dataset>.parquet" ones
    return data_files, period_files


@app.cell
def _(DATA_DIR, WORKSPACE, data_files, describe_periods, mo, period_files, pl):
    # Welcome / status cell. Hailer adds analysis cells below this one.
    _header = mo.md(
        f"""
    # __HAILER_TITLE__

    Chat with your data in the terminal (`uvx hailer`): ask a question in plain English and Hailer
    loads the files, explores them and puts the tables, charts and summaries here.

    **Workspace:** `{WORKSPACE}`

    **Data:** `{DATA_DIR}`
    """
    )
    _parts = [_header]
    if data_files:
        _periods = {pf.path.name: pf.period.label for pf in period_files}
        _columns = {
            "file": [f.name for f in data_files],
            "type": [f.suffix.lstrip(".").lower() for f in data_files],
            "size_kb": [round(f.stat().st_size / 1024, 1) for f in data_files],
        }
        if _periods:
            _columns["period"] = [_periods.get(f.name, "") for f in data_files]
        _parts.append(mo.ui.table(pl.DataFrame(_columns), selection=None, label="Data files"))
        _parts.append(
            mo.md(
                "Try asking: *what is in these files?* · *show the ten largest values* · "
                "*chart the totals by month*"
            )
        )
        if period_files:
            _parts.append(mo.md("**Monthly files**\\n\\n```text\\n" + describe_periods(period_files) + "\\n```"))
    else:
        _parts.append(
            mo.callout(
                mo.md(
                    "No data files yet. Put the files you want to analyse (CSV, Parquet, JSON or Excel, any "
                    "name) in the data folder above, then ask in the terminal, for example "
                    "*load sales.csv and chart revenue by month*."
                ),
                kind="info",
            )
        )
    welcome = mo.vstack(_parts)
    welcome
    return (welcome,)


if __name__ == "__main__":
    app.run()
'''
"""The starter notebook: the same cells as notebooks/analysis.py (imports, hailer.periods
helpers, WORKSPACE / DATA_DIR, data_files and period_files, welcome cell) with the notebook title
as heading."""


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #


def check_notebook_name(name: str) -> str:
    """``name`` as a notebook name (either separator accepted, returned with ``/``), or
    ``NotebookPathError``.

    A name is a POSIX path relative to the notebooks folder that stays inside it: no absolute path
    or drive, no ``.`` or ``..`` or empty part, no part starting with ``.`` (``.git``) or ``__``
    (``__marimo__``), no character Windows refuses or reads specially (``:`` would name a data
    stream) and no invisible formatting character (a right-to-left override), no part ending in a
    dot or space or longer than :data:`NAME_PART_MAX_LEN`, no Windows device name (``con``, ``nul``,
    ``com1``, ``conin$`` ...) as a part's stem, and a ``.py`` file name.
    """
    text = str(name or "").replace("\\", "/")
    hint = "Give the notebook's name inside the notebooks folder, for example sales.py or q3/review.py."
    invisible = any(unicodedata.category(char) in ("Cc", "Cf") for char in text)
    if not text or text.startswith("/") or _BAD_NAME_CHARS.search(text) or invisible:
        raise NotebookPathError(f"{name!r} is not a notebook name inside the notebooks folder.", hint=hint)
    parts = text.split("/")
    for part in parts:
        stem = part.split(".", 1)[0].rstrip(" ").lower()
        if (
            part in ("", ".", "..")
            or part.startswith((".", "__"))
            or part != part.rstrip(". ")
            or len(part) > NAME_PART_MAX_LEN
            or stem in RESERVED_NAMES
        ):
            raise NotebookPathError(f"{name!r} is not a notebook name inside the notebooks folder.", hint=hint)
    if not parts[-1].lower().endswith(".py"):
        raise NotebookPathError(f"{name!r} is not a notebook file name.", hint="A notebook is a .py file. " + hint)
    return text


def _valid_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return check_notebook_name(value)
    except NotebookPathError:
        return None


def same_name(a: str, b: str) -> bool:
    """Whether two notebook names name the same file here (names ignore case on Windows)."""
    return a.casefold() == b.casefold() if os.name == "nt" else a == b


def notebook_name(config: HailerConfig, path: Path | str) -> str | None:
    """The name of the host file ``path`` (absolute, or relative to the workspace) in the notebooks
    folder, or ``None`` when it is outside it (links resolved) or not a valid name. Nothing is read."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(config.workspace) / candidate
    root = notebook_file_key(Path(config.notebooks_root))
    return _valid_name(name_in_folder(root, str(candidate.expanduser().absolute()), native=True) or "")


def default_notebook(config: HailerConfig) -> str:
    """The configured notebook (``[hailer].notebook``) as a name; its file name when it lies outside
    the notebooks folder (``hailer doctor`` reports that)."""
    return notebook_name(config, config.notebook) or _valid_name(config.notebook.name) or "analysis.py"


def notebook_display_name(config: HailerConfig, path: Path) -> str:
    """A host path as shown to people: relative to the workspace with forward slashes."""
    try:
        resolved = Path(path).resolve()
    except OSError:
        resolved = Path(path)
    try:
        return resolved.relative_to(Path(config.workspace).resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


# --------------------------------------------------------------------------- #
# Active-notebook state
# --------------------------------------------------------------------------- #


def state_path(workspace: Path) -> Path:
    """``<workspace>/.hailer/notebook.json``."""
    return state_dir(workspace) / STATE_FILENAME


def _read_state(config: HailerConfig) -> tuple[str | None, list[str]]:
    """(active, recent) from the state file, as names. A file without ``version`` is the old
    format (host paths, absolute or relative to the workspace): entries inside the notebooks
    folder become names, anything else is dropped. Invalid entries, entries of the wrong type and a
    file of an unknown (newer) version are ignored, and names that differ only in case on Windows
    are one."""
    try:
        raw = json.loads(state_path(config.workspace).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, []
    if not isinstance(raw, dict):
        return None, []
    version = raw.get("version")
    if version == STATE_VERSION:
        convert = _valid_name
    elif version is None:
        def convert(value: object) -> str | None:
            return notebook_name(config, value) if isinstance(value, str) and value.strip() else None
    else:
        return None, []

    stored = raw.get("recent")
    recent: list[str] = []
    for value in stored if isinstance(stored, list) else []:
        name = convert(value)
        if name is not None and not any(same_name(name, seen) for seen in recent):
            recent.append(name)
    return convert(raw.get("active")), recent


def load_active_notebook(config: HailerConfig) -> str:
    """The name of the notebook the chat is working in: the state file's ``active`` entry, else the
    configured notebook (:func:`default_notebook`).

    Only the state file is read (never the notebooks folder or the environment): the CLI and the
    agent's tools both switch notebooks through it, so every reader re-reads it and never caches
    it. Whether the notebook still exists is the sandbox's to say.
    """
    active, _recent = _read_state(config)
    return active if active is not None else default_notebook(config)


def load_recent(config: HailerConfig) -> list[str]:
    """Recently active notebooks, most recent first (names; they may no longer exist)."""
    return _read_state(config)[1]


def save_active_notebook(config: HailerConfig, name: str) -> None:
    """Persist ``name`` as the active notebook (atomic write) and push it onto the recent list.
    Writes the current format, so an old file is converted by the first save."""
    name = check_notebook_name(name)
    path = ensure_state_dir(config.workspace) / STATE_FILENAME
    previous = [r for r in load_recent(config) if not same_name(r, name)]
    payload = {"version": STATE_VERSION, "active": name, "recent": [name, *previous][:RECENT_LIMIT]}
    # A temporary file of its own: the CLI and a tool call finishing after Ctrl+C may save at once.
    fd, tmp = tempfile.mkstemp(prefix=f".{STATE_FILENAME}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------- #
# Finding notebooks
# --------------------------------------------------------------------------- #


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
    if not slug[0].isalpha() or slug in RESERVED_NAMES:
        slug = "nb_" + slug
    return slug[:SLUG_MAX_LEN].rstrip("_")


def _available_hint(names: Sequence[str]) -> str:
    if not names:
        return "There are no notebooks in the notebooks folder yet; create one first."
    shown = ", ".join(names[:10])
    more = f" (and {len(names) - 10} more)" if len(names) > 10 else ""
    return f"Notebooks: {shown}{more}."


def reference_folders(config: HailerConfig, *kernel_paths: str) -> tuple[str, ...]:
    """The prefixes a reference to a notebook may start with: ``kernel_paths`` (the notebooks folder
    as the kernel knows it), the host folder and its workspace-relative form (``notebooks``)."""
    host = Path(os.path.normpath(str(Path(config.notebooks_root).expanduser().absolute()))).as_posix()
    return (*kernel_paths, host, notebook_display_name(config, config.notebooks_root))


def _strip_folder(text: str, folders: Sequence[str]) -> str:
    for folder in folders:
        prefix = folder.replace("\\", "/").rstrip("/") + "/"
        if prefix != "/" and (text.startswith(prefix) or (os.name == "nt" and text.lower().startswith(prefix.lower()))):
            return text[len(prefix):]
    return text


def resolve_notebook(ref: str, names: Sequence[str], *, folders: Sequence[str] = ()) -> str:
    """Map a user's reference to one of ``names`` (the notebooks that exist, from the sandbox).

    Accepted forms: a bare name (``q2 churn``, ``q2_churn``), a file name (``q2_churn.py``), a path
    inside the notebooks folder (``sub/x.py``, ``sub/x``, either separator), and any of those after
    one of ``folders`` (:func:`reference_folders`: ``/work/notebooks/x.py``, ``notebooks/x.py``, the
    host path). Tried in order: the exact name (with and without ``.py``), the same ignoring case,
    ``<slug>.py``, then a unique match on the file name without ``.py`` in any folder. Raises
    ``NotebookPathError`` when the reference points outside the folder (an absolute path elsewhere,
    ``..``) and ``NotebookNotFoundError`` when nothing matches.
    """
    text = (ref or "").strip().strip("\"'").strip().replace("\\", "/")
    if not text:
        raise NotebookNotFoundError("No notebook was given.", hint=_available_hint(names))
    text = _strip_folder(text, folders)
    outside_hint = "Hailer only opens notebooks in the notebooks folder; give the name or the path inside it."
    if text.startswith(("/", "~")) or re.match(r"^[A-Za-z]:", text):
        raise NotebookPathError(f"{text} is outside the notebooks folder.", hint=outside_hint)
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if ".." in parts:
        # An explicit escape ("../x.py") is refused even when a notebook of that name exists inside
        # the folder; silently mapping it to the inside twin would hide the user's intent.
        raise NotebookPathError(f"{text} points outside the notebooks folder.", hint=outside_hint)
    text = "/".join(parts)
    with_py = text if text.lower().endswith(".py") else f"{text}.py"
    for candidate in (text, with_py):
        if candidate in names:
            return candidate
    folded = {name.lower(): name for name in names}
    for candidate in (text, with_py):
        if candidate.lower() in folded:
            return folded[candidate.lower()]
    try:
        slug: str | None = slugify(text.rsplit("/", 1)[-1])
    except ValueError:
        slug = None
    if slug is not None and f"{slug}.py" in names:
        return f"{slug}.py"
    wanted = {text.lower(), text.lower().removesuffix(".py")} | ({slug} if slug else set())
    matches = [name for name in names if name.rsplit("/", 1)[-1].lower().removesuffix(".py") in wanted]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise NotebookNotFoundError(
            f"Several notebooks match {text!r}: {', '.join(matches)}.",
            hint="Give the path instead, for example the first one listed.",
        )
    raise NotebookNotFoundError(f"No notebook named {text!r} in the notebooks folder.", hint=_available_hint(names))


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


def new_notebook(name: str, *, kind: TemplateKind = "starter") -> tuple[str, str]:
    """(``<slug>.py``, its source from the template) for a human name; the caller writes it into
    the sandbox (``write_notebook(..., replace=False)``). ``NotebookPathError`` when the name yields
    no usable file name."""
    title = " ".join(str(name).split())
    try:
        slug = slugify(title)
    except ValueError as err:
        raise NotebookPathError(
            f"{name!r} is not a usable notebook name.",
            hint="The name needs at least one ASCII letter or digit, for example 'q2 churn' (saved as q2_churn.py).",
        ) from err
    return f"{slug}.py", render_template(kind, title=title)


def ensure_notebook(path: Path, *, title: str, kind: TemplateKind = "starter") -> bool:
    """Write ``kind``'s template to ``path`` unless something is already there; True when it wrote.

    Workspace setup (``hailer init``, the first run) uses it for the configured notebook, on the
    host, before any kernel starts. Parent folders are created and an existing file is never
    touched. ``OSError`` when the file cannot be written.
    """
    target = Path(path)
    if target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_new_file(target, render_template(kind, title=title))
    except FileExistsError:  # appeared between the check and the write: leave it alone
        return False
    return True


def _write_new_file(path: Path, text: str) -> None:
    """Create ``path`` with ``text``; ``FileExistsError`` when it already exists (no overwrite)."""
    with open(path, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


__all__ = [
    "EMPTY_TEMPLATE",
    "RECENT_LIMIT",
    "RESERVED_NAMES",
    "SLUG_MAX_LEN",
    "STARTER_TEMPLATE",
    "STATE_FILENAME",
    "TEMPLATE_KINDS",
    "check_notebook_name",
    "default_notebook",
    "ensure_notebook",
    "load_active_notebook",
    "load_recent",
    "marimo_version",
    "new_notebook",
    "notebook_display_name",
    "notebook_name",
    "reference_folders",
    "same_name",
    "render_template",
    "resolve_notebook",
    "save_active_notebook",
    "slugify",
    "state_path",
]
