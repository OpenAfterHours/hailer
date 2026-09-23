"""Copying notebooks between the workspace's notebooks folder and a sandbox that keeps its own.

The docker kernel's notebooks folder is the container's (an in-memory folder, gone when it stops),
so nothing notebook code writes there reaches this machine by itself. :class:`NotebookSync` copies
the workspace's notebooks in when the kernel starts and copies the ones the sandbox changed back:
after every agent turn, on a notebook switch, every :data:`SYNC_INTERVAL_SEC` seconds from a daemon
thread (edits made in the browser) and once more when the kernel stops, before its container is
removed. It only ever uses the sandbox's file operations (marimo's file API), so a kernel that is
not on this machine fits too.

Whatever runs in the kernel can replace the marimo server, so every answer may be forged. Both
directions copy only what :func:`sync_refusal` allows: marimo notebooks with plain names that no
tool on this machine runs or imports by name (pytest collects ``test_*.py`` and ``conftest.py``, a
``json.py`` shadows the module for scripts run in that folder, marimo runs a notebook's setup cell
when it is imported). Every copy has a deadline (:data:`SYNC_DEADLINE_SEC` at the start and the
stop, :data:`SYNC_CYCLE_SEC` per background pass, which also reads at most
:data:`SYNC_CYCLE_BYTES`), enforced per request and while a reply is read
(:func:`~hailer.marimo_client.request_deadline`), so a server cannot hold Hailer's exit or make it
rewrite the host's files without bound. A write on this machine is atomic (a temporary file
replaced into place), stays inside the notebooks folder and never goes through a link or junction;
when the file here changed since it was last copied, it is first saved under
``.hailer/notebook-backups``. Nothing is ever deleted here. Every failure is a warning line
(:meth:`~hailer.sandbox.MarimoSandbox.notify`) and leaves the file here as it was.

Standard library only (the kernel image's package names are read from :mod:`hailer.kernel_image`
when first needed).
"""

from __future__ import annotations

import fnmatch
import functools
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from hailer.errors import MarimoUnavailableError, NotebookExistsError, NotebookNotFoundError, NotebookPathError
from hailer.marimo_client import request_deadline
from hailer.notebooks import check_notebook_name
from hailer.sandbox import MAX_NOTEBOOK_BYTES, MAX_NOTEBOOK_DEPTH, MAX_NOTEBOOKS, SKIPPED_FOLDERS, notebook_digest
from hailer.statedir import ensure_state_dir, mark_sandbox_wrote

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from hailer.sandbox import MarimoSandbox

#: How often the daemon thread copies notebooks back while the kernel runs (browser edits).
SYNC_INTERVAL_SEC = 15.0
#: The whole copy in at the start, and the last copy back at the stop, end after this many seconds.
SYNC_DEADLINE_SEC = 30.0
#: One background pass ends after this many seconds or this many bytes read, whichever comes first
#: (the rest waits for the next pass).
SYNC_CYCLE_SEC = 10.0
SYNC_CYCLE_BYTES = 10 * MAX_NOTEBOOK_BYTES
#: How long a copy waits for another one under way (they share one lock) before it gives up.
SYNC_LOCK_SEC = SYNC_CYCLE_SEC + 5
#: The most a docker kernel's stop waits for the last copy (the background pass to end, the lock,
#: the copy itself) before it removes the containers anyway.
SYNC_STOP_SEC = 2 * SYNC_LOCK_SEC + SYNC_DEADLINE_SEC
#: Per session: at most this many notebooks new to this machine, and this many bytes written here;
#: past either, new names are no longer copied back (changes to known notebooks within the byte
#: budget still are), with one warning.
SESSION_NEW_FILES = 200
SESSION_BYTES = 100 * 1024 * 1024
#: ``.hailer/notebook-backups``: the copies of files here that a sandbox's version replaced.
BACKUP_FOLDER = "notebook-backups"
#: File names other tools run or import as code however they look (test runners, task runners,
#: packaging, Sphinx, Django, gunicorn, IPython and Jupyter configuration, Python's start-up hooks,
#: package markers). Patterns, compared with the name casefolded.
TOOLING_NAMES = (
    "conftest.py", "test_*.py", "*_test.py", "setup.py", "noxfile.py", "tasks.py", "fabfile.py", "dodo.py",
    "conf.py", "manage.py", "gunicorn.conf.py", "ipython_config.py", "jupyter_*_config.py",
    "sitecustomize.py", "usercustomize.py", "__init__.py", "__main__.py",
)  # fmt: skip
#: A Windows 8.3 short name (``PROGRA~1``): it can name another file than it seems to.
_SHORT_NAME = re.compile(r"~\d")
#: How many names a "not copied" warning shows before "and N more".
SHOWN_NAMES = 5
#: The prefix of the temporary file a write here goes through (a dot file: never copied itself).
TEMP_PREFIX = ".hailer-sync-"


#: Modules scripts next to notebooks commonly import (besides the standard library and the image's).
COMMON_MODULES = (
    "hailer", "pandas", "numpy", "scipy", "matplotlib", "seaborn", "sklearn", "statsmodels", "pyarrow", "requests",
    "httpx", "openpyxl", "xlsxwriter", "sqlalchemy", "pytest", "ipython", "jupyter", "yaml", "toml", "attr", "pydantic",
)  # fmt: skip


@functools.cache
def module_names() -> frozenset[str]:
    """Top-level module names (casefolded) Python may import instead of a notebook of that name:
    the standard library, the kernel image's packages and :data:`COMMON_MODULES`. A fixed list, so
    the rule is the same on every machine whatever Hailer's environment holds."""
    from hailer.kernel_image import IMAGE_PACKAGES  # lazy: only the name list is needed

    return frozenset(name.casefold() for name in (*sys.stdlib_module_names, *IMAGE_PACKAGES, *COMMON_MODULES))


def sync_refusal(name: str, text: str | None = None) -> str | None:
    """Why the notebook ``name`` (with ``text``, when given) is not copied between this machine and
    a sandbox, in either direction; ``None`` when it may be. The one allow-list:

    - a relative POSIX path (``/`` only) that :func:`~hailer.notebooks.check_notebook_name`
      accepts: no drive, ``..`` or empty part, no part starting with ``.`` or ``__`` (``.git``,
      ``__marimo__``, ``__pycache__``, ``__init__.py``), nothing Windows reads specially; no part
      that looks like an 8.3 short name (``~`` and a digit);
    - a ``.py`` file (that spelling) at most :data:`~hailer.sandbox.MAX_NOTEBOOK_DEPTH` folders deep;
    - not a name other tools run (:data:`TOOLING_NAMES`, any case);
    - no part (a folder, or the file without ``.py``) named like a module Python would import
      instead (:func:`module_names`: ``json``, ``pandas``, ``polars``, any case);
    - with ``text``: at most :data:`~hailer.sandbox.MAX_NOTEBOOK_BYTES` of UTF-8 text without NUL
      characters that looks like a marimo notebook (``import marimo`` and ``marimo.App(``).
    """
    raw = str(name or "")
    if "\\" in raw:
        return "not a notebook name"
    try:
        check_notebook_name(raw)
    except NotebookPathError:
        return "not a notebook name"
    parts = raw.split("/")
    base = parts[-1]
    if not base.endswith(".py"):
        return "not a .py file"
    if any(_SHORT_NAME.search(part) for part in parts):
        return "not a notebook name"
    if len(parts) - 1 > MAX_NOTEBOOK_DEPTH:
        return f"more than {MAX_NOTEBOOK_DEPTH} folders deep"
    folded = base.casefold()
    if any(fnmatch.fnmatchcase(folded, pattern) for pattern in TOOLING_NAMES):
        return "a file name other tools run"
    shadowed = next((part for part in [*parts[:-1], base.removesuffix(".py")] if part.casefold() in module_names()), None)
    if shadowed is not None:
        return f"the name of the Python module {shadowed}"
    if text is None:
        return None
    try:
        size = len(text.encode("utf-8"))
    except UnicodeEncodeError:
        return "not UTF-8 text"
    if size > MAX_NOTEBOOK_BYTES:
        return f"larger than {MAX_NOTEBOOK_BYTES // (1024 * 1024)} MiB"
    if "\x00" in text:
        return "not text"
    if "import marimo" not in text or "marimo.App(" not in text:
        return "not a marimo notebook"
    return None


def _is_link(path: Path) -> bool:
    try:
        return path.is_symlink() or path.is_junction()
    except OSError:
        return True


def _inside(path: Path, root: Path) -> bool:
    """``path`` resolved is ``root`` resolved or below it."""
    try:
        target, base = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(root))
    except OSError:
        return False
    return target == base or target.startswith(base.rstrip(os.sep) + os.sep)


def _write_atomically(target: Path, data: bytes) -> None:
    """Write ``data`` to ``target`` through a temporary file in its folder that replaces it, so a
    reader (or a crash) never sees half a file. A replaced file keeps its permissions; a new one is
    ``0644``. The temporary file is removed when anything fails."""
    fd, temp = tempfile.mkstemp(dir=target.parent, prefix=TEMP_PREFIX, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        if target.is_file():
            shutil.copymode(target, temp)
        else:
            os.chmod(temp, 0o644)
        os.replace(temp, target)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


def _printable(text: str) -> str:
    """``text`` without control or formatting characters (a name from the sandbox may carry escape
    sequences or a right-to-left override meant for the terminal)."""
    return "".join(char if char.isprintable() else "?" for char in text)


def _names(refused: Iterable[tuple[str, str]]) -> str:
    """``name`` for each refused file, with the reason for a module's name (the user would not guess it)."""
    shown = [
        _printable(f"{name} ({reason})" if reason.startswith("the name of the Python module") else name) for name, reason in sorted(refused)
    ]
    text = ", ".join(shown[:SHOWN_NAMES])
    return text + (f" and {len(shown) - SHOWN_NAMES} more" if len(shown) > SHOWN_NAMES else "")


class _OutOfTime(Exception):
    """A copy reached its deadline or byte budget; the rest waits (or, at the stop, is not copied)."""


class NotebookSync:
    """The notebook copies between ``folder`` (the workspace's notebooks folder) and ``sandbox``.

    It remembers, per notebook, the digest (:func:`~hailer.sandbox.notebook_digest`) of the text
    last copied in or out, which is how a change here is told from one in the sandbox, and the
    sandbox listing's time and size, so an unchanged notebook is not read again (without that, a
    background pass would read every notebook every time and, with its byte budget, could never
    reach the ones after the first :data:`SYNC_CYCLE_BYTES`). Only in memory: a new kernel starts
    from what is here. All copies are serialised; :meth:`start` runs the periodic one on a daemon
    thread and :meth:`close` stops it and copies once more.
    """

    def __init__(self, sandbox: MarimoSandbox, folder: Path, workspace: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.sandbox = sandbox
        self.folder = Path(folder)
        self.workspace = Path(workspace)
        self._clock = clock
        self._known: dict[str, str] = {}
        self._seen: dict[str, tuple[float, int]] = {}
        self._named: set[str] = set()  # refused names already in a "not copied" line
        self._new_files = 0  # notebooks this session created here
        self._written = 0  # bytes this session wrote here
        self._said: set[str] = set()  # warnings already shown (a dead kernel would repeat them every pass)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

    # -- in ------------------------------------------------------------------- #

    def host_notebooks(self) -> tuple[list[str], list[tuple[str, str]]]:
        """(names to copy in, (name, reason) of ``.py`` files here that are not copied), sorted.
        Folders starting with ``.`` or ``__``, :data:`~hailer.sandbox.SKIPPED_FOLDERS`, links and
        junctions are never looked into; files that are not ``.py`` are left alone without a word."""
        found: list[str] = []
        refused: list[tuple[str, str]] = []
        pending: list[tuple[Path, str, int]] = [(self.folder, "", 0)]
        while pending:
            current, prefix, depth = pending.pop(0)
            try:
                entries = sorted(os.scandir(current), key=lambda item: item.name)
            except OSError:
                continue
            for entry in entries:
                name = prefix + entry.name
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False) and not _is_link(path):
                    if depth < MAX_NOTEBOOK_DEPTH and not entry.name.startswith((".", "__")) and entry.name.lower() not in SKIPPED_FOLDERS:
                        pending.append((path, name + "/", depth + 1))
                    continue
                if not entry.name.lower().endswith(".py"):
                    continue
                reason = "a link" if _is_link(path) or not entry.is_file(follow_symlinks=False) else sync_refusal(name)
                if reason is None:
                    found.append(name)
                else:
                    refused.append((name, reason))
        return sorted(found), sorted(refused)

    def sync_in(self) -> list[str]:
        """Copy every allowed notebook here into the sandbox (created, or replaced) and remember
        what was copied, within :data:`SYNC_DEADLINE_SEC`; the names copied. At most
        :data:`~hailer.sandbox.MAX_NOTEBOOKS`. One warning line names what was not copied; never
        raises."""
        with self._lock:
            copied: list[str] = []
            names, refused = self.host_notebooks()
            try:
                with request_deadline(SYNC_DEADLINE_SEC):
                    for name in names:
                        if len(copied) >= MAX_NOTEBOOKS:
                            refused.append((name, "more than 500 notebooks"))
                            continue
                        text = self._read_here(name)
                        reason = "unreadable" if text is None else sync_refusal(name, text)
                        if reason is not None:
                            refused.append((name, reason))
                            continue
                        try:
                            info = self.sandbox.write_notebook(name, text)
                        except (NotebookPathError, NotebookExistsError):  # this file only (marimo refused its name)
                            refused.append((name, "refused by marimo"))
                            continue
                        self._known[name] = notebook_digest(text)
                        self._seen[name] = (info.modified, info.size)
                        copied.append(name)
            except Exception as err:  # noqa: BLE001 - the chat must still start
                self.sandbox.notify(
                    f"Warning: could not copy the notebooks in {self.folder} into the kernel ({err}); "
                    f"{len(copied)} copied. The files there are unchanged."
                )
            if refused:
                self._named.update(name for name, _reason in refused)
                self.sandbox.notify(f"Not copied into the kernel (only marimo notebooks with plain names are): {_names(refused)}.")
            return copied

    def _read_here(self, name: str) -> str | None:
        path = self.folder.joinpath(*name.split("/"))
        try:
            if _is_link(path) or path.stat().st_size > MAX_NOTEBOOK_BYTES:
                return None
            return path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    # -- out ------------------------------------------------------------------ #

    def sync_out(self, *, seconds: float | None = None, budget: int | None = None) -> list[str]:
        """Copy the notebooks the sandbox changed since they were last copied back here, within
        ``seconds`` (default :data:`SYNC_CYCLE_SEC`; every request and reply counts) and ``budget``
        bytes read (default :data:`SYNC_CYCLE_BYTES`); the names written.
        Never raises; failures, and a copy cut short, are warning lines and leave the files here as
        they were. Waits at most :data:`SYNC_LOCK_SEC` for a copy under way."""
        if not self._lock.acquire(timeout=SYNC_LOCK_SEC):
            self._say_once("Warning: an earlier copy of the notebooks back from the kernel did not finish; this one was skipped.")
            return []
        seconds = SYNC_CYCLE_SEC if seconds is None else seconds
        budget = SYNC_CYCLE_BYTES if budget is None else budget
        try:
            copied: list[str] = []
            try:
                with request_deadline(seconds):
                    self._sync_out(copied, time.monotonic() + seconds, budget)
            except (_OutOfTime, MarimoUnavailableError) as err:
                if isinstance(err, _OutOfTime) or "out of time" in str(err):
                    self._say_once(
                        f"Warning: copying the notebooks back from the kernel stopped after {seconds:g} s or "
                        f"{budget // (1024 * 1024)} MiB; the files in {self.folder} not copied yet are unchanged."
                    )
                else:
                    self._say_once(f"Warning: could not copy the notebooks back from the kernel ({err}). The files in {self.folder} are unchanged.")
            except Exception as err:  # noqa: BLE001 - a copy must never break the chat
                self._say_once(f"Warning: could not copy the notebooks back from the kernel ({err}). The files in {self.folder} are unchanged.")
            return copied
        finally:
            self._lock.release()

    def _sync_out(self, copied: list[str], end: float, budget: int) -> None:
        refused: list[tuple[str, str]] = []
        candidates = []
        folded: set[str] = set()
        for entry in self.sandbox.list_tree():
            leaf = entry.name.rsplit("/", 1)[-1]
            if entry.folder:
                if leaf.startswith(".") or (leaf.startswith("__") and leaf not in ("__marimo__", "__pycache__")):
                    refused.append((entry.name + "/", "a hidden folder"))
                continue
            reason = sync_refusal(entry.name)
            if reason is None and entry.name.casefold() in folded:
                reason = "differs from another name only in case"  # it would overwrite that one here
            if reason is None and len(candidates) >= MAX_NOTEBOOKS:
                reason = "more than 500 notebooks"
            if reason is not None:
                refused.append((entry.name, reason))
                continue
            folded.add(entry.name.casefold())
            candidates.append(entry)
        failures: list[str] = []
        read = 0
        try:
            for entry in candidates:
                stamp = (entry.modified, entry.size)
                if self._seen.get(entry.name) == stamp:
                    continue
                if time.monotonic() > end or read > budget:
                    raise _OutOfTime
                try:
                    text = self.sandbox.read_notebook(entry.name)
                except NotebookNotFoundError:
                    continue  # deleted meanwhile: nothing here is deleted
                except NotebookPathError:
                    text = None
                read += len(text) if text is not None else 0
                reason = "unreadable or too large" if text is None else sync_refusal(entry.name, text)
                if reason is not None:
                    refused.append((entry.name, reason))
                    self._seen[entry.name] = stamp
                    continue
                try:
                    if self._write_here(entry.name, text):
                        copied.append(entry.name)
                except (OSError, NotebookPathError) as err:
                    failures.append(f"{entry.name} ({err})")
                    continue
                self._seen[entry.name] = stamp
        finally:
            fresh = [(name, reason) for name, reason in refused if name not in self._named]
            if fresh:
                self._named.update(name for name, _reason in fresh)
                self.sandbox.notify(
                    "Not copied back from the kernel (only marimo notebooks with plain names are; everything else "
                    f"stays in the kernel and is gone when it stops): {_names(fresh)}."
                )
            if failures:
                self._say_once(f"Warning: could not copy back {', '.join(failures)}; the files in {self.folder} are unchanged.")

    def _target(self, name: str) -> Path:
        """The file here for ``name``: folders are created as needed, but never through a link or
        junction, and the resolved path stays inside the notebooks folder. ``NotebookPathError``."""
        self.folder.mkdir(parents=True, exist_ok=True)
        parts = name.split("/")
        current = self.folder
        for part in parts[:-1]:
            current = current / part
            if _is_link(current):
                raise NotebookPathError(f"{current} is a link or junction")
            if not current.exists():
                current.mkdir()
            if _is_link(current) or not current.is_dir():
                raise NotebookPathError(f"{current} is not a folder")
        target = current / parts[-1]
        if _is_link(target) or (os.path.lexists(target) and not target.is_file()):
            raise NotebookPathError(f"{target} is a link or not a file")
        if not _inside(target, self.folder):
            raise NotebookPathError(f"{target} leads outside {self.folder}")
        return target

    def _write_here(self, name: str, text: str) -> bool:
        """Write the sandbox's ``text`` of ``name`` here (False: it already is). A file here that
        changed since it was last copied is saved under ``.hailer/notebook-backups`` first; when
        that fails, nothing is written."""
        digest = notebook_digest(text)
        if self._known.get(name) == digest:
            return False
        target = self._target(name)
        data = text.encode("utf-8")
        if not target.exists() and (self._new_files >= SESSION_NEW_FILES or self._written + len(data) > SESSION_BYTES):
            self._say_once(
                f"Warning: this session already copied back {self._new_files} new notebooks "
                f"({self._written // (1024 * 1024)} MiB); new notebooks from the kernel are no longer copied back."
            )
            return False
        backup = None
        if target.is_file():
            current = target.read_bytes()
            here = notebook_digest(current.decode("utf-8", "replace"))
            if here == digest:
                self._known[name] = digest
                return False
            if here != self._known.get(name):
                backup = self._backup(name, current)
        if not target.exists():
            self._new_files += 1
        _write_atomically(target, data)
        self._written += len(data)
        self._known[name] = digest
        mark_sandbox_wrote(self.workspace)  # the unsafe-local runtime asks before it runs them
        if backup is not None:
            self.sandbox.notify(
                f"Warning: {name} changed both here and in the kernel. The kernel's version replaced it; "
                f"the copy that was here is saved as {backup}."
            )
        return True

    def _backup(self, name: str, data: bytes) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._clock()))
        folder = ensure_state_dir(self.workspace) / BACKUP_FOLDER
        stem = name.removesuffix(".py")
        for attempt in range(100):
            path = folder.joinpath(*f"{stem}.{stamp}{f'-{attempt}' if attempt else ''}.py".split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(path, "xb") as handle:
                    handle.write(data)
            except FileExistsError:
                continue
            return path
        raise NotebookPathError(f"no free backup name for {name} in {folder}")

    def _say_once(self, text: str) -> None:
        if text not in self._said:
            self._said.add(text)
            self.sandbox.notify(text)

    # -- the periodic copy ---------------------------------------------------- #

    def start(self, interval: float | None = None) -> None:
        """Copy back every ``interval`` seconds (default :data:`SYNC_INTERVAL_SEC`), and whenever
        :meth:`soon` asks, on a daemon thread."""
        if self._thread is not None:
            return
        every = SYNC_INTERVAL_SEC if interval is None else interval

        def run() -> None:
            while not self._stopping.is_set():
                self._wake.wait(every)
                self._wake.clear()
                if self._stopping.is_set():
                    return
                self.sync_out()

        self._thread = threading.Thread(target=run, name="hailer-notebook-sync", daemon=True)
        self._thread.start()

    def soon(self) -> None:
        """Wake the periodic copy now (returns at once)."""
        self._wake.set()

    def close(self, *, final: bool = True) -> list[str]:
        """Stop the periodic copy (a pass under way ends within :data:`SYNC_CYCLE_SEC`) and, with
        ``final``, copy back once more within :data:`SYNC_DEADLINE_SEC`; the names written by that
        copy. Idempotent."""
        self._stopping.set()
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(SYNC_CYCLE_SEC + 5)
        return self.sync_out(seconds=SYNC_DEADLINE_SEC, budget=MAX_NOTEBOOKS * MAX_NOTEBOOK_BYTES) if final else []


__all__ = [
    "BACKUP_FOLDER",
    "SYNC_CYCLE_BYTES",
    "SYNC_CYCLE_SEC",
    "SYNC_DEADLINE_SEC",
    "SYNC_INTERVAL_SEC",
    "TOOLING_NAMES",
    "NotebookSync",
    "module_names",
    "sync_refusal",
]
