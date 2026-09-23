"""The sandbox: a started kernel, as everything above the runtime layer sees it.

:class:`MarimoSandbox` is the contract: what a kernel runtime's ``start`` returns and what the chat
and the agent's tools hold for as long as the chat runs. It owns the server (URL and token),
stopping, a description, and the files: the notebooks and the data. Notebooks are *names*, POSIX
paths relative to the notebooks folder (``"sales.py"``, ``"q3/review.py"``); data files are names
relative to the data folder. Nothing that holds a sandbox reads or writes the notebooks or data
folders itself, so a kernel whose files are not on this machine (a container with its own
notebooks, a cloud kernel) fits the same contract. Every runtime runs marimo, so every runtime
subclasses :class:`MarimoSandbox` for its lifecycle (stop, log, wait).

Notebook files go through marimo's own file API (:class:`~hailer.marimo_client.MarimoClient`), so
they work the same wherever the kernel runs; the data folder, a host folder in both runtimes today,
is listed on this machine. A sandbox whose notebooks are not the host's own (the docker kernel's
in-memory folder) copies them in when it starts and back out while it runs and when it stops
(:mod:`hailer.notebook_sync`); for the local kernel the two are one folder and the ``sync_*``
methods do nothing.

Standard library only at import time (``data_schema`` imports Polars when it is called).
"""

from __future__ import annotations

import hashlib
import re
import urllib.error
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from hailer.errors import (
    HailerError,
    MalformedParquetError,
    MarimoUnavailableError,
    NotebookExistsError,
    NotebookNotFoundError,
    NotebookPathError,
)
from hailer.marimo_client import MAX_NOTEBOOK_BYTES, MarimoClient, home_url, kernel_path, name_in_folder, open_notebook_url
from hailer.models import MarimoServer
from hailer.notebooks import check_notebook_name

#: MAX_NOTEBOOK_BYTES (from marimo_client, which sizes replies from it): the largest notebook
#: :meth:`MarimoSandbox.read_notebook` reads (marimo's own limit is far higher).
#: How many folder levels below the notebooks folder a notebook may be (listing and writing).
MAX_NOTEBOOK_DEPTH = 4
#: At most this many notebooks are listed, and at most this many folders are looked into.
MAX_NOTEBOOKS = 500
MAX_NOTEBOOK_FOLDERS = 200
#: At most this many files and folders are listed by :meth:`MarimoSandbox.list_tree`.
MAX_TREE_ENTRIES = 5000
#: Folders never looked into, besides those starting with ``.`` (``.git``, ``.venv``) or ``__``
#: (``__marimo__``, ``__pycache__``).
SKIPPED_FOLDERS = frozenset({"node_modules", "venv", "site-packages"})
_BAD_DATA_CHARS = re.compile(r'[\x00-\x1f<>:"|?*]')


@dataclass(frozen=True)
class NotebookFile:
    """A marimo notebook in the sandbox's notebooks folder."""

    name: str  # POSIX path relative to the notebooks folder, e.g. "q3/review.py"
    modified: float = 0.0  # seconds since the epoch
    size: int = 0  # bytes

    @property
    def stem(self) -> str:
        """The file name without ``.py`` (``"review"``)."""
        return self.name.rsplit("/", 1)[-1].removesuffix(".py")


@dataclass(frozen=True)
class DataFile:
    """A file in the sandbox's data folder."""

    name: str  # relative to the data folder, e.g. "25-03 sales.parquet"
    modified: float = 0.0
    size: int = 0


@dataclass(frozen=True)
class SandboxEntry:
    """A file or folder in the sandbox's notebooks folder, whatever it holds
    (:meth:`MarimoSandbox.list_tree`)."""

    name: str  # POSIX path relative to the notebooks folder
    folder: bool = False
    modified: float = 0.0
    size: int = 0
    marimo: bool = False  # marimo's listing says it is a notebook (``import marimo`` and ``marimo.App``)


def notebook_digest(source: str) -> str:
    """The sha256 of a notebook's text with its line endings normalised to ``\\n``: marimo's
    ``update`` writes ``\\r\\n`` on a Windows kernel while ``create`` stores the bytes as given, so
    only a normalised digest says whether two copies differ."""
    return hashlib.sha256(source.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _check_data_name(name: str) -> str:
    """``name`` as a file directly in the data folder, or ``HailerError``."""
    text = str(name or "")
    if not text or "/" in text or "\\" in text or text.startswith(".") or _BAD_DATA_CHARS.search(text):
        raise HailerError(f"{name!r} is not a file name in the data folder.", hint="Use a name list_periods reported.")
    return text


def _print(text: str) -> None:
    print(text, flush=True)


def _skipped(folder: str) -> bool:
    return folder.startswith((".", "__")) or folder.lower() in SKIPPED_FOLDERS


@dataclass
class MarimoSandbox:
    """A started kernel and its files.

    ``server`` has the URL and the token. ``notebooks_path`` is the kernel's path of the notebooks
    folder (the unsafe-local runtime: the host folder as :func:`~hailer.marimo_client.notebook_file_key`
    writes it; docker: ``/work/notebooks``, the container's own) and ``native_paths`` says the
    kernel's paths are this machine's; ``data_path`` is the data folder as notebook code reaches it
    and ``data_dir`` the host folder behind it (a read-only mount in a container). ``log_hint``
    says where the kernel's log is; ``ended`` why :meth:`wait` returned when the kernel stopped by
    itself (empty after Ctrl+C). ``notice`` shows a warning line (the chat points it at its own
    console; default: print); the notebook copies call it from any thread.
    """

    server: MarimoServer
    notebooks_path: str = ""
    data_path: str = ""
    data_dir: Path | None = None
    native_paths: bool = False
    log_hint: str = ""
    ended: str = ""
    notice: Callable[[str], None] | None = None
    #: marimo's server token (its page's skew token), read once and shared by this sandbox's clients.
    _tokens: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    # -- the kernel ---------------------------------------------------------- #

    def describe(self) -> str:
        """The text after ``Kernel:`` (runtime, isolation, network)."""
        from hailer.kernel import describe_runtime  # lazy: hailer.kernel imports this module
        from hailer.models import KernelConfig

        return describe_runtime(KernelConfig(runtime=self.server.runtime, network=self.server.network_access))

    def stop(self) -> None:
        """Stop the kernel and remove what it left behind. Idempotent, never raises."""

    def log_tail(self, lines: int = 15) -> list[str]:
        return []

    def wait(self) -> int:
        """Block while the kernel runs (``--foreground``); Ctrl+C ends the wait. Returns an exit code."""
        raise NotImplementedError

    def client(self, *, notebook: str | None = None, token_in_links: bool = False, timeout: float = 10.0) -> MarimoClient:
        """A client for this kernel that knows notebooks by name (``notebook``: its default)."""
        return MarimoClient(
            self.server.url,
            self.server.token,
            timeout=timeout,
            notebooks_path=self.notebooks_path or None,
            native_paths=self.native_paths,
            notebook=notebook,
            token_in_links=token_in_links,
            token_cache=self._tokens,
        )

    def notebook_url(self, name: str, *, with_token: bool = False) -> str:
        """The URL that opens ``name`` in app view; signed in only with ``with_token``."""
        return open_notebook_url(self.server, kernel_path(self.notebooks_path, check_notebook_name(name)), with_token=with_token)

    def home_url(self, *, with_token: bool = False) -> str:
        return home_url(self.server, with_token=with_token)

    def notify(self, text: str) -> None:
        """Show the warning line ``text`` through :attr:`notice` (never raises)."""
        try:
            (self.notice or _print)(text)
        except Exception:  # noqa: BLE001 - a warning must never break what reported it
            pass

    # -- notebooks ----------------------------------------------------------- #

    def list_notebooks(self) -> list[NotebookFile]:
        """Every marimo notebook (``.py``) under the notebooks folder, sorted by name, from
        :meth:`list_tree` (so within its bounds; at most :data:`MAX_NOTEBOOKS`); nothing in a folder
        starting with ``.`` or ``__`` or in :data:`SKIPPED_FOLDERS`, nor (local kernel) behind a link
        or junction that leads out of the notebooks folder. ``MarimoUnavailableError`` when marimo
        cannot list."""
        found: list[NotebookFile] = []
        for entry in self.list_tree():
            if entry.folder or not entry.marimo or not entry.name.lower().endswith(".py"):
                continue
            parts = entry.name.split("/")
            if any(part.startswith((".", "__")) or part.lower() in SKIPPED_FOLDERS for part in parts):
                continue
            try:
                check_notebook_name(entry.name)
            except NotebookPathError:
                continue
            found.append(NotebookFile(entry.name, entry.modified, entry.size))
        return sorted(found, key=lambda info: info.name.lower())[:MAX_NOTEBOOKS]

    def list_tree(self) -> list[SandboxEntry]:
        """Every file and folder in the notebooks folder, whatever it holds, sorted by name: what
        :meth:`list_notebooks` and copying notebooks out look at. Folders are looked into at most
        :data:`MAX_NOTEBOOK_DEPTH` levels deep, at most :data:`MAX_NOTEBOOK_FOLDERS` of them, and
        never one starting with ``.`` or ``__`` (``.git``, ``__marimo__``) or in
        :data:`SKIPPED_FOLDERS`: those are listed themselves. At most :data:`MAX_TREE_ENTRIES`
        entries. A local kernel's names never lead out of the folder through a link or junction.
        ``MarimoUnavailableError`` when marimo cannot list."""
        client = self.client()
        found: list[SandboxEntry] = []
        pending: list[tuple[str, int]] = [(self.notebooks_path, 0)]
        folders = 0
        while pending and folders < MAX_NOTEBOOK_FOLDERS and len(found) < MAX_TREE_ENTRIES:
            folder, depth = pending.pop(0)
            folders += 1
            for entry in client.list_files(folder):
                path = entry.get("path")
                name = client.name_of(path) if isinstance(path, str) else None
                if not name:
                    continue
                directory = bool(entry.get("isDirectory"))
                found.append(
                    SandboxEntry(
                        name, directory, float(entry.get("lastModified") or 0.0), int(entry.get("size") or 0),
                        marimo=not directory and bool(entry.get("isMarimoFile")),
                    )
                )  # fmt: skip
                if directory and depth < MAX_NOTEBOOK_DEPTH and not _skipped(name.rsplit("/", 1)[-1]):
                    pending.append((path, depth + 1))
                if len(found) >= MAX_TREE_ENTRIES:
                    break
        return sorted(found, key=lambda item: item.name)

    def _inside(self, client: MarimoClient, name: str) -> str:
        """The kernel path of ``name`` after refusing, for a local kernel, one whose resolved target
        leaves the notebooks folder (a link or junction on the way): marimo follows them."""
        key = client.file_key(name)
        if self.native_paths and name_in_folder(self.notebooks_path, key, native=True) is None:
            raise NotebookPathError(
                f"{name} leads outside the notebooks folder (through a link or junction).",
                hint="Hailer only reads and writes notebooks inside the notebooks folder.",
            )
        return key

    def _entry(self, client: MarimoClient, key: str) -> dict | None:
        """marimo's listing entry for the file at ``key``, or ``None`` when there is none. Names
        ignore case on a Windows kernel."""
        folder, _, base = key.rpartition("/")
        wanted = base.casefold() if client.folds_case else base
        for entry in client.list_files(folder):
            label = str(entry.get("name") or "")
            if (label.casefold() if client.folds_case else label) == wanted:
                return entry
        return None

    def _spelling(self, client: MarimoClient, name: str, entry: dict | None) -> str:
        """``name`` as the kernel's listing spells it (a Windows kernel ignores case)."""
        listed = client.name_of(entry.get("path")) if entry else None
        return listed or name

    def has_notebook(self, name: str) -> bool:
        """Whether the file ``name`` exists (one request; for startup's active-notebook check)."""
        client = self.client()
        try:
            entry = self._entry(client, self._inside(client, check_notebook_name(name)))
        except NotebookPathError:
            return False
        return entry is not None and not entry.get("isDirectory")

    def read_notebook(self, name: str) -> str:
        """The text of the notebook ``name``, for copying it out of the sandbox.
        ``NotebookNotFoundError`` when there is no such file,
        ``NotebookPathError`` when it is not UTF-8 text, larger than :data:`MAX_NOTEBOOK_BYTES` or
        (local kernel) reached through a link out of the folder."""
        name = check_notebook_name(name)
        client = self.client()
        key = self._inside(client, name)
        try:
            details = client.file_details(key, max_bytes=MAX_NOTEBOOK_BYTES)
        except urllib.error.HTTPError as err:
            raise NotebookNotFoundError(f"No notebook named {name} in the notebooks folder (HTTP {err.code}).") from err
        if details.get("isTooLarge"):
            raise NotebookPathError(f"{name} is larger than {MAX_NOTEBOOK_BYTES // (1024 * 1024)} MiB.")
        contents = details.get("contents")
        if details.get("isBase64") or not isinstance(contents, str):
            raise NotebookPathError(f"{name} is not a UTF-8 text file.")
        return contents

    def writable_name(self, name: str) -> str:
        """``name`` as a notebook this sandbox may create (``/`` separators), or ``NotebookPathError``:
        :func:`~hailer.notebooks.check_notebook_name`, at most :data:`MAX_NOTEBOOK_DEPTH` folders
        deep; a runtime may add rules (the docker kernel: names it would copy back)."""
        name = check_notebook_name(name)
        if name.count("/") > MAX_NOTEBOOK_DEPTH:
            raise NotebookPathError(
                f"{name} is more than {MAX_NOTEBOOK_DEPTH} folders deep.",
                hint="Keep notebooks at most four folders below the notebooks folder.",
            )
        return name

    def write_notebook(self, name: str, source: str, *, replace: bool = True) -> NotebookFile:
        """Create the notebook ``name`` with ``source``, or replace its text when it exists (marimo
        reloads an open session of it). ``replace=False`` refuses an existing file with
        ``NotebookExistsError``; folders in ``name`` are created, at most :data:`MAX_NOTEBOOK_DEPTH`
        deep. On a Windows kernel a name that differs only in case is the existing file, and the
        result has the listing's spelling. A local kernel never writes through a link or junction
        that leaves the notebooks folder. A new file holds ``source``'s bytes; a replaced one is
        written by marimo with the kernel's line endings (compare with :func:`notebook_digest`).
        The name goes through :meth:`writable_name` first."""
        name = self.writable_name(name)
        client = self.client()
        key = self._inside(client, name)
        existing = self._entry(client, key)
        name = self._spelling(client, name, existing)
        if existing is not None and (existing.get("isDirectory") or not replace):
            raise NotebookExistsError(
                f"{name} already exists in the notebooks folder.",
                hint="Pick another name, or open the existing notebook instead.",
            )
        if existing is None:
            folder, _, base = key.rpartition("/")
            reply = client.create_file(folder, base, source.encode("utf-8"))
            info = reply.get("info") if isinstance(reply.get("info"), dict) else {}
            created = client.name_of(info.get("path")) if reply.get("success") else None
            if created is not None and (created.casefold() == name.casefold() if client.folds_case else created == name):
                return NotebookFile(created, float(info.get("lastModified") or 0.0), int(info.get("size") or 0))
            if created is not None:  # the name was taken meanwhile; marimo wrote <stem>_1.py instead
                client.delete_file(str(info.get("path")))
                if not replace:
                    raise NotebookExistsError(f"{name} already exists in the notebooks folder.")
            elif not reply.get("success"):
                raise NotebookPathError(
                    f"Could not create {name}: {reply.get('message') or 'marimo refused'}.",
                    hint="Try a shorter or simpler name, and check that the notebooks folder is writable.",
                )
            key = self._inside(client, name)
        reply = client.update_file(existing.get("path") if existing else key, source)
        if not reply.get("success"):
            raise NotebookPathError(f"Could not write {name}: {reply.get('message') or 'marimo refused'}.")
        info = reply.get("info") if isinstance(reply.get("info"), dict) else {}
        return NotebookFile(name, float(info.get("lastModified") or 0.0), int(info.get("size") or 0))

    def sync_soon(self) -> None:
        """Ask for the sandbox's notebooks to be copied back to the workspace in the background
        (after a turn, on a notebook switch); returns at once. Nothing to do while the notebooks
        folder is the host's own (the local kernel); the docker kernel wakes its
        :class:`~hailer.notebook_sync.NotebookSync`."""

    # -- data ---------------------------------------------------------------- #

    def list_data(self) -> list[DataFile]:
        """The files directly in the data folder (not hidden), sorted by name. ``HailerError`` when
        the folder does not exist."""
        folder = self.data_dir
        if folder is None or not folder.is_dir():
            raise HailerError(f"Data directory does not exist: {self.data_path or folder}")
        found: list[DataFile] = []
        for path in folder.iterdir():
            if path.name.startswith("."):
                continue
            try:
                if not path.is_file():
                    continue
                stat = path.stat()
            except OSError:
                continue
            found.append(DataFile(path.name, stat.st_mtime, stat.st_size))
        return sorted(found, key=lambda info: info.name.lower())

    def data_schema(self, name: str) -> dict[str, str]:
        """The columns of the Parquet file ``name`` in the data folder, with their types as text.
        ``MalformedParquetError`` when it cannot be read."""
        name = _check_data_name(name)
        if self.data_dir is None:
            raise MarimoUnavailableError("This kernel has no data folder.")
        import polars as pl  # lazy: heavy, and only list_periods needs it

        path = self.data_dir / name
        try:
            return {column: str(dtype) for column, dtype in pl.read_parquet_schema(path).items()}
        except (pl.exceptions.PolarsError, OSError, ValueError, TypeError) as err:
            raise MalformedParquetError(
                f"Could not read {name}: {err}",
                hint=f"Check that {name} is a valid parquet file (it may be truncated, not parquet, or still being written).",
            ) from err


__all__ = [
    "MAX_NOTEBOOK_BYTES",
    "MAX_NOTEBOOK_DEPTH",
    "MAX_NOTEBOOK_FOLDERS",
    "MAX_NOTEBOOKS",
    "MAX_TREE_ENTRIES",
    "SKIPPED_FOLDERS",
    "DataFile",
    "MarimoSandbox",
    "NotebookFile",
    "SandboxEntry",
    "notebook_digest",
]
