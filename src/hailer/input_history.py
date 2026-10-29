"""What the chat remembers of what you typed: ``.hailer/input_history``, recalled with Up.

The composer (:mod:`hailer.chat_ui`) and plain mode's prompt (``cli.common._LineReader``) take their
history from :func:`input_history`: prompt_toolkit's in-memory history when ``[hailer].input_history``
is off, else an :class:`InputHistory` that prompt_toolkit's ``ThreadedHistory`` reads on a worker
thread, so the first render never waits for the disk.

The file is in prompt_toolkit's ``FileHistory`` format: each entry after a ``# <time>`` line, each of
its lines behind a ``+``, so a multi-line message comes back whole. Reading keeps the last
:data:`MAX_ENTRIES` entries (and at most the file's last :data:`MAX_READ_BYTES`) and rewrites a longer
file. An entry that :func:`hailer.log.redact` would change (it looks like it holds a key or token) is
never written: it stays in this session's memory. Reads, appends and the rewrite hold a lock on
``input_history.lock`` beside the file, so two chats in one workspace never mix their entries; the
rewrite replaces the file in one step. The files are created readable by their owner only (where the
OS has such permissions). Nothing here raises: an unreadable, unwritable or corrupt file only means
less history.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from prompt_toolkit.history import History, InMemoryHistory, ThreadedHistory

from hailer.log import redact
from hailer.statedir import STATE_DIRNAME, ensure_state_dir, state_dir

HISTORY_FILENAME = "input_history"
MAX_ENTRIES = 1000
#: Only the end of a larger file is read; the entry it cuts in two is dropped.
MAX_READ_BYTES = 2 * 1024 * 1024
#: How long an append or a read waits for the other chat's lock before it gives up.
LOCK_WAIT_SECONDS = 0.25

_OWNER_ONLY = 0o600
_BINARY = getattr(os, "O_BINARY", 0)  # Windows: no newline translation

log = logging.getLogger(__name__)


def input_history_path(workspace: Path) -> Path:
    """``<workspace>/.hailer/input_history``."""
    return state_dir(workspace) / HISTORY_FILENAME


def input_history(path: Path | None) -> History:
    """The prompt's history: ``path``'s, read on a worker thread, or (``None``) this session's only."""
    return ThreadedHistory(InputHistory(path)) if path is not None else InMemoryHistory()


def storable(text: str) -> bool:
    """Whether ``text`` may be written to the history file: not blank and nothing in it that looks
    like a secret (what :func:`hailer.log.redact` masks: bearer tokens, ``sk-`` keys, the values of
    ``*_KEY``/``*_TOKEN``/... variables)."""
    return bool(text.strip()) and redact(text) == text


class InputHistory(History):
    """prompt_toolkit's history kept in ``path`` (see the module docstring)."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    def load_history_strings(self) -> Iterable[str]:
        """The entries, newest first."""
        try:
            if not self.path.exists():
                return []
            with _locked(self.lock_path):
                entries, rewrite = self._read()
                if rewrite:
                    self._rewrite(entries)
        except Exception as exc:  # noqa: BLE001 - history must never break the chat
            log.debug("Input history not read from %s: %s", self.path, exc)
            return []
        return entries[::-1]

    def store_string(self, string: str) -> None:
        if not storable(string):
            return
        data = _entry(string, str(datetime.datetime.now())).encode("utf-8", errors="replace")
        try:
            _ensure_folder(self.path)
            with _locked(self.lock_path):
                descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | _BINARY, _OWNER_ONLY)
                try:
                    _write_all(descriptor, data)
                finally:
                    os.close(descriptor)
        except Exception as exc:  # noqa: BLE001 - a read-only workspace keeps this session's history only
            log.debug("Input history not written to %s: %s", self.path, exc)

    def _read(self) -> tuple[list[str], bool]:
        """(the last MAX_ENTRIES entries, oldest first; whether the file holds more than that)."""
        try:
            with open(self.path, "rb") as handle:
                size = os.fstat(handle.fileno()).st_size
                cut = size > MAX_READ_BYTES
                if cut:
                    handle.seek(size - MAX_READ_BYTES)
                data = handle.read()
        except FileNotFoundError:
            return [], False
        entries = _parse(data, cut=cut)
        return entries[-MAX_ENTRIES:], cut or len(entries) > MAX_ENTRIES

    def _rewrite(self, entries: list[str]) -> None:
        """Replace the file with ``entries`` in one step (a crash leaves the old file or the new one)."""
        temporary = self.path.with_name(self.path.name + ".tmp")
        data = "".join(_entry(entry, "") for entry in entries).encode("utf-8", errors="replace")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _BINARY, _OWNER_ONLY)
        try:
            _write_all(descriptor, data)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, self.path)


def _parse(data: bytes, *, cut: bool = False) -> list[str]:
    """The entries in ``data``, oldest first: runs of ``+`` lines; any other line ends an entry (so
    stray or damaged lines are skipped). ``cut``: ``data`` starts inside the file, so its first line
    may be a piece of one and its first run the end of an entry: both are dropped."""
    entries: list[str] = []
    pieces = data.split(b"\n")
    lines: list[str] | None = None if cut else []
    for raw in pieces[1:] if cut else pieces:
        if raw.startswith(b"+"):
            if lines is not None:
                lines.append(raw[1:].decode("utf-8", errors="replace"))
        else:
            if lines:
                entries.append("\n".join(lines))
            lines = []
    if lines:
        entries.append("\n".join(lines))
    return entries


def _entry(text: str, stamp: str) -> str:
    """One entry as the file holds it: a ``#`` line, then each line behind a ``+``."""
    return f"\n# {stamp}\n" + "".join(f"+{line}\n" for line in text.split("\n"))


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(descriptor, view):]


def _ensure_folder(path: Path) -> None:
    folder = path.parent
    if folder.name == STATE_DIRNAME:
        ensure_state_dir(folder.parent)  # with its .gitignore, like everything Hailer writes there
    else:
        folder.mkdir(parents=True, exist_ok=True)


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold the lock file ``path`` for this block, waiting at most :data:`LOCK_WAIT_SECONDS` (then
    ``TimeoutError``). A file system without locks (some network drives) is used unlocked."""
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | _BINARY, _OWNER_ONLY)
    try:
        deadline = time.monotonic() + LOCK_WAIT_SECONDS
        while not _try_lock(descriptor):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{path} is held by another Hailer")
            time.sleep(0.005)
        try:
            yield
        finally:
            _unlock(descriptor)
    finally:
        os.close(descriptor)


if sys.platform == "win32":
    import msvcrt

    def _try_lock(descriptor: int) -> bool:
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(descriptor: int) -> None:
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass

else:
    import fcntl

    def _try_lock(descriptor: int) -> bool:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        except OSError:  # no locks on this file system: carry on without
            return True
        return True

    def _unlock(descriptor: int) -> None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
