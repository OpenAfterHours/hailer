"""What the chat remembers of what you typed: ``.hailer/input_history``, recalled with Up.

The composer (:mod:`hailer.chat_ui`) and plain mode's prompt (``cli.common._LineReader``) take their
history from :func:`input_history`: prompt_toolkit's in-memory history when ``[hailer].input_history``
is off, else an :class:`InputHistory` that prompt_toolkit's ``ThreadedHistory`` reads on a worker
thread, so the first render never waits for the disk. Saving never runs on the UI loop either: a
submitted message goes to one ordered writer thread, which checks it, waits for the lock and appends
it; at exit Hailer gives that thread :data:`EXIT_FLUSH_SECONDS` to finish, then leaves it.

The file is in prompt_toolkit's ``FileHistory`` format: each entry after a ``# <time>`` line, each of
its lines behind a ``+``, so a multi-line message comes back whole. Reading keeps the last
:data:`MAX_ENTRIES` entries and rewrites a longer file (never one it could read nothing from). Two
kinds of entry stay in this session's memory and are never written: one over
:data:`MAX_ENTRY_BYTES` (a large paste), and one that looks like it holds a credential
(:func:`storable`). Reads, appends and the rewrite hold a lock on ``input_history.lock`` beside the
file, so two chats in one workspace never mix their entries; the rewrite goes through a new temporary
file and replaces the history in one step. Both files are created readable by their owner only (where
the OS has such permissions), and a history or lock file that is a link or anything but a regular file
is never opened: history then stays in memory. Nothing here raises: an unreadable, unwritable or
corrupt file only means less history.
"""

from __future__ import annotations

import atexit
import datetime
import errno
import logging
import os
import re
import stat
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from prompt_toolkit.history import History, InMemoryHistory, ThreadedHistory

from hailer.log import redact
from hailer.statedir import STATE_DIRNAME, ensure_state_dir

MAX_ENTRIES = 1000
#: A larger entry (a pasted file) is kept in memory for the session only.
MAX_ENTRY_BYTES = 64 * 1024
#: Never read more than a capped file can hold; from a larger one (not written by Hailer) the end.
MAX_READ_BYTES = MAX_ENTRIES * (MAX_ENTRY_BYTES + 64)
#: How long the writer (or a read) waits for another chat's lock before it gives up on that entry.
LOCK_WAIT_SECONDS = 2.0
#: How long exit waits for entries still being written.
EXIT_FLUSH_SECONDS = 0.5

_OWNER_ONLY = 0o600
_FLAGS = getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)  # no newline translation; never through a link
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)  # a planted FIFO must not block the open

#: Credential shapes :func:`hailer.log.redact` does not mask: a URL with a password in it
#: (``scheme://user:password@``), GitHub tokens, AWS access key IDs.
_CREDENTIALS = re.compile(
    r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s/@:]+:[^\s/@]+@"
    r"|\bgh[pousr]_[A-Za-z0-9]{20,}"
    r"|\bgithub_pat_[A-Za-z0-9_]{20,}"
    r"|\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"
)

log = logging.getLogger(__name__)


def input_history(path: Path | None) -> History:
    """The prompt's history: ``path``'s, read on a worker thread, or (``None``) this session's only."""
    return ThreadedHistory(InputHistory(path)) if path is not None else InMemoryHistory()


def storable(text: str) -> bool:
    """Whether ``text`` may be written to the history file: not blank, at most
    :data:`MAX_ENTRY_BYTES`, and nothing in it that looks like a credential: what
    :func:`hailer.log.redact` masks (a bearer token or ``Authorization:`` value, an ``sk-`` key, the
    value of a ``*_KEY``/``*_TOKEN``/``*_SECRET``/``*_PASSWORD`` environment variable), a URL with a
    password (``user:password@``), a GitHub token (``ghp_``, ``gho_``, ``ghs_``, ``ghu_``, ``ghr_``,
    ``github_pat_``) or an AWS access key ID (``AKIA``/``ASIA``)."""
    if not text.strip() or len(text) > MAX_ENTRY_BYTES or len(text.encode("utf-8", errors="replace")) > MAX_ENTRY_BYTES:
        return False
    return not _CREDENTIALS.search(text) and redact(text) == text


class InputHistory(History):
    """prompt_toolkit's history kept in ``path`` (see the module docstring)."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._refused = False

    def load_history_strings(self) -> Iterable[str]:
        """The entries, newest first (after this process's earlier entries are written)."""
        _WRITER.flush(LOCK_WAIT_SECONDS)
        try:
            if self._refused or not self._usable(must_exist=True):
                return []
            with _locked(self.lock_path):
                entries, rewrite = self._read()
                if rewrite:
                    try:
                        self._rewrite(entries)
                    except Exception as exc:  # noqa: BLE001 - the file stays long until a later session trims it
                        log.debug("Input history at %s not trimmed: %s", self.path, exc)
        except Exception as exc:  # noqa: BLE001 - history must never break the chat
            log.debug("Input history not read from %s: %s", self.path, exc)
            return []
        return entries[::-1]

    def store_string(self, string: str) -> None:
        """Queue ``string`` for the writer thread (nothing is checked or written on the caller's)."""
        if not self._refused:
            _WRITER.submit(self, string)

    def flush(self, timeout: float = LOCK_WAIT_SECONDS) -> bool:
        """Wait until the entries queued so far are written (or dropped); False on timeout."""
        return _WRITER.flush(timeout)

    # -- on the writer thread, or the reading one --------------------------------- #

    def _append(self, text: str) -> None:
        if self._refused or not storable(text):
            return
        data = _entry(text, str(datetime.datetime.now())).encode("utf-8", errors="replace")
        try:
            _ensure_folder(self.path)
            if not self._usable(must_exist=False):
                return
            with _locked(self.lock_path):
                descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | _FLAGS | _NONBLOCK, _OWNER_ONLY)
                try:
                    if self._regular(descriptor):
                        _write_all(descriptor, data)
                finally:
                    os.close(descriptor)
        except Exception as exc:  # noqa: BLE001 - a read-only workspace keeps this session's history only
            log.debug("Input history not written to %s: %s", self.path, exc)

    def _usable(self, *, must_exist: bool) -> bool:
        """Whether the history and lock files are regular files (or, unless ``must_exist``, missing)."""
        for path in (self.path, self.lock_path):
            try:
                mode = os.lstat(path).st_mode
            except FileNotFoundError:
                if must_exist and path == self.path:
                    return False
                continue
            if not stat.S_ISREG(mode):
                return self._refuse(f"{path} is not a regular file")
        return True

    def _regular(self, descriptor: int) -> bool:
        return stat.S_ISREG(os.fstat(descriptor).st_mode) or self._refuse(f"{self.path} is not a regular file")

    def _refuse(self, reason: str) -> bool:
        if not self._refused:
            self._refused = True
            log.info("Input history stays in memory: %s (a link or a special file is never followed).", reason)
        return False

    def _read(self) -> tuple[list[str], bool]:
        """(the last MAX_ENTRIES entries, oldest first; whether the file should be rewritten: it holds
        more, or an entry too large to keep, and something could be read from it)."""
        try:
            descriptor = os.open(self.path, os.O_RDONLY | _FLAGS | _NONBLOCK)
        except FileNotFoundError:
            return [], False
        with os.fdopen(descriptor, "rb") as handle:
            if not self._regular(handle.fileno()):
                return [], False
            size = os.fstat(handle.fileno()).st_size
            cut = size > MAX_READ_BYTES
            if cut:
                handle.seek(size - MAX_READ_BYTES)
            data = handle.read()
        entries, dropped = _parse(data, cut=cut)
        kept = entries[-MAX_ENTRIES:]
        return kept, bool(kept) and (cut or dropped or len(entries) > MAX_ENTRIES)

    def _rewrite(self, entries: list[str]) -> None:
        """Replace the file with ``entries`` in one step (a crash leaves the old file or the new one),
        through a new temporary file: nothing already in the folder is opened for writing."""
        data = "".join(_entry(entry, "") for entry in entries).encode("utf-8", errors="replace")
        descriptor, name = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp")
        temporary = Path(name)
        try:
            try:
                _write_all(descriptor, data)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temporary, self.path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


class _Writer:
    """One ordered thread that writes every history's entries, started with the first one."""

    def __init__(self) -> None:
        self._jobs: deque[tuple[InputHistory, str]] = deque()
        self._condition = threading.Condition()
        self._writing = False
        self._thread: threading.Thread | None = None

    def submit(self, history: InputHistory, text: str) -> None:
        with self._condition:
            self._jobs.append((history, text))
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="hailer-input-history", daemon=True)
                self._thread.start()
            self._condition.notify_all()

    def flush(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._jobs or self._writing:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
        return True

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._jobs:
                    self._condition.wait()
                history, text = self._jobs.popleft()
                self._writing = True
            try:
                history._append(text)
            except Exception as exc:  # noqa: BLE001 - _append logs its own; this keeps the thread alive
                log.debug("Input history entry not written: %s", exc)
            finally:
                with self._condition:
                    self._writing = False
                    self._condition.notify_all()


_WRITER = _Writer()
# Entries still queued at exit get a moment; the daemon thread is then left behind.
atexit.register(_WRITER.flush, EXIT_FLUSH_SECONDS)


def _parse(data: bytes, *, cut: bool = False) -> tuple[list[str], bool]:
    """(the entries in ``data``, oldest first; whether one over MAX_ENTRY_BYTES was left out). An
    entry is a run of ``+`` lines; any other line ends it (so stray or damaged lines are skipped).
    ``cut``: ``data`` starts inside the file, so its first line may be a piece of one and its first
    run the end of an entry: both are dropped."""
    entries: list[str] = []
    dropped = False
    pieces = [piece.removesuffix(b"\r") for piece in data.split(b"\n")]  # CRLF too (an editor, Windows text mode)
    lines: list[bytes] | None = None if cut else []
    size = 0

    def end() -> None:
        nonlocal dropped
        if lines:
            if size > MAX_ENTRY_BYTES:
                dropped = True
            else:
                entries.append("\n".join(line.decode("utf-8", errors="replace") for line in lines))

    for raw in pieces[1:] if cut else pieces:
        if raw.startswith(b"+"):
            if lines is not None:
                lines.append(raw[1:])
                size += len(raw)
        else:
            end()
            lines, size = [], 0
    end()
    return entries, dropped


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
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | _FLAGS | _NONBLOCK, _OWNER_ONLY)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"{path} is not a regular file")
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


# Windows (msvcrt.locking): only "the region is locked" means held; a file system without locks
# raises other errors, and is then used unlocked, as on POSIX.
_WINDOWS_HELD = frozenset({errno.EACCES, getattr(errno, "EDEADLOCK", errno.EDEADLK)})


def _windows_try_lock(descriptor: int) -> bool:
    import msvcrt

    os.lseek(descriptor, 0, os.SEEK_SET)
    try:
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        return exc.errno not in _WINDOWS_HELD
    return True


def _windows_unlock(descriptor: int) -> None:
    import msvcrt

    os.lseek(descriptor, 0, os.SEEK_SET)
    try:
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    except OSError:
        pass


def _posix_try_lock(descriptor: int) -> bool:
    import fcntl

    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    except OSError:  # no locks on this file system: carry on without
        return True
    return True


def _posix_unlock(descriptor: int) -> None:
    import fcntl

    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    except OSError:
        pass


_try_lock, _unlock = (_windows_try_lock, _windows_unlock) if sys.platform == "win32" else (_posix_try_lock, _posix_unlock)
