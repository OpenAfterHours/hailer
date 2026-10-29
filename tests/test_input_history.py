"""The input history file (.hailer/input_history): format, cap, secrets, links, failures, the writer
thread, locks and two processes."""

from __future__ import annotations

import errno
import logging
import os
import stat
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.history import InMemoryHistory, ThreadedHistory

from hailer import input_history as module
from hailer.input_history import InputHistory, input_history, storable
from hailer.statedir import GITIGNORE_NAME, input_history_path

SRC = Path(__file__).resolve().parents[1] / "src"
POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions and links")


def entries(path: Path) -> list[str]:
    """What a later session loads, oldest first."""
    return list(InputHistory(path).load_history_strings())[::-1]


def store(history: InputHistory, *texts: str) -> None:
    """Queue ``texts`` and wait until the writer thread has handled them."""
    for text in texts:
        history.store_string(text)
    assert history.flush(10)


def lines(count: int, prefix: str = "message") -> str:
    return "".join(f"\n# t\n+{prefix} {index}\n" for index in range(count))


def test_entries_round_trip_with_their_lines_and_unicode(tmp_path):
    path = input_history_path(tmp_path)
    texts = [
        "Which region grew fastest?",
        "line one\n  indented two\n\nafter a blank line",
        "+ starts like a marker\n# and a comment line",
        "Unicode λ ✓ and a\rcarriage return",
        "/exec\nprint(1)\n",
    ]
    store(InputHistory(path), *texts)
    assert entries(path) == texts
    assert (tmp_path / ".hailer" / GITIGNORE_NAME).is_file(), "the folder is made as everything in .hailer is"


def test_the_file_is_prompt_toolkits_file_history_format(tmp_path):
    from prompt_toolkit.history import FileHistory

    path = tmp_path / "history"
    store(InputHistory(path), "first", "two\nlines")
    assert list(FileHistory(str(path)).load_history_strings()) == ["two\nlines", "first"]


def test_saving_happens_on_the_writer_thread_not_the_callers(tmp_path, monkeypatch):
    checked_on: list[threading.Thread] = []
    redact = module.redact

    def recording(text: str) -> str:
        checked_on.append(threading.current_thread())
        return redact(text)

    monkeypatch.setattr(module, "redact", recording)
    path = tmp_path / "history"
    history = InputHistory(path)
    started = time.perf_counter()
    history.store_string("a question")
    history.store_string("x" * 5_000_000)  # a pasted 5 MB file: the caller only queues it
    assert time.perf_counter() - started < 0.05
    assert history.flush(10)
    assert entries(path) == ["a question"]
    assert checked_on and threading.current_thread() not in checked_on


@pytest.mark.parametrize(
    "text",
    [
        "   ",
        "use the key sk-abcdefghijklmnopqrstuvwxyz123456",
        "curl -H 'Authorization: Bearer abc.def.ghi' https://example.com",
        "the password is hunter2-correct-horse, keep it",
        "connect to postgres://analyst:s3cret@db.internal:5432/sales",
        "token ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "gho_abcdefghijklmnopqrstuvwxyz0123456789 and ghs_abcdefghijklmnopqrstuvwxyz0123456789",
        "the fine-grained one github_pat_11ABCDEFG0123456789_abcdefghijklmnop",
        "aws key AKIAIOSFODNN7EXAMPLE in the config",
    ],
)
def test_entries_that_look_like_credentials_are_not_written(tmp_path, monkeypatch, text):
    monkeypatch.setenv("WAREHOUSE_PASSWORD", "hunter2-correct-horse")
    path = tmp_path / "history"
    store(InputHistory(path), text, "a normal question about revenue")
    assert entries(path) == ["a normal question about revenue"]
    assert not storable(text)


@pytest.mark.parametrize(
    "text",
    [
        "/exec\nimport os\nprint(os.getcwd())",
        "email the report to analyst@example.com",
        "clone ssh://git@github.com/acme/sales.git and https://docs.pola.rs/api/python/",
        "postgres://localhost:5432/sales has no password",
        "the sk-1 risk segment, tokens by region, the AKIA prefix",
    ],
)
def test_ordinary_messages_are_written(text):
    assert storable(text)


def test_an_entry_over_the_size_limit_stays_in_memory_and_never_pushes_others_out(tmp_path):
    path = tmp_path / "history"
    history = InputHistory(path)
    store(history, *(f"question {index}" for index in range(50)))
    store(history, "a,b,c\n" * 400_000, "after the paste")  # a 2.4 MB pasted CSV
    store(history, "é" * (module.MAX_ENTRY_BYTES // 2 + 1))  # few characters, too many bytes
    assert entries(path) == [*(f"question {index}" for index in range(50)), "after the paste"]
    assert path.stat().st_size < 5_000


def test_an_oversized_entry_an_older_file_holds_is_dropped_and_the_rest_kept(tmp_path):
    path = tmp_path / "history"
    path.write_text(lines(50, "question") + "\n# t\n" + "+a,b,c\n" * 400_000 + lines(1, "last"), encoding="utf-8")
    assert entries(path) == [*(f"question {index}" for index in range(50)), "last 0"]
    assert path.stat().st_size < 5_000, "rewritten without it"


def test_a_file_nothing_can_be_read_from_is_never_rewritten(tmp_path):
    path = tmp_path / "history"
    data = b"\x00\x01 not history\n" * 2_000
    path.write_bytes(data)
    assert entries(path) == []
    assert path.read_bytes() == data


def test_reading_keeps_the_last_thousand_and_trims_the_file(tmp_path):
    path = tmp_path / "history"
    path.write_text(lines(1_050), encoding="utf-8")
    loaded = list(InputHistory(path).load_history_strings())
    assert len(loaded) == module.MAX_ENTRIES == 1_000
    assert loaded[0] == "message 1049" and loaded[-1] == "message 50", "newest first, the oldest 50 dropped"
    assert entries(path) == [f"message {index}" for index in range(50, 1_050)], "the file itself is trimmed"
    store(InputHistory(path), "one more")
    assert entries(path)[-1] == "one more" and len(entries(path)) == 1_000


def test_a_trim_that_fails_still_loads_the_history(tmp_path, monkeypatch):
    path = tmp_path / "history"
    path.write_text(lines(1_010), encoding="utf-8")

    def locked(*_args):
        raise PermissionError("another program has the file open")

    monkeypatch.setattr(module.os, "replace", locked)
    loaded = list(InputHistory(path).load_history_strings())
    assert len(loaded) == 1_000 and loaded[0] == "message 1009"
    assert path.read_text(encoding="utf-8").count("+message") == 1_010, "left as it was"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["history", "history.lock"], "no temporary file left"


def test_only_the_end_of_a_file_larger_than_any_hailer_writes_is_read(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "MAX_READ_BYTES", 200)
    path = tmp_path / "history"
    path.write_text("".join(f"\n# t\n+entry {index} first line\n+entry {index} second line\n" for index in range(40)), encoding="utf-8")
    loaded = entries(path)
    assert loaded and loaded[-1] == "entry 39 first line\nentry 39 second line"
    assert all(text.startswith("entry ") and text.count("\n") == 1 for text in loaded), "no half entries"


def test_a_file_with_windows_line_endings_loads_without_carriage_returns(tmp_path):
    path = tmp_path / "history"
    path.write_bytes(b"\r\n# t\r\n+first line\r\n+second line\r\n\r\n# t\r\n+question\r\n")
    assert entries(path) == ["first line\nsecond line", "question"]


def test_a_corrupt_file_loads_what_it_can(tmp_path):
    path = tmp_path / "history"
    path.write_bytes(b"\x00\xff\xfe garbage\n+kept \xff entry\nnot an entry\n++double\n\n\n+last")
    assert entries(path) == ["kept � entry", "+double", "last"]


@pytest.mark.parametrize("problem", ["parent-is-a-file", "path-is-a-folder"])
def test_read_and_write_failures_never_raise(tmp_path, problem):
    if problem == "parent-is-a-file":
        (tmp_path / "file").write_text("x", encoding="utf-8")
        path = tmp_path / "file" / "history"
    else:
        path = tmp_path / "history"
        path.mkdir()
    history = InputHistory(path)
    store(history, "not stored, and no exception")
    assert list(history.load_history_strings()) == []


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions and a non-root user")
def test_a_read_only_workspace_keeps_history_in_memory(tmp_path):
    folder = tmp_path / ".hailer"
    folder.mkdir()
    folder.chmod(0o500)
    try:
        history = InputHistory(folder / "input_history")
        store(history, "kept in memory only")
        assert list(history.load_history_strings()) == []
    finally:
        folder.chmod(0o700)


@POSIX_ONLY
def test_the_files_are_readable_by_their_owner_only(tmp_path):
    path = input_history_path(tmp_path)
    store(InputHistory(path), "private")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.with_name("input_history.lock").stat().st_mode) == 0o600
    path.write_text(lines(1_001), encoding="utf-8")
    path.chmod(0o644)
    InputHistory(path).load_history_strings()  # trims: the new file is the owner's only
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@POSIX_ONLY
def test_a_planted_temporary_link_is_never_written_through_by_the_trim(tmp_path):
    folder = tmp_path / ".hailer"
    folder.mkdir()
    victim = tmp_path / "victim_bashrc"
    victim.write_text("export PATH=...\n", encoding="utf-8")
    path = folder / "input_history"
    path.write_text(lines(1_001) + "\n# t\n+;echo PWNED\n", encoding="utf-8")
    for name in ("input_history.tmp", ".input_history.tmp"):
        (folder / name).symlink_to(victim)
    assert len(entries(path)) == 1_000
    assert victim.read_text(encoding="utf-8") == "export PATH=...\n"
    assert all((folder / name).is_symlink() for name in ("input_history.tmp", ".input_history.tmp"))


@POSIX_ONLY
@pytest.mark.parametrize("linked", ["input_history", "input_history.lock"])
def test_a_linked_history_or_lock_file_means_memory_only_logged_once(tmp_path, caplog, linked):
    folder = tmp_path / ".hailer"
    folder.mkdir()
    victim = tmp_path / "victim"
    victim.write_text("\n# t\n+keep\n", encoding="utf-8")
    (folder / linked).symlink_to(victim)
    if linked != "input_history":
        (folder / "input_history").write_text(lines(3), encoding="utf-8")
    history = InputHistory(folder / "input_history")
    with caplog.at_level(logging.INFO, logger="hailer.input_history"):
        store(history, "my question", "another")
        assert list(history.load_history_strings()) == []
        store(history, "a third")
    assert victim.read_text(encoding="utf-8") == "\n# t\n+keep\n"
    assert len([record for record in caplog.records if "stays in memory" in record.getMessage()]) == 1


@POSIX_ONLY
def test_a_fifo_in_place_of_the_history_is_refused_without_blocking(tmp_path):
    path = tmp_path / "history"
    os.mkfifo(path)
    history = InputHistory(path)
    store(history, "not into the pipe")
    assert list(history.load_history_strings()) == []


def test_a_short_hold_of_the_lock_delays_the_write_but_loses_nothing(tmp_path):
    assert module.LOCK_WAIT_SECONDS >= 1.0, "another chat's trim may hold the lock for a moment"
    path = tmp_path / "history"
    history = InputHistory(path)
    store(history, "first")
    held, release = threading.Event(), threading.Event()

    def other_chat() -> None:
        with module._locked(history.lock_path):
            held.set()
            release.wait(5)

    holder = threading.Thread(target=other_chat)
    holder.start()
    held.wait(5)
    started = time.perf_counter()
    history.store_string("while another chat holds the lock")
    assert time.perf_counter() - started < 0.05, "the caller never waits for the lock"
    time.sleep(0.3)
    release.set()
    holder.join(5)
    assert history.flush(10)
    assert entries(path) == ["first", "while another chat holds the lock"]


def test_a_lock_held_past_the_wait_drops_that_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "LOCK_WAIT_SECONDS", 0.05)
    path = tmp_path / "history"
    history = InputHistory(path)
    store(history, "first")
    with module._locked(history.lock_path):
        store(history, "while another chat holds the lock")
    assert entries(path) == ["first"]


@pytest.mark.parametrize(
    ("raised", "locked"),
    [(None, True), (errno.EACCES, False), (getattr(errno, "EDEADLOCK", errno.EDEADLK), False),
     (errno.EINVAL, True), (errno.ENOLCK, True), (errno.EBADF, True)],
    ids=["locked", "held", "held-deadlock", "no-locks-einval", "no-locks-enolck", "other-error"],
)
def test_windows_locking_counts_only_the_lock_held_errors_as_held(tmp_path, monkeypatch, raised, locked):
    calls: list[tuple[int, int]] = []

    def locking(descriptor: int, mode: int, nbytes: int) -> None:
        calls.append((mode, nbytes))
        if raised is not None and mode == fake.LK_NBLCK:
            raise OSError(raised, os.strerror(raised))

    fake = SimpleNamespace(LK_NBLCK=2, LK_UNLCK=0, locking=locking)
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    descriptor = os.open(tmp_path / "lock", os.O_RDWR | os.O_CREAT)
    try:
        assert module._windows_try_lock(descriptor) is locked
        module._windows_unlock(descriptor)  # never raises, held or not
    finally:
        os.close(descriptor)
    assert calls[0] == (fake.LK_NBLCK, 1) and calls[-1] == (fake.LK_UNLCK, 1)


def test_two_processes_append_and_trim_without_mixing_or_losing_entries(tmp_path):
    path = tmp_path / "history"
    path.write_text(lines(1_100, "old"), encoding="utf-8")
    writer = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(SRC)!r})
        from hailer.input_history import InputHistory
        history = InputHistory({str(path)!r})
        for index in range(150):
            history.store_string(f"{{sys.argv[1]}} {{index}} first line\\n{{sys.argv[1]}} {{index}} second line")
            if sys.argv[1] == "b" and index % 10 == 0:
                list(history.load_history_strings())  # trims while the other appends
        assert history.flush(60)
    """)
    script = tmp_path / "writer.py"
    script.write_text(writer, encoding="utf-8")
    processes = [subprocess.Popen([sys.executable, str(script), name]) for name in ("a", "b")]
    assert [process.wait(120) for process in processes] == [0, 0]
    loaded = entries(path)
    assert len(loaded) <= 1_000
    for name in ("a", "b"):
        mine = [text for text in loaded if text.startswith(f"{name} ")]
        assert mine == [f"{name} {index} first line\n{name} {index} second line" for index in range(150)]
    assert all(text.startswith("old ") or text.count("\n") == 1 for text in loaded)


def test_exit_writes_what_is_queued_but_never_waits_long_for_a_held_lock(tmp_path):
    path = tmp_path / "history"
    child = textwrap.dedent(f"""
        import atexit, sys, time
        started = []
        atexit.register(lambda: print(f"{{time.monotonic() - started[0]:.2f}}"))  # runs after Hailer's flush
        sys.path.insert(0, {str(SRC)!r})
        from hailer.input_history import InputHistory
        InputHistory({str(path)!r}).store_string(sys.argv[1])
        started.append(time.monotonic())
    """)
    script = tmp_path / "child.py"
    script.write_text(child, encoding="utf-8")
    quick = subprocess.run([sys.executable, str(script), "written at exit"], capture_output=True, text=True, timeout=60)
    assert quick.returncode == 0, quick.stderr
    assert entries(path) == ["written at exit"]
    with module._locked(InputHistory(path).lock_path):
        held = subprocess.run([sys.executable, str(script), "abandoned"], capture_output=True, text=True, timeout=60)
    assert held.returncode == 0, held.stderr
    assert float(held.stdout.split()[-1]) < module.EXIT_FLUSH_SECONDS + 0.5, "exit waited about EXIT_FLUSH_SECONDS"
    assert entries(path) == ["written at exit"]


def test_the_setting_off_means_memory_only(tmp_path):
    assert isinstance(input_history(None), InMemoryHistory)
    history = input_history(tmp_path / "history")
    assert isinstance(history, ThreadedHistory) and isinstance(history.history, InputHistory)
