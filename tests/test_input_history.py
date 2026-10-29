"""The input history file (.hailer/input_history): format, cap, secrets, failures, two processes."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from prompt_toolkit.history import InMemoryHistory, ThreadedHistory

from hailer import input_history as module
from hailer.input_history import InputHistory, input_history, storable
from hailer.statedir import GITIGNORE_NAME, input_history_path

SRC = Path(__file__).resolve().parents[1] / "src"


def entries(path: Path) -> list[str]:
    """What a later session loads, oldest first."""
    return list(InputHistory(path).load_history_strings())[::-1]


def test_entries_round_trip_with_their_lines_and_unicode(tmp_path):
    path = input_history_path(tmp_path)
    history = InputHistory(path)
    texts = [
        "Which region grew fastest?",
        "line one\n  indented two\n\nafter a blank line",
        "+ starts like a marker\n# and a comment line",
        "Unicode λ ✓ and a\rcarriage return",
        "/exec\nprint(1)\n",
    ]
    for text in texts:
        history.store_string(text)
    assert entries(path) == texts
    assert (tmp_path / ".hailer" / GITIGNORE_NAME).is_file(), "the folder is made as everything in .hailer is"


def test_the_file_is_prompt_toolkits_file_history_format(tmp_path):
    from prompt_toolkit.history import FileHistory

    path = tmp_path / "history"
    InputHistory(path).store_string("first")
    InputHistory(path).store_string("two\nlines")
    assert list(FileHistory(str(path)).load_history_strings()) == ["two\nlines", "first"]


def test_blank_entries_and_entries_that_look_like_secrets_are_not_written(tmp_path, monkeypatch):
    monkeypatch.setenv("WAREHOUSE_PASSWORD", "hunter2-correct-horse")
    path = tmp_path / "history"
    history = InputHistory(path)
    for text in (
        "   ",
        "use the key sk-abcdefghijklmnopqrstuvwxyz123456",
        "curl -H 'Authorization: Bearer abc.def.ghi' https://example.com",
        "the password is hunter2-correct-horse, keep it",
        "a normal question about revenue",
    ):
        history.store_string(text)
    assert entries(path) == ["a normal question about revenue"]
    assert "hunter2" not in path.read_text(encoding="utf-8")
    assert not storable("Bearer abcdef0123") and storable("the sk-1 risk segment, tokens by region")


def test_reading_keeps_the_last_thousand_and_trims_the_file(tmp_path):
    path = tmp_path / "history"
    path.write_text("".join(f"\n# t\n+message {index}\n" for index in range(1_050)), encoding="utf-8")
    loaded = list(InputHistory(path).load_history_strings())
    assert len(loaded) == module.MAX_ENTRIES == 1_000
    assert loaded[0] == "message 1049" and loaded[-1] == "message 50", "newest first, the oldest 50 dropped"
    assert entries(path) == [f"message {index}" for index in range(50, 1_050)], "the file itself is trimmed"
    InputHistory(path).store_string("one more")
    assert entries(path)[-1] == "one more" and len(entries(path)) == 1_000


def test_a_trim_that_fails_still_loads_the_history(tmp_path, monkeypatch):
    path = tmp_path / "history"
    path.write_text("".join(f"\n# t\n+message {index}\n" for index in range(1_010)), encoding="utf-8")

    def locked(*_args):
        raise PermissionError("another program has the file open")

    monkeypatch.setattr(module.os, "replace", locked)
    loaded = list(InputHistory(path).load_history_strings())
    assert len(loaded) == 1_000 and loaded[0] == "message 1009"
    assert path.read_text(encoding="utf-8").count("+message") == 1_010, "left as it was"
    assert not (tmp_path / "history.tmp").exists()


def test_only_the_end_of_a_huge_file_is_read_and_a_cut_entry_is_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "MAX_READ_BYTES", 200)
    path = tmp_path / "history"
    path.write_text("".join(f"\n# t\n+entry {index} first line\n+entry {index} second line\n" for index in range(40)), encoding="utf-8")
    loaded = entries(path)
    assert loaded and loaded[-1] == "entry 39 first line\nentry 39 second line"
    assert all(text.startswith("entry ") and text.count("\n") == 1 for text in loaded), "no half entries"
    assert len(path.read_bytes()) <= 200, "rewritten to what was kept"


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
    history.store_string("not stored, and no exception")
    assert list(history.load_history_strings()) == []


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions and a non-root user")
def test_a_read_only_workspace_keeps_history_in_memory(tmp_path):
    folder = tmp_path / ".hailer"
    folder.mkdir()
    folder.chmod(0o500)
    try:
        history = InputHistory(folder / "input_history")
        history.store_string("kept in memory only")
        assert list(history.load_history_strings()) == []
    finally:
        folder.chmod(0o700)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_the_files_are_readable_by_their_owner_only(tmp_path):
    path = input_history_path(tmp_path)
    InputHistory(path).store_string("private")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.with_name("input_history.lock").stat().st_mode) == 0o600
    path.write_text("".join(f"\n# t\n+m{index}\n" for index in range(1_001)), encoding="utf-8")
    path.chmod(0o644)
    InputHistory(path).load_history_strings()  # trims: the new file is the owner's only
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_a_held_lock_skips_the_write_instead_of_waiting(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "LOCK_WAIT_SECONDS", 0.05)
    path = tmp_path / "history"
    history = InputHistory(path)
    history.store_string("first")
    with module._locked(history.lock_path):
        history.store_string("while another chat holds the lock")
        assert list(history.load_history_strings()) == []
    assert entries(path) == ["first"]


def test_two_processes_append_and_trim_without_mixing_entries(tmp_path):
    path = tmp_path / "history"
    path.write_text("".join(f"\n# t\n+old {index}\n" for index in range(1_100)), encoding="utf-8")
    writer = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(SRC)!r})
        from hailer import input_history
        input_history.LOCK_WAIT_SECONDS = 30  # a slow disk must not turn a wait into a skipped write here
        history = input_history.InputHistory({str(path)!r})
        for index in range(150):
            history.store_string(f"{{sys.argv[1]}} {{index}} first line\\n{{sys.argv[1]}} {{index}} second line")
            if sys.argv[1] == "b" and index % 10 == 0:
                list(history.load_history_strings())  # trims while the other appends
    """)
    script = tmp_path / "writer.py"
    script.write_text(writer, encoding="utf-8")
    processes = [subprocess.Popen([sys.executable, str(script), name]) for name in ("a", "b")]
    assert [process.wait(60) for process in processes] == [0, 0]
    loaded = entries(path)
    assert len(loaded) <= 1_000
    for name in ("a", "b"):
        mine = [text for text in loaded if text.startswith(f"{name} ")]
        assert mine == [f"{name} {index} first line\n{name} {index} second line" for index in range(150)]
    assert all(text.startswith("old ") or text.count("\n") == 1 for text in loaded)


def test_the_setting_off_means_memory_only(tmp_path):
    assert isinstance(input_history(None), InMemoryHistory)
    history = input_history(tmp_path / "history")
    assert isinstance(history, ThreadedHistory) and isinstance(history.history, InputHistory)
