"""Tests for slash-command parsing and session persistence."""

from __future__ import annotations

import json

from hailer.models import Command, SessionState
from hailer.session import (
    COMMANDS,
    EXIT_COMMANDS,
    help_text,
    load_session,
    parse_command,
    save_session,
    session_path,
)


def test_non_slash_text_is_conversation():
    assert parse_command("compare the latest two periods") is None
    assert parse_command("   plain text with / inside") is None
    assert parse_command("") is None


def test_simple_commands():
    assert parse_command("/help") == Command("help", "")
    assert parse_command("/exit") == Command("exit", "")
    assert parse_command("/quit") == Command("quit", "")
    assert parse_command("  /status  ") == Command("status", "")


def test_command_with_args():
    assert parse_command("/model gpt-5.5") == Command("model", "gpt-5.5")
    assert parse_command("/model internal:foo") == Command("model", "internal:foo")
    assert parse_command("/skill pra101-reconciliation compare the months") == Command(
        "skill", "pra101-reconciliation compare the months"
    )
    assert parse_command("/PROMPT monthly-pack 2025-03") == Command("prompt", "monthly-pack 2025-03")


def test_a_pasted_exec_block_keeps_its_lines():
    """The name ends at the first whitespace, a line break included: /exec code can be pasted."""
    assert parse_command("/exec\nimport polars as pl\nprint(pl.__version__)") == Command(
        "exec", "import polars as pl\nprint(pl.__version__)"
    )
    assert parse_command("/exec for x in range(2):\n    print(x)") == Command("exec", "for x in range(2):\n    print(x)")
    assert parse_command("/exec\n    x = 1\n    print(x)\n") == Command("exec", "    x = 1\n    print(x)"), "indentation kept for dedent"


def test_bare_slash_and_unknown():
    assert parse_command("/") == Command("", "")
    assert parse_command("/wat 1 2") == Command("wat", "1 2")


def test_help_lists_every_command():
    text = help_text()
    for name in COMMANDS:
        assert f"/{name}" in text
    assert all(name in COMMANDS for name in EXIT_COMMANDS)


def test_session_round_trip(tmp_path):
    state = SessionState(thread_id="t-1", model="gpt-5.5", provider="openai", turns=3, input_tokens=10, output_tokens=4)
    save_session(tmp_path, state)
    assert session_path(tmp_path).exists()
    assert load_session(tmp_path) == state


def test_missing_session_is_fresh(tmp_path):
    assert load_session(tmp_path) == SessionState()


def test_corrupt_session_is_fresh(tmp_path):
    path = session_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert load_session(tmp_path) == SessionState()
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert load_session(tmp_path) == SessionState()


def test_partial_or_wrong_typed_fields_are_ignored(tmp_path):
    path = session_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"thread_id": 5, "turns": "many", "model": "m", "output_tokens": -1}), encoding="utf-8")
    state = load_session(tmp_path)
    assert state.thread_id is None
    assert state.turns == 0
    assert state.model == "m"
    assert state.output_tokens == 0


def test_help_text_does_not_repeat_the_command_in_its_description():
    for line in help_text().splitlines():
        if line.strip().startswith("/"):
            name, _, desc = line.strip().partition(" ")
            assert not desc.strip().startswith(name), line
    assert "/model <name>" in COMMANDS["model"]  # usage still shown, after the description


def test_a_session_file_from_an_older_release_still_loads(tmp_path):
    """Releases up to 0.2 stored a prompt fingerprint next to the state; it is ignored and dropped."""
    path = session_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"thread_id": "t", "turns": 3, "prompt_hash": "abc"}), encoding="utf-8")
    state = load_session(tmp_path)
    assert (state.thread_id, state.turns) == ("t", 3)
    save_session(tmp_path, state)
    assert "prompt_hash" not in json.loads(path.read_text(encoding="utf-8"))
