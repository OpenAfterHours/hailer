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
