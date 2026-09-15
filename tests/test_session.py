"""Tests for slash-command parsing and session persistence."""

from __future__ import annotations

import json

from hailer.models import Command, SessionState
from hailer.session import (
    COMMANDS,
    EXIT_COMMANDS,
    help_text,
    load_prompt_hash,
    load_session,
    parse_command,
    prompt_hash,
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


def test_help_text_does_not_repeat_the_command_in_its_description():
    for line in help_text().splitlines():
        if line.strip().startswith("/"):
            name, _, desc = line.strip().partition(" ")
            assert not desc.strip().startswith(name), line
    assert "/model <name>" in COMMANDS["model"]  # usage still shown, after the description


def test_prompt_hash_is_stable_and_persisted(tmp_path):
    h1 = prompt_hash("You are Hailer.")
    assert h1 == prompt_hash("You are Hailer.") and h1 != prompt_hash("You are Hailer!")
    assert load_prompt_hash(tmp_path) is None
    save_session(tmp_path, SessionState(thread_id="t-1"), prompt_hash=h1)
    assert load_prompt_hash(tmp_path) == h1
    # saving without a hash keeps the stored one; state fields still round-trip
    save_session(tmp_path, SessionState(thread_id="t-1", turns=2))
    assert load_prompt_hash(tmp_path) == h1
    assert load_session(tmp_path).turns == 2
    # an explicit empty hash clears it
    save_session(tmp_path, SessionState(thread_id="t-2"), prompt_hash="")
    assert load_prompt_hash(tmp_path) is None


def test_prompt_hash_tolerates_absence_and_garbage(tmp_path):
    path = session_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"thread_id": "t", "prompt_hash": 42}), encoding="utf-8")
    assert load_prompt_hash(tmp_path) is None
    assert load_session(tmp_path).thread_id == "t"
