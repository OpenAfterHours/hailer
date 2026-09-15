"""Conversational session state: slash-command parsing and thread persistence.

The CLI owns all printing; this module is pure logic so it is easy to test.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from hailer.models import Command, SessionState

SESSION_DIRNAME = ".hailer"
SESSION_FILENAME = "session.json"
PROMPT_HASH_KEY = "prompt_hash"

# name -> one-line description (order is the order shown by /help)
COMMANDS: dict[str, str] = {
    "help": "Show this help.",
    "status": "Show model, provider, credentials, thread, token usage and marimo status.",
    "new": "Start a new conversation thread (context files are re-read).",
    "model": "Switch model and start a new thread. Usage: /model <name>  or  /model <provider>:<name>",
    "notebook": "Show or switch the active notebook. Usage: /notebook [list | new <name> [--empty] | open <name> | close [name]]",
    "context": "List loaded context files, skills, prompts and the web allowlist.",
    "skill": "Run a turn with a project skill attached. Usage: /skill <name> [message]",
    "prompt": "Send a saved prompt from .config/hailer/prompts. Usage: /prompt <name> [args]",
    "reload": "Re-read .config/hailer; applies to the next thread (/new).",
    "clear": "Clear the screen.",
    "exit": "Exit Hailer.",
    "quit": "Exit Hailer.",
}

EXIT_COMMANDS = ("exit", "quit")


def parse_command(line: str) -> Command | None:
    """Parse a slash command. Returns ``None`` for ordinary conversation text.

    ``"/model gpt-5.5"`` -> ``Command("model", "gpt-5.5")``; ``"/"`` -> ``Command("", "")``.
    Unknown commands are still returned so the CLI can report them.
    """
    text = line.strip()
    if not text.startswith("/"):
        return None
    body = text[1:].strip()
    if not body:
        return Command("", "")
    name, _, rest = body.partition(" ")
    return Command(name.strip().lower(), rest.strip())


def help_text() -> str:
    width = max(len(name) for name in COMMANDS) + 1
    lines = ["Commands:"]
    for name, desc in COMMANDS.items():
        lines.append(f"  /{name.ljust(width)} {desc}")
    lines.append("")
    lines.append("Anything else is sent to the conversation.")
    return "\n".join(lines)


def session_path(workspace: Path) -> Path:
    return Path(workspace) / SESSION_DIRNAME / SESSION_FILENAME


def prompt_hash(text: str) -> str:
    """Stable fingerprint of the system prompt a thread was started with."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_raw(workspace: Path) -> dict:
    path = session_path(workspace)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def load_session(workspace: Path) -> SessionState:
    """Load the saved session; a missing or corrupt file yields a fresh state."""
    raw = _read_raw(workspace)
    state = SessionState()
    for key in ("thread_id", "model", "provider"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            setattr(state, key, value)
    for key in ("turns", "input_tokens", "output_tokens"):
        value = raw.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            setattr(state, key, value)
    return state


def load_prompt_hash(workspace: Path) -> str | None:
    """The prompt fingerprint stored with the session, if any (tolerant of absence)."""
    value = _read_raw(workspace).get(PROMPT_HASH_KEY)
    return value if isinstance(value, str) and value else None


def save_session(workspace: Path, state: SessionState, *, prompt_hash: str | None = None) -> None:
    """Persist ``state``. ``prompt_hash`` is stored when given, otherwise the existing one is kept."""
    path = session_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    stored = prompt_hash if prompt_hash is not None else load_prompt_hash(workspace)
    if stored:
        payload[PROMPT_HASH_KEY] = stored
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
