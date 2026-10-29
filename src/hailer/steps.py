"""What a tool call is in the user's words, and which calls of a turn are still running.

The chat shows each call as one row: a present-tense verb while it runs ("Running code"), then a
past-tense verb with what it acted on and how it went ("Ran code  12 lines · 4 rows  1.5s"). The
raw tool name is in the debug log (``--verbose``) with the call's id, the arguments nowhere outside
the conversation (they can hold data). Pure logic, no terminal: the composer (:mod:`hailer.chat_ui`)
and the plain display (``cli.common._TurnDisplay``) both use it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from hailer.models import AgentEvent
from hailer.text import plural, single_line, url_host


@dataclass(frozen=True)
class Step:
    """A tool call in words: ``live`` while it runs, ``done`` once it has, on ``target`` (may be "")."""

    live: str
    done: str
    target: str = ""


#: tool name -> (live verb, done verb). An unknown tool is shown by its name.
VERBS: dict[str, tuple[str, str]] = {
    "marimo_execute": ("Running code", "Ran code"),
    "marimo_status": ("Checking the kernel", "Checked the kernel"),
    "notebook_cells": ("Reading the notebook", "Read the notebook"),
    "notebook_check": ("Checking the notebook", "Checked the notebook"),
    "notebook_list": ("Listing notebooks", "Listed notebooks"),
    "notebook_create": ("Creating a notebook", "Created a notebook"),
    "notebook_open": ("Opening a notebook", "Opened a notebook"),
    "notebook_close": ("Closing a notebook", "Closed a notebook"),
    "list_periods": ("Scanning the data", "Scanned the data"),
    "load_skill": ("Loading a skill", "Loaded a skill"),
    "read_skill_file": ("Reading a skill file", "Read a skill file"),
    "fetch_page": ("Fetching a page", "Fetched a page"),
    "project_files": ("Listing project files", "Listed project files"),
    "project_search": ("Searching the project", "Searched the project"),
    "project_read": ("Reading a project file", "Read a project file"),
}

# Code-mode calls in a marimo_execute script, and how a script that only makes one kind reads.
_CELL_CALLS = {
    "create_cell(": ("Adding", "Added"),
    "edit_cell(": ("Editing", "Edited"),
    "delete_cell(": ("Deleting", "Deleted"),
    "move_cell(": ("Moving", "Moved"),
}


def _describe_code(code: str) -> Step:
    calls = {verbs: code.count(call) for call, verbs in _CELL_CALLS.items() if call in code}
    if len(calls) > 1:
        return Step("Changing cells", "Changed cells")
    if calls:
        (live, done), count = next(iter(calls.items()))
        cells = "a cell" if count == 1 else f"{count} cells"
        return Step(f"{live} {cells}", f"{done} {cells}")
    lines = sum(1 for line in code.splitlines() if line.strip())
    live, done = VERBS["marimo_execute"]
    return Step(live, done, plural(lines, "line") if lines else "")


def describe_call(name: str, args: Mapping[str, Any] | None = None) -> Step:
    """``name``'s call with ``args`` in words. Missing or odd arguments only lose the target."""
    args = args if isinstance(args, Mapping) else {}
    name = single_line(name)
    if name == "marimo_execute":
        return _describe_code(str(args.get("code") or ""))

    def arg(key: str) -> str:
        value = args.get(key)
        return str(value) if isinstance(value, (str, int, float)) else ""

    targets: dict[str, Callable[[], str]] = {
        "notebook_cells": lambda: arg("pattern"),
        "notebook_create": lambda: arg("name"),
        "notebook_open": lambda: arg("notebook"),
        "notebook_close": lambda: arg("notebook"),
        "list_periods": lambda: arg("name"),
        "load_skill": lambda: arg("name"),
        "read_skill_file": lambda: "/".join(part for part in (arg("name"), arg("path")) if part),
        "fetch_page": lambda: url_host(arg("url"), port=False),
        "project_files": lambda: arg("pattern"),
        "project_search": lambda: arg("query"),
        "project_read": lambda: arg("path") + (f":{arg('start_line') or 1}-{arg('end_line')}" if arg("end_line") not in ("", "0") else ""),
    }
    live, done = VERBS.get(name, (name, name))
    target = targets[name]() if name in targets else ""
    return Step(live, done, single_line(target))


StepStatus = Literal["ok", "failed", "cancelled", "unknown"]

CANCELLED_NOTE = "cancelled, effect unknown"
UNKNOWN_NOTE = "no result, effect unknown"
_JOINERS = " ,;:·-"


@dataclass(frozen=True)
class Receipt:
    """A step that has ended, as one transcript row: ``outcome`` is the tool's summary or a note."""

    status: StepStatus
    label: str
    target: str = ""
    outcome: str = ""
    seconds: float | None = None


@dataclass
class _Running:
    step: Step
    started: float


def _outcome(summary: Any, target: str) -> str:
    """The tool's one-line summary without a leading repeat of the step's target (the row already
    shows it): ``"sales, 6 months"`` on target ``sales`` reads ``"6 months"``."""
    text = single_line(str(summary or ""))
    if target and text.startswith(target) and (len(text) == len(target) or text[len(target)] in _JOINERS):
        text = text[len(target):].lstrip(_JOINERS)
    return text


class StepTracker:
    """The steps of one turn, keyed by the tool-call id of their events.

    The agent makes ids unique in a turn; several calls may be outstanding at once and each
    ``tool_result`` closes the call with its id. A ``tool_call`` without a usable id still gets a
    row, but no result can close it: the turn's end does (:meth:`finish`).
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._running: dict[str, _Running] = {}
        self._serial = 0
        #: Steps started in this turn.
        self.count = 0

    @property
    def running(self) -> bool:
        return bool(self._running)

    def current(self) -> tuple[Step, float, int] | None:
        """(the oldest running step, when it started, how many others run) or None."""
        if not self._running:
            return None
        oldest = next(iter(self._running.values()))
        return oldest.step, oldest.started, len(self._running) - 1

    def call(self, event: AgentEvent) -> None:
        """Start the step of a ``tool_call`` event."""
        detail = event.detail if isinstance(event.detail, Mapping) else {}
        key = detail.get("id")
        if not isinstance(key, str) or not key or key in self._running:
            self._serial += 1
            key = f"#{self._serial}"
        args = detail.get("args")
        self._running[key] = _Running(describe_call(event.text, args if isinstance(args, Mapping) else None), self._clock())
        self.count += 1

    def result(self, event: AgentEvent) -> Receipt | None:
        """Close the step of a ``tool_result`` event; None when no running step has its id (one
        already shown as cancelled, or a result for another turn)."""
        detail = event.detail if isinstance(event.detail, Mapping) else {}
        key = detail.get("id")
        running = self._running.pop(key, None) if isinstance(key, str) else None
        if running is None:
            return None
        seconds = detail.get("seconds")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds < 0:
            seconds = self._clock() - running.started
        step = running.step
        return Receipt(
            "failed" if detail.get("ok") is False else "ok", step.done, step.target,
            _outcome(detail.get("summary"), step.target), float(seconds),
        )

    def cancel(self) -> list[Receipt]:
        """End every running step as cancelled: the call may still finish and change things."""
        return self._close("cancelled", CANCELLED_NOTE)

    def finish(self) -> list[Receipt]:
        """End what is left when the turn is over without its results (an error): effect unknown."""
        return self._close("unknown", UNKNOWN_NOTE)

    def _close(self, status: StepStatus, note: str) -> list[Receipt]:
        now = self._clock()
        closed = [Receipt(status, r.step.done, r.step.target, note, now - r.started) for r in self._running.values()]
        self._running.clear()
        return closed
