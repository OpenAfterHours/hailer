"""The composer's command menu: slash commands, then each command's own arguments.

``/`` at the start of the draft lists the commands with their short descriptions
(:data:`hailer.session.COMMAND_USAGE`). After a command and a space it lists that command's arguments:
``/notebook`` subcommands, then notebook names after ``open`` or ``close``; skill and prompt names;
for ``/model`` the current model and each provider as ``<provider>:``. They come from
:class:`~hailer.models.CompletionData`, which the controller builds from state it already holds:
nothing here asks marimo or the disk. A submission still goes through
:func:`~hailer.session.parse_command`; the menu only writes the draft. :func:`menu_fragments` draws
the menu under the composer's frame.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.utils import get_cwidth

from hailer.models import CompletionData
from hailer.session import COMMAND_USAGE, NOTEBOOK_SUBCOMMANDS
from hailer.steps import single_line
from hailer.transcript import Glyphs

#: What Enter runs at once when the menu writes it: the commands that take no arguments.
RUN_ON_ENTER = frozenset({f"/{name}" for name, (usage, _summary) in COMMAND_USAGE.items() if not usage} | {"/notebook list"})
#: The menu's rows; more completions scroll, with a "N more" line under them.
MENU_ROWS = 6


class SlashCompleter(Completer):
    """Completions for a draft that starts with ``/`` (never for one with a line break)."""

    def __init__(self, data: Callable[[], CompletionData] | None = None) -> None:
        self._data = data

    def data(self) -> CompletionData:
        if self._data is None:
            return CompletionData()
        try:
            data = self._data()
        except Exception:  # noqa: BLE001 - a menu must never break typing
            return CompletionData()
        return data if isinstance(data, CompletionData) else CompletionData()

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/") or "\n" in text:
            return
        head, space, rest = text[1:].partition(" ")
        # A command or subcommand that takes arguments gets its space, so their menu follows, but
        # never when it is typed in full: then the menu steps aside and Enter sends what is typed.
        if not space:
            for name, (usage, summary) in COMMAND_USAGE.items():
                if name.startswith(head.lower()):
                    then = " " if usage and name != head else ""
                    yield Completion(f"/{name}{then}", -len(text), display=f"/{name}", display_meta=summary)
            return
        words = rest.split(" ")
        partial = words[-1]
        for value, meta, then in self._arguments(head.lower(), words[:-1]):
            if value.casefold().startswith(partial.casefold()):
                then = then if value != partial else ""
                yield Completion(value + then, -len(partial), display=single_line(value), display_meta=single_line(meta))

    def _arguments(self, command: str, before: list[str]) -> list[tuple[str, str, str]]:
        """(value, description, what follows it) for the word after ``/command`` and ``before``."""
        if command == "notebook" and not before:
            return [(sub, summary, "" if sub == "list" else " ") for sub, summary in NOTEBOOK_SUBCOMMANDS.items()]
        data = self.data()
        if command == "notebook" and len(before) == 1 and before[0].lower() in ("open", "close"):
            return [(name, "active" if name == data.active_notebook else "", "") for name in data.notebooks]
        if before:
            return []
        if command == "model":
            current = [(data.model, "current", "")] if data.model else []
            return current + [(f"{provider}:", "provider", "") for provider in data.providers]
        if command == "skill":
            return [(name, description, "") for name, description in data.skills]
        if command == "prompt":
            return [(name, "saved prompt", "") for name in data.prompts]
        return []


def menu_open(buffer: Any) -> bool:
    """Whether the buffer has completions to show (the menu is on screen)."""
    state = buffer.complete_state
    return bool(state is not None and state.completions)


def _window(buffer: Any) -> tuple[list[Completion], int | None, int]:
    """(completions, the chosen one's index, the first row shown): the chosen one stays in view."""
    state = buffer.complete_state
    items = list(state.completions) if state is not None else []
    current = state.complete_index if state is not None else None
    start = 0 if current is None else max(0, min(current - MENU_ROWS + 1, len(items) - MENU_ROWS))
    return items, current, start


def menu_height(buffer: Any) -> int:
    count = len(_window(buffer)[0])
    return min(count, MENU_ROWS) + (1 if count > MENU_ROWS else 0)


def menu_fragments(buffer: Any, glyphs: Glyphs, columns: int) -> list[tuple[str, str]]:
    """The menu's rows: a mark on the chosen completion (the first while none is: Enter takes it),
    its name, and its description shortened with an ellipsis to the terminal's width."""
    items, current, start = _window(buffer)
    shown = items[start:start + MENU_ROWS]
    chosen = 0 if current is None else current
    names = [_fit(item.display_text, max(4, columns // 2), glyphs.ellipsis) for item in shown]
    width = max((get_cwidth(name) for name in names), default=0) + 3
    room = columns - 4 - width - 1  # the mark's four cells, the name's column, one cell spare
    out: list[tuple[str, str]] = []
    for index, (item, name) in enumerate(zip(shown, names, strict=True), start):
        picked = index == chosen
        meta = _fit(item.display_meta_text, room, glyphs.ellipsis) if room > 4 else ""
        out += [
            ("class:palette.pick", f"  {glyphs.prompt if picked else ' '} "),
            ("class:palette.name.current" if picked else "class:palette.name", name + " " * (width - get_cwidth(name))),
            ("class:palette.meta", meta),
            ("", "\n"),
        ]
    if len(items) > MENU_ROWS:
        out.append(("class:muted", f"    {len(items) - MENU_ROWS} more"))
    elif out:
        out.pop()
    return out


def _fit(text: str, cells: int, ellipsis: str) -> str:
    """``text`` in at most ``cells`` terminal cells, shortened with ``ellipsis``."""
    if get_cwidth(text) <= cells:
        return text
    out, used, room = "", 0, cells - get_cwidth(ellipsis)
    for char in text:
        used += get_cwidth(char)
        if used > room:
            break
        out += char
    return out.rstrip() + ellipsis
