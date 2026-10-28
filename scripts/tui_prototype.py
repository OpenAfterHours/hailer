"""A runnable prototype of a more refined chat terminal, built on the real composer.

Nothing here is wired into ``uvx hailer``. Turns are simulated (no model, API key, kernel or
browser), so the look and feel can be tried before any of it moves into ``hailer.chat_ui``.
``PrototypeUI`` subclasses :class:`hailer.chat_ui.ChatUI`: the transcript writer, paste mode,
startup gating and cancellation are the real ones; only the layout and what is printed change.

    uv run python scripts/tui_prototype.py
    uv run python scripts/tui_prototype.py --unsafe-local --ascii

Try: ask anything (a simulated analysis of the sample sales data), type / for the command
menu, Tab after "/notebook open ", Up for history (Right accepts the grey suggestion),
/status, /help, /notebook list, /fail, Esc or Ctrl+C during a turn, /exit.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar
from urllib.parse import urlparse

from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, VSplit, Window, WindowAlign
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.processors import Processor, Transformation, TransformationInput
from prompt_toolkit.styles import Style
from rich import box
from rich.cells import cell_len
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.markdown import CodeBlock, Heading, Markdown, TableElement
from rich.panel import Panel
from rich.segment import Segment
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from hailer.chat_ui import ChatUI
from hailer.models import AgentEvent
from hailer.session import COMMANDS, EXIT_COMMANDS, parse_command

# --------------------------------------------------------------------------- #
# Glyphs and colours
# --------------------------------------------------------------------------- #

UNICODE = {
    "user": "▌", "reply": "●", "ok": "✓", "fail": "✗", "warn": "▲", "resume": "↺",
    "on": "●", "off": "○", "sep": "·", "link": "↗", "prompt": "›", "rule": "─", "bar": "│",
    "corners": "╭╮╰╯", "spinner": "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", "brand": "◆", "up_down": "↑↓", "up": "↑",
}
ASCII = {
    "user": "|", "reply": "*", "ok": "+", "fail": "x", "warn": "!", "resume": "~",
    "on": "*", "off": "o", "sep": "-", "link": "->", "prompt": ">", "rule": "-", "bar": "|",
    "corners": "++++", "spinner": "|/-\\", "brand": "#", "up_down": "up/down", "up": "up",
}
G = UNICODE


def _unicode_ok() -> bool:
    """A cp1252 console (Windows pipes, older terminals) gets the ASCII set, never mojibake."""
    try:
        "".join(UNICODE.values()).encode(sys.stdout.encoding or "ascii")
    except (UnicodeEncodeError, LookupError):
        return False
    return True


# ANSI names only, so the user's terminal theme (light or dark) decides the actual colours.
THEME = Theme({
    "accent": "cyan",
    "muted": "dim",
    "ok": "green",
    "warn": "yellow",
    "fail": "red",
    "markdown.h1": "bold",
    "markdown.h2": "bold",
    "markdown.h3": "bold",
    "markdown.h4": "bold",
    "markdown.code": "cyan",
    "markdown.link": "cyan underline",
    "markdown.link_url": "dim",
    "markdown.item.bullet": "dim",
    "markdown.item.number": "dim",
    "markdown.block_quote": "italic",
    "markdown.hr": "dim",
    "markdown.table.header": "bold",
    "markdown.table.border": "dim",
})

STYLE = Style.from_dict({
    "frame.idle": "ansicyan",
    "frame.busy": "ansibrightblack",
    "frame.failed": "ansired",
    "prompt.idle": "ansicyan bold",
    "prompt.busy": "ansibrightblack",
    "placeholder": "ansibrightblack",
    "auto-suggestion": "ansibrightblack",
    "spinner": "ansicyan",
    "activity": "",
    "muted": "ansibrightblack",
    "hint": "ansibrightblack",
    "hint.flash": "ansiyellow",
    "ok": "ansigreen",
    "warn": "ansiyellow",
    "palette.pick": "ansicyan bold",
    "palette.name": "",
    "palette.name.current": "ansicyan bold",
    "palette.meta": "ansibrightblack",
})


def _seconds(value: float) -> str:
    if value < 10:
        return f"{value:.1f}s"
    if value < 60:
        return f"{value:.0f}s"
    minutes, rest = divmod(int(value), 60)
    return f"{minutes}m {rest:02d}s"


def _tokens(value: int) -> str:
    return f"{value / 1000:.1f}k" if value >= 1000 else str(value)


def _join(*parts: str) -> str:
    return f" {G['sep']} ".join(part for part in parts if part)


# --------------------------------------------------------------------------- #
# Transcript pieces (Rich renderables committed to native scrollback)
# --------------------------------------------------------------------------- #


class Gutter:
    """``body`` rendered narrower, with ``first`` before its first line and ``rest`` before the others."""

    def __init__(self, body: RenderableType, first: str, rest: str | None = None, style: str = "") -> None:
        self.body, self.first, self.style = body, first, style
        self.rest = " " * cell_len(first) if rest is None else rest

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = max(10, options.max_width - cell_len(self.first))
        lines = console.render_lines(self.body, options.update_width(width), pad=False)
        style = console.get_style(self.style) if self.style else None
        for index, line in enumerate(lines):
            yield Segment(self.first if index == 0 else self.rest, style)
            yield from line
            yield Segment.line()


class _Heading(Heading):
    LEVEL_ALIGN: ClassVar[dict[str, str]] = {}  # Rich centres h1/h2; a reply reads left to right


class _Code(CodeBlock):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        code = str(self.text).rstrip()
        syntax = Syntax(code, self.lexer_name, theme="ansi_dark", word_wrap=True, background_color="default")
        if self.lexer_name not in ("", "text", "default"):
            yield Text(self.lexer_name, style="muted")
        yield Gutter(syntax, f"{G['bar']} ", style="muted")


ASCII_SIMPLE_HEAD = box.Box("    \n    \n -- \n    \n    \n    \n    \n    \n", ascii=True)


class _Table(TableElement):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        table = Table(
            box=box.SIMPLE_HEAD if G is UNICODE else ASCII_SIMPLE_HEAD, show_edge=False, pad_edge=False,
            border_style="markdown.table.border", header_style="markdown.table.header",
        )
        if self.header is not None and self.header.row is not None:
            for cell in self.header.row.cells:
                table.add_column(cell.content.copy(), justify=cell.content.justify or "left")
        if self.body is not None:
            for row in self.body.rows:
                table.add_row(*(cell.content for cell in row.cells))
        yield table


class ReplyMarkdown(Markdown):
    elements: ClassVar[dict[str, type[Any]]] = {**Markdown.elements, "heading_open": _Heading, "fence": _Code, "code_block": _Code, "table_open": _Table}


def user_message(text: str) -> RenderableType:
    return Gutter(Text(text), f"{G['user']} ", f"{G['user']} ", style="accent")


def reply(markdown: str) -> RenderableType:
    body = ReplyMarkdown(markdown) if markdown.strip() else Text("(no reply)", style="muted")
    return Gutter(body, f"{G['reply']} ", style="accent")


def row(icon: str, icon_style: str, label: str, detail: str = "", seconds: float | None = None, *, label_width: int = 20) -> Table:
    """One aligned line: a status icon, what happened, on what, how long. Durations line up
    across separately printed rows because every column has a fixed width."""
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(width=3, no_wrap=True)
    grid.add_column(width=label_width, no_wrap=True)
    grid.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    grid.add_column(width=7, justify="right", no_wrap=True)
    grid.add_row(
        Text(f"  {icon}", style=icon_style),
        Text(label, style=icon_style if icon_style in ("warn", "fail") else ""),
        Text(detail, style="muted"),
        Text(_seconds(seconds) if seconds is not None else "", style="muted"),
    )
    return grid


def callout(icon: str, style: str, message: str, hint: str | Text = "") -> RenderableType:
    """A problem in one line, then what to do about it."""
    body = Text(message, style=style)
    if hint:
        body.append("\n")
        body.append(hint if isinstance(hint, Text) else Text(hint))
    return Gutter(body, f"  {icon} ", style=style)


# What a tool call is in the user's words, while it runs and once it is done. The raw tool name
# and arguments stay available to --verbose and the log.
VERBS = {
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
}


def describe(tool: str, args: dict[str, Any]) -> tuple[str, str, str]:
    """(live verb, done verb, target) for a tool call."""
    live, done = VERBS.get(tool, (tool, tool))
    if tool == "marimo_execute":
        code = str(args.get("code", ""))
        lines = sum(1 for line in code.splitlines() if line.strip())
        if "create_cell" in code:
            return "Adding a cell", "Added a cell", ""
        if "edit_cell" in code:
            return "Editing a cell", "Edited a cell", ""
        return live, done, f"{lines} lines"
    target = {
        "notebook_create": args.get("name", ""),
        "notebook_open": args.get("notebook", ""),
        "notebook_close": args.get("notebook", ""),
        "list_periods": args.get("name", ""),
        "load_skill": args.get("name", ""),
        "read_skill_file": f"{args.get('name', '')}/{args.get('path', '')}",
        "fetch_page": urlparse(str(args.get("url", ""))).hostname or "",
        "notebook_cells": args.get("notebook", ""),
    }.get(tool, "")
    return live, done, str(target or "")


# --------------------------------------------------------------------------- #
# The composer
# --------------------------------------------------------------------------- #


@dataclass
class SessionView:
    """What the footer and the menus show; in Hailer this comes from the controller."""

    notebook: str = "analysis.py"
    notebook_open: bool = False
    notebooks: list[str] = field(default_factory=lambda: ["analysis.py", "q2-review.py", "returns.py"])
    model: str = "gpt-5.5"
    provider: str = "openai"
    models: list[str] = field(default_factory=lambda: ["gpt-5.5", "gpt-5.5-mini", "internal:llama-4-70b"])
    skills: list[str] = field(default_factory=lambda: ["example-skill"])
    prompts: list[str] = field(default_factory=lambda: ["first-look"])
    unsafe_local: bool = False
    turns: int = 3
    tokens: int = 14_820
    started: float = field(default_factory=time.monotonic)


# Short forms for the menu and /help; in Hailer they would sit next to session.COMMANDS.
SHORT = {
    "help": ("", "Commands and keys"),
    "status": ("", "Model, kernel, notebook and usage"),
    "new": ("", "Start a new conversation"),
    "model": ("<name>", "Switch model; starts a new conversation"),
    "notebook": ("[list|new|open|close]", "List, create, open or close notebooks"),
    "exec": ("<code>", "Run Python in the notebook's kernel; the model does not see it"),
    "context": ("", "Context files, skills and prompts"),
    "skill": ("<name> [message]", "Run a turn with a project skill"),
    "prompt": ("<name> [args]", "Send a saved prompt"),
    "reload": ("", "Re-read .config/hailer"),
    "clear": ("", "Clear the screen"),
    "exit": ("", "Leave Hailer"),
}
assert set(SHORT) == set(COMMANDS) - {"quit"}
MENU_COMMANDS = [(name, SHORT[name][1]) for name in COMMANDS if name != "quit"]
TAKES_ARGUMENTS = {"model", "notebook", "exec", "skill", "prompt"}
RUN_ON_ENTER = {"/help", "/status", "/new", "/context", "/reload", "/clear", "/exit", "/notebook list"}
MENU_ROWS = 6


class SlashCompleter(Completer):
    """Commands after "/", then each command's own arguments (notebooks, skills, models)."""

    def __init__(self, view: SessionView) -> None:
        self.view = view

    def get_completions(self, document: Any, complete_event: Any) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/") or "\n" in text:
            return
        head, space, rest = text[1:].partition(" ")
        if not space:
            for name, summary in MENU_COMMANDS:
                if name.startswith(head.lower()):
                    suffix = " " if name in TAKES_ARGUMENTS else ""
                    yield Completion(f"/{name}{suffix}", -len(text), display=f"/{name}", display_meta=summary)
            return
        words = rest.split(" ")
        if head == "notebook" and len(words) == 1:
            options = [("list", "Every notebook, with its state"), ("new", "Create one from the starter template"),
                       ("open", "Switch the active notebook"), ("close", "Free a notebook's kernel session")]
        elif head == "notebook" and len(words) == 2 and words[0] in ("open", "close"):
            options = [(name, "active" if name == self.view.notebook else "") for name in self.view.notebooks]
        elif head == "model" and len(words) == 1:
            options = [(name, "current" if name == self.view.model else "") for name in self.view.models]
        elif head == "skill" and len(words) == 1:
            options = [(name, "project skill") for name in self.view.skills]
        elif head == "prompt" and len(words) == 1:
            options = [(name, "saved prompt") for name in self.view.prompts]
        else:
            return
        partial = words[-1]
        for value, meta in options:
            if value.startswith(partial):
                yield Completion(value + (" " if head == "notebook" and value in ("open", "close", "new") else ""),
                                 -len(partial), display=value, display_meta=meta)


class _Placeholder(Processor):
    def __init__(self, text: Callable[[], str]) -> None:
        self.text = text

    def apply_transformation(self, ti: TransformationInput) -> Transformation:
        if ti.document.text or ti.lineno != 0:
            return Transformation(ti.fragments)
        return Transformation([*ti.fragments, ("class:placeholder", self.text())])


class PrototypeUI(ChatUI):
    """ChatUI with a rounded composer, a live activity line, a command menu and a footer."""

    def __init__(self, console: Console, on_submit: Callable[[str], Awaitable[None]], view: SessionView, **kwargs: Any) -> None:
        super().__init__(console, on_submit, lambda: "", **kwargs)
        self.view = view
        self._live: tuple[str, str, int] = ("", "", 20)  # label, detail, label width
        self._live_since = self._since = time.monotonic()
        self._flash: tuple[str, float] | None = None
        self._step: tuple[str, str, str, float] | None = None

        self.buffer.completer = SlashCompleter(view)
        self.buffer.complete_while_typing = Condition(lambda: self.buffer.text.startswith("/"))
        self.buffer.auto_suggest = AutoSuggestFromHistory()
        for past in ("Which region grew fastest between Q1 and Q2?", "Chart weekly revenue by channel"):
            self.buffer.history.append_string(past)
        self.composer.window.get_line_prefix = self._line_prefix
        self.composer.control.input_processors.append(_Placeholder(self._placeholder))

        self.application.layout = Layout(self._container(), focused_element=self.composer)
        self.application.style = STYLE
        self.application.key_bindings = merge_key_bindings([self.application.key_bindings, self._menu_bindings()])
        self.application.timeoutlen = 0.25  # Esc alone cancels quickly; Alt+Enter arrives as one sequence

    # -- layout ------------------------------------------------------------ #

    def _container(self) -> HSplit:
        tl, tr, bl, br = G["corners"]
        rule, bar = G["rule"], G["bar"]
        border = self._border_style
        top = VSplit([
            Window(width=1, height=1, char=tl, style=border),
            Window(height=1, char=rule, style=border),
            Window(width=1, height=1, char=tr, style=border),
        ])
        middle = VSplit([
            Window(width=1, char=bar, style=border), Window(width=1),
            self.composer,
            Window(width=1), Window(width=1, char=bar, style=border),
        ])
        bottom = VSplit([
            Window(width=1, height=1, char=bl, style=border),
            Window(width=1, height=1, char=rule, style=border),
            Window(FormattedTextControl(self._hints), height=1, dont_extend_width=True),
            Window(height=1, char=rule, style=border),
            Window(width=1, height=1, char=br, style=border),
        ])
        menu = ConditionalContainer(
            Window(FormattedTextControl(self._menu), height=self._menu_height),
            filter=Condition(self._menu_visible),
        )
        footer = Window(FormattedTextControl(self._footer), height=1)
        # The live row lines up with the step rows above it, so a running step turns into its
        # finished row in place. It takes no space while idle.
        activity = ConditionalContainer(
            VSplit([
                Window(FormattedTextControl(self._activity), height=1),
                Window(FormattedTextControl(self._activity_elapsed), height=1, width=8, align=WindowAlign.RIGHT),
            ]),
            filter=Condition(lambda: (self.busy or self._preparing) and bool(self._live[0])),
        )
        return HSplit([activity, top, middle, bottom, menu, footer])

    def _state(self) -> str:
        if self._startup_failed:
            return "failed"
        return "busy" if self.busy or self._preparing else "idle"

    def _border_style(self) -> str:
        return f"class:frame.{self._state()}"

    def _line_prefix(self, line: int, wrap: int) -> list[tuple[str, str]]:
        if line == 0 and wrap == 0:
            return [(f"class:prompt.{'busy' if self._state() != 'idle' else 'idle'}", f"{G['prompt']} ")]
        return [("", "  ")]

    def _placeholder(self) -> str:
        if self._preparing:
            return "Ask about your data; your first message waits for startup"
        if self.busy:
            return "Draft your next message"
        return "Ask about your data, or type / for commands"

    def _activity(self) -> list[tuple[str, str]]:
        label, detail, width = self._live
        frames = G["spinner"]
        frame = frames[int(time.monotonic() / 0.08) % len(frames)]
        return [("class:spinner", f"  {frame} "), ("class:activity", label.ljust(width)), ("class:muted", f" {detail}")]

    def _activity_elapsed(self) -> list[tuple[str, str]]:
        elapsed = time.monotonic() - self._live_since
        return [("class:muted", _seconds(elapsed))] if elapsed >= 1 else []

    def _hints(self) -> list[tuple[str, str]]:
        sep = f" {G['sep']} "
        if self._flash is not None and time.monotonic() < self._flash[1]:
            return [("class:hint.flash", f" {self._flash[0]} ")]
        if self._menu_visible():
            text = sep.join([f"{G['up_down']} choose", "tab complete", "enter run", "esc close"])
        elif self._cancelling:
            text = "cancelling"
        elif self.busy and self._executing:
            text = sep.join(["esc cancel", "your draft is kept"])
        elif self._preparing:
            text = sep.join(["starting", "you can type now"])
        elif self._startup_failed:
            text = sep.join(["startup failed", "/exit to leave"])
        elif self.buffer.suggestion is not None and self.buffer.suggestion.text:
            text = sep.join(["→ accept" if G is UNICODE else "right accept", "enter send", "alt+enter new line"])
        elif self.buffer.text:
            text = sep.join(["enter send", "alt+enter new line"])
        else:
            text = sep.join(["/ commands", f"{G['up']} history", "ctrl+c exit"])
        return [("class:hint", f" {text} ")]

    def _footer(self) -> list[tuple[str, str]]:
        """Notebook, kernel and model on the left, usage on the right. When the terminal is too
        narrow the least important parts go first, never the kernel warning."""
        view = self.view
        dot = ("class:ok", G["on"]) if view.notebook_open else ("class:muted", G["off"])
        kernel = (("class:warn", f"{G['warn']} unsafe-local {G['sep']} not isolated") if view.unsafe_local
                  else ("class:muted", f"docker {G['sep']} isolated"))
        groups = [[dot, ("", f" {view.notebook}")], [kernel], [("class:muted", view.model)]]
        if self.busy and self._executing:
            right = [f"working {_seconds(time.monotonic() - self._since)}"]
        else:
            right = [f"{view.turns} turns", f"{_tokens(view.tokens)} tokens"]
        width = self.application.output.get_size().columns - 2
        sep = ("class:muted", f"  {G['sep']}  ")

        def length(fragments: list[tuple[str, str]]) -> int:
            return sum(len(text) for _style, text in fragments)

        while True:
            left = [part for index, group in enumerate(groups) for part in ([sep] if index else []) + group]
            tail = _join(*right)
            if length(left) + len(tail) + 2 <= width or (len(groups) == 2 and not right):
                break
            if right:
                right.pop()
            else:
                groups.pop()
        gap = max(2, width - length(left) - len(tail))
        return [("", " "), *left, ("", " " * gap), ("class:muted", tail)]

    # -- the command menu --------------------------------------------------- #

    def _menu_visible(self) -> bool:
        state = self.buffer.complete_state
        return bool(state and state.completions)

    def _menu_window(self) -> tuple[list[Completion], int | None, int]:
        state = self.buffer.complete_state
        items = list(state.completions) if state else []
        current = state.complete_index if state else None
        start = 0 if current is None else max(0, min(current - MENU_ROWS + 1, len(items) - MENU_ROWS))
        return items, current, start

    def _menu_height(self) -> int:
        items, _current, _start = self._menu_window()
        return min(len(items), MENU_ROWS) + (1 if len(items) > MENU_ROWS else 0)

    def _menu(self) -> list[tuple[str, str]]:
        items, current, start = self._menu_window()
        shown = items[start:start + MENU_ROWS]
        width = max((len(c.display_text) for c in shown), default=0) + 3
        room = self.application.output.get_size().columns - width - 5
        out: list[tuple[str, str]] = []
        for index, completion in enumerate(shown, start):
            picked = index == (0 if current is None else current)  # Enter takes the first one
            meta = completion.display_meta_text
            if len(meta) > room:
                meta = meta[: max(0, room - 1)].rstrip() + "…"
            out += [
                ("class:palette.pick", f"  {G['prompt'] if picked else ' '} "),
                ("class:palette.name.current" if picked else "class:palette.name", completion.display_text.ljust(width)),
                ("class:palette.meta", meta),
                ("", "\n"),
            ]
        if len(items) > MENU_ROWS:
            out.append(("class:muted", f"    {len(items) - MENU_ROWS} more"))
        elif out:
            out.pop()
        return out

    def _menu_bindings(self) -> KeyBindings:
        bindings = KeyBindings()
        menu = Condition(self._menu_visible)

        @bindings.add("enter", filter=menu)
        def accept(_event: Any) -> None:
            state = self.buffer.complete_state
            assert state is not None
            self.buffer.apply_completion(state.current_completion or state.completions[0])
            if self.buffer.text.strip() in RUN_ON_ENTER:
                self.buffer.cancel_completion()
                self.submit()

        @bindings.add("tab", filter=menu)
        def next_item(_event: Any) -> None:
            self.buffer.complete_next()

        @bindings.add("s-tab", filter=menu)
        def previous_item(_event: Any) -> None:
            self.buffer.complete_previous()

        @bindings.add("tab", filter=~menu)
        def open_menu(_event: Any) -> None:
            self.buffer.start_completion(select_first=False)

        @bindings.add("escape", filter=menu)
        def close_menu(_event: Any) -> None:
            self.buffer.cancel_completion()

        @bindings.add("escape", filter=~menu & Condition(lambda: self.busy and self._executing))
        def cancel_turn(_event: Any) -> None:
            self.cancel()

        return bindings

    # -- behaviour ----------------------------------------------------------- #

    def live(self, label: str, detail: str = "", width: int = 20) -> None:
        """What the live row shows: the step in progress, in the same columns as finished steps."""

        def update() -> None:
            self._live = (label, detail, width)
            self._live_since = time.monotonic()
            self.application.invalidate()

        self._on_ui_thread(update)

    def set_activity(self, text: str) -> None:
        # ChatUI's own states, in the live row's words.
        named = {"Ready": "", "Working...": "Working", "Thinking...": "Thinking",
                 "Preparing chat...": "Starting", "Cancelling...": "Cancelling"}
        self.live(named.get(text, text))

    def flash(self, text: str, seconds: float = 2.5) -> None:
        self._flash = (text, time.monotonic() + seconds)
        self.application.invalidate()

    def submit(self) -> None:
        """ChatUI.submit with the new echo and busy hints (a real change would add one echo hook)."""
        text = self.buffer.text.strip()
        command = parse_command(text)
        if (self._preparing or self._startup_failed) and (
            text.startswith("\x1a") or command is not None and command.name in EXIT_COMMANDS
        ):
            self.console.print("Bye.", markup=False)
            self.exit()
            return
        if self.busy:
            self.flash("Starting up · your first message is queued" if self._preparing
                       else _join("Still working", "esc cancels", "your draft is kept"))
            return
        if self._startup_failed:
            self.flash("Startup failed · your draft is kept · /exit to leave")
            return
        if text.startswith("\x1a"):
            self.console.print("Bye.", markup=False)
            self.exit()
            return
        if not text:
            return
        self.buffer.reset(append_to_history=True)
        self.console.print(user_message(text))
        self.console.print()
        if self._preparing:
            self._pending_submission = text
            self.flash("Queued · it sends once startup finishes")
            return
        self._launch_submission(text)

    def _launch_submission(self, text: str) -> None:
        self._since = time.monotonic()
        super()._launch_submission(text)

    async def _submit(self, text: str) -> None:
        try:
            await self._on_submit(text)
        except asyncio.CancelledError:
            self.console.print(callout(
                G["warn"], "warn", "Cancelled.",
                "The conversation continues; the model is told the last step's effect is unknown.",
            ))
            self.console.print()
        except Exception as exc:  # noqa: BLE001 - keep the composer usable, as ChatUI does
            self.console.print(callout(G["fail"], "fail", f"Unexpected error: {type(exc).__name__}: {exc}"))
        finally:
            self._executing = False
            await self.flush()
            self._cancelling = False
            self.set_activity("Ready")
            if self._exit_requested:
                self._exit_application()

    def on_event(self, event: AgentEvent) -> None:
        if event.kind == "tool_call":
            live, done, target = describe(event.text, event.detail.get("args", {}))
            self._step = (event.text, done, target, time.monotonic())
            self.live(live, target)
        elif event.kind == "tool_result":  # prototype only: Hailer emits no tool-finished event yet
            if self._step is not None:
                _tool, done, target, started = self._step
                detail = _join(target, event.detail.get("summary", ""))
                ok = event.detail.get("ok", True)
                self.console.print(row(G["ok"] if ok else G["fail"], "ok" if ok else "fail", done, detail, time.monotonic() - started))
                self._step = None
            self.live("Thinking")
        elif event.kind == "message_delta":
            if self._live[0] != "Writing the answer":
                self.live("Writing the answer")
        elif event.kind == "status" and event.text:
            self.live(event.text)

    def step_cancelled(self) -> None:
        if self._step is not None:
            _tool, done, target, started = self._step
            self.console.print(row(G["warn"], "warn", done, _join(target, "cancelled, effect unknown"), time.monotonic() - started))
            self._step = None

    def finish(self, final_response: str, *, steps: int = 0, input_tokens: int = 0, output_tokens: int = 0, link: str = "") -> None:
        self.console.print()
        self.console.print(reply(final_response))
        self.console.print()
        stats = _join(
            f"{steps} steps" if steps else "",
            _seconds(time.monotonic() - self._since),
            f"{_tokens(input_tokens)} in" if input_tokens else "",
            f"{_tokens(output_tokens)} out" if output_tokens else "",
        )
        footer = Table.grid(expand=True)
        footer.add_column(ratio=1)
        footer.add_column(justify="right")
        target = Text(f"{G['link']} {self.view.notebook}", style=f"dim link {link}" if link else "dim")
        footer.add_row(Text(f"  {stats}", style="muted"), target if link else "")
        self.console.print(footer)
        self.console.print()
        self.view.turns += 1
        self.view.tokens += input_tokens + output_tokens

    async def run(self) -> None:
        async def tick() -> None:
            while True:
                await asyncio.sleep(0.08)
                flashing = self._flash is not None and time.monotonic() < self._flash[1] + 0.1
                if self.application.is_running and (self.busy or self._preparing or flashing):
                    self.application.invalidate()

        ticker = asyncio.create_task(tick())
        try:
            await super().run()
        finally:
            ticker.cancel()


# --------------------------------------------------------------------------- #
# The simulated session
# --------------------------------------------------------------------------- #

NOTEBOOK_URL = "http://127.0.0.1:2718/?file=analysis.py"

TOTALS_CODE = """\
from hailer.periods import load_periods
df = load_periods(DATA_DIR, "sales")
q = df.with_columns(quarter=pl.when(pl.col("period") <= "25-03").then(pl.lit("Q1")).otherwise(pl.lit("Q2")))
by_region = (
    q.group_by("region", "quarter").agg(pl.col("revenue").sum())
    .pivot(on="quarter", index="region", values="revenue")
    .with_columns(change=pl.col("Q2") / pl.col("Q1") - 1)
    .sort("change", descending=True)
)
by_region
"""
CELL_CODE = """\
import marimo._code_mode as cm
async with cm.get_context() as ctx:
    ctx.create_cell(GROWTH_CELL, name="growth_by_region")
"""


@dataclass
class Step:
    tool: str
    args: dict[str, Any]
    seconds: float
    summary: str
    ok: bool = True


def analysis_steps() -> list[Step]:
    clean = f"ruff {G['ok']} ty {G['ok']}" if G is UNICODE else "checks clean"
    return [
        Step("list_periods", {"name": "sales"}, 0.7, "6 months, 25-01 to 25-06; discount from 25-04"),
        Step("notebook_cells", {}, 0.4, "analysis.py, 6 cells"),
        Step("marimo_execute", {"code": TOTALS_CODE}, 1.5, "4 rows"),
        Step("marimo_execute", {"code": CELL_CODE}, 1.9, f"growth_by_region {G['sep']} {clean}"),
    ]


ANSWER = """\
**North grew fastest**: Q2 revenue was **39,608**, up **29.6%** on Q1. West is the only \
region that fell. Total revenue rose 6.9% to 123,189.

| Region | Q1 | Q2 | Change |
|:--|--:|--:|--:|
| North | 30,573 | 39,608 | +29.6% |
| East | 27,805 | 28,647 | +3.0% |
| South | 27,532 | 28,189 | +2.4% |
| West | 29,332 | 26,745 | -8.8% |

Discounts start in 25-04 (the `discount` column first appears there), so every region's Q2 \
figure is net of them. The table is in `analysis.py` as `growth_by_region`, with a bar chart.
"""


class Session:
    def __init__(self, console: Console, view: SessionView, delay: float) -> None:
        self.console, self.view, self.delay = console, view, delay
        self.ui = PrototypeUI(console, self.handle, view)

    async def pause(self, seconds: float) -> None:
        await asyncio.sleep(seconds * self.delay)

    def header(self) -> None:
        out = self.console
        title = Text.assemble((f"{G['brand']} ", "accent"), ("Hailer", "bold"), (" 0.3.0", "muted"),
                              (f"   {G['sep']}   ~/work/acme-analytics", "muted"))
        out.print(title)
        out.print()
        if self.view.unsafe_local:
            out.print(row(G["warn"], "warn", "Kernel", _join("unsafe-local", "runs as you, not isolated"), 0.9, label_width=10))
        else:
            out.print(row(G["ok"], "ok", "Kernel", _join("docker", "isolated", "no network", "data read-only"), 1.9, label_width=10))

    async def prepare(self) -> bool:
        ui, out = self.ui, self.ui.console
        ui.live("Notebook", "opening analysis.py in your browser", 10)
        started = time.monotonic()
        await self.pause(2.2)
        self.view.notebook_open = True
        out.print(row(G["ok"], "ok", "Notebook", _join("analysis.py", "open in your browser"), time.monotonic() - started, label_width=10))
        out.print(Gutter(Text("Shows results only; Ctrl+. in the notebook shows the code", style="muted"), " " * 15))
        ui.live("Agent", f"starting {self.view.model}", 10)
        started = time.monotonic()
        await self.pause(0.9)
        out.print(row(G["ok"], "ok", "Agent", _join(self.view.model, self.view.provider, "key from keychain"), time.monotonic() - started, label_width=10))
        out.print(row(G["resume"], "muted", "Resumed", _join(f"{self.view.turns} earlier turns", "/new starts fresh"), label_width=10))
        out.print()
        return True

    async def handle(self, text: str) -> None:
        command = parse_command(text)
        if command is None:
            await self.turn(analysis_steps())
        elif command.name in EXIT_COMMANDS:
            self.ui.exit()
        elif command.name == "fail":
            await self.turn(analysis_steps()[:1], fail=True)
        else:
            await self.pause(0.15)
            self.command(command.name, command.args)

    async def turn(self, steps: list[Step], *, fail: bool = False) -> None:
        ui = self.ui
        ui.on_event(AgentEvent("status", "Thinking"))
        await self.pause(0.9)
        try:
            for step in steps:
                ui.on_event(AgentEvent("tool_call", step.tool, {"args": step.args}))
                await self.pause(step.seconds)
                ui.on_event(AgentEvent("tool_result", step.tool, {"summary": step.summary, "ok": step.ok}))  # type: ignore[arg-type]
                await self.pause(0.35)
        except asyncio.CancelledError:
            ui.step_cancelled()
            raise
        if fail:
            ui.console.print()
            hint = Text.assemble("Run ", ("hailer login openai", "accent"), " to store a new key, or set OPENAI_API_KEY in this terminal.\n",
                                 ("The endpoint said: Incorrect API key provided.", "muted"))
            ui.console.print(callout(G["fail"], "fail", "The model endpoint rejected the API key for provider 'openai'.", hint))
            ui.console.print()
            return
        for _ in range(6):
            ui.on_event(AgentEvent("message_delta", "..."))
            await self.pause(0.15)
        ui.finish(ANSWER, steps=len(steps), input_tokens=3_140, output_tokens=612, link=NOTEBOOK_URL)

    def command(self, name: str, args: str) -> None:
        out, view = self.ui.console, self.view
        if name == "help":
            out.print(help_card())
        elif name == "status":
            out.print(status_card(view))
        elif name == "notebook" and args.strip() == "list":
            out.print(notebook_table(view))
        elif name == "notebook" and args.startswith("open "):
            target = args.split(None, 1)[1].strip()
            target = target if target.endswith(".py") else f"{target}.py"
            view.notebook, view.notebook_open = target, True
            out.print(row(G["ok"], "ok", "Notebook", _join(target, "active", "open in your browser"), 0.6, label_width=10))
        elif name == "new":
            view.turns = 0
            out.print(row(G["resume"], "muted", "New", "conversation started; context files re-read", label_width=10))
        elif name == "model" and args:
            view.model = args.split(":")[-1]
            out.print(row(G["ok"], "ok", "Model", _join(view.model, "new conversation"), label_width=10))
        elif name == "clear":
            out.clear()
            return
        else:
            out.print(Text(f"  /{name} behaves as in Hailer; this prototype only fakes /help, /status, "
                           "/notebook list|open, /new, /model, /fail and /clear.", style="muted"))
        out.print()

    def goodbye(self) -> None:
        out, view = self.console, self.view
        elapsed = _seconds(time.monotonic() - view.started)
        out.print(Text(f"  Session ended {G['sep']} {view.turns} turns {G['sep']} {elapsed} {G['sep']} {_tokens(view.tokens)} tokens", style="muted"))
        out.print(Text.assemble(
            (f"  {G['ok']} ", "ok"), ("Notebooks copied back to notebooks/", "muted"),
            (f"   {G['ok']} ", "ok"), ("Kernel stopped", "muted"),
        ))

    async def run(self) -> None:
        self.header()
        self.ui.configure_startup(self.prepare)
        await self.ui.run()
        self.goodbye()


def help_card() -> RenderableType:
    groups = {
        "Conversation": ["new", "model", "skill", "prompt"],
        "Notebook": ["notebook", "exec"],
        "Session": ["status", "context", "reload", "clear", "exit"],
    }
    grid = Table.grid(padding=(0, 2))
    grid.add_column(width=12, style="muted")
    grid.add_column(no_wrap=True)
    grid.add_column(ratio=1)
    for group, names in groups.items():
        for index, name in enumerate(names):
            usage, summary = SHORT[name]
            command = Text.assemble((f"/{name}", "accent"), (f" {usage}" if usage else "", "muted"))
            grid.add_row(group if index == 0 else "", command, summary)
        grid.add_row("", "", "")
    keys = Table.grid(padding=(0, 2))
    keys.add_column(width=12, style="muted")
    keys.add_column(ratio=1, style="muted")
    keys.add_row("Keys", _join("enter send", "alt+enter new line", "esc cancel", f"{G['up']} history", "tab complete"))
    return Gutter(Group(grid, keys), "  ")


def status_card(view: SessionView) -> RenderableType:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=1)
    grid.add_column(width=10, style="muted")
    grid.add_column(ratio=1)

    def line(ok: bool | None, label: str, value: str) -> None:
        dot = Text(G["on"], style="ok") if ok else Text(G["warn"], style="warn") if ok is False else Text(" ")
        grid.add_row(dot, label, value)

    line(True, "Model", _join(view.model, view.provider, "OPENAI_API_KEY from keychain"))
    if view.unsafe_local:
        line(False, "Kernel", _join("unsafe-local", "runs as you, not isolated"))
    else:
        line(True, "Kernel", _join("docker", "no network", "data read-only", "127.0.0.1:2718"))
    line(view.notebook_open, "Notebook", _join(view.notebook, "open in your browser" if view.notebook_open else "not open"))
    line(None, "Thread", _join("3f9c2a1e", f"{view.turns} turns", f"{_tokens(view.tokens)} tokens"))
    line(None, "Context", _join("1 file", "1 skill", "1 prompt"))
    line(None, "Web", "no domains allowed")
    box_style = box.ROUNDED if G is UNICODE else box.ASCII
    return Gutter(Panel(grid, title="Session", title_align="left", box=box_style, border_style="muted", expand=False, padding=(0, 1)), "  ")


def notebook_table(view: SessionView) -> RenderableType:
    table = Table(box=box.SIMPLE_HEAD if G is UNICODE else ASCII_SIMPLE_HEAD, show_edge=False, pad_edge=False,
                  header_style="muted", border_style="muted")
    table.add_column("Notebook")
    table.add_column("Modified", style="muted")
    table.add_column("Size", justify="right", style="muted")
    table.add_column("")
    rows = [("analysis.py", "today 14:02", "6.1 KB"), ("q2-review.py", "yesterday", "11.8 KB"), ("returns.py", "12 Oct", "3.4 KB")]
    for name, modified, size in rows:
        state = Text()
        if name == view.notebook:
            state.append(f"{G['on']} open", style="ok" if view.notebook_open else "muted")
            state.append("  active", style="accent")
        table.add_row(name, modified, size, state)
    return Gutter(table, "  ")


def main() -> None:
    global G
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unsafe-local", action="store_true", help="show the unsafe-local kernel warnings")
    parser.add_argument("--ascii", action="store_true", help="ASCII glyphs (chosen automatically for consoles that cannot print Unicode)")
    parser.add_argument("--delay", type=float, default=1.0, help="multiplier for the simulated durations")
    parser.add_argument("--timeout", type=float, default=600, help="maximum lifetime in seconds")
    args = parser.parse_args()
    G = ASCII if args.ascii or not _unicode_ok() else UNICODE
    console = Console(highlight=False, theme=THEME)
    session = Session(console, SessionView(unsafe_local=args.unsafe_local), max(args.delay, 0.0))
    try:
        asyncio.run(asyncio.wait_for(session.run(), args.timeout))
    except TimeoutError:
        console.print(f"Prototype stopped after {args.timeout:g} seconds.", markup=False)


if __name__ == "__main__":
    main()
