"""What the chat prints: one glyph set, step rows, callouts, gutters, cards and the footer's data.

The composer (:mod:`hailer.chat_ui`) and the plain line display (``cli.common._TurnDisplay``) share
these, so a turn reads the same in both. Only Rich is imported here (the reply's Markdown typography
lives with the composer). Colours are ANSI names, so the terminal's own theme decides them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rich import box
from rich.cells import cell_len
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.panel import Panel
from rich.segment import Segment
from rich.style import Style
from rich.table import Table
from rich.text import Text

from hailer.session import COMMAND_GROUPS, COMMAND_USAGE
from hailer.steps import Receipt, plural, single_line

ACCENT, MUTED, OK, WARN, FAIL = "cyan", "dim", "green", "yellow", "red"
STATUS_STYLES = {"ok": OK, "failed": FAIL, "cancelled": WARN, "unknown": WARN}
#: Column widths shared by committed step rows and the composer's live row, so they line up.
ICON_WIDTH, LABEL_WIDTH, SECONDS_WIDTH = 3, 20, 7


# --------------------------------------------------------------------------- #
# Glyphs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Glyphs:
    """The symbols the chat draws with: one Unicode set and one ASCII set (:func:`glyphs_for`)."""

    user: str
    reply: str
    ok: str
    fail: str
    warn: str
    on: str
    off: str
    sep: str
    link: str
    prompt: str
    rule: str
    bar: str
    corners: str  # top-left, top-right, bottom-left, bottom-right
    spinner: str  # one character per frame
    up: str
    ellipsis: str
    ascii: bool = False
    brand: str = "◆"  # the startup header's mark
    resume: str = "↺"  # a resumed or new conversation

    def join(self, *parts: str) -> str:
        """The non-empty ``parts`` with the separator between them."""
        return f" {self.sep} ".join(part for part in parts if part)

    def icon(self, status: str) -> str:
        """A step's or session row's status as its mark: "ok", "failed"/"fail", "resume" (a resumed
        conversation), "wait" (not there yet); anything else (cancelled, unknown, warn) is a warning."""
        return {"ok": self.ok, "failed": self.fail, "fail": self.fail, "resume": self.resume, "wait": self.off}.get(status, self.warn)

    def characters(self) -> str:
        return "".join(str(value) for value in vars(self).values() if isinstance(value, str))


UNICODE = Glyphs(
    user="▌", reply="●", ok="✓", fail="✗", warn="▲", on="●", off="○", sep="·", link="↗", prompt="›",
    rule="─", bar="│", corners="╭╮╰╯", spinner="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", up="↑", ellipsis="…",
)
ASCII = Glyphs(
    user="|", reply="*", ok="+", fail="x", warn="!", on="*", off="o", sep="-", link="->", prompt=">",
    rule="-", bar="|", corners="++++", spinner="|/-\\", up="up", ellipsis="...",
    ascii=True, brand="#", resume="~",
)


def glyphs_for(encoding: str | None) -> Glyphs:
    """The Unicode set when ``encoding`` can write all of it, else ASCII (a cp1252 Windows console
    or pipe), so a terminal never shows replacement characters."""
    try:
        UNICODE.characters().encode(encoding or "ascii")
    except (UnicodeEncodeError, LookupError):
        return ASCII
    return UNICODE


def console_glyphs(console: Any) -> Glyphs:
    """The set for ``console`` (a Rich console, or the composer's queued one), by its encoding."""
    return glyphs_for(getattr(console, "encoding", None))


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #


def seconds(value: float) -> str:
    if value < 10:
        return f"{value:.1f}s"
    if value < 60:
        return f"{value:.0f}s"
    minutes, rest = divmod(int(value), 60)
    return f"{minutes}m {rest:02d}s"


def tokens(value: int) -> str:
    return f"{value / 1000:.1f}k" if value >= 1000 else str(value)


# --------------------------------------------------------------------------- #
# Layout pieces
# --------------------------------------------------------------------------- #


class Gutter:
    """``body`` with ``first`` before its first line and ``rest`` before the others.

    On a console that wraps (the composer's) the body is laid out narrower, so wrapped lines stay
    indented. On a soft-wrapped console (plain mode, pipes) nothing is wrapped or cut: each line is
    written whole and the terminal, if any, wraps it.
    """

    def __init__(self, body: RenderableType, first: str, rest: str | None = None, style: str = "") -> None:
        self.body, self.first, self.style = body, first, style
        self.rest = " " * cell_len(first) if rest is None else rest

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if options.no_wrap:
            lines = list(Segment.split_lines(console.render(self.body, options)))
        elif isinstance(self.body, Text) and "://" in self.body.plain:
            lines = self._url_lines(console, options)
        else:
            width = max(10, options.max_width - cell_len(self.first))
            lines = console.render_lines(self.body, options.update(width=width, no_wrap=False), pad=False)
        style = console.get_style(self.style) if self.style else None
        for index, line in enumerate(lines):
            line = _rstrip(line)
            prefix = self.first if index == 0 else self.rest
            yield Segment(prefix if line else prefix.rstrip(), style)
            yield from line
            yield Segment.line()

    def _url_lines(self, console: Console, options: ConsoleOptions) -> list[list[Segment]]:
        """A text's lines, wrapped inside the gutter except those with a URL: Rich would cut a URL
        into lines that no longer open or copy as one, so such a line is written whole and the
        terminal wraps it."""
        assert isinstance(self.body, Text)
        width = max(10, options.max_width - cell_len(self.first))
        lines: list[list[Segment]] = []
        for part in self.body.split("\n"):
            if "://" in part.plain:
                lines += Segment.split_lines(console.render(part, options.update(no_wrap=True, overflow="ignore")))
            else:
                lines += console.render_lines(part, options.update(width=width, no_wrap=False), pad=False)
        return lines


def _rstrip(line: list[Segment]) -> list[Segment]:
    """``line`` without trailing padding (spaces with no background), so copied text has none."""
    line = list(line)
    while line:
        text, style, control = line[-1]
        if control or (style is not None and style.bgcolor is not None):
            break
        stripped = text.rstrip(" ")
        if stripped:
            line[-1] = Segment(stripped, style)
            break
        line.pop()
    return line


def user_message(glyphs: Glyphs, text: str) -> RenderableType:
    """The user's message as sent, every line behind the accent bar."""
    bar = f"{glyphs.user} "
    return Gutter(Text(text), bar, bar, style=ACCENT)


def callout(glyphs: Glyphs, kind: str, message: str, hints: Sequence[str] = ()) -> RenderableType:
    """A problem in one line (``kind``: "fail" or "warn"), then what to do about it, indented."""
    style = FAIL if kind == "fail" else WARN
    body = Text(message, style=style)
    for hint in hints:
        body.append("\n")
        body.append(hint)
    return Gutter(body, f"  {glyphs.fail if kind == 'fail' else glyphs.warn} ", style=style)


@dataclass(frozen=True)
class _Row:
    """What a step row and a session row share: an icon, a fixed-width label, the detail and the
    seconds, with the styles of the icon, the label and the detail."""

    icon: str
    styles: tuple[str, str, str]
    label: str
    label_width: int
    detail: str = ""
    took: float | None = None
    #: One line, its detail cut with an ellipsis; otherwise (a failure, a warning) the detail wraps.
    worked: bool = True

    def grid(self) -> Table:
        """Every column has a fixed width, so durations line up across separately printed rows and
        with the composer's live row."""
        icon_style, label_style, detail_style = self.styles
        grid = Table.grid(expand=True, padding=(0, 1))
        grid.add_column(width=ICON_WIDTH, no_wrap=True)
        grid.add_column(width=self.label_width, no_wrap=True)
        grid.add_column(ratio=1, no_wrap=self.worked, overflow="ellipsis" if self.worked else "fold")
        grid.add_column(width=SECONDS_WIDTH, justify="right", no_wrap=True)
        grid.add_row(
            Text(f"  {self.icon}", style=icon_style), Text(self.label, style=label_style),
            Text(self.detail, style=detail_style), Text(seconds(self.took) if self.took is not None else "", style=MUTED),
        )
        return grid

    def line(self, width: int) -> Text:
        """One line of text that is never cut; the seconds go to the right edge when they fit there."""
        icon_style, label_style, detail_style = self.styles
        line = Text.assemble(
            (f"  {self.icon} ", icon_style),
            (self.label.ljust(self.label_width), label_style),
            (f" {self.detail}" if self.detail else "", detail_style),
        )
        line.rstrip()
        if self.took is not None:
            took = seconds(self.took)
            room = width - line.cell_len
            line.append(took.rjust(room) if room >= len(took) + 2 else f"  {took}", style=MUTED)
        return line


def _step(glyphs: Glyphs, receipt: Receipt) -> _Row:
    style = STATUS_STYLES.get(receipt.status, WARN)
    ok = receipt.status == "ok"
    return _Row(
        glyphs.icon(receipt.status), (style, "" if ok else style, MUTED), receipt.label, LABEL_WIDTH,
        glyphs.join(receipt.target, receipt.outcome), receipt.seconds, worked=ok,
    )


def step_row(glyphs: Glyphs, receipt: Receipt) -> Table:
    """One finished step: icon, done verb, ``target · outcome``, seconds. A step that worked keeps
    to one line; the detail of one that did not (its error, "cancelled, effect unknown") wraps in
    its column rather than being cut."""
    return _step(glyphs, receipt).grid()


def plain_step_row(glyphs: Glyphs, receipt: Receipt, width: int) -> Text:
    """:func:`step_row` as one line of text that is never cut."""
    return _step(glyphs, receipt).line(width)


def turn_stats(glyphs: Glyphs, *, steps: int, elapsed: float | None, input_tokens: int | None, output_tokens: int | None) -> str:
    """``4 steps · 7.7s · 3.1k in · 612 out``; parts that are unknown or zero are left out."""
    return glyphs.join(
        plural(steps, "step") if steps else "",
        seconds(elapsed) if elapsed is not None else "",
        f"{tokens(input_tokens)} in" if input_tokens else "",
        f"{tokens(output_tokens)} out" if output_tokens else "",
    )


def turn_footer(glyphs: Glyphs, stats: str, notebook: str = "", url: str = "") -> Table:
    """The muted line under a reply, with a terminal hyperlink to the notebook on the right. The
    stats keep their room; a narrow terminal shortens the link."""
    grid = Table.grid(expand=True)
    grid.add_column(no_wrap=True, width=cell_len(stats) + 4)  # "  " before the stats, two spaces after
    grid.add_column(ratio=1, justify="right", no_wrap=True, overflow="ellipsis")
    # A Style, not a style string: a URL is never parsed as style words.
    link = Text(f"{glyphs.link} {notebook}", style=Style(dim=True, link=url)) if notebook and url else Text("")
    grid.add_row(Text(f"  {stats}", style=MUTED), link)
    return grid


# --------------------------------------------------------------------------- #
# Session rows: the startup checklist, notebook switches and the exit summary
# --------------------------------------------------------------------------- #

#: The session rows' label column (Kernel, Notebook, Agent, Resumed): narrower than a step's.
SESSION_LABEL_WIDTH = 10
#: Where a session row's detail starts: the notes under a row (a tip, a link) line up with it.
SESSION_DETAIL_COLUMN = ICON_WIDTH + 1 + SESSION_LABEL_WIDTH + 1
#: A session row's status: the style of its icon, label and detail.
SESSION_STYLES = {"ok": (OK, "", MUTED), "warn": (WARN, WARN, WARN), "fail": (FAIL, FAIL, FAIL), "resume": (MUTED, "", MUTED), "wait": (MUTED, "", MUTED)}


def _printable(text: str) -> str:
    """``text`` without control characters (an escape sequence, a newline), its spaces kept."""
    return "".join(char if char.isprintable() else " " if char.isspace() else "" for char in text)


class SessionRow:
    """One line of what the session did: ``✓ Kernel  docker · isolated  1.9s``. ``status`` is "ok",
    "warn", "fail", "resume" (a resumed conversation) or "wait" (not there yet: a notebook to open).
    The label and the detail are made one plain line (names and warnings come from outside).

    On a terminal its columns line up with each other and with the composer's live row (seconds at
    the right edge); a row that worked keeps to one line (its detail cut with ``…``), a warning's
    or failure's detail wraps in its column. Elsewhere (plain mode, pipes) it is one line of text
    that is never cut, so nothing depends on the width or on redraws."""

    def __init__(self, glyphs: Glyphs, status: str, label: str, detail: str = "", took: float | None = None) -> None:
        self.row = _Row(
            glyphs.icon(status), SESSION_STYLES.get(status, SESSION_STYLES["warn"]), single_line(label),
            SESSION_LABEL_WIDTH, single_line(detail), took, worked=status in ("ok", "resume", "wait"),
        )

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield self.row.grid() if console.is_terminal else self.row.line(options.max_width)


def session_row(glyphs: Glyphs, status: str, label: str, detail: str = "", took: float | None = None) -> SessionRow:
    """A :class:`SessionRow`."""
    return SessionRow(glyphs, status, label, detail, took)


def session_note(text: str, style: str = MUTED) -> RenderableType:
    """A line under a session row's detail (a tip, a hint, a link to open by hand), without
    control characters. A URL in it is written whole and never becomes a hyperlink: printed with
    ``soft_wrap=True`` the terminal wraps it, so it copies as one."""
    return Gutter(Text(_printable(text), style=style), " " * SESSION_DETAIL_COLUMN)


def session_header(glyphs: Glyphs, version: str, workspace: str) -> Text:
    """``◆ Hailer 0.3.0 · ~/work/sales``: the first line of a session."""
    return Text.assemble((f"{glyphs.brand} ", ACCENT), ("Hailer", "bold"), (f" {version}", MUTED), (f"   {glyphs.sep}   {workspace}", MUTED))


def session_ended(glyphs: Glyphs, *, turns: int, elapsed: float, tokens_used: int) -> Text:
    """``Session ended · 4m 12s · this session: 2 turns, 1.1k tokens``: the time, then the turns and
    tokens of this session (the footer and ``/status`` count the whole conversation)."""
    counts = ", ".join([plural(turns, "turn"), *([f"{tokens(tokens_used)} tokens"] if tokens_used else [])])
    return Text("  " + glyphs.join("Session ended", seconds(elapsed), f"this session: {counts}"), style=MUTED)


def done_items(glyphs: Glyphs, items: Sequence[tuple[str, str]]) -> Text:
    """``✓ Notebooks copied back to notebooks/   ✓ Kernel stopped``: (status, text) on one line."""
    line = Text("  ")
    for index, (status, text) in enumerate(items):
        icon_style = SESSION_STYLES.get(status, SESSION_STYLES["warn"])[0]
        line.append(("   " if index else "") + f"{glyphs.icon(status)} ", style=icon_style)
        line.append(text, style=MUTED if status == "ok" else icon_style)
    return line


# --------------------------------------------------------------------------- #
# Cards
# --------------------------------------------------------------------------- #


ASCII_SIMPLE_HEAD = box.Box("    \n    \n -- \n    \n    \n    \n    \n    \n", ascii=True)


def simple_head(glyphs: Glyphs) -> box.Box:
    """A table box with only a rule under the header."""
    return ASCII_SIMPLE_HEAD if glyphs.ascii else box.SIMPLE_HEAD


def status_card(glyphs: Glyphs, rows: Sequence[tuple[str | None, str, str]], *, title: str = "Session") -> RenderableType:
    """``rows`` of (health, label, value) in a rounded card. Health: "ok" (a green dot), "warn" (a
    yellow triangle; the value is yellow too), "off" (an open dot) or None (no mark)."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=1, no_wrap=True)
    grid.add_column(style=MUTED, no_wrap=True, width=max((cell_len(label) for _h, label, _v in rows), default=0) + 1)
    grid.add_column(ratio=1, overflow="fold")  # a long path (a kernel log) wraps whole instead of being cut
    marks = {"ok": (glyphs.on, OK), "warn": (glyphs.warn, WARN), "off": (glyphs.off, MUTED)}
    for health, label, value in rows:
        mark, style = marks.get(health or "", (" ", ""))
        grid.add_row(Text(mark, style=style), Text(label), Text(value, style=WARN if health == "warn" else ""))
    card = Panel(
        grid, title=Text(title), title_align="left", box=box.ASCII if glyphs.ascii else box.ROUNDED,
        border_style=MUTED, expand=False, padding=(0, 1),
    )
    return Gutter(card, "  ")


class _HelpCard:
    """The commands grouped by task, with their arguments, then the keys. Groups sit in a column
    of their own when the descriptions still get room beside them, else above their commands; in
    a very narrow terminal each description goes under its command."""

    #: The narrowest description column beside the groups, and beside the commands alone.
    MIN_DESCRIPTION, MIN_BESIDE = 30, 20

    def __init__(self, glyphs: Glyphs) -> None:
        self.glyphs = glyphs

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        commands = {
            name: Text.assemble((f"/{name}", ACCENT), (f" {usage}" if usage else "", MUTED))
            for name, (usage, _summary) in COMMAND_USAGE.items()
        }
        group_width = max(cell_len(group) for group in [*COMMAND_GROUPS, "Keys"])
        command_width = max(command.cell_len for command in commands.values())
        keys = self.glyphs.join(
            "enter send", "alt+enter or ctrl+j new line", "esc cancel", f"{self.glyphs.up} history", "ctrl+c exit",
        )
        side_by_side = options.max_width - group_width - command_width - 4 >= self.MIN_DESCRIPTION
        beside = side_by_side or options.max_width - command_width - 2 >= self.MIN_BESIDE
        grid = Table.grid(padding=(0, 2))
        if side_by_side:
            grid.add_column(width=group_width, style=MUTED, no_wrap=True)
        grid.add_column(width=command_width if beside else None, no_wrap=beside)
        if beside:
            grid.add_column(ratio=1)
        for group, names in COMMAND_GROUPS.items():
            if not side_by_side:
                grid.add_row(Text(group, style=MUTED), *[Text("")] * (len(grid.columns) - 1))
            for index, name in enumerate(names):
                summary = COMMAND_USAGE[name][1]
                if side_by_side:
                    grid.add_row(Text(group if index == 0 else ""), commands[name], Text(summary))
                elif beside:
                    grid.add_row(commands[name], Text(summary))
                else:
                    grid.add_row(Group(commands[name], Gutter(Text(summary), "  ")))
            grid.add_row(*[Text("")] * len(grid.columns))
        footer = Table.grid(padding=(0, 2))
        footer.add_column(width=group_width, style=MUTED, no_wrap=True)
        footer.add_column(ratio=1, style=MUTED)
        footer.add_row(Text("Keys"), Text(keys))
        yield Group(grid, footer)


def help_card(glyphs: Glyphs) -> RenderableType:
    """The ``/help`` card (:class:`_HelpCard`)."""
    return Gutter(_HelpCard(glyphs), "  ")


@dataclass(frozen=True)
class NotebookRow:
    name: str
    modified: str
    size: str
    active: bool = False
    open: bool = False


def notebook_table(glyphs: Glyphs, rows: Sequence[NotebookRow], note: str = "") -> RenderableType:
    """The notebooks with headers; the active one and those with a kernel session are marked."""
    table = Table(box=simple_head(glyphs), show_edge=False, pad_edge=False, header_style=MUTED, border_style=MUTED)
    table.add_column("Notebook", no_wrap=True)
    table.add_column("Modified", style=MUTED, no_wrap=True)
    table.add_column("Size", justify="right", style=MUTED, no_wrap=True)
    table.add_column("", no_wrap=True)
    for row in rows:
        state = Text()
        if row.open:
            state.append(f"{glyphs.on} open", style=OK)
        elif row.active:
            state.append(f"{glyphs.off} not open", style=MUTED)
        if row.active:
            state.append("  active", style=ACCENT)
        table.add_row(Text(row.name), Text(row.modified), Text(row.size), state)  # Text: a name is never markup
    parts: list[RenderableType] = [table]
    if note:
        parts += [Text(""), Text(note, style=MUTED)]
    return Gutter(Group(*parts), "  ")


def detail_card(glyphs: Glyphs, sections: Sequence[tuple[str, Sequence[RenderableType | str]]], warnings: Sequence[str] = ()) -> RenderableType:
    """Labelled sections, one value per line (``/context``); warnings after them."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=MUTED, no_wrap=True)
    grid.add_column(ratio=1)
    for label, values in sections:
        for index, value in enumerate(values or [""]):
            grid.add_row(Text(label if index == 0 else ""), Text(value) if isinstance(value, str) else value)
    parts: list[RenderableType] = [grid]
    for warning in warnings:
        parts.append(Text.assemble((f"{glyphs.warn} ", WARN), (warning, WARN)))
    return Gutter(Group(*parts), "  ")


# --------------------------------------------------------------------------- #
# The composer's footer
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FooterData:
    """What the composer's footer shows. The controller builds it from state it already has
    (nothing is asked of marimo for it): it is read on every redraw."""

    notebook: str = ""
    #: Whether the notebook has a kernel session (a browser tab); None: not known.
    notebook_open: bool | None = None
    #: The kernel runtime ("docker", "unsafe-local"), "" for none.
    kernel: str = ""
    model: str = ""
    turns: int = 0
    tokens: int = 0
    #: The notebook's URL without the access token, for the link under each reply.
    notebook_url: str = ""
