"""The plain-text helpers the agent side and the terminal share (hailer.text): no terminal needed."""

from __future__ import annotations

import pytest
from prompt_toolkit.utils import get_cwidth

from hailer import text


def test_single_line_drops_controls_and_folds_whitespace():
    assert text.single_line("notebook_open\nq2") == "notebook_open q2"
    assert text.single_line("a\x1b[2J‮b  \t c\r\n") == "a[2Jb c"
    assert text.single_line("") == ""


def test_printable_drops_controls_but_keeps_the_spacing():
    assert text.printable("a\x1b[31mb‮c") == "a[31mbc"
    assert text.printable("one\ntwo  three\t") == "one two  three "


def test_clean_reply_removes_escape_sequences_and_controls_and_keeps_lines():
    raw = "\x1b[31mred\x1b[0m \x1b]0;title\x07x\x1b]8;;http://h\x1b\\link\x1b]8;;\x1b\\ \x9b2Jy\x00\x08\r\n-\titem"
    assert text.clean_reply(raw) == "red xlink y\n-   item"
    assert text.clean_reply("") == ""


@pytest.mark.parametrize(
    ("args", "kwargs", "expected"),
    [
        ((1, "cell"), {}, "1 cell"),
        ((2, "cell"), {}, "2 cells"),
        ((0, "turn"), {}, "0 turns"),
        ((12345, "row"), {}, "12345 rows"),
        ((12345, "row"), {"grouped": True}, "12,345 rows"),
        ((3, "child", "children"), {}, "3 children"),
        ((1, "child", "children"), {}, "1 child"),
    ],
)
def test_plural(args, kwargs, expected):
    assert text.plural(*args, **kwargs) == expected


@pytest.mark.parametrize(
    ("url", "with_port", "without_port"),
    [
        ("https://user:secret@docs.pola.rs/api/?q=1", "docs.pola.rs", "docs.pola.rs"),
        (" http://127.0.0.1:2718/?access_token=t ", "127.0.0.1:2718", "127.0.0.1"),
        ("HTTPS://Example.COM", "example.com", "example.com"),
        ("http://host:notaport/", "", "host"),  # a bad port only matters when it is shown
        ("not a url", "", ""),
        ("", "", ""),
    ],
)
def test_url_host_never_shows_more_than_the_host(url, with_port, without_port):
    assert text.url_host(url) == with_port
    assert text.url_host(url, port=False) == without_port


def test_fit_shortens_by_characters_or_by_terminal_cells():
    assert text.fit("short", 10, "...") == "short"
    assert text.fit("a summary that is far too long", 18, "...") == "a summary that..."  # cut, trailing space gone
    assert text.fit("a summary that is far too long", 16, "...") == "a summary tha..."
    assert text.fit("abcdef", 4) == "abcd"  # no ellipsis: just the start
    assert text.fit("Ready  ", 5) == "Ready"
    wide = "表格表格表格"  # two cells each
    assert text.fit(wide, 7, "…", width=get_cwidth) == "表格表…"
    assert get_cwidth(text.fit(wide, 7, "…", width=get_cwidth)) <= 7
    assert text.fit("abc", 0, "...") == "..."  # the ellipsis is all there is room for
