"""The command menu's completions and rows, from cached data only."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from prompt_toolkit.buffer import CompletionState
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from hailer.completion import MENU_ROWS, RUN_ON_ENTER, SlashCompleter, menu_fragments, menu_height
from hailer.models import CompletionData
from hailer.session import COMMAND_USAGE
from hailer.transcript import ASCII, UNICODE

DATA = CompletionData(
    notebooks=("analysis.py", "q2-review.py", "returns.py"),
    active_notebook="analysis.py",
    skills=(("example-skill", "How this team writes period reports"),),
    prompts=("first-look",),
    model="gpt-5.5",
    providers=("internal", "openai"),
)


def complete(text: str, data: CompletionData = DATA) -> list[tuple[str, str, str]]:
    """(the draft after taking each completion, its name, its description)."""
    completer = SlashCompleter(lambda: data)
    out = []
    for completion in completer.get_completions(Document(text), CompleteEvent(text_inserted=True)):
        result = text[: len(text) + completion.start_position] + completion.text
        out.append((result, completion.display_text, completion.display_meta_text))
    return out


def test_slash_lists_every_command_with_its_short_description_and_quit_stays_an_alias():
    listed = complete("/")
    assert [name for _result, name, _meta in listed] == [f"/{name}" for name in COMMAND_USAGE]
    assert listed[0] == ("/help", "/help", "Commands and keys")
    assert ("/model ", "/model", "Switch model; starts a new conversation") in listed, "arguments follow a space"
    assert "/quit" not in [name for _result, name, _meta in listed]
    assert [result for result, _name, _meta in complete("/e")] == ["/exec ", "/exit"]


def test_a_command_typed_in_full_gets_no_space_so_enter_sends_it():
    assert [result for result, *_ in complete("/model")] == ["/model"]
    assert [result for result, *_ in complete("/notebook")] == ["/notebook"]
    assert [result for result, *_ in complete("/notebook open")] == ["/notebook open"]


def test_arguments_come_from_the_cached_data():
    assert complete("/notebook ") == [
        ("/notebook list", "list", "Every notebook, with its state"),
        ("/notebook new ", "new", "Create one from the starter template"),
        ("/notebook open ", "open", "Switch the active notebook"),
        ("/notebook close ", "close", "Free a notebook's kernel session"),
    ]
    assert complete("/notebook open ") == [
        ("/notebook open analysis.py", "analysis.py", "active"),
        ("/notebook open q2-review.py", "q2-review.py", ""),
        ("/notebook open returns.py", "returns.py", ""),
    ]
    assert [result for result, *_ in complete("/notebook close Q2")] == ["/notebook close q2-review.py"]
    assert complete("/notebook new ") == [] and complete("/notebook open a.py ") == []
    assert complete("/model ") == [("/model gpt-5.5", "gpt-5.5", "current"), ("/model internal:", "internal:", "provider"),
                                   ("/model openai:", "openai:", "provider")]
    assert complete("/skill ") == [("/skill example-skill", "example-skill", "How this team writes period reports")]
    assert complete("/prompt f") == [("/prompt first-look", "first-look", "saved prompt")]
    assert complete("/skill example-skill summarise it") == []
    assert complete("/exec ") == [] and complete("/unknown ") == []


@pytest.mark.parametrize("text", ["", "hello /help", "/notebook open\nmore", " /help"])
def test_nothing_completes_outside_a_one_line_slash_command(text):
    assert complete(text) == []


def test_a_failing_or_missing_data_source_still_completes_commands():
    def broken() -> CompletionData:
        raise RuntimeError("the controller is gone")

    for completer in (SlashCompleter(broken), SlashCompleter(None), SlashCompleter(lambda: "not data")):  # type: ignore[arg-type]
        found = list(completer.get_completions(Document("/no"), CompleteEvent()))
        assert [c.text for c in found] == ["/notebook "]
        assert list(completer.get_completions(Document("/notebook open "), CompleteEvent())) == []


def test_names_from_outside_cannot_add_rows_or_controls():
    data = CompletionData(notebooks=("odd\x1b[31m\nname.py",), skills=(("s", "line\none"),))
    [(_result, name, _meta)] = complete("/notebook open ", data)
    assert "\n" not in name and "\x1b" not in name
    [(_result, _name, meta)] = complete("/skill ", data)
    assert "\n" not in meta


def test_enter_runs_at_once_only_the_commands_without_arguments():
    assert RUN_ON_ENTER == {"/help", "/status", "/new", "/context", "/reload", "/clear", "/exit", "/notebook list"}


def buffer_with(text: str, data: CompletionData = DATA, index: int | None = None) -> SimpleNamespace:
    completions = list(SlashCompleter(lambda: data).get_completions(Document(text), CompleteEvent()))
    state = CompletionState(Document(text), completions, complete_index=index)
    return SimpleNamespace(complete_state=state)


def rows(fragments: list[tuple[str, str]]) -> list[str]:
    return "".join(text for _style, text in fragments).split("\n")


def test_menu_rows_mark_the_chosen_item_scroll_and_count_the_rest():
    buffer = buffer_with("/")
    assert menu_height(buffer) == MENU_ROWS + 1
    shown = rows(menu_fragments(buffer, UNICODE, 100))
    assert shown[0] == "  › /help       Commands and keys"
    assert shown[5].startswith("    /exec       Run Python") and shown[6] == "    6 more"
    # Choosing the eighth command scrolls it into the last row.
    shown = rows(menu_fragments(buffer_with("/", index=7), UNICODE, 100))
    assert shown[0].startswith("    /new") and shown[5].startswith("  › /skill")
    styles = {style for style, _text in menu_fragments(buffer, UNICODE, 100)}
    assert not any(style.startswith("class:menu") for style in styles), "prompt_toolkit paints class:menu grey"


def test_menu_descriptions_shorten_to_the_width_with_the_glyph_sets_ellipsis():
    for glyphs, ellipsis in ((UNICODE, "…"), (ASCII, "...")):
        shown = rows(menu_fragments(buffer_with("/ex"), glyphs, 40))
        assert all(len(line) <= 40 for line in shown)
        assert shown[0].endswith(ellipsis) and shown[0].startswith(f"  {glyphs.prompt} /exec")
    few = buffer_with("/notebook open ")
    assert menu_height(few) == 3 and rows(menu_fragments(few, UNICODE, 64))[-1].startswith("    returns.py")


def test_nothing_completes_unless_the_cursor_is_at_the_end_of_the_draft():
    completer = SlashCompleter(lambda: DATA)
    middle = Document("/notebo open analysis.py", cursor_position=len("/notebo"))
    assert list(completer.get_completions(middle, CompleteEvent())) == []
    assert completer.enter_pick(middle) is None, "Enter sends it as typed"
    spaces = Document("/no   ", cursor_position=3)
    assert [c.text for c in completer.get_completions(spaces, CompleteEvent())] == ["/notebook "]


def test_names_that_match_as_typed_come_first():
    data = CompletionData(notebooks=("Sales.py", "sales-2024.py", "sales.py"))
    assert [name for _result, name, _meta in complete("/notebook open sa", data)] == ["sales-2024.py", "sales.py", "Sales.py"]
    assert [name for _result, name, _meta in complete("/notebook open SA", data)] == ["Sales.py", "sales-2024.py", "sales.py"]


@pytest.mark.parametrize(
    ("text", "picked"),
    [
        ("/n", "/new"), ("/", "/help"), ("/exi", "/exit"), ("/no", "/notebook "), ("/notebook ", "/notebook list"),
        ("/notebook open SALES.PY", "/notebook open Sales.py"), ("/skill rep", "/skill report-weekly"),
        # whole as typed, or nothing completes it: Enter sends the draft
        ("/new", None), ("/Help", None), ("/model", None), ("/notebook", None), ("/notebook open", None),
        ("/notebook open sales.py", None), ("/notebook open Sales.py", None), ("/skill report", None),
        ("/prompt first-look", None), ("/model gpt-5.5", None), ("/exec print(1)", None), ("/unknown", None),
        ("a question", None), ("/skill report summarise it", None),
        # free text: a model name, code, a skill's message, a prompt's arguments, a new notebook's name
        ("/model gpt-5", None), ("/model ", None), ("/model inter", None), ("/exec pri", None),
        ("/skill report summarise th", None), ("/prompt first-look cust", None), ("/notebook new sal", None),
    ],
)
def test_enter_without_a_choice_completes_a_partial_command_and_sends_a_whole_one(text, picked):
    data = CompletionData(
        notebooks=("Sales.py", "sales.py"), skills=(("report-weekly", "w"), ("report", "r")),
        prompts=("first-look", "first-look-v2"), model="gpt-5.5",
    )
    pick = SlashCompleter(lambda: data).enter_pick(Document(text))
    result = None if pick is None else text[: len(text) + pick.start_position] + pick.text
    assert result == picked


def test_the_hint_says_what_enter_will_do():
    from hailer.completion import enter_action

    def action(text: str, index: int | None = None) -> str:
        buffer = buffer_with(text, index=index)
        buffer.document = Document(text)
        return enter_action(buffer)

    assert action("/") == "complete"
    assert action("/", index=0) == "run", "a chosen /help runs"
    assert action("/", index=3) == "complete", "a chosen /model takes arguments"
    assert action("/prompt first-look") == "send"
    assert action("/model gpt") == "send", "a model name is free text: sent as typed"
