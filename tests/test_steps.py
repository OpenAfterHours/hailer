"""Tool calls in plain words: every tool the agent has gets its own, with a target from its arguments."""

from __future__ import annotations

from hailer.steps import VERBS, describe_call
from hailer.tools import TOOL_NAMES


def test_every_tool_has_its_live_and_done_words():
    assert set(VERBS) == set(TOOL_NAMES), "a new tool needs its words in steps.VERBS, or its raw name shows"


def test_the_project_tools_name_what_they_look_at():
    read = describe_call("project_read", {"path": "src/billing.py", "start_line": 40, "end_line": 80})
    assert (read.live, read.done, read.target) == ("Reading a project file", "Read a project file", "src/billing.py:40-80")
    assert describe_call("project_read", {"path": "src/billing.py", "end_line": 0}).target == "src/billing.py"
    assert describe_call("project_search", {"query": "def net_revenue", "pattern": "*.py"}).target == "def net_revenue"
    assert describe_call("project_files", {"pattern": "*.sql"}).target == "*.sql"
    assert describe_call("project_files", {}).done == "Listed project files"
