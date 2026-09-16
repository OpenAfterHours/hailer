"""Tests for hailer.wire: the Responses <-> Chat Completions bridge."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from hailer import wire as wire_mod
from hailer.wire import (
    ChatBridge,
    ChatStreamTranslator,
    ChatToolMap,
    ChatUpstream,
    _reason,
    chat_request_from_responses,
    iter_sse_data,
    sse_frame,
)

# --------------------------------------------------------------------------- #
# Request translation
# --------------------------------------------------------------------------- #


RESPONSES_REQUEST: dict[str, Any] = {
    "model": "risk-analyst-v3",
    "instructions": "You are Hailer.",
    "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "[Hailer] notice"}, {"type": "input_text", "text": "hello"}]},
        {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "thinking"}], "encrypted_content": "xxx"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Let me check."}]},
        {"type": "function_call", "call_id": "call_a", "name": "shell", "arguments": '{"cmd": "ls"}'},
        {"type": "function_call", "call_id": "call_b", "name": "mcp__hailer__marimo_status", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_a", "output": "file.py"},
        {"type": "function_call_output", "call_id": "call_b", "output": [{"type": "input_text", "text": "ok"}]},
        {"type": "custom_tool_call", "call_id": "call_c", "name": "apply_patch", "input": "*** Begin Patch"},
        {"type": "custom_tool_call_output", "call_id": "call_c", "output": "Done"},
        {"type": "message", "role": "developer", "content": "Be brief."},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "look"}, {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "low"}]},
    ],
    "tools": [
        {"type": "function", "name": "shell", "description": "Run a command", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}, "strict": False},
        {"type": "custom", "name": "apply_patch", "description": "Patch files", "format": {"type": "grammar"}},
        {"type": "web_search"},
    ],
    "tool_choice": "auto",
    "parallel_tool_calls": False,
    "reasoning": {"effort": "high", "summary": "auto"},
    "store": False,
    "stream": True,
    "include": ["reasoning.encrypted_content"],
    "prompt_cache_key": "abc",
    "client_metadata": {"x": "y"},
}


def test_request_translation_builds_chat_messages():
    chat, tools = chat_request_from_responses(RESPONSES_REQUEST)
    assert tools.custom == {"apply_patch"} and tools.namespaced == {}
    assert chat["model"] == "risk-analyst-v3"
    assert chat["stream"] is True and chat["stream_options"] == {"include_usage": True}
    assert chat["reasoning_effort"] == "high"
    assert chat["tool_choice"] == "auto" and chat["parallel_tool_calls"] is False
    for key in ("store", "include", "prompt_cache_key", "client_metadata", "instructions", "input"):
        assert key not in chat

    msgs = chat["messages"]
    assert msgs[0] == {"role": "system", "content": "You are Hailer."}
    assert msgs[1] == {"role": "user", "content": "[Hailer] notice\nhello"}
    # reasoning item dropped; assistant text and its two parallel calls merge into one message
    assert msgs[2]["role"] == "assistant" and msgs[2]["content"] == "Let me check."
    assert [c["id"] for c in msgs[2]["tool_calls"]] == ["call_a", "call_b"]
    assert msgs[2]["tool_calls"][0] == {"id": "call_a", "type": "function", "function": {"name": "shell", "arguments": '{"cmd": "ls"}'}}
    assert msgs[3] == {"role": "tool", "tool_call_id": "call_a", "content": "file.py"}
    assert msgs[4] == {"role": "tool", "tool_call_id": "call_b", "content": "ok"}
    # a custom tool call after tool outputs opens a fresh assistant message with an `input` argument
    assert msgs[5]["role"] == "assistant" and msgs[5]["content"] is None
    assert json.loads(msgs[5]["tool_calls"][0]["function"]["arguments"]) == {"input": "*** Begin Patch"}
    assert msgs[6] == {"role": "tool", "tool_call_id": "call_c", "content": "Done"}
    assert msgs[7] == {"role": "system", "content": "Be brief."}  # developer -> system
    assert msgs[8]["role"] == "user" and msgs[8]["content"] == [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "low"}},
    ]


def test_request_translation_tools():
    chat, _ = chat_request_from_responses(RESPONSES_REQUEST)
    tools = chat["tools"]
    assert [t["function"]["name"] for t in tools] == ["shell", "apply_patch"]  # hosted web_search dropped
    assert tools[0] == {
        "type": "function",
        "function": {"name": "shell", "description": "Run a command", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}},
    }
    assert tools[1]["function"]["parameters"]["required"] == ["input"]


def test_request_translation_minimal_and_string_input():
    chat, tools = chat_request_from_responses({"model": "m", "input": "hi"})
    assert chat == {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True, "stream_options": {"include_usage": True}}
    assert tools == ChatToolMap()


def test_request_translation_without_streaming_sends_stream_false():
    chat, _ = chat_request_from_responses(RESPONSES_REQUEST, stream=False)
    assert chat["stream"] is False
    assert "stream_options" not in chat  # only meaningful with stream=true; some gateways reject it otherwise
    assert chat["tools"][0]["function"]["name"] == "shell"  # everything else is translated as usual


def test_request_translation_named_tool_choice_and_json_schema():
    chat, _ = chat_request_from_responses(
        {
            "model": "m",
            "input": [],
            "tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}, "strict": True}],
            "tool_choice": {"type": "function", "name": "f"},
            "text": {"format": {"type": "json_schema", "name": "out", "schema": {"type": "object"}, "strict": True}, "verbosity": "low"},
            "max_output_tokens": 50,
        }
    )
    assert chat["tool_choice"] == {"type": "function", "function": {"name": "f"}}
    assert chat["tools"][0]["function"]["strict"] is True
    assert chat["response_format"] == {"type": "json_schema", "json_schema": {"name": "out", "schema": {"type": "object"}, "strict": True}}
    assert chat["verbosity"] == "low" and chat["max_tokens"] == 50


# Codex 0.154 sends an MCP server's tools as one "namespace" tool; the bridge flattens it.
NAMESPACE_TOOLS: list[dict[str, Any]] = [
    {"type": "function", "name": "shell", "parameters": {"type": "object"}},
    {
        "type": "namespace",
        "name": "mcp__hailer",
        "description": "Hailer tools. A long explanation of the notebook workflow that must not be sent.",
        "tools": [
            {"type": "function", "name": "marimo_status", "description": "Is marimo running?", "parameters": {"type": "object", "properties": {}}, "strict": False},
            {"type": "function", "name": "shell", "description": "Clashes with the top-level tool", "parameters": {"type": "object"}},
            {"type": "function", "name": "notebook_cells", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}}, "strict": True},
            {"type": "hosted_thing", "name": "ignored"},
            {"type": "function", "name": ""},
            "not a tool",
        ],
    },
    {"type": "namespace", "name": "mcp__other", "tools": [{"type": "function", "name": "marimo_status", "parameters": {"type": "object"}}]},
    {"type": "namespace", "tools": [{"type": "function", "name": "nameless_namespace"}]},
]


def test_request_translation_flattens_namespace_tools():
    chat, tools = chat_request_from_responses({"model": "m", "input": [], "tools": NAMESPACE_TOOLS})
    names = [t["function"]["name"] for t in chat["tools"]]
    assert names == ["shell", "marimo_status", "mcp__hailer__shell", "notebook_cells", "mcp__other__marimo_status"]
    assert tools.custom == frozenset()
    assert tools.namespaced == {
        "marimo_status": ("mcp__hailer", "marimo_status"),
        "mcp__hailer__shell": ("mcp__hailer", "shell"),
        "notebook_cells": ("mcp__hailer", "notebook_cells"),
        "mcp__other__marimo_status": ("mcp__other", "marimo_status"),
    }
    assert chat["tools"][1]["function"] == {"name": "marimo_status", "description": "Is marimo running?", "parameters": {"type": "object", "properties": {}}}
    assert chat["tools"][2]["function"]["description"] == "Clashes with the top-level tool"
    assert chat["tools"][3]["function"]["strict"] is True
    assert "Hailer tools" not in json.dumps(chat)  # the namespace description is not sent
    assert tools.chat_name("shell", "mcp__hailer") == "mcp__hailer__shell"
    assert tools.chat_name("shell") == "shell"
    assert tools.chat_name("gone", "mcp__hailer") == "gone"  # unknown tool: bare name, never an error


def test_chat_tool_map_is_a_read_only_value_object():
    tools = ChatToolMap(custom={"apply_patch"}, namespaced={"a": ("ns", "a")})
    assert tools == ChatToolMap(custom=frozenset({"apply_patch"}), namespaced={"a": ("ns", "a")})
    assert tools.namespaced == {"a": ("ns", "a")} and tools.namespaced.get("a") == ("ns", "a")
    with pytest.raises(TypeError):
        tools.namespaced["b"] = ("ns", "b")  # type: ignore[index]
    assert tools.source("a") == ("ns", "a") and tools.source("ns__a") == ("ns", "a")
    assert tools.source("ns__b") is None and tools.source("a__x") is None


def test_request_translation_duplicate_name_inside_one_namespace_gets_the_prefixed_form():
    dup = {"type": "namespace", "name": "ns", "tools": [{"type": "function", "name": "f"}, {"type": "function", "name": "f"}]}
    chat, tools = chat_request_from_responses({"model": "m", "input": [], "tools": [dup]})
    assert [t["function"]["name"] for t in chat["tools"]] == ["f", "ns__f"]
    assert tools.namespaced == {"f": ("ns", "f"), "ns__f": ("ns", "f")}


def test_request_translation_top_level_tools_win_even_when_listed_after_a_namespace():
    chat, tools = chat_request_from_responses({"model": "m", "input": [], "tools": [NAMESPACE_TOOLS[1], NAMESPACE_TOOLS[0]]})
    assert [t["function"]["name"] for t in chat["tools"]] == ["marimo_status", "mcp__hailer__shell", "notebook_cells", "shell"]
    assert tools.namespaced["mcp__hailer__shell"] == ("mcp__hailer", "shell")


def test_request_translation_replays_namespaced_calls_with_the_advertised_names():
    chat, _ = chat_request_from_responses(
        {
            "model": "m",
            "input": [
                {"type": "function_call", "call_id": "c1", "name": "marimo_status", "namespace": "mcp__hailer", "arguments": "{}"},
                {"type": "function_call", "call_id": "c2", "name": "shell", "namespace": "mcp__hailer", "arguments": "{}"},
                {"type": "function_call", "call_id": "c3", "name": "shell", "arguments": '{"cmd": "ls"}'},
                {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            ],
            "tools": NAMESPACE_TOOLS,
            "tool_choice": {"type": "function", "name": "shell", "namespace": "mcp__hailer"},
        }
    )
    calls = chat["messages"][0]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["marimo_status", "mcp__hailer__shell", "shell"]
    assert chat["messages"][1] == {"role": "tool", "tool_call_id": "c1", "content": "ok"}
    assert chat["tool_choice"] == {"type": "function", "function": {"name": "mcp__hailer__shell"}}


# --------------------------------------------------------------------------- #
# Stream translation
# --------------------------------------------------------------------------- #


def _chunk(**delta: Any) -> dict[str, Any]:
    return {"id": "chatcmpl-1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


def _types(events: list[dict[str, Any]]) -> list[str]:
    return [e["type"] for e in events]


def test_stream_text_reply():
    t = ChatStreamTranslator(model="m")
    started = t.start()
    assert _types(started) == ["response.created", "response.in_progress"]
    rid = started[0]["response"]["id"]
    assert rid.startswith("resp_") and started[0]["response"]["status"] == "in_progress"

    ev = t.feed(_chunk(role="assistant", content=""))
    assert ev == []  # empty deltas do not open an item
    ev = t.feed(_chunk(content="Hel"))
    assert _types(ev) == ["response.output_item.added", "response.content_part.added", "response.output_text.delta"]
    assert ev[0]["item"]["type"] == "message" and ev[2]["delta"] == "Hel"
    ev = t.feed(_chunk(content="lo"))
    assert _types(ev) == ["response.output_text.delta"]
    ev = t.feed({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    assert ev == []
    ev = t.feed({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "prompt_tokens_details": {"cached_tokens": 4}}})
    assert ev == []
    done = t.finish()
    assert _types(done) == ["response.output_text.done", "response.output_item.done", "response.completed"]
    item = done[1]["item"]
    assert item["type"] == "message" and item["role"] == "assistant"
    assert item["content"] == [{"type": "output_text", "text": "Hello", "annotations": []}]
    completed = done[2]["response"]
    assert completed["id"] == rid and completed["status"] == "completed"
    assert completed["output"] == [item]
    assert completed["usage"] == {
        "input_tokens": 10,
        "input_tokens_details": {"cached_tokens": 4},
        "output_tokens": 2,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 12,
    }
    assert t.finish() == []  # idempotent


def test_stream_namespaced_tool_calls_come_back_with_namespace_and_bare_name():
    _, tools = chat_request_from_responses({"model": "m", "input": [], "tools": NAMESPACE_TOOLS})
    t = ChatStreamTranslator(tools=tools)
    t.start()
    t.feed(_chunk(tool_calls=[{"index": 0, "id": "call_1", "type": "function", "function": {"name": "marimo_status", "arguments": "{}"}}]))
    t.feed(_chunk(tool_calls=[{"index": 1, "id": "call_2", "type": "function", "function": {"name": "mcp__hailer__shell", "arguments": '{"cmd": "ls"}'}}]))
    t.feed(_chunk(tool_calls=[{"index": 2, "id": "call_3", "type": "function", "function": {"name": "shell", "arguments": "{}"}}]))
    done = t.finish()
    items = [e["item"] for e in done if e["type"] == "response.output_item.done"]
    assert items[0] == {
        "type": "function_call",
        "id": items[0]["id"],
        "call_id": "call_1",
        "name": "marimo_status",
        "namespace": "mcp__hailer",
        "arguments": "{}",
        "status": "completed",
    }
    assert items[1]["name"] == "shell" and items[1]["namespace"] == "mcp__hailer" and items[1]["arguments"] == '{"cmd": "ls"}'
    assert items[2]["name"] == "shell" and "namespace" not in items[2]  # a plain function stays plain
    added = [e["item"] for e in done if e["type"] == "response.output_item.added"]
    assert added[1]["namespace"] == "mcp__hailer" and added[1]["name"] == "shell" and added[1]["status"] == "in_progress"
    assert [o.get("namespace") for o in done[-1]["response"]["output"]] == ["mcp__hailer", "mcp__hailer", None]


def test_stream_prefixed_reply_name_for_a_bare_advertised_tool_is_still_routed():
    _, tools = chat_request_from_responses({"model": "m", "input": [], "tools": NAMESPACE_TOOLS})
    assert "mcp__hailer__marimo_status" not in tools.namespaced  # advertised as the bare name
    t = ChatStreamTranslator(tools=tools)
    t.start()
    t.feed(_chunk(tool_calls=[{"index": 0, "id": "call_1", "type": "function", "function": {"name": "mcp__hailer__marimo_status", "arguments": "{}"}}]))
    t.feed(_chunk(tool_calls=[{"index": 1, "id": "call_2", "type": "function", "function": {"name": "mcp__hailer__nope", "arguments": "{}"}}]))
    done = t.finish()
    items = [e["item"] for e in done if e["type"] == "response.output_item.done"]
    assert items[0]["name"] == "marimo_status" and items[0]["namespace"] == "mcp__hailer"
    assert items[1]["name"] == "mcp__hailer__nope" and "namespace" not in items[1]  # unknown remainder: left alone


def test_stream_whole_body_reply_with_a_namespaced_tool_call():
    _, tools = chat_request_from_responses({"model": "m", "input": [], "tools": NAMESPACE_TOOLS})
    t = ChatStreamTranslator(tools=tools)
    t.start()
    whole = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "notebook_cells", "arguments": "{}"}}]},
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }
    t.feed(whole)
    done = t.finish()
    output = done[-1]["response"]["output"]
    assert output == [
        {"type": "function_call", "id": output[0]["id"], "call_id": "call_1", "name": "notebook_cells", "namespace": "mcp__hailer", "arguments": "{}", "status": "completed"}
    ]


def test_stream_tool_calls_including_custom_and_parallel():
    t = ChatStreamTranslator(tools=ChatToolMap(custom=frozenset({"apply_patch"})))
    t.start()
    t.feed(_chunk(content="Running."))
    ev = t.feed(_chunk(tool_calls=[{"index": 0, "id": "call_1", "type": "function", "function": {"name": "shell", "arguments": ""}}]))
    # the text message closes before the first tool call
    assert _types(ev) == ["response.output_text.done", "response.output_item.done"]
    t.feed(_chunk(tool_calls=[{"index": 0, "function": {"arguments": '{"cmd":'}}]))
    t.feed(_chunk(tool_calls=[{"index": 0, "function": {"arguments": ' "ls"}'}}]))
    t.feed(_chunk(tool_calls=[{"index": 1, "id": "call_2", "function": {"name": "apply_patch", "arguments": '{"input": "*** Begin Patch"}'}}]))
    done = t.finish()
    assert _types(done) == [
        "response.output_item.added",
        "response.output_item.done",
        "response.output_item.added",
        "response.output_item.done",
        "response.completed",
    ]
    fc = done[1]["item"]
    assert fc == {"type": "function_call", "id": fc["id"], "call_id": "call_1", "name": "shell", "arguments": '{"cmd": "ls"}', "status": "completed"}
    custom = done[3]["item"]
    assert custom["type"] == "custom_tool_call" and custom["call_id"] == "call_2" and custom["input"] == "*** Begin Patch"
    output = done[4]["response"]["output"]
    assert [o["type"] for o in output] == ["message", "function_call", "custom_tool_call"]
    assert done[4]["response"].get("usage") is None


def test_stream_tool_call_without_id_gets_one():
    t = ChatStreamTranslator()
    t.start()
    t.feed(_chunk(tool_calls=[{"index": 0, "function": {"name": "f", "arguments": "{}"}}]))
    done = t.finish()
    call_id = done[1]["item"]["call_id"]
    assert call_id.startswith("call_")
    assert done[0]["item"]["call_id"] == call_id  # the `added` event carries the same id, never null


def test_stream_tool_call_deltas_without_index_continue_the_open_call():
    t = ChatStreamTranslator()
    t.start()
    t.feed(_chunk(tool_calls=[{"id": "c1", "type": "function", "function": {"name": "shell", "arguments": ""}}]))
    t.feed(_chunk(tool_calls=[{"function": {"arguments": '{"cmd":'}}]))
    t.feed(_chunk(tool_calls=[{"function": {"arguments": ' "ls"}'}}]))
    t.feed(_chunk(tool_calls=[{"id": "c1", "function": {"arguments": ""}}]))  # repeated id: same call
    t.feed(_chunk(tool_calls=[{"id": "c2", "function": {"name": "read", "arguments": "{}"}}]))  # new id: new call
    t.feed(_chunk(tool_calls=[{"function": {"name": "write", "arguments": "{}"}}]))  # name while one is open: new call
    output = t.finish()[-1]["response"]["output"]
    assert [(o["name"], o["arguments"], o["call_id"]) for o in output] == [
        ("shell", '{"cmd": "ls"}', "c1"),
        ("read", "{}", "c2"),
        ("write", "{}", output[2]["call_id"]),
    ]


def test_stream_reasoning_after_text_closes_the_message_first():
    t = ChatStreamTranslator()
    t.start()
    added = t.feed(_chunk(content="Let me"))
    ev = t.feed(_chunk(reasoning_content="hmm"))
    assert _types(ev)[:2] == ["response.output_text.done", "response.output_item.done"]
    assert ev[1]["output_index"] == added[0]["output_index"] == 0
    assert ev[2]["type"] == "response.output_item.added" and ev[2]["output_index"] == 1
    t.feed(_chunk(content=" check"))
    output = t.finish()[-1]["response"]["output"]
    assert [o["type"] for o in output] == ["message", "reasoning", "message"]


def test_stream_string_error_fails_the_response():
    t = ChatStreamTranslator()
    t.start()
    ev = t.feed({"error": "Rate limit exceeded"})
    assert _types(ev) == ["response.failed"] and ev[0]["response"]["error"] == {"message": "Rate limit exceeded"}
    t = ChatStreamTranslator()
    t.start()
    assert t.feed({"error": {}}) == [] and not t.failed  # an empty error object is not an error


def test_stream_content_part_with_null_text_is_ignored():
    t = ChatStreamTranslator()
    t.start()
    assert t.feed({"choices": [{"index": 0, "delta": {"content": [{"type": "text", "text": None}, {"type": "text", "text": "ok"}]}}]})
    assert t.finish()[-1]["response"]["output"][0]["content"][0]["text"] == "ok"


def test_reason_wording_matches_the_connection_signals():
    import socket
    import urllib.error

    assert _reason(urllib.error.URLError(socket.gaierror(-2, "Name or service not known"))).startswith("dns error (")
    assert _reason(urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))) == "[Errno 111] Connection refused"
    assert _reason(TimeoutError("The read operation timed out")) == "The read operation timed out"
    assert _reason(TimeoutError()) == "timed out (TimeoutError)"


def test_stream_reasoning_content_becomes_a_reasoning_item():
    t = ChatStreamTranslator()
    t.start()
    ev = t.feed(_chunk(reasoning_content="Consider "))
    assert _types(ev) == ["response.output_item.added", "response.reasoning_summary_part.added", "response.reasoning_summary_text.delta"]
    t.feed(_chunk(reasoning="the data."))  # alternate field name
    ev = t.feed(_chunk(content="Answer"))
    assert _types(ev)[:2] == ["response.reasoning_summary_text.done", "response.output_item.done"]
    assert ev[1]["item"] == {"type": "reasoning", "id": ev[1]["item"]["id"], "summary": [{"type": "summary_text", "text": "Consider the data."}], "content": []}
    done = t.finish()
    assert [o["type"] for o in done[-1]["response"]["output"]] == ["reasoning", "message"]


def test_stream_error_chunk_fails_the_response():
    t = ChatStreamTranslator()
    t.start()
    t.feed(_chunk(content="partial"))
    ev = t.feed({"error": {"message": "context length exceeded", "code": "context_length_exceeded"}})
    assert _types(ev) == ["response.failed"]
    assert ev[0]["response"]["error"] == {"message": "context length exceeded", "code": "context_length_exceeded"}
    assert t.failed and t.finish() == [] and t.feed(_chunk(content="more")) == []


def test_stream_content_parts_and_message_shape():
    t = ChatStreamTranslator()
    t.start()
    t.feed({"choices": [{"index": 0, "message": {"role": "assistant", "content": [{"type": "text", "text": "non-streamed"}]}}]})
    done = t.finish()
    assert done[1]["item"]["content"][0]["text"] == "non-streamed"


def test_sse_helpers():
    frame = sse_frame({"type": "response.created", "response": {"id": "r"}})
    assert frame.startswith(b"event: response.created\ndata: {") and frame.endswith(b"}\n\n")
    lines = [b": comment\n", b"data: {\"a\": 1}\n", b"\n", b"event: x\n", b"data:[DONE]\n", b"\n", b"data: tail\n"]
    assert list(iter_sse_data(lines)) == ['{"a": 1}', "[DONE]", "tail"]


# --------------------------------------------------------------------------- #
# The loopback server against a fake Chat Completions endpoint
# --------------------------------------------------------------------------- #


class FakeUpstream:
    """A Chat Completions server that records requests and streams scripted chunks."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        status: int = 200,
        body: bytes | None = None,
        done: bool = True,
        content_type: str = "text/event-stream",
    ) -> None:
        self.chunks = chunks
        self.status = status
        self.body = body
        self.done = done
        self.content_type = content_type
        self.requests: list[dict[str, Any]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:
                pass

            def do_GET(self) -> None:
                server.requests.append({"method": "GET", "path": self.path, "headers": dict(self.headers)})
                payload = b'{"object": "list", "data": []}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                server.requests.append({"method": "POST", "path": self.path, "headers": dict(self.headers), "json": json.loads(raw)})
                if server.status != 200:
                    payload = server.body or b'{"error": {"message": "nope"}}'
                    self.send_response(server.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(200)
                self.send_header("Content-Type", server.content_type)
                self.send_header("Connection", "close")
                self.end_headers()
                if server.body is not None:
                    self.wfile.write(server.body)
                    self.wfile.flush()
                    self.close_connection = True
                    return
                for chunk in server.chunks:
                    self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                    self.wfile.flush()
                if server.done:
                    self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "FakeUpstream":
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def _post(url: str, body: dict[str, Any], headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def _events(body: bytes) -> list[dict[str, Any]]:
    return [json.loads(d) for d in iter_sse_data(body.splitlines(keepends=True))]


@pytest.fixture
def no_proxy(monkeypatch):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


def test_bridge_translates_a_streamed_turn(no_proxy):
    chunks = [
        _chunk(role="assistant", content="Hi "),
        _chunk(content="there"),
        _chunk(tool_calls=[{"index": 0, "id": "call_9", "type": "function", "function": {"name": "shell", "arguments": "{}"}}]),
        _chunk(tool_calls=[{"index": 1, "id": "call_10", "type": "function", "function": {"name": "marimo_status", "arguments": "{}"}}]),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}},
    ]
    tools = [
        {"type": "function", "name": "shell", "parameters": {}},
        {"type": "namespace", "name": "mcp__hailer", "description": "Hailer tools.", "tools": [{"type": "function", "name": "marimo_status", "parameters": {"type": "object"}}]},
    ]
    with FakeUpstream(chunks) as up, ChatBridge({"internal": up.base_url}) as bridge:
        assert bridge.urls == {"internal": f"http://127.0.0.1:{bridge.port}/internal"}
        status, headers, body = _post(
            bridge.urls["internal"] + "/responses?api-version=2025-04-01",
            {"model": "m", "instructions": "sys", "input": "hi", "tools": tools, "stream": True},
            {"Authorization": "Bearer sk-secret", "X-Team": "risk", "OpenAI-Beta": "responses=experimental", "Accept": "text/event-stream"},
        )
    assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
    events = _events(body)
    kinds = [e["type"] for e in events]
    assert kinds[:2] == ["response.created", "response.in_progress"]
    assert kinds[-1] == "response.completed"
    assert "response.output_text.delta" in kinds and "response.output_item.done" in kinds
    completed = events[-1]["response"]
    assert [o["type"] for o in completed["output"]] == ["message", "function_call", "function_call"]
    assert completed["output"][0]["content"][0]["text"] == "Hi there"
    assert completed["output"][1]["call_id"] == "call_9" and "namespace" not in completed["output"][1]
    # the MCP tool call goes back to Codex as (namespace, bare name), which is how it routes to the server
    assert completed["output"][2]["call_id"] == "call_10"
    assert completed["output"][2]["name"] == "marimo_status" and completed["output"][2]["namespace"] == "mcp__hailer"
    assert completed["usage"]["input_tokens"] == 5 and completed["usage"]["output_tokens"] == 3

    (req,) = up.requests
    assert req["path"] == "/v1/chat/completions?api-version=2025-04-01"
    assert req["headers"]["Authorization"] == "Bearer sk-secret"
    assert req["headers"]["X-Team"] == "risk"
    assert req["headers"]["Accept"] == "text/event-stream"
    assert req["json"]["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert req["json"]["stream"] is True
    assert [t["function"]["name"] for t in req["json"]["tools"]] == ["shell", "marimo_status"]  # the namespace is flattened


def test_bridge_relays_upstream_http_errors(no_proxy):
    with FakeUpstream([], status=401, body=b'{"error": {"message": "invalid api key"}}') as up, ChatBridge({"p": up.base_url}) as bridge:
        status, _, body = _post(bridge.urls["p"] + "/responses", {"model": "m", "input": "hi"})
    assert status == 401 and json.loads(body)["error"]["message"] == "invalid api key"


def test_bridge_reports_unreachable_upstream_as_502(no_proxy):
    with ChatBridge({"p": "http://127.0.0.1:9/v1"}) as bridge:
        status, _, body = _post(bridge.urls["p"] + "/responses", {"model": "m", "input": "hi"})
    assert status == 502
    message = json.loads(body)["error"]["message"]
    assert "could not reach http://127.0.0.1:9/v1/chat/completions" in message


def test_bridge_passes_models_probe_through_and_rejects_unknown_paths(no_proxy):
    with FakeUpstream([]) as up, ChatBridge({"p": up.base_url}) as bridge:
        with urllib.request.urlopen(bridge.urls["p"] + "/models", timeout=10) as resp:
            assert resp.status == 200 and json.loads(resp.read())["object"] == "list"
        assert up.requests[-1]["path"] == "/v1/models"
        status, _, _ = _post(bridge.urls["p"] + "/responses/compact", {})
        assert status == 404
        status, _, _ = _post(f"http://127.0.0.1:{bridge.port}/other/responses", {})
        assert status == 404
        status, _, _ = _post(bridge.urls["p"] + "/responses", {"model": "m", "input": "hi"}, {})
        assert status == 200


def test_bridge_detects_sse_with_a_wrong_content_type(no_proxy):
    chunks = [_chunk(content="streamed"), {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
    with FakeUpstream(chunks, content_type="text/plain") as up, ChatBridge({"p": up.base_url}) as bridge:
        status, _, body = _post(bridge.urls["p"] + "/responses", {"model": "m", "input": "hi"})
    assert status == 200
    events = _events(body)
    assert "response.output_text.delta" in [e["type"] for e in events]
    assert events[-1]["response"]["output"][0]["content"][0]["text"] == "streamed"


def test_bridge_accepts_a_non_streamed_json_reply(no_proxy):
    reply = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "whole"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    with FakeUpstream([], content_type="application/json", body=json.dumps(reply).encode()) as up, ChatBridge({"p": up.base_url}) as bridge:
        status, _, body = _post(bridge.urls["p"] + "/responses", {"model": "m", "input": "hi"})
    assert status == 200
    completed = _events(body)[-1]
    assert completed["type"] == "response.completed"
    assert completed["response"]["output"][0]["content"][0]["text"] == "whole"
    assert completed["response"]["usage"]["total_tokens"] == 2


def test_bridge_asks_for_one_json_reply_when_streaming_is_off(no_proxy):
    reply = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "whole", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "shell", "arguments": "{\"cmd\": \"ls\"}"}}]},
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }
    upstream = FakeUpstream([], content_type="application/json", body=json.dumps(reply).encode())
    with upstream as up, ChatBridge({"p": ChatUpstream(up.base_url, stream=False)}) as bridge:
        assert bridge.streaming == {"p": False}
        status, headers, body = _post(
            bridge.urls["p"] + "/responses",
            {"model": "m", "input": "hi", "tools": [{"type": "function", "name": "shell", "parameters": {}}], "stream": True},
            {"Accept": "text/event-stream"},
        )
    assert status == 200 and headers["Content-Type"].startswith("text/event-stream")  # Codex still gets SSE
    (req,) = up.requests
    assert req["json"]["stream"] is False and "stream_options" not in req["json"]
    assert req["headers"]["Accept"] == "application/json"
    events = _events(body)
    assert events[0]["type"] == "response.created" and events[-1]["type"] == "response.completed"
    output = events[-1]["response"]["output"]
    assert output[0]["content"][0]["text"] == "whole"
    assert output[1]["type"] == "function_call" and output[1]["call_id"] == "call_1" and output[1]["arguments"] == '{"cmd": "ls"}'
    assert events[-1]["response"]["usage"]["total_tokens"] == 7


def test_bridge_non_streamed_reply_routes_a_namespaced_tool_call(no_proxy):
    reply = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None, "tool_calls": [{"id": "call_7", "type": "function", "function": {"name": "marimo_status", "arguments": "{}"}}]},
                "finish_reason": "tool_calls",
            }
        ]
    }
    tools = [{"type": "namespace", "name": "mcp__hailer", "tools": [{"type": "function", "name": "marimo_status", "parameters": {"type": "object"}}]}]
    upstream = FakeUpstream([], content_type="application/json", body=json.dumps(reply).encode())
    with upstream as up, ChatBridge({"p": ChatUpstream(up.base_url, stream=False)}) as bridge:
        status, _, body = _post(bridge.urls["p"] + "/responses", {"model": "m", "input": "hi", "tools": tools, "stream": True})
    assert status == 200
    (req,) = up.requests
    assert req["json"]["stream"] is False and [t["function"]["name"] for t in req["json"]["tools"]] == ["marimo_status"]
    (call,) = _events(body)[-1]["response"]["output"]
    assert call["type"] == "function_call" and call["call_id"] == "call_7"
    assert call["name"] == "marimo_status" and call["namespace"] == "mcp__hailer"


def test_bridge_streams_by_default_for_plain_string_upstreams():
    bridge = ChatBridge({"p": "http://127.0.0.1:1/v1/", "q": ChatUpstream("http://127.0.0.1:2/v1/")})
    assert bridge.upstreams == {"p": "http://127.0.0.1:1/v1", "q": "http://127.0.0.1:2/v1"}
    assert bridge.streaming == {"p": True, "q": True}


def test_bridge_ends_the_stream_cleanly_when_translation_raises(no_proxy, monkeypatch):
    def boom(self, chunk):
        raise TypeError("sequence item 0: expected str instance, NoneType found")

    monkeypatch.setattr(wire_mod.ChatStreamTranslator, "feed", boom)
    with FakeUpstream([_chunk(content="x")]) as up, ChatBridge({"p": up.base_url}) as bridge:
        status, _, body = _post(bridge.urls["p"] + "/responses", {"model": "m", "input": "hi"})
    assert status == 200  # headers were already out; the failure travels inside the stream
    events = _events(body)
    assert events[-1]["type"] == "response.failed"
    assert "NoneType" in events[-1]["response"]["error"]["message"]


def test_bridge_stream_without_done_still_completes(no_proxy):
    with FakeUpstream([_chunk(content="x")], done=False) as up, ChatBridge({"p": up.base_url}) as bridge:
        status, _, body = _post(bridge.urls["p"] + "/responses", {"model": "m", "input": "hi"})
    assert status == 200
    assert _events(body)[-1]["type"] == "response.completed"


def test_bridge_close_is_idempotent_and_urls_empty_before_start():
    bridge = ChatBridge({"p": "http://127.0.0.1:1/v1/"})
    assert bridge.urls == {} and bridge.port is None
    assert bridge.upstreams == {"p": "http://127.0.0.1:1/v1"}
    bridge.close()
    bridge.start()
    assert bridge.port
    bridge.close()
    bridge.close()


def test_bridge_logs_upstream_http_errors_at_debug(no_proxy):
    import logging

    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger = logging.getLogger("hailer.wire")
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    body = b'{"detail": [{"loc": ["body", "model"], "msg": "String should match pattern"}]}'
    try:
        with FakeUpstream([], status=422, body=body) as up, ChatBridge({"p": up.base_url}) as bridge:
            status, _, relayed = _post(bridge.urls["p"] + "/responses", {"model": "m", "input": "hi"})
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
    assert status == 422 and relayed == body  # relay unchanged
    messages = [r.getMessage() for r in records]
    assert any(
        "answered 422" in m and "String should match pattern" in m and up.base_url + "/chat/completions" in m
        for m in messages
    ), messages


# --------------------------------------------------------------------------- #
# Strict-gateway request shaping: merged messages, optional fields
# --------------------------------------------------------------------------- #


def _codex_like_request():
    """Codex's real shape: instructions plus a developer message, then environment context plus the user's text."""
    return {
        "model": "m",
        "instructions": "# Hailer agent instructions",
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "<skills_instructions>...</skills_instructions>"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<environment_context>cwd</environment_context>"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Reply with pong"}]},
        ],
        "tools": [{"type": "function", "name": "shell", "parameters": {"type": "object"}}],
        "parallel_tool_calls": True,
        "stream": True,
    }


def test_request_translation_merges_consecutive_system_and_user_messages():
    chat, _ = chat_request_from_responses(_codex_like_request())
    assert [m["role"] for m in chat["messages"]] == ["system", "user"]
    assert chat["messages"][0]["content"] == "# Hailer agent instructions\n\n<skills_instructions>...</skills_instructions>"
    assert chat["messages"][1]["content"] == "<environment_context>cwd</environment_context>\n\nReply with pong"


def test_request_translation_merge_can_be_turned_off():
    chat, _ = chat_request_from_responses(_codex_like_request(), merge_messages=False)
    assert [m["role"] for m in chat["messages"]] == ["system", "system", "user", "user"]


def test_request_translation_merge_keeps_text_and_image_parts_in_order():
    body = {
        "model": "m",
        "input": [
            {"type": "message", "role": "user", "content": "Look at this"},
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "the chart"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "low"},
                ],
            },
            {"type": "message", "role": "user", "content": "and say what you see"},
        ],
    }
    chat, _ = chat_request_from_responses(body)
    (user,) = chat["messages"]
    assert user["role"] == "user"
    assert user["content"] == [
        {"type": "text", "text": "Look at this"},
        {"type": "text", "text": "the chart"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "low"}},
        {"type": "text", "text": "and say what you see"},
    ]


def test_request_translation_never_merges_assistant_or_tool_messages():
    body = {
        "model": "m",
        "input": [
            {"type": "message", "role": "user", "content": "run two things"},
            {"type": "message", "role": "assistant", "content": "First."},
            {"type": "message", "role": "assistant", "content": "Second."},
            {"type": "function_call", "call_id": "a", "name": "shell", "arguments": "{}"},
            {"type": "function_call", "call_id": "b", "name": "shell", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "a", "output": "one"},
            {"type": "function_call_output", "call_id": "b", "output": "two"},
            {"type": "message", "role": "user", "content": "thanks"},
        ],
    }
    chat, _ = chat_request_from_responses(body)
    assert [m["role"] for m in chat["messages"]] == ["user", "assistant", "assistant", "tool", "tool", "user"]
    assert chat["messages"][1]["content"] == "First." and chat["messages"][2]["content"] == "Second."
    assert [c["id"] for c in chat["messages"][2]["tool_calls"]] == ["a", "b"]  # calls still attach to the trailing assistant message
    assert [m["tool_call_id"] for m in chat["messages"][3:5]] == ["a", "b"]


def test_request_translation_merge_drops_empty_text():
    body = {
        "model": "m",
        "instructions": "",
        "input": [
            {"type": "message", "role": "system", "content": ""},
            {"type": "message", "role": "system", "content": "rules"},
            {"type": "message", "role": "user", "content": "hi"},
        ],
    }
    chat, _ = chat_request_from_responses(body)
    assert chat["messages"] == [{"role": "system", "content": "rules"}, {"role": "user", "content": "hi"}]


def test_request_translation_can_omit_stream_options():
    chat, _ = chat_request_from_responses(_codex_like_request(), stream_options=False)
    assert chat["stream"] is True and "stream_options" not in chat
    chat, _ = chat_request_from_responses(_codex_like_request())
    assert chat["stream_options"] == {"include_usage": True}  # the default is unchanged


def test_request_translation_can_omit_parallel_tool_calls():
    chat, _ = chat_request_from_responses(_codex_like_request(), parallel_tool_calls=False)
    assert chat["tools"] and "parallel_tool_calls" not in chat  # left out, never sent as false
    body = dict(_codex_like_request(), parallel_tool_calls=False)
    chat, _ = chat_request_from_responses(body, parallel_tool_calls=False)
    assert "parallel_tool_calls" not in chat
    chat, _ = chat_request_from_responses(body)
    assert chat["parallel_tool_calls"] is False  # by default Codex's own value passes through


def test_chat_upstream_request_options_default_on():
    up = ChatUpstream("http://gateway.example/v1/")
    assert (up.stream, up.merge_messages, up.stream_options, up.parallel_tool_calls) == (True, True, True, True)


def test_bridge_applies_the_upstream_request_options(no_proxy):
    chunks = [_chunk(role="assistant", content="ok"), {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
    with FakeUpstream(chunks) as up:
        strict = ChatUpstream(up.base_url, merge_messages=False, stream_options=False, parallel_tool_calls=False)
        with ChatBridge({"internal": strict}) as bridge:
            status, _, body = _post(bridge.urls["internal"] + "/responses", _codex_like_request(), {"Accept": "text/event-stream"})
    assert status == 200 and _events(body)[-1]["type"] == "response.completed"
    sent = up.requests[-1]["json"]
    assert [m["role"] for m in sent["messages"]] == ["system", "system", "user", "user"]
    assert sent["stream"] is True and "stream_options" not in sent and "parallel_tool_calls" not in sent


def test_bridge_merges_messages_and_keeps_the_optional_fields_by_default(no_proxy):
    chunks = [_chunk(role="assistant", content="ok"), {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
    with FakeUpstream(chunks) as up, ChatBridge({"internal": up.base_url}) as bridge:
        _post(bridge.urls["internal"] + "/responses", _codex_like_request(), {"Accept": "text/event-stream"})
    sent = up.requests[-1]["json"]
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert sent["stream_options"] == {"include_usage": True} and sent["parallel_tool_calls"] is True
