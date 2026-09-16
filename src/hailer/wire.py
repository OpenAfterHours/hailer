"""Chat Completions bridge for providers declared with ``wire_api = "chat"``.

The bundled Codex runtime speaks only the OpenAI Responses API (it refuses
``wire_api = "chat"`` outright). To let a gateway that offers Chat Completions
only be used without an external proxy, Hailer runs a small loopback HTTP server
(:class:`ChatBridge`), points Codex at it, and translates every streamed
``POST /responses`` into ``POST {base_url}/chat/completions`` and the Chat
Completions reply back into Responses SSE events. Upstream streaming is per
provider (:class:`ChatUpstream`): normally the bridge asks for ``stream: true`` and
relays the chunks as they arrive; with ``stream=False`` (``stream = false`` in
``hailer.toml``) it sends ``stream: false``, reads the single JSON reply and emits
the same Responses events for it in one go, for gateways that reject or cannot
deliver server-sent events.

Codex still attaches the provider's credentials and custom headers itself
(``env_key`` → ``Authorization``, ``http_headers``, ``env_http_headers``,
``query_params``); the bridge forwards them upstream unchanged and holds no
secret of its own.

The translation functions are pure (dict in, dicts out) so they are testable
without sockets.
"""

from __future__ import annotations

import json
import logging
import socket
import ssl
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import MappingProxyType
from typing import Any

from hailer.models import VALID_WIRE_APIS, WIRE_API_CHAT, WIRE_API_RESPONSES  # noqa: F401 - re-exported

try:  # keep importable when hailer.log is absent (see agent.py)
    from hailer.log import get_logger
except Exception:  # pragma: no cover

    def get_logger(name: str = "hailer") -> logging.Logger:
        return logging.getLogger(name)


log = get_logger("hailer.wire")

#: Seconds without a byte from the upstream stream before the bridge gives up.
UPSTREAM_READ_TIMEOUT_SEC = 300
#: How often the server loop checks for shutdown; keeps ``close()`` (and process exit) prompt.
_SERVE_POLL_INTERVAL_SEC = 0.05
#: Byte prefixes that identify an SSE body whatever Content-Type the gateway declared.
_SSE_PREFIXES = (b"data:", b"event:", b"id:", b"retry:", b":")


@dataclass(frozen=True)
class ChatUpstream:
    """One Chat Completions endpoint behind the bridge.

    ``stream`` is whether to ask the gateway for server-sent events (``stream: true``); when
    False the bridge sends ``stream: false`` and expects one JSON ``chat.completion`` body.
    """

    base_url: str
    stream: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))


# Hop-by-hop or request-specific headers that must not be copied upstream.
_DROP_REQUEST_HEADERS = {
    "host",
    "content-length",
    "content-type",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "accept",
    "accept-encoding",
    "expect",
}


# --------------------------------------------------------------------------- #
# Responses request -> Chat Completions request (pure)
# --------------------------------------------------------------------------- #


def _text_of(content: Any) -> str:
    """Flatten Responses message content (string or list of parts) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    out.append(text)
        return "\n".join(out)
    return str(content)


def _message_content(role: str, content: Any) -> Any:
    """Chat content for a Responses message: a plain string, or parts when images are present."""
    if not isinstance(content, list):
        return _text_of(content)
    images = [p for p in content if isinstance(p, Mapping) and p.get("type") == "input_image"]
    if role != "user" or not images:
        return _text_of(content)
    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        if part.get("type") == "input_image":
            url = part.get("image_url")
            if isinstance(url, Mapping):
                url = url.get("url")
            if not url:
                continue
            image: dict[str, Any] = {"url": url}
            if part.get("detail"):
                image["detail"] = part["detail"]
            parts.append({"type": "image_url", "image_url": image})
        elif isinstance(part.get("text"), str):
            parts.append({"type": "text", "text": part["text"]})
    return parts


@dataclass(frozen=True)
class ChatToolMap:
    """How one request's Responses tools were advertised to Chat Completions.

    ``custom`` names the custom (free-form) tools; their calls come back as ``custom_tool_call``
    items. ``namespaced`` maps each chat-side function name that stands for a tool inside a
    Responses ``namespace`` tool (Codex sends every MCP server's tools as one such tool) to
    ``(namespace, tool name)``; their calls come back as ``function_call`` items carrying
    ``namespace`` and the bare tool name, which is how Codex routes them to the MCP server.
    """

    custom: frozenset[str] = frozenset()
    namespaced: Mapping[str, tuple[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # A frozen value object must not hand out a mutable dict: keep a read-only view.
        object.__setattr__(self, "custom", frozenset(self.custom))
        object.__setattr__(self, "namespaced", MappingProxyType(dict(self.namespaced)))

    def chat_name(self, name: str, namespace: str | None = None) -> str:
        """The chat-side function name for a Responses tool call, as advertised in ``tools``.

        Falls back to ``name`` when the call is not namespaced or the namespace is unknown (for
        example a call recorded before the tool set changed).
        """
        if namespace:
            for chat, (ns, bare) in self.namespaced.items():
                if ns == namespace and bare == name:
                    return chat
        return name

    def source(self, chat_name: str) -> tuple[str, str] | None:
        """The ``(namespace, bare name)`` a function name in the reply stands for; None for a plain function.

        Accepts the advertised name and, as a fallback, the prefixed form ``<namespace>__<name>``
        that a model may produce even when the bare name was advertised.
        """
        mapped = self.namespaced.get(chat_name)
        if mapped is not None:
            return mapped
        for ns, bare in self.namespaced.values():
            if chat_name == f"{ns}__{bare}":
                return (ns, bare)
        return None


def _function_tool(name: str, tool: Mapping[str, Any]) -> dict[str, Any]:
    fn: dict[str, Any] = {"name": name}
    if tool.get("description"):
        fn["description"] = tool["description"]
    fn["parameters"] = tool.get("parameters") or {"type": "object", "properties": {}}
    if tool.get("strict") is True:
        fn["strict"] = True
    return {"type": "function", "function": fn}


def _chat_tools(tools: Any) -> tuple[list[dict[str, Any]], ChatToolMap]:
    """Translate Responses tools to Chat Completions tools; returns them plus their :class:`ChatToolMap`.

    Custom (free-form) tools have no Chat Completions equivalent, so each becomes a function
    with a single required string parameter ``input``; the bridge maps such calls back to a
    ``custom_tool_call`` item. A ``namespace`` tool (the shape Codex uses for an MCP server's
    tools) is flattened: every function inside it becomes an ordinary function tool under its
    bare name, or under ``<namespace>__<name>`` when that name is already taken by a top-level
    tool, by an earlier namespace or by an earlier tool of the same namespace, so the order of
    the tools decides deterministically. The namespace's own description is not sent; each tool
    keeps its own. Hosted tools (web search, local shell, ...) are dropped.
    """
    entries = [t for t in tools or [] if isinstance(t, Mapping)]
    out: list[dict[str, Any]] = []
    custom: set[str] = set()
    namespaced: dict[str, tuple[str, str]] = {}
    # Top-level names win over namespaced ones wherever they appear in the list.
    used: set[str] = {t["name"] for t in entries if t.get("type") in ("function", "custom") and isinstance(t.get("name"), str)}
    for tool in entries:
        kind = tool.get("type")
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        if kind == "function":
            out.append(_function_tool(name, tool))
        elif kind == "custom":
            custom.add(name)
            fn = {
                "name": name,
                "parameters": {
                    "type": "object",
                    "properties": {"input": {"type": "string", "description": "The raw tool input."}},
                    "required": ["input"],
                },
            }
            if tool.get("description"):
                fn["description"] = tool["description"]
            out.append({"type": "function", "function": fn})
        elif kind == "namespace":
            if tool.get("description"):
                log.debug("namespace %r description (%d chars) is not sent to chat completions; each tool keeps its own", name, len(str(tool["description"])))
            for inner in tool.get("tools") or []:
                if not isinstance(inner, Mapping):
                    continue
                bare = inner.get("name")
                if not isinstance(bare, str) or not bare:
                    continue
                if inner.get("type", "function") != "function":
                    log.debug("dropping unsupported tool type %r inside namespace %r for chat completions", inner.get("type"), name)
                    continue
                chat_name = bare if bare not in used else f"{name}__{bare}"
                while chat_name in used:  # a literal "<namespace>__<name>" tool is unlikely but must not collide
                    chat_name += "_"
                used.add(chat_name)
                namespaced[chat_name] = (name, bare)
                out.append(_function_tool(chat_name, inner))
        else:
            log.debug("dropping unsupported tool type %r for chat completions", kind)
    return out, ChatToolMap(custom=frozenset(custom), namespaced=namespaced)


def _tool_choice(choice: Any, tool_map: ChatToolMap) -> Any:
    if isinstance(choice, str):
        return choice
    if isinstance(choice, Mapping):
        if choice.get("type") == "function" and choice.get("name"):
            return {"type": "function", "function": {"name": tool_map.chat_name(choice["name"], choice.get("namespace"))}}
        return None
    return None


def _append_tool_call(messages: list[dict[str, Any]], call: dict[str, Any]) -> None:
    """Attach a tool call to the trailing assistant message, or open a new one."""
    if messages and messages[-1].get("role") == "assistant":
        last = messages[-1]
        last.setdefault("tool_calls", []).append(call)
        if not last.get("content"):
            last["content"] = None
        return
    messages.append({"role": "assistant", "content": None, "tool_calls": [call]})


def chat_request_from_responses(body: Mapping[str, Any], *, stream: bool = True) -> tuple[dict[str, Any], ChatToolMap]:
    """Build the Chat Completions request body for a Responses API request.

    Returns the body and the :class:`ChatToolMap` describing how the tools were advertised;
    hand the map to :class:`ChatStreamTranslator` so the reply's tool calls are translated
    back consistently. Earlier calls replayed in ``input`` use the same chat-side names.
    """
    tools, tool_map = _chat_tools(body.get("tools"))

    messages: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})

    items = body.get("input")
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    for item in items or []:
        if not isinstance(item, Mapping):
            continue
        kind = item.get("type") or ("message" if "role" in item else None)
        if kind == "message":
            role = item.get("role") or "user"
            if role == "developer":
                role = "system"
            if role not in ("system", "user", "assistant"):
                role = "user"
            messages.append({"role": role, "content": _message_content(role, item.get("content"))})
        elif kind == "function_call":
            args = item.get("arguments")
            name = tool_map.chat_name(item.get("name") or "", item.get("namespace"))
            _append_tool_call(
                messages,
                {
                    "id": item.get("call_id") or item.get("id") or f"call_{len(messages)}",
                    "type": "function",
                    "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args or {})},
                },
            )
        elif kind == "custom_tool_call":
            _append_tool_call(
                messages,
                {
                    "id": item.get("call_id") or item.get("id") or f"call_{len(messages)}",
                    "type": "function",
                    "function": {"name": item.get("name") or "", "arguments": json.dumps({"input": item.get("input") or ""})},
                },
            )
        elif kind in ("function_call_output", "custom_tool_call_output"):
            messages.append({"role": "tool", "tool_call_id": item.get("call_id") or "", "content": _text_of(item.get("output"))})
        else:
            # reasoning items, hosted tool calls, compaction markers: nothing to send.
            log.debug("dropping input item of type %r for chat completions", kind)

    chat: dict[str, Any] = {"model": body.get("model"), "messages": messages, "stream": stream}
    if stream:
        chat["stream_options"] = {"include_usage": True}

    if tools:
        chat["tools"] = tools
        choice = _tool_choice(body.get("tool_choice"), tool_map)
        if choice is not None:
            chat["tool_choice"] = choice
        if isinstance(body.get("parallel_tool_calls"), bool):
            chat["parallel_tool_calls"] = body["parallel_tool_calls"]

    reasoning = body.get("reasoning")
    if isinstance(reasoning, Mapping) and reasoning.get("effort"):
        chat["reasoning_effort"] = reasoning["effort"]

    text = body.get("text")
    if isinstance(text, Mapping):
        fmt = text.get("format")
        if isinstance(fmt, Mapping) and fmt.get("type") == "json_schema":
            schema: dict[str, Any] = {"name": fmt.get("name") or "output", "schema": fmt.get("schema") or {}}
            if fmt.get("strict") is True:
                schema["strict"] = True
            chat["response_format"] = {"type": "json_schema", "json_schema": schema}
        if text.get("verbosity"):
            chat["verbosity"] = text["verbosity"]

    if isinstance(body.get("max_output_tokens"), int):
        chat["max_tokens"] = body["max_output_tokens"]
    return chat, tool_map


# --------------------------------------------------------------------------- #
# Chat Completions stream -> Responses SSE events (pure)
# --------------------------------------------------------------------------- #


def _usage_from_chat(usage: Mapping[str, Any]) -> dict[str, Any]:
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    pd = usage.get("prompt_tokens_details") or {}
    cd = usage.get("completion_tokens_details") or {}
    return {
        "input_tokens": prompt,
        "input_tokens_details": {"cached_tokens": int((pd.get("cached_tokens") if isinstance(pd, Mapping) else 0) or 0)},
        "output_tokens": completion,
        "output_tokens_details": {"reasoning_tokens": int((cd.get("reasoning_tokens") if isinstance(cd, Mapping) else 0) or 0)},
        "total_tokens": int(usage.get("total_tokens") or (prompt + completion)),
    }


class ChatStreamTranslator:
    """Turn Chat Completions stream chunks into Responses API events, one chunk at a time.

    Usage: ``events = t.start()``; then ``events = t.feed(chunk)`` per parsed chunk; finally
    ``events = t.finish()``. Each returned event is a dict with a ``type`` key, ready to be
    serialised as the ``data:`` of an SSE frame. ``tools`` is the :class:`ChatToolMap` of the
    request being answered, so custom and namespaced tool calls are translated back correctly.
    """

    def __init__(self, *, model: str | None = None, tools: ChatToolMap | None = None) -> None:
        self.response_id = "resp_" + uuid.uuid4().hex
        self.model = model
        self._tools = tools or ChatToolMap()
        self._output_index = 0
        self._items: list[dict[str, Any]] = []  # completed output items, in order
        # in-flight pieces
        self._reasoning: list[str] = []
        self._reasoning_id: str | None = None
        self._text: list[str] = []
        self._message_id: str | None = None
        self._calls: dict[int, dict[str, Any]] = {}
        self._usage: dict[str, Any] | None = None
        self._finished = False
        self._failed = False

    # ---- helpers -----------------------------------------------------------

    def _response(self, status: str, **extra: Any) -> dict[str, Any]:
        resp: dict[str, Any] = {"id": self.response_id, "object": "response", "status": status, "output": list(self._items)}
        if self.model:
            resp["model"] = self.model
        resp.update(extra)
        return resp

    def _close_reasoning(self) -> list[dict[str, Any]]:
        if self._reasoning_id is None:
            return []
        text = "".join(self._reasoning)
        item = {"type": "reasoning", "id": self._reasoning_id, "summary": [{"type": "summary_text", "text": text}], "content": []}
        index = self._output_index
        self._output_index += 1
        self._items.append(item)
        self._reasoning_id = None
        self._reasoning = []
        return [
            {"type": "response.reasoning_summary_text.done", "item_id": item["id"], "output_index": index, "summary_index": 0, "text": text},
            {"type": "response.output_item.done", "output_index": index, "item": item},
        ]

    def _close_message(self) -> list[dict[str, Any]]:
        if self._message_id is None:
            return []
        text = "".join(self._text)
        item = {
            "type": "message",
            "id": self._message_id,
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
        index = self._output_index
        self._output_index += 1
        self._items.append(item)
        self._message_id = None
        self._text = []
        return [
            {"type": "response.output_text.done", "item_id": item["id"], "output_index": index, "content_index": 0, "text": text},
            {"type": "response.output_item.done", "output_index": index, "item": item},
        ]

    def _close_calls(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for _, call in sorted(self._calls.items()):
            name = call["name"]
            arguments = "".join(call["arguments"])
            if not call["call_id"]:
                call["call_id"] = "call_" + uuid.uuid4().hex[:12]
            index = self._output_index
            self._output_index += 1
            if name in self._tools.custom:
                try:
                    parsed = json.loads(arguments) if arguments else {}
                    raw_input = parsed.get("input", arguments) if isinstance(parsed, Mapping) else arguments
                except ValueError:
                    raw_input = arguments
                item: dict[str, Any] = {
                    "type": "custom_tool_call",
                    "id": call["item_id"],
                    "call_id": call["call_id"],
                    "name": name,
                    "input": raw_input if isinstance(raw_input, str) else json.dumps(raw_input),
                    "status": "completed",
                }
            else:
                item = {
                    "type": "function_call",
                    "id": call["item_id"],
                    "call_id": call["call_id"],
                    "name": name,
                    "arguments": arguments or "{}",
                    "status": "completed",
                }
                mapped = self._tools.source(name)
                if mapped is not None:
                    # Codex routes a namespaced call by (namespace, bare name); a prefixed name alone is unknown to it.
                    item["namespace"], item["name"] = mapped
            self._items.append(item)
            events.append({"type": "response.output_item.added", "output_index": index, "item": dict(item, status="in_progress")})
            events.append({"type": "response.output_item.done", "output_index": index, "item": item})
        self._calls = {}
        return events

    # ---- public ------------------------------------------------------------

    def start(self) -> list[dict[str, Any]]:
        created = self._response("in_progress")
        return [
            {"type": "response.created", "response": created},
            {"type": "response.in_progress", "response": created},
        ]

    def feed(self, chunk: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Translate one parsed stream chunk (the JSON after ``data:``)."""
        if self._finished:
            return []
        error = chunk.get("error")
        if isinstance(error, Mapping):
            if error:
                return self.fail(str(error.get("message") or error), code=error.get("code") or error.get("type"))
        elif error:
            return self.fail(str(error))

        events: list[dict[str, Any]] = []
        usage = chunk.get("usage")
        if isinstance(usage, Mapping) and usage:
            self._usage = _usage_from_chat(usage)

        for choice in chunk.get("choices") or []:
            if not isinstance(choice, Mapping):
                continue
            delta = choice.get("delta") or choice.get("message") or {}
            if not isinstance(delta, Mapping):
                continue

            reasoning = delta.get("reasoning_content")
            if reasoning is None:
                reasoning = delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                if self._reasoning_id is None:
                    events.extend(self._close_message())
                    self._reasoning_id = "rs_" + uuid.uuid4().hex
                    events.append(
                        {
                            "type": "response.output_item.added",
                            "output_index": self._output_index,
                            "item": {"type": "reasoning", "id": self._reasoning_id, "summary": []},
                        }
                    )
                    events.append(
                        {
                            "type": "response.reasoning_summary_part.added",
                            "item_id": self._reasoning_id,
                            "output_index": self._output_index,
                            "summary_index": 0,
                            "part": {"type": "summary_text", "text": ""},
                        }
                    )
                self._reasoning.append(reasoning)
                events.append(
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": self._reasoning_id,
                        "output_index": self._output_index,
                        "summary_index": 0,
                        "delta": reasoning,
                    }
                )

            content = delta.get("content")
            if isinstance(content, list):  # some gateways stream content parts
                content = "".join(p["text"] for p in content if isinstance(p, Mapping) and isinstance(p.get("text"), str))
            if isinstance(content, str) and content:
                events.extend(self._close_reasoning())
                if self._message_id is None:
                    self._message_id = "msg_" + uuid.uuid4().hex
                    events.append(
                        {
                            "type": "response.output_item.added",
                            "output_index": self._output_index,
                            "item": {"type": "message", "id": self._message_id, "role": "assistant", "status": "in_progress", "content": []},
                        }
                    )
                    events.append(
                        {
                            "type": "response.content_part.added",
                            "item_id": self._message_id,
                            "output_index": self._output_index,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": "", "annotations": []},
                        }
                    )
                self._text.append(content)
                events.append(
                    {
                        "type": "response.output_text.delta",
                        "item_id": self._message_id,
                        "output_index": self._output_index,
                        "content_index": 0,
                        "delta": content,
                    }
                )

            for call in delta.get("tool_calls") or []:
                if not isinstance(call, Mapping):
                    continue
                events.extend(self._close_reasoning())
                events.extend(self._close_message())
                fn = call.get("function") or {}
                if not isinstance(fn, Mapping):
                    fn = {}
                index = call.get("index")
                if not isinstance(index, int):
                    index = self._slot_for(call.get("id"), fn.get("name"))
                slot = self._calls.get(index)
                if slot is None:
                    slot = {
                        "item_id": "fc_" + uuid.uuid4().hex,
                        "call_id": None,
                        "name": "",
                        "arguments": [],
                    }
                    self._calls[index] = slot
                if call.get("id") and not slot["call_id"]:
                    slot["call_id"] = str(call["id"])
                if fn.get("name") and not slot["name"]:
                    slot["name"] = str(fn["name"])
                if isinstance(fn.get("arguments"), str):
                    slot["arguments"].append(fn["arguments"])
        return events

    def _slot_for(self, call_id: Any, name: Any) -> int:
        """Pick the slot for a tool-call delta that carries no ``index``.

        A delta with a new id, or a name while the open call already has one, starts a new
        call; anything else (argument fragments, a repeated id) continues the latest call.
        """
        if not self._calls:
            return 0
        last = max(self._calls)
        open_call = self._calls[last]
        if call_id:
            return last if str(call_id) == open_call["call_id"] else last + 1
        if name and open_call["name"]:
            return last + 1
        return last

    def fail(self, message: str, *, code: Any = None) -> list[dict[str, Any]]:
        if self._finished:
            return []
        self._finished = True
        self._failed = True
        error: dict[str, Any] = {"message": message}
        if code:
            error["code"] = str(code)
        return [{"type": "response.failed", "response": self._response("failed", error=error)}]

    def finish(self) -> list[dict[str, Any]]:
        if self._finished:
            return []
        self._finished = True
        events = self._close_reasoning() + self._close_message() + self._close_calls()
        completed = self._response("completed")
        if self._usage is not None:
            completed["usage"] = self._usage
        events.append({"type": "response.completed", "response": completed})
        return events

    @property
    def failed(self) -> bool:
        return self._failed


# --------------------------------------------------------------------------- #
# SSE helpers
# --------------------------------------------------------------------------- #


def sse_frame(event: Mapping[str, Any]) -> bytes:
    kind = str(event.get("type") or "message")
    return f"event: {kind}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")


def iter_sse_data(lines: Iterable[bytes]) -> Iterator[str]:
    """Yield the joined ``data:`` payload of each SSE event in a byte-line stream."""
    buffer: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if line == "":
            if buffer:
                yield "\n".join(buffer)
                buffer = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if field == "data":
            buffer.append(value[1:] if value.startswith(" ") else value)
    if buffer:
        yield "\n".join(buffer)


# --------------------------------------------------------------------------- #
# The bridge server
# --------------------------------------------------------------------------- #


class _BridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: "ChatBridge._Server"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        log.debug("bridge: " + format, *args)

    # ---- routing -----------------------------------------------------------

    def _route(self) -> tuple[ChatUpstream | None, str, str]:
        """Split ``/<provider>/<rest>?<query>`` into (upstream, rest, query)."""
        parsed = urllib.parse.urlsplit(self.path)
        parts = parsed.path.lstrip("/").split("/", 1)
        pid = urllib.parse.unquote(parts[0]) if parts and parts[0] else ""
        rest = "/" + (parts[1] if len(parts) > 1 else "")
        upstream = self.server.upstreams.get(pid)
        return upstream, rest, parsed.query

    def _forward_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        for name, value in self.headers.items():
            if name.lower() in _DROP_REQUEST_HEADERS:
                continue
            headers[name] = value
        return headers

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str, kind: str = "hailer_bridge_error") -> None:
        self._send_json(status, {"error": {"message": message, "type": kind}})

    def _relay(self, status: int, content_type: str | None, payload: bytes) -> None:
        """Send an upstream response (typically an error) back to Codex unchanged."""
        self.send_response(status)
        self.send_header("Content-Type", content_type or "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length > 0 else b""

    # ---- verbs -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        upstream, rest, query = self._route()
        if upstream is None:
            self._send_error_json(404, f"unknown provider path {self.path!r}")
            return
        self._passthrough("GET", upstream.base_url, rest, query, None)

    def do_POST(self) -> None:  # noqa: N802
        upstream, rest, query = self._route()
        body = self._read_body()
        if upstream is None:
            self._send_error_json(404, f"unknown provider path {self.path!r}")
            return
        if rest == "/responses":
            self._responses(upstream, query, body)
            return
        # /responses/compact and anything else the Responses API offers has no chat equivalent.
        self._send_error_json(404, f"{rest} is not available through the Chat Completions bridge", "unsupported_endpoint")

    # ---- upstream calls ----------------------------------------------------

    def _open_upstream(self, method: str, url: str, data: bytes | None, headers: Mapping[str, str]) -> Any:
        req = urllib.request.Request(url, data=data, headers=dict(headers), method=method)
        return self.server.opener.open(req, timeout=UPSTREAM_READ_TIMEOUT_SEC)  # noqa: S310 - user-configured endpoint

    def _passthrough(self, method: str, upstream: str, rest: str, query: str, body: bytes | None) -> None:
        url = upstream + rest + (f"?{query}" if query else "")
        headers = self._forward_headers()
        if body is not None:
            headers["Content-Type"] = self.headers.get("Content-Type") or "application/json"
        try:
            with self._open_upstream(method, url, body, headers) as resp:
                self._relay(resp.status, resp.headers.get("Content-Type"), resp.read())
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            log.debug("bridge: upstream %s %s answered %d: %s", method, url, exc.code, _body_snippet(payload))
            self._relay(exc.code, exc.headers.get("Content-Type"), payload)
        except (urllib.error.URLError, OSError) as exc:
            log.debug("bridge: could not reach %s: %s", url, _reason(exc))
            self._send_error_json(502, f"could not reach {url}: {_reason(exc)}")

    def _responses(self, upstream: ChatUpstream, query: str, body: bytes) -> None:
        try:
            request = json.loads(body.decode("utf-8")) if body else {}
        except ValueError as exc:
            self._send_error_json(400, f"invalid JSON in request body: {exc}", "invalid_request_error")
            return
        if not isinstance(request, Mapping):
            self._send_error_json(400, "request body must be a JSON object", "invalid_request_error")
            return

        chat_body, tool_map = chat_request_from_responses(request, stream=upstream.stream)
        url = upstream.base_url + "/chat/completions" + (f"?{query}" if query else "")
        headers = self._forward_headers()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream" if upstream.stream else "application/json"
        data = json.dumps(chat_body).encode("utf-8")
        log.debug(
            "bridge: POST %s (%d messages, %d tools, %s)",
            url,
            len(chat_body["messages"]),
            len(chat_body.get("tools") or []),
            "streamed" if upstream.stream else "stream=false",
        )

        try:
            resp = self._open_upstream("POST", url, data, headers)
        except urllib.error.HTTPError as exc:
            # Relayed unchanged; logged here because Codex's error text names the bridge URL, not this one.
            payload = exc.read()
            log.debug("bridge: upstream POST %s answered %d: %s", url, exc.code, _body_snippet(payload))
            self._relay(exc.code, exc.headers.get("Content-Type"), payload)
            return
        except (urllib.error.URLError, OSError) as exc:
            log.debug("bridge: could not reach %s: %s", url, _reason(exc))
            self._send_error_json(502, f"could not reach {url}: {_reason(exc)}")
            return

        translator = ChatStreamTranslator(model=chat_body.get("model"), tools=tool_map)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def emit(events: list[dict[str, Any]]) -> None:
            if not events:
                return
            frames = [sse_frame(event) for event in events]
            self.wfile.write(b"".join(f"{len(f):x}\r\n".encode("ascii") + f + b"\r\n" for f in frames))
            self.wfile.flush()

        try:
            with resp:
                emit(translator.start())
                if _looks_like_sse(resp):
                    for payload in iter_sse_data(resp):
                        if payload.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except ValueError:
                            log.debug("bridge: skipping non-JSON stream payload")
                            continue
                        if isinstance(chunk, Mapping):
                            emit(translator.feed(chunk))
                        if translator.failed:
                            break
                else:
                    # stream=false was requested, or the gateway ignored stream=true: one JSON body, one chunk.
                    whole = json.loads(resp.read().decode("utf-8"))
                    if not isinstance(whole, Mapping):
                        raise ValueError("upstream returned a non-object JSON body")
                    emit(translator.feed(whole))
                emit(translator.finish())
        except OSError as exc:
            # The Codex side or the upstream socket went away; nothing sensible left to send.
            log.debug("bridge: stream aborted: %s", _reason(exc))
            return
        except Exception as exc:  # noqa: BLE001 - any translation failure must end the stream cleanly
            log.debug("bridge: upstream stream from %s failed: %s", url, _reason(exc))
            try:
                emit(translator.fail(f"upstream stream from {url} failed: {_reason(exc)}"))
            except OSError:
                return
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError:
            pass


def _body_snippet(payload: bytes, limit: int = 500) -> str:
    """The start of an upstream error body on one line, for debug logs (the log filter masks secrets)."""
    text = " ".join(payload.decode("utf-8", errors="replace").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _looks_like_sse(resp: Any) -> bool:
    """True when the upstream body is an event stream, by Content-Type or by its first bytes."""
    if "text/event-stream" in (resp.headers.get("Content-Type") or "").lower():
        return True
    peek = getattr(resp, "peek", None)
    if peek is None:
        return False
    try:
        head = peek(16).lstrip()
    except (OSError, ValueError):
        return False
    return head.startswith(_SSE_PREFIXES)


def _reason(exc: BaseException) -> str:
    """A short cause for an upstream failure, worded so Hailer's error mapping recognises it."""
    reason = getattr(exc, "reason", None)
    cause = reason if reason is not None else exc
    text = str(cause) or type(cause).__name__
    if isinstance(cause, socket.gaierror):
        return f"dns error ({text})"
    if isinstance(cause, TimeoutError) and "timed out" not in text.lower():
        return f"timed out ({text})"
    return text


def _build_opener() -> urllib.request.OpenerDirector:
    """One opener (and one TLS context) for the bridge's lifetime, honouring the usual proxy variables."""
    context = ssl.create_default_context()
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))


class ChatBridge:
    """A loopback server translating Codex's Responses calls for Chat Completions providers.

    ``upstreams`` maps a provider id to its real ``base_url`` (a string, streamed) or to a
    :class:`ChatUpstream` that also says whether to stream. After :meth:`start`,
    :attr:`urls` maps the same ids to the loopback base URL Codex should use instead.
    """

    class _Server(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True
        upstreams: dict[str, ChatUpstream]
        opener: urllib.request.OpenerDirector

    def __init__(self, upstreams: Mapping[str, str | ChatUpstream], *, host: str = "127.0.0.1") -> None:
        self._upstreams = {pid: up if isinstance(up, ChatUpstream) else ChatUpstream(up) for pid, up in upstreams.items()}
        self._host = host
        self._server: ChatBridge._Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def upstreams(self) -> dict[str, str]:
        """Provider id -> upstream ``base_url``."""
        return {pid: up.base_url for pid, up in self._upstreams.items()}

    @property
    def streaming(self) -> dict[str, bool]:
        """Provider id -> whether the bridge asks that upstream for server-sent events."""
        return {pid: up.stream for pid, up in self._upstreams.items()}

    @property
    def port(self) -> int | None:
        return self._server.server_address[1] if self._server is not None else None

    @property
    def urls(self) -> dict[str, str]:
        if self._server is None:
            return {}
        host, port = self._server.server_address[:2]
        return {pid: f"http://{host}:{port}/{urllib.parse.quote(pid, safe='')}" for pid in self._upstreams}

    def start(self) -> "ChatBridge":
        if self._server is not None:
            return self
        server = ChatBridge._Server((self._host, 0), _BridgeHandler)
        server.upstreams = dict(self._upstreams)
        server.opener = _build_opener()
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": _SERVE_POLL_INTERVAL_SEC}, name="hailer-chat-bridge", daemon=True
        )
        self._thread.start()
        log.debug("chat bridge listening on port %s for %s", self.port, ", ".join(sorted(self._upstreams)))
        return self

    def close(self) -> None:
        server, self._server = self._server, None
        if server is None:
            return
        try:
            server.shutdown()
            server.server_close()
        except Exception as exc:  # pragma: no cover - best effort
            log.debug("chat bridge close failed: %s", type(exc).__name__)
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "ChatBridge":
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        self.close()
