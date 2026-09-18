"""Prototype of a LangChain-backed HailerAgent (same surface the CLI uses today).

start / new_thread / set_model / run_turn / close, AgentEvent callbacks, TurnSummary.
Provider settings are the ones hailer.toml already has (base_url, wire_api, stream, stream_options,
parallel_tool_calls, http_headers, query_params, env_key).

Sync surface for the CLI, one asyncio loop inside: Runner.run() turns Ctrl+C into task cancellation,
which aborts the in-flight HTTP request (LangGraph's sync runner blocks in an untimed wait that
Windows cannot interrupt; measured 5-9 s late in the spike).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import aiosqlite
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


@dataclass(frozen=True)
class Provider:
    id: str
    base_url: str | None = None
    wire_api: str = "responses"  # "responses" | "chat"
    stream: bool = True
    stream_options: bool = True
    parallel_tool_calls: bool = True
    http_headers: dict[str, str] = field(default_factory=dict)
    query_params: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentEvent:
    kind: str
    text: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnSummary:
    final_response: str
    thread_id: str
    duration_ms: int
    tool_calls: list[str]
    input_tokens: int | None
    output_tokens: int | None
    status: str = "completed"


def build_model(provider: Provider, model: str, api_key: str, reasoning_effort: str | None = None) -> ChatOpenAI:
    """hailer.toml provider -> ChatOpenAI. This replaces wire.py + the -c override builder."""
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "use_responses_api": provider.wire_api == "responses",
        "max_retries": 2,
    }
    if provider.base_url:
        kwargs["base_url"] = provider.base_url
    if provider.http_headers:
        kwargs["default_headers"] = dict(provider.http_headers)
    if provider.query_params:
        kwargs["default_query"] = dict(provider.query_params)
    if not provider.stream:
        kwargs["disable_streaming"] = True
    kwargs["stream_usage"] = bool(provider.stream and provider.stream_options)
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    return ChatOpenAI(**kwargs)


def make_tools(log: list[str], delay: dict[str, float]) -> list[Any]:
    """Stand-ins for HailerTools methods; the real ones are plain methods returning str already."""

    @tool
    def marimo_status() -> str:
        """Report whether marimo is running and which notebook is active."""
        log.append("marimo_status")
        return "marimo running at http://127.0.0.1:2718; active notebook notebooks/analysis.py (session s1)"

    @tool
    def marimo_execute(code: str) -> str:
        """Run Python in the active notebook's kernel and return stdout."""
        log.append(f"marimo_execute:{code}")
        time.sleep(delay.get("marimo_execute", 0.2))
        log.append("marimo_execute:done")
        return "2"

    return [marimo_status, marimo_execute]


class LangChainHailerAgent:
    def __init__(self, provider: Provider, model: str, api_key: str, system_prompt: str, db_path: Path, tool_log: list[str]):
        self.provider, self._model, self._api_key = provider, model, api_key
        self._system_prompt = system_prompt
        self._db_path = db_path
        self.tool_delay: dict[str, float] = {}
        self._tools = make_tools(tool_log, self.tool_delay)
        self._runner = asyncio.Runner()
        self._conn: Any = None
        self._saver: Any = None
        self._graph: Any = None
        self.thread_id: str | None = None
        self.last_repaired = 0

    # ---- lifecycle
    async def _aensure(self) -> None:
        if self._saver is None:
            self._conn = await aiosqlite.connect(str(self._db_path))
            self._saver = AsyncSqliteSaver(self._conn)
        if self._graph is None:
            model = build_model(self.provider, self._model, self._api_key)
            self._graph = create_agent(model, self._tools, system_prompt=self._system_prompt, checkpointer=self._saver)

    def start(self, *, resume_thread_id: str | None = None) -> str:
        async def go() -> str:
            await self._aensure()
            if resume_thread_id and await self._saver.aget_tuple({"configurable": {"thread_id": resume_thread_id}}) is not None:
                self.thread_id = resume_thread_id
                return resume_thread_id
            return self.new_thread()

        return self._runner.run(go())

    def new_thread(self) -> str:
        self.thread_id = uuid.uuid4().hex
        return self.thread_id

    def set_model(self, name: str, provider: Provider | None = None, api_key: str | None = None) -> bool:
        self._model = name
        if provider is not None:
            self.provider = provider
        if api_key is not None:
            self._api_key = api_key
        self._graph = None  # rebuilt lazily; the thread (= message history) carries on
        return False

    def close(self) -> None:
        async def go() -> None:
            if self._conn is not None:
                await self._conn.close()

        try:
            self._runner.run(go())
        finally:
            self._runner.close()

    @property
    def _config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def messages(self) -> list[Any]:
        async def go() -> list[Any]:
            await self._aensure()
            return list((await self._graph.aget_state(self._config)).values.get("messages", []))

        return self._runner.run(go())

    async def _arepair(self) -> int:
        """After an interrupted turn: answer tool calls that never got a ToolMessage."""
        msgs = (await self._graph.aget_state(self._config)).values.get("messages", [])
        answered = {m.tool_call_id for m in msgs if isinstance(m, ToolMessage)}
        patches = [
            ToolMessage(content="[interrupted by the user before this tool finished]", tool_call_id=tc["id"])
            for m in msgs
            if isinstance(m, AIMessage)
            for tc in (m.tool_calls or [])
            if tc["id"] not in answered
        ]
        if patches:
            await self._graph.aupdate_state(self._config, {"messages": patches}, as_node="tools")
        return len(patches)

    # ---- turns
    def run_turn(self, text: str, *, on_event: Callable[[AgentEvent], None] | None = None, preamble: str | None = None, repair: bool = True) -> TurnSummary:
        if self.thread_id is None:
            self.start()
        emit = on_event or (lambda _e: None)
        content = f"{preamble}\n\n{text}" if preamble else text
        payload = {"messages": [HumanMessage(content)]}
        started = time.monotonic()
        acc: dict[str, Any] = {"final": "", "tools": [], "in": 0, "out": 0, "repaired": 0}

        def handle(mode: str, data: Any) -> None:
            if mode == "messages":
                chunk, _meta = data
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    emit(AgentEvent("message_delta", chunk.content if isinstance(chunk.content, str) else ""))
                return
            for _node, update in data.items():
                for msg in (update or {}).get("messages", []):
                    if isinstance(msg, AIMessage):
                        for tc in msg.tool_calls or []:
                            acc["tools"].append(tc["name"])
                            emit(AgentEvent("tool_call", tc["name"], {"arguments": str(tc["args"])[:80]}))
                        if msg.usage_metadata:
                            acc["in"] += msg.usage_metadata.get("input_tokens", 0)
                            acc["out"] += msg.usage_metadata.get("output_tokens", 0)
                        if not msg.tool_calls:
                            acc["final"] = msg.content if isinstance(msg.content, str) else str(msg.content)

        async def go() -> None:
            await self._aensure()
            if repair:
                acc["repaired"] = await self._arepair()
            async for mode, data in self._graph.astream(payload, self._config, stream_mode=["updates", "messages"]):
                handle(mode, data)

        self._runner.run(go())  # Runner.run turns SIGINT into task cancellation, then raises KeyboardInterrupt
        self.last_repaired = acc["repaired"]
        return TurnSummary(
            final_response=acc["final"],
            thread_id=self.thread_id or "",
            duration_ms=int((time.monotonic() - started) * 1000),
            tool_calls=acc["tools"],
            input_tokens=acc["in"] or None,
            output_tokens=acc["out"] or None,
        )
