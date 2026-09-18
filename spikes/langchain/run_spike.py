"""Scenarios: strict chat-only gateway, streaming on/off, resume, model switch, errors, interrupt."""

from __future__ import annotations

import _thread
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

from fake_gateway import FakeGateway
from lc_agent import LangChainHailerAgent, Provider

RESULTS: list[tuple[str, bool, str]] = []
SYSTEM = "You are Hailer, a local data-analysis assistant."


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {('- ' + detail) if detail else ''}", flush=True)


def provider_for(gw: FakeGateway, **over) -> Provider:
    base = dict(
        id="corp", base_url=gw.base_url, wire_api="chat", stream=True, stream_options=False, parallel_tool_calls=False,
        http_headers={"X-Client-Id": "hailer-spike"}, query_params={"api-version": "2025-04-01-preview"},
    )
    base.update(over)
    return Provider(**base)


def scenario_streaming(tmp: Path) -> None:
    print("\n== A. strict chat-only gateway, streaming ==")
    with FakeGateway() as gw:
        tool_log: list[str] = []
        agent = LangChainHailerAgent(provider_for(gw), "corp-gpt", "sk-spike", SYSTEM, tmp / "a.sqlite", tool_log)
        events = []
        agent.start()
        summary = agent.run_turn("what is the status?", on_event=events.append)
        kinds = [e.kind for e in events]
        check("A1 tool ran in-process", tool_log == ["marimo_status"], str(tool_log))
        check("A2 tool_call event surfaced", "tool_call" in kinds and summary.tool_calls == ["marimo_status"], str(summary.tool_calls))
        check("A3 text deltas streamed", kinds.count("message_delta") > 3, f"{kinds.count('message_delta')} deltas")
        check("A4 final answer carries the tool result", "marimo running" in summary.final_response, summary.final_response[:70])
        check("A5 only /v1/chat/completions was called", {r['path'] for r in gw.requests} == {"/v1/chat/completions"}, str({r['path'] for r in gw.requests}))
        bodies = [r["body"] for r in gw.requests]
        check("A6 no stream_options / parallel_tool_calls sent", all("stream_options" not in b and "parallel_tool_calls" not in b for b in bodies), str([sorted(b) for b in bodies]))
        check("A7 every request used the configured model", {b["model"] for b in bodies} == {"corp-gpt"}, str({b["model"] for b in bodies}))
        check("A8 exactly 2 requests for a 1-tool turn (no side requests)", len(bodies) == 2, str(len(bodies)))
        check("A9 tools arrive as plain functions", [t["function"]["name"] for t in bodies[0]["tools"]] == ["marimo_status", "marimo_execute"], str(bodies[0].get("tools"))[:120])
        check("A10 system prompt is one system message", [m["role"] for m in bodies[0]["messages"]] == ["system", "user"], str([m["role"] for m in bodies[0]["messages"]]))
        check("A11 token usage captured without stream_options", summary.input_tokens == 22 and summary.output_tokens == 14, f"{summary.input_tokens}/{summary.output_tokens}")
        agent.close()


def scenario_no_stream(tmp: Path) -> None:
    print("\n== B. gateway rejects stream=true ==")
    with FakeGateway(allow_stream=False) as gw:
        agent = LangChainHailerAgent(provider_for(gw, stream=False), "corp-gpt", "sk-spike", SYSTEM, tmp / "b.sqlite", [])
        events = []
        summary = agent.run_turn("please run it", on_event=events.append)
        check("B1 turn completes with stream=false", summary.final_response.startswith("Tool said: 2"), summary.final_response)
        check("B2 no request asked for streaming", all(not r["body"].get("stream") for r in gw.requests), str([r["body"].get("stream") for r in gw.requests]))
        check("B3 tool_call event still surfaced", [e.text for e in events if e.kind == "tool_call"] == ["marimo_execute"], str([e.kind for e in events]))
        check("B4 usage captured", summary.input_tokens == 22, f"{summary.input_tokens}/{summary.output_tokens}")
        agent.close()


def scenario_resume_and_switch(tmp: Path) -> None:
    print("\n== C. resume across restarts + model switch on the same thread ==")
    with FakeGateway() as gw:
        db = tmp / "c.sqlite"
        agent = LangChainHailerAgent(provider_for(gw), "corp-gpt", "sk-spike", SYSTEM, db, [])
        thread_id = agent.start()
        agent.run_turn("hello there")
        agent.close()

        agent2 = LangChainHailerAgent(provider_for(gw), "corp-gpt", "sk-spike", SYSTEM, db, [])
        resumed = agent2.start(resume_thread_id=thread_id)
        summary = agent2.run_turn("and again")
        check("C1 thread resumed from sqlite after restart", resumed == thread_id and "I see 4 messages: system,user,assistant,user" in summary.final_response, summary.final_response)
        check("C2 unknown thread id falls back to a new thread", agent2.start(resume_thread_id="nope") != "nope")
        agent2.thread_id = thread_id
        agent2.set_model("corp-mini")
        summary = agent2.run_turn("third")
        check("C3 /model switch keeps the conversation", gw.requests[-1]["body"]["model"] == "corp-mini" and "I see 6 messages" in summary.final_response, summary.final_response)
        agent2.close()


def scenario_errors(tmp: Path) -> None:
    print("\n== D. error surface (typed exceptions instead of Codex text) ==")
    import openai

    with FakeGateway() as gw:
        # one agent, one event loop: providers/models are switched the way /model does it
        agent = LangChainHailerAgent(provider_for(gw), "corp-gpt", "sk-spike", SYSTEM, tmp / "d.sqlite", [])
        for label, kwargs, model, key in (
            ("D1 bad model -> 422", {}, "gpt-5.5", "sk-spike"),
            ("D2 bad key -> 401", {}, "corp-gpt", "sk-wrong"),
            ("D3 stream_options rejected -> 422", {"stream_options": True}, "corp-gpt", "sk-spike"),
            ("D4 responses API on chat-only gateway -> 404", {"wire_api": "responses"}, "corp-gpt", "sk-spike"),
        ):
            agent.set_model(model, provider_for(gw, **kwargs), api_key=key)
            agent.new_thread()
            before = len(gw.requests)
            try:
                agent.run_turn("hello")
                check(label, False, "no exception")
            except openai.APIStatusError as exc:
                body = str(exc.body)[:110]
                check(label, True, f"{type(exc).__name__} status={exc.status_code} requests={len(gw.requests) - before} body={body}")
            except Exception as exc:  # noqa: BLE001
                check(label, False, f"{type(exc).__name__}: {exc}"[:200])
        agent.set_model("corp-gpt", provider_for(gw), api_key="sk-spike")
        summary = agent.run_turn("hello")
        check("D6 the same agent recovers after the failed turns", "I see" in summary.final_response, summary.final_response)
        agent.close()
    agent = LangChainHailerAgent(Provider(id="corp", base_url="http://127.0.0.1:9/v1", wire_api="chat"), "corp-gpt", "sk-spike", SYSTEM, tmp / "d5.sqlite", [])
    try:
        agent.run_turn("hello")
        check("D5 unreachable endpoint", False, "no exception")
    except openai.APIConnectionError as exc:
        check("D5 unreachable endpoint -> APIConnectionError", True, type(exc).__name__)
    except Exception as exc:  # noqa: BLE001
        check("D5 unreachable endpoint", False, f"{type(exc).__name__}: {exc}"[:200])
    agent.close()


def _interrupt_after(seconds: float) -> threading.Timer:
    timer = threading.Timer(seconds, _thread.interrupt_main)  # what Ctrl+C does: SIGINT for the main thread
    timer.daemon = True
    timer.start()
    return timer


def scenario_interrupt(tmp: Path, *, streaming: bool) -> None:
    label = "stream" if streaming else "no-stream"
    print(f"\n== E. Ctrl+C during a model call ({label}) ==")
    gw_kwargs = {"chunk_delay": 0.25} if streaming else {"allow_stream": False, "reply_delay": 6.0}
    with FakeGateway(**gw_kwargs) as gw:
        agent = LangChainHailerAgent(provider_for(gw, stream=streaming), "corp-gpt", "sk-spike", SYSTEM, tmp / f"e_{streaming}.sqlite", [])
        agent.start()
        agent.run_turn("hello first")
        started = time.monotonic()
        timer = _interrupt_after(1.0)
        try:
            agent.run_turn("a slow answer please")
            check(f"E[{label}] KeyboardInterrupt surfaced", False, "turn completed instead")
        except KeyboardInterrupt:
            latency = time.monotonic() - started - 1.0
            check(f"E[{label}] KeyboardInterrupt surfaced promptly", latency < 1.0, f"{latency:.2f}s after the signal")
        except BaseException as exc:  # noqa: BLE001
            check(f"E[{label}] KeyboardInterrupt surfaced", False, f"{type(exc).__name__}: {exc}"[:200])
        finally:
            timer.cancel()
        time.sleep(0.5)
        if streaming:
            check(f"E[{label}] the HTTP request was really aborted", gw.aborted >= 1, f"gateway saw {gw.aborted} aborted stream(s)")
        roles = [type(m).__name__ for m in agent.messages()]
        try:
            summary = agent.run_turn("hello after interrupt")
            check(f"E[{label}] next turn works on the same thread", "I see" in summary.final_response, f"{summary.final_response} | state before: {roles}")
        except BaseException as exc:  # noqa: BLE001
            check(f"E[{label}] next turn works on the same thread", False, f"{type(exc).__name__}: {exc}"[:300])
        agent.close()


def scenario_interrupt_in_tool(tmp: Path) -> None:
    print("\n== F. Ctrl+C while a tool is running -> dangling tool call ==")
    with FakeGateway() as gw:
        tool_log: list[str] = []
        agent = LangChainHailerAgent(provider_for(gw), "corp-gpt", "sk-spike", SYSTEM, tmp / "f.sqlite", tool_log)
        agent.start()
        agent.tool_delay["marimo_execute"] = 4.0  # the interrupt lands inside the tool
        timer = _interrupt_after(1.0)
        started = time.monotonic()
        try:
            agent.run_turn("please run it")
            check("F1 interrupt inside a tool", False, "turn completed")
        except KeyboardInterrupt:
            latency = time.monotonic() - started - 1.0
            check("F1 KeyboardInterrupt surfaced promptly while a tool was running", latency < 1.0, f"{latency:.2f}s after the signal")
        finally:
            timer.cancel()
        agent.tool_delay["marimo_execute"] = 0.0
        state = [type(m).__name__ + (":tool_calls" if getattr(m, "tool_calls", None) else "") for m in agent.messages()]
        check("F2 checkpoint holds a dangling tool call", bool(state) and state[-1] == "AIMessage:tool_calls", str(state))
        try:
            summary = agent.run_turn("hello without repair", repair=False)
            check("F3 without repair: what happens", True, f"accepted by LangGraph; reply={summary.final_response[:80]}")
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001
            check("F3 without repair the next turn is rejected", True, f"{type(exc).__name__}: {str(exc)[:160]}")
        summary = agent.run_turn("hello after repair")
        check("F4 with the repair step the next turn works", "I see" in summary.final_response, f"patched={agent.last_repaired}; {summary.final_response}")
        time.sleep(3.5)
        check("F5 the abandoned tool thread finished on its own", "marimo_execute:done" in tool_log, str(tool_log))
        agent.close()


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="hailer-spike-"))
    for fn, kwargs in (
        (scenario_streaming, {}),
        (scenario_no_stream, {}),
        (scenario_resume_and_switch, {}),
        (scenario_errors, {}),
        (scenario_interrupt, {"streaming": True}),
        (scenario_interrupt, {"streaming": False}),
        (scenario_interrupt_in_tool, {}),
    ):
        try:
            fn(tmp, **kwargs)
        except BaseException:  # noqa: BLE001
            check(f"{fn.__name__} {kwargs}", False, "scenario crashed")
            traceback.print_exc()
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
