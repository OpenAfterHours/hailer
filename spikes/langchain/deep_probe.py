"""What does create_deep_agent put on the wire for the same 1-tool turn?"""

from __future__ import annotations

import inspect
import json

from deepagents import create_deep_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from fake_gateway import ALLOWED_FIELDS, FakeGateway

print("signature:", inspect.signature(create_deep_agent))

ALLOWED_FIELDS.update({"stream_options", "parallel_tool_calls"})  # only observe here, do not reject


@tool
def marimo_status() -> str:
    """Report whether marimo is running and which notebook is active."""
    return "marimo running; active notebook notebooks/analysis.py"


SYSTEM = "You are Hailer, a local data-analysis assistant."

with FakeGateway() as gw:
    model = ChatOpenAI(
        model="corp-gpt", api_key="sk-spike", base_url=gw.base_url, use_responses_api=False, stream_usage=False,
        default_headers={"X-Client-Id": "hailer-spike"}, default_query={"api-version": "2025-04-01-preview"},
    )
    agent = create_deep_agent(model=model, tools=[marimo_status], system_prompt=SYSTEM)
    result = agent.invoke({"messages": [{"role": "user", "content": "what is the status?"}]})
    print("final:", result["messages"][-1].content[:80])
    body = gw.requests[0]["body"]
    tools = [t["function"]["name"] for t in body.get("tools", [])]
    system = "\n".join(m["content"] if isinstance(m["content"], str) else json.dumps(m["content"]) for m in body["messages"] if m["role"] == "system")
    print("requests:", len(gw.requests))
    print("tools sent:", tools)
    print("tool schema bytes:", len(json.dumps(body.get("tools", []))))
    print("system prompt chars:", len(system), "(Hailer's own part:", len(SYSTEM), ")")
    print("system prompt head:", system[:300].replace("\n", " | "))
    print("state keys:", sorted(result.keys()))
