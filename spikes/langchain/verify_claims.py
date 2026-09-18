"""Check the research claims that change the design, against the installed packages."""

import importlib
import importlib.util
import inspect
import json
import os

import langchain_openai
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI

print("langchain-openai", langchain_openai.__version__ if hasattr(langchain_openai, "__version__") else "?")

# (a) private ChatGPT OAuth support
for mod in ("langchain_openai.chatgpt_oauth", "langchain_openai.chat_models.codex"):
    spec = importlib.util.find_spec(mod)
    print(f"(a) {mod}: {'present' if spec else 'absent'}")
    if spec:
        m = importlib.import_module(mod)
        names = [n for n in dir(m) if "login" in n.lower() or "Codex" in n]
        print("    names:", names)
        doc = (inspect.getdoc(m) or "")[:300].replace("\n", " ")
        print("    doc:", doc)

# (b) summarization fraction trigger without a model profile
from langchain.agents.middleware import SummarizationMiddleware

model = ChatOpenAI(model="corp-gpt", api_key="x", base_url="http://127.0.0.1:1/v1", use_responses_api=False)
print("(b) model.profile for an unknown deployment name:", getattr(model, "profile", "n/a"))
for trigger in (("fraction", 0.8), ("tokens", 100_000)):
    try:
        SummarizationMiddleware(model=model, trigger=trigger)
        print(f"(b) trigger={trigger}: constructed")
    except Exception as exc:  # noqa: BLE001
        print(f"(b) trigger={trigger}: {type(exc).__name__}: {str(exc)[:200]}")
print("(b) default trigger:", inspect.signature(SummarizationMiddleware.__init__).parameters["trigger"].default)

# (c) LANGSMITH_GATEWAY vs an explicit base_url
os.environ["LANGSMITH_GATEWAY"] = "true"
try:
    m2 = ChatOpenAI(model="corp-gpt", api_key="x", base_url="http://127.0.0.1:1/v1", use_responses_api=False)
    print("(c) with LANGSMITH_GATEWAY set, explicit base_url ->", m2.openai_api_base)
    m3 = ChatOpenAI(model="corp-gpt", api_key="x", use_responses_api=False)
    print("(c) with LANGSMITH_GATEWAY set, no base_url      ->", m3.openai_api_base)
except Exception as exc:  # noqa: BLE001
    print("(c)", type(exc).__name__, str(exc)[:200])
finally:
    os.environ.pop("LANGSMITH_GATEWAY", None)

# (d) model-name routing when use_responses_api is left unset
for name in ("corp-gpt", "gpt-5.5", "gpt-5-codex", "gpt-5.5-pro"):
    m4 = ChatOpenAI(model=name, api_key="x", base_url="http://127.0.0.1:1/v1")
    try:
        routed = m4._use_responses_api({})
    except Exception as exc:  # noqa: BLE001
        routed = f"{type(exc).__name__}"
    print(f"(d) use_responses_api unset, model={name!r}: routes to Responses = {routed}")
m5 = ChatOpenAI(model="gpt-5-codex", api_key="x", base_url="http://127.0.0.1:1/v1", use_responses_api=False)
print("(d) explicit False, model='gpt-5-codex':", m5._use_responses_api({}))

# (e) tool schema size for the plain agent's two spike tools
from lc_agent import make_tools

tools = [convert_to_openai_tool(t) for t in make_tools([], {})]
print("(e) plain agent, marimo_status alone:", len(json.dumps(tools[:1])), "bytes; deepagents sent 10721 bytes for the same tool plus its own")
