# LangChain spike (evidence for docs/LANGCHAIN_MIGRATION.md)

Throwaway code. Not packaged, not collected by pytest. It answers one question: can a plain LangChain
agent do what Hailer needs against the strict Chat-Completions-only gateways that the Codex bridge
(`src/hailer/wire.py`) was written for?

- `fake_gateway.py`: a strict fake gateway. Only `POST /v1/chat/completions` exists; it rejects unknown
  fields (`stream_options`, `parallel_tool_calls`) and unknown model names with 422, requires a custom
  header and an `api-version` query parameter, and can refuse `stream: true`.
- `lc_agent.py`: a prototype `HailerAgent` on `langchain.agents.create_agent` with the surface the CLI
  uses today (`start`, `new_thread`, `set_model`, `run_turn`, `close`, `AgentEvent`, `TurnSummary`).
- `run_spike.py`: 34 checks (streaming, no streaming, resume after restart, `/model` switch, typed
  errors, Ctrl+C during a model call and during a tool call).
- `deep_probe.py`: the same one-tool turn through `deepagents.create_deep_agent`, printing what it
  puts on the wire.
- `verify_claims.py`: checks against the installed packages for the design rules in the proposal
  (model-name routing to the Responses API, `LANGSMITH_GATEWAY`, the summarisation trigger, the private
  ChatGPT-OAuth model).
- `exit_probe.py`: Ctrl+C during a long tool call followed by an immediate `close()`.

Run it in a scratch environment, not the project venv:

```
uv venv --python 3.12 .venv-spike
uv pip install --python .venv-spike langchain langchain-openai langgraph-checkpoint-sqlite
NO_PROXY=127.0.0.1,localhost .venv-spike/Scripts/python.exe spikes/langchain/run_spike.py
```

`deep_probe.py` additionally needs `deepagents`.

Verified on 2026-09-18, Windows 11, Python 3.12, with langchain 1.4.1, langchain-core 1.6.3,
langchain-openai 1.6.2, langgraph 1.2.11, langgraph-checkpoint-sqlite 3.1.1, deepagents 0.7.15:
34/34 checks pass.
