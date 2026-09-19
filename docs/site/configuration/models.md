# Models and endpoints {#model-configuration}

All model settings live in `hailer.toml` (repo root, or `.config/hailer/hailer.toml`). `uvx hailer init`
writes a commented starter file.

## OpenAI

```toml
[model]
name = "gpt-5.5"
provider = "openai"
```

Store an API key once with `uvx hailer login openai`, or set `OPENAI_API_KEY` in the terminal. Hailer
talks to `https://api.openai.com/v1` over the Responses API. To use Chat Completions instead, or to turn
streaming off, declare the provider without a `base_url`:

```toml
[model_providers.openai]
wire_api = "chat"
```

## A custom or internal endpoint

Hailer supports endpoints that implement the OpenAI Responses API or Chat Completions with function
calling, including compatible internal gateways and local model servers. Use the model name, URL and
protocol required by your endpoint.

```toml
[model]
name     = "analyst-v3"               # whatever model id the endpoint expects
provider = "internal"
# reasoning_effort = "medium"         # minimal | low | medium | high | xhigh; "" sends no reasoning effort
# summarize_after_tokens = 100000     # summarise older turns past this size; lower it for small context windows

[model_providers.internal]
base_url         = "https://llm.example.internal/v1"
wire_api         = "responses"               # or "chat" for a Chat Completions endpoint
env_key          = "INTERNAL_MODEL_API_KEY"  # env var name; value from `hailer login internal` or the terminal
# stream         = true                      # false if the endpoint rejects stream = true
# stream_options = true                      # false omits stream_options (token counts may be lost)
# name             = "Internal"
# http_headers     = { "X-Team" = "data-analytics" }
# env_http_headers = { "X-Client-Id" = "INTERNAL_CLIENT_ID" }
# query_params     = { "api-version" = "2025-04-01-preview" }
```

Then:

```bash
uvx hailer login internal     # stores the key in the OS credential store (hidden prompt)
uvx hailer doctor             # config / notebook / credentials / marimo / session, with fixes
uvx hailer
```

Or set `INTERNAL_MODEL_API_KEY` in the terminal instead of logging in; an environment variable always wins
over the credential store. The startup panel and `status` show the provider with its base URL; `doctor` and
`status` show where the key came from (`from keyring` or `from env`), never the value.

**`wire_api` chooses the protocol the endpoint speaks:**

| `wire_api` | What Hailer sends | Use it when |
|---|---|---|
| `"responses"` (default) | `POST {base_url}/responses` | The endpoint implements the OpenAI Responses API. |
| `"chat"` | `POST {base_url}/chat/completions` | The endpoint implements Chat Completions (vLLM, Ollama, LiteLLM, most internal gateways). |

Hailer uses LangChain's `ChatOpenAI` client to send requests to `base_url`: the key as
`Authorization: Bearer ...`, your `http_headers`, each `env_http_headers` entry whose variable is set,
and your `query_params`. The body follows the selected protocol: Chat Completions uses `messages`,
while Responses uses `input`. Both carry the configured model, conversation and Hailer's tool definitions.
Every request names the model you configured; there are no side requests under other model names.
`HTTP_PROXY`, `HTTPS_PROXY` and `NO_PROXY` are honoured.

Two per-provider switches help with strict endpoints; both default to `true` and apply to either
`wire_api`:

- `stream = false` sends `stream: false` and reads one JSON reply, for endpoints that reject streamed
  requests or cannot deliver server-sent events (some gateways and proxies buffer or refuse them). The
  answer then appears when the turn finishes rather than as it is generated. It is also the workaround
  for an endpoint that streams tool calls in a shape the client does not understand. The provider shows
  as `internal (https://..., chat completions, no streaming)` in the startup panel and `/status`.
- `stream_options = false` omits the `stream_options` field from streamed requests (some Azure API
  versions and proxies reject it). Token counts are then whatever the final chunk carries, often nothing.

Independently of the provider, `reasoning_effort = ""` under `[model]` stops the reasoning effort being
sent at all, for endpoints or models that reject it.

Long conversations are kept inside the model's context window by summarising older turns once the
conversation passes `[model].summarize_after_tokens` (default 100000; the last 20 messages are always kept
as they are). The summary is written by the model you configured. Lower the number for a model with a
small context window; `0` turns summarising off.

Several providers can be declared; switch inside a session with `/model <name>` or
`/model <provider>:<name>` (this starts a new thread). One-off overrides: `HAILER_MODEL`,
`HAILER_MODEL_PROVIDER`.
