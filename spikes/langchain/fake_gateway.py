"""A strict, Chat-Completions-only fake gateway for the LangChain spike.

Mimics the corporate gateways Hailer's chat bridge was written for:
- only POST /v1/chat/completions exists (POST /v1/responses -> 404)
- the model name must match a pattern (422 otherwise, FastAPI/pydantic wording)
- unknown request fields are rejected with 422 ("extra inputs"): stream_options, parallel_tool_calls
- a custom header and a query parameter are required (401 / 400 otherwise)
- optionally rejects stream=true
Every request is recorded in ``gateway.requests`` for assertions.
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

MODEL_PATTERN = re.compile(r"^(corp-gpt|corp-mini)$")
ALLOWED_FIELDS = {"model", "messages", "tools", "tool_choice", "stream", "temperature", "max_tokens", "reasoning_effort"}


class FakeGateway:
    def __init__(self, *, allow_stream: bool = True, usage_without_stream_options: bool = True, chunk_delay: float = 0.0, reply_delay: float = 0.0):
        self.allow_stream = allow_stream
        self.usage_without_stream_options = usage_without_stream_options
        self.chunk_delay = chunk_delay
        self.reply_delay = reply_delay
        self.requests: list[dict] = []
        self.aborted = 0
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_a):  # quiet
                pass

            def _json(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # noqa: N802
                parsed = urlparse(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                record = {"path": parsed.path, "query": parse_qs(parsed.query), "headers": dict(self.headers), "body": body}
                gateway.requests.append(record)
                if parsed.path != "/v1/chat/completions":
                    return self._json(404, {"detail": "Not Found"})
                if self.headers.get("Authorization") != "Bearer sk-spike":
                    return self._json(401, {"error": {"message": "Invalid API key", "type": "invalid_request_error"}})
                if self.headers.get("X-Client-Id") != "hailer-spike":
                    return self._json(401, {"error": {"message": "missing X-Client-Id", "type": "invalid_request_error"}})
                if parse_qs(parsed.query).get("api-version") != ["2025-04-01-preview"]:
                    return self._json(400, {"error": {"message": "api-version is required", "type": "invalid_request_error"}})
                extra = sorted(set(body) - ALLOWED_FIELDS)
                if extra:
                    return self._json(422, {"detail": [{"type": "extra_forbidden", "loc": ["body", extra[0]], "msg": "Extra inputs are not permitted"}]})
                if not MODEL_PATTERN.match(str(body.get("model"))):
                    return self._json(422, {"detail": [{"type": "string_pattern_mismatch", "loc": ["body", "model"], "msg": "String should match pattern '^(corp-gpt|corp-mini)$'"}]})
                if body.get("stream") and not gateway.allow_stream:
                    return self._json(422, {"detail": [{"loc": ["body", "stream"], "msg": "streaming is not supported"}]})
                reply = gateway.script(body)
                if gateway.reply_delay:
                    time.sleep(gateway.reply_delay)
                if body.get("stream"):
                    return self._stream(body, reply)
                message = {"role": "assistant", "content": reply.get("text")}
                if reply.get("tool"):
                    message["tool_calls"] = [{"id": "call_1", "type": "function", "function": {"name": reply["tool"], "arguments": json.dumps(reply.get("args", {}))}}]
                self._json(200, {
                    "id": "chatcmpl-spike", "object": "chat.completion", "created": 0, "model": body["model"],
                    "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if reply.get("tool") else "stop"}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                })

            def _stream(self, body: dict, reply: dict) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def send(obj) -> None:
                    data = b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n"
                    self.wfile.write(hex(len(data))[2:].encode() + b"\r\n" + data + b"\r\n")
                    self.wfile.flush()

                base = {"id": "chatcmpl-spike", "object": "chat.completion.chunk", "created": 0, "model": body["model"]}
                try:
                    if reply.get("tool"):
                        args = json.dumps(reply.get("args", {}))
                        send({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": None, "tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": reply["tool"], "arguments": ""}}]}, "finish_reason": None}]})
                        for piece in (args[: len(args) // 2], args[len(args) // 2:]):
                            send({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": piece}}]}, "finish_reason": None}]})
                        finish = "tool_calls"
                    else:
                        send({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
                        for word in reply["text"].split(" "):
                            if gateway.chunk_delay:
                                time.sleep(gateway.chunk_delay)
                            send({**base, "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}]})
                        finish = "stop"
                    final = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
                    if gateway.usage_without_stream_options:
                        final["usage"] = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
                    send(final)
                    send(b"[DONE]")
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionError, OSError):
                    gateway.aborted += 1  # the client went away mid-stream (a cancelled turn)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    # The "model": decide the reply from the conversation so far.
    def script(self, body: dict) -> dict:
        messages = body["messages"]
        last = messages[-1]
        if last["role"] == "tool":
            return {"text": f"Tool said: {str(last['content'])[:60]}"}
        text = last["content"] if isinstance(last["content"], str) else json.dumps(last["content"])
        if "status" in text:
            return {"tool": "marimo_status", "args": {}}
        if "run" in text:
            return {"tool": "marimo_execute", "args": {"code": "print(1 + 1)"}}
        if "slow" in text:
            return {"text": " ".join(f"word{i}" for i in range(40))}
        roles = [m["role"] for m in messages]
        return {"text": f"I see {len(messages)} messages: {','.join(roles)}"}

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._server.shutdown()
        self._server.server_close()
