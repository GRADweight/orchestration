"""Lokaler, OpenAI-kompatibler Testserver (nur 127.0.0.1, kein externes Netz).

Antworten werden als Warteschlange vorgegeben; jede Anfrage (Header + JSON)
wird aufgezeichnet. Ein Eintrag ist entweder eine :class:`Reply` oder eine
Funktion ``request_json -> Reply``.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class Reply:
    status: int = 200
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    delay_s: float = 0.0


def chat(
    content: str | None = None,
    tool_calls: list[tuple[str, Any]] | None = None,
    *,
    usage: tuple[int, int] | None = (10, 5),
    model: str = "fake-7b",
) -> Reply:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args if isinstance(args, str) else json.dumps(args),
                },
            }
            for i, (name, args) in enumerate(tool_calls)
        ]
    body: dict[str, Any] = {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
    }
    if usage is not None:
        body["usage"] = {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": sum(usage),
        }
    return Reply(body=body)


Responder = Reply | Callable[[dict[str, Any]], Reply]


class FakeOpenAI:
    def __init__(self) -> None:
        self.queue: list[Responder] = []
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(payload)
                outer.headers.append({k.lower(): v for k, v in self.headers.items()})
                if self.path != "/v1/chat/completions":
                    reply = Reply(status=404, body={"error": "not found"})
                elif not outer.queue:
                    reply = Reply(status=500, body={"error": "Warteschlange leer"})
                else:
                    item = outer.queue.pop(0)
                    reply = item(payload) if callable(item) else item
                if reply.delay_s:
                    time.sleep(reply.delay_s)
                raw = (
                    reply.body if isinstance(reply.body, bytes) else json.dumps(reply.body).encode()
                )
                try:
                    self.send_response(reply.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    for key, value in reply.headers.items():
                        self.send_header(key, value)
                    self.end_headers()
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, format: str, *args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> FakeOpenAI:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
