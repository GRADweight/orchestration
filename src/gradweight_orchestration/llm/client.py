"""Client für OpenAI-kompatible Chat-Completions (Ollama, vLLM, LiteLLM, OpenAI).

Standardmässig nur ``urllib`` aus der Standardbibliothek; mit dem Extra
``httpx`` steht :class:`HttpxTransport` zur Verfügung. Wiederholungen mit
exponentiellem Backoff und vollem Jitter bei Netzwerkfehlern, 408/409/425/429
und 5xx; ``Retry-After`` wird beachtet (gedeckelt).

Hinweis zu Zeitlimits: ``timeout_s`` gilt bei ``urllib`` pro blockierender
Socket-Operation, nicht für die Gesamtdauer einer Anfrage. Die Gesamtdauer
begrenzt das Zeitbudget des Agenten.

``tool_choice`` wird bewusst nicht gesendet: Ollama unterstützt das Feld nicht.
"""

from __future__ import annotations

import http.client
import json
import math
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .._json import JSONObject, JSONValue
from .types import (
    LLMHTTPError,
    LLMResponse,
    LLMResponseError,
    LLMTransportError,
    Message,
    ToolCall,
    ToolsNotSupportedError,
    Usage,
    estimate_tokens,
    parse_tool_arguments,
)

__all__ = [
    "HTTPResponse",
    "HttpxTransport",
    "OpenAICompatClient",
    "RetryPolicy",
    "Transport",
    "UrllibTransport",
]

_TOOLS_UNSUPPORTED = (
    "does not support tools",
    "tools are not supported",
    "tool calling is not supported",
    "tool use is not supported",
    "enable-auto-tool-choice",
)


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class Transport(Protocol):
    """Schickt einen POST-Request. Netzwerkfehler als :class:`LLMTransportError`."""

    def post(
        self, url: str, body: bytes, headers: Mapping[str, str], timeout_s: float
    ) -> HTTPResponse: ...


class UrllibTransport:
    """Transport über ``urllib`` (Standardbibliothek).

    ``trust_env=True`` übernimmt Proxy-Einstellungen aus der Umgebung
    (``HTTPS_PROXY``, ``NO_PROXY`` ...); ``False`` verbindet immer direkt.
    """

    def __init__(self, *, trust_env: bool = True) -> None:
        handlers: list[urllib.request.BaseHandler] = []
        if not trust_env:
            handlers.append(urllib.request.ProxyHandler({}))
        self._opener = urllib.request.build_opener(*handlers)

    def post(
        self, url: str, body: bytes, headers: Mapping[str, str], timeout_s: float
    ) -> HTTPResponse:
        request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")  # noqa: S310 - Schema im Client geprüft
        try:
            with self._opener.open(request, timeout=timeout_s) as response:
                return HTTPResponse(
                    status=int(response.status),
                    headers={k.lower(): v for k, v in response.headers.items()},
                    body=response.read(),
                )
        except urllib.error.HTTPError as exc:
            headers_ = {k.lower(): v for k, v in exc.headers.items()} if exc.headers else {}
            return HTTPResponse(status=exc.code, headers=headers_, body=exc.read() or b"")
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            raise LLMTransportError(f"Verbindung zu {url} fehlgeschlagen: {exc}") from exc


class HttpxTransport:
    """Transport über ``httpx`` (Extra ``httpx``)."""

    def __init__(self, client: Any = None, *, trust_env: bool = True) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - nur ohne Extra
            raise ImportError(
                "HttpxTransport braucht: pip install 'gradweight-orchestration[httpx]'"
            ) from exc
        self._httpx = httpx
        self._client = client or httpx.Client(trust_env=trust_env)

    def post(
        self, url: str, body: bytes, headers: Mapping[str, str], timeout_s: float
    ) -> HTTPResponse:
        try:
            response = self._client.post(
                url, content=body, headers=dict(headers), timeout=timeout_s
            )
        except self._httpx.HTTPError as exc:
            raise LLMTransportError(f"Verbindung zu {url} fehlgeschlagen: {exc}") from exc
        return HTTPResponse(
            status=response.status_code,
            headers={k.lower(): v for k, v in response.headers.items()},
            body=response.content,
        )


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    retry_statuses: frozenset[int] = field(
        default_factory=lambda: frozenset({408, 409, 425, 429, 500, 502, 503, 504})
    )

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts muss ≥ 1 sein")

    def delay(self, attempt: int, rng: Callable[[], float], retry_after: float | None) -> float:
        """Wartezeit vor Versuch ``attempt + 1`` (volle Streuung, gedeckelt)."""
        if retry_after is not None:
            return max(0.0, min(retry_after, self.max_delay_s))
        ceiling = min(self.max_delay_s, self.base_delay_s * (2.0 ** (attempt - 1)))
        return rng() * ceiling


def _retry_after(headers: Mapping[str, str]) -> float | None:
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None  # HTTP-Datum wird nicht ausgewertet; normales Backoff greift


class OpenAICompatClient:
    """Chat-Completions-Client mit Werkzeugaufrufen, JSON-Modus und Wiederholungen."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        retry: RetryPolicy | None = None,
        transport: Transport | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        extra_body: Mapping[str, JSONValue] | None = None,
        headers: Mapping[str, str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        scheme = urllib.parse.urlparse(base_url).scheme
        if scheme not in ("http", "https"):
            raise ValueError(f"base_url muss http(s) sein, nicht {scheme!r}")
        if not model:
            raise ValueError("model fehlt")
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._api_key = api_key
        self._timeout = timeout_s
        self._retry = RetryPolicy() if retry is None else retry
        self._transport = UrllibTransport() if transport is None else transport
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra = dict(extra_body or {})
        self._headers = dict(headers or {})
        self._sleep = sleep
        self._rng = rng

    @classmethod
    def from_env(
        cls, *, environ: Mapping[str, str] | None = None, prefix: str = "GW_LLM_", **kwargs: Any
    ) -> OpenAICompatClient:
        """Konfiguration aus der Umgebung.

        ``GW_LLM_BASE_URL`` (Standard: lokales Ollama), ``GW_LLM_MODEL`` (Pflicht),
        ``GW_LLM_API_KEY`` (optional).
        """
        env = os.environ if environ is None else environ
        model = env.get(f"{prefix}MODEL")
        if not model:
            raise ValueError(f"{prefix}MODEL ist nicht gesetzt")
        return cls(
            base_url=env.get(f"{prefix}BASE_URL", "http://localhost:11434/v1"),
            model=model,
            api_key=env.get(f"{prefix}API_KEY") or None,
            **kwargs,
        )

    @property
    def model_name(self) -> str:
        return self._model

    def __repr__(self) -> str:
        key = "gesetzt" if self._api_key else "keiner"
        return f"OpenAICompatClient(url={self._url!r}, model={self._model!r}, api_key={key})"

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[JSONObject] | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        body: JSONObject = {"model": self._model, "messages": list(messages)}
        if tools:
            body["tools"] = list(tools)
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self._temperature is not None:
            body["temperature"] = self._temperature
        if self._max_tokens is not None:
            body["max_tokens"] = self._max_tokens
        body.update(self._extra)
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        headers.update(self._headers)
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._transport.post(self._url, payload, headers, self._timeout)
            except LLMTransportError:
                if attempt >= self._retry.max_attempts:
                    raise
                self._sleep(self._retry.delay(attempt, self._rng, None))
                continue
            if 200 <= response.status < 300:
                return _parse_response(response.body, prompt_chars=len(payload))
            text = response.body.decode("utf-8", "replace")
            if response.status in self._retry.retry_statuses and attempt < self._retry.max_attempts:
                self._sleep(self._retry.delay(attempt, self._rng, _retry_after(response.headers)))
                continue
            lowered = text.lower()
            if (
                tools
                and response.status in (400, 404, 422, 500, 501)
                and any(marker in lowered for marker in _TOOLS_UNSUPPORTED)
            ):
                raise ToolsNotSupportedError(response.status, text)
            raise LLMHTTPError(response.status, text)


def _content_text(content: object) -> str | None:
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):  # "content parts"
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        return "".join(str(p) for p in parts)
    raise LLMResponseError(f"Unerwarteter content-Typ: {type(content).__name__}")


def _parse_response(raw: bytes, *, prompt_chars: int) -> LLMResponse:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LLMResponseError(f"Antwort ist kein JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMResponseError("Antwort ist kein JSON-Objekt")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LLMResponseError("Antwort enthält keine 'choices'")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise LLMResponseError("Antwort enthält keine 'message'")
    content = _content_text(message.get("content"))
    calls: list[ToolCall] = []
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        raw_calls = []
    if not isinstance(raw_calls, list):
        raise LLMResponseError("'tool_calls' ist keine Liste")
    for index, item in enumerate(raw_calls):
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise LLMResponseError(f"tool_calls[{index}] ohne Funktionsnamen")
        arguments, raw_args, error = parse_tool_arguments(function.get("arguments"))
        call_id = item.get("id") if isinstance(item.get("id"), str) and item.get("id") else None
        calls.append(
            ToolCall(
                id=call_id or f"call_{index}",
                name=function["name"],
                arguments=arguments,
                raw_arguments=raw_args,
                error=error,
            )
        )
    usage_raw = data.get("usage")
    if isinstance(usage_raw, dict) and isinstance(usage_raw.get("prompt_tokens"), int):
        usage = Usage(
            prompt_tokens=int(usage_raw.get("prompt_tokens", 0)),
            completion_tokens=int(usage_raw.get("completion_tokens", 0) or 0),
        )
    else:
        completion = (content or "") + "".join(c.raw_arguments for c in calls)
        usage = Usage(
            prompt_tokens=math.ceil(prompt_chars / 4),
            completion_tokens=estimate_tokens(completion),
            estimated=True,
        )
    finish = choices[0].get("finish_reason")
    model = data.get("model")
    return LLMResponse(
        content=content,
        tool_calls=tuple(calls),
        finish_reason=finish if isinstance(finish, str) else None,
        usage=usage,
        model=model if isinstance(model, str) else None,
    )
