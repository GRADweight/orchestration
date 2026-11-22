"""OpenAI-kompatibler Client gegen einen lokalen Fake-Server."""

from __future__ import annotations

import socket

import pytest

from fake_openai import FakeOpenAI, Reply, chat
from gradweight_orchestration.llm import (
    HttpxTransport,
    LLMHTTPError,
    LLMResponseError,
    LLMTransportError,
    OpenAICompatClient,
    RetryPolicy,
    ToolsNotSupportedError,
    UrllibTransport,
)


def make_client(server: FakeOpenAI, **kwargs: object) -> tuple[OpenAICompatClient, list[float]]:
    sleeps: list[float] = []
    options: dict[str, object] = {
        "base_url": server.base_url,
        "model": "qwen2.5:7b-instruct",
        "sleep": sleeps.append,
        "rng": lambda: 1.0,
        "timeout_s": 5.0,
    }
    options.update(kwargs)
    return OpenAICompatClient(**options), sleeps  # type: ignore[arg-type]


TOOL = {
    "type": "function",
    "function": {
        "name": "rechnung_lesen",
        "description": "x",
        "parameters": {"type": "object", "properties": {}},
    },
}


def test_plain_answer(fake_openai: FakeOpenAI) -> None:
    fake_openai.queue.append(chat("Grüezi"))
    client, _ = make_client(fake_openai)
    response = client.complete([{"role": "user", "content": "Hallo"}])
    assert response.content == "Grüezi"
    assert response.tool_calls == ()
    assert response.usage.prompt_tokens == 10
    assert response.usage.estimated is False
    assert response.model == "fake-7b"
    assert response.finish_reason == "stop"
    request = fake_openai.requests[0]
    assert request["model"] == "qwen2.5:7b-instruct"
    assert request["temperature"] == 0.0
    assert "tools" not in request
    assert "tool_choice" not in request
    assert "response_format" not in request
    assert "authorization" not in fake_openai.headers[0]


def test_tool_calls_and_request_fields(fake_openai: FakeOpenAI) -> None:
    fake_openai.queue.append(
        chat(None, [("rechnung_lesen", {"dokument_id": "RE-1"}), ("x", "{kaputt")])
    )
    client, _ = make_client(
        fake_openai,
        api_key="geheim",
        max_tokens=256,
        extra_body={"seed": 7},
        headers={"X-Mandant": "K1"},
    )
    response = client.complete([{"role": "user", "content": "x"}], tools=[TOOL], json_mode=True)
    first, second = response.tool_calls
    assert first.name == "rechnung_lesen"
    assert first.arguments == {"dokument_id": "RE-1"}
    assert first.id == "call_0"
    assert second.arguments is None
    assert second.error is not None
    request = fake_openai.requests[0]
    assert request["tools"] == [TOOL]
    assert "tool_choice" not in request  # Ollama kennt tool_choice nicht
    assert request["response_format"] == {"type": "json_object"}
    assert request["max_tokens"] == 256
    assert request["seed"] == 7
    headers = fake_openai.headers[0]
    assert headers["authorization"] == "Bearer geheim"
    assert headers["x-mandant"] == "K1"
    assert "geheim" not in repr(client)


def test_arguments_as_object_and_missing_id(fake_openai: FakeOpenAI) -> None:
    body = chat().body
    body["choices"][0]["message"]["tool_calls"] = [
        {"type": "function", "function": {"name": "f", "arguments": {"a": 1}}}
    ]
    body["choices"][0]["message"]["content"] = [
        {"type": "text", "text": "Teil "},
        {"type": "text", "text": "zwei"},
    ]
    del body["usage"]
    fake_openai.queue.append(Reply(body=body))
    response = make_client(fake_openai)[0].complete([{"role": "user", "content": "x"}])
    assert response.tool_calls[0].arguments == {"a": 1}
    assert response.tool_calls[0].id == "call_0"
    assert response.content == "Teil zwei"
    assert response.usage.estimated is True
    assert response.usage.prompt_tokens > 0


def test_retries_on_server_errors_with_backoff(fake_openai: FakeOpenAI) -> None:
    fake_openai.queue += [Reply(status=503, body={}), Reply(status=502, body={}), chat("ok")]
    client, sleeps = make_client(fake_openai)
    assert client.complete([{"role": "user", "content": "x"}]).content == "ok"
    assert sleeps == [0.5, 1.0]


def test_retry_after_is_respected_and_capped(fake_openai: FakeOpenAI) -> None:
    fake_openai.queue += [
        Reply(status=429, body={}, headers={"Retry-After": "2"}),
        Reply(status=429, body={}, headers={"Retry-After": "120"}),
        Reply(status=429, body={}, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
        chat("ok"),
    ]
    client, sleeps = make_client(fake_openai, retry=RetryPolicy(max_attempts=4))
    assert client.complete([{"role": "user", "content": "x"}]).content == "ok"
    assert sleeps == [2.0, 8.0, 2.0]


def test_client_errors_are_not_retried(fake_openai: FakeOpenAI) -> None:
    fake_openai.queue.append(Reply(status=400, body={"error": {"message": "schlechte Anfrage"}}))
    client, sleeps = make_client(fake_openai)
    with pytest.raises(LLMHTTPError) as info:
        client.complete([{"role": "user", "content": "x"}])
    assert info.value.status == 400
    assert "schlechte Anfrage" in str(info.value)
    assert sleeps == []


def test_exhausted_retries_raise_last_status(fake_openai: FakeOpenAI) -> None:
    fake_openai.queue += [Reply(status=500, body={})] * 2
    client, _ = make_client(fake_openai, retry=RetryPolicy(max_attempts=2))
    with pytest.raises(LLMHTTPError):
        client.complete([{"role": "user", "content": "x"}])


def test_tools_not_supported_is_detected(fake_openai: FakeOpenAI) -> None:
    message = {"error": {"message": "registry.ollama.ai/library/gemma:2b does not support tools"}}
    fake_openai.queue += [Reply(status=400, body=message), Reply(status=400, body=message)]
    client, _ = make_client(fake_openai)
    with pytest.raises(ToolsNotSupportedError):
        client.complete([{"role": "user", "content": "x"}], tools=[TOOL])
    with pytest.raises(LLMHTTPError) as info:  # ohne tools: normaler HTTP-Fehler
        client.complete([{"role": "user", "content": "x"}])
    assert not isinstance(info.value, ToolsNotSupportedError)


def test_timeout_is_retried_then_raised(fake_openai: FakeOpenAI) -> None:
    fake_openai.queue += [Reply(body={}, delay_s=1.0), Reply(body={}, delay_s=1.0)]
    client, sleeps = make_client(fake_openai, timeout_s=0.2, retry=RetryPolicy(max_attempts=2))
    with pytest.raises(LLMTransportError):
        client.complete([{"role": "user", "content": "x"}])
    assert sleeps == [0.5]


def test_connection_refused() -> None:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    client = OpenAICompatClient(
        base_url=f"http://127.0.0.1:{port}/v1",
        model="m",
        retry=RetryPolicy(max_attempts=1),
        transport=UrllibTransport(trust_env=False),
    )
    with pytest.raises(LLMTransportError):
        client.complete([{"role": "user", "content": "x"}])


@pytest.mark.parametrize(
    "body",
    [
        b"kein json",
        b"[]",
        b"{}",
        b'{"choices": [{}]}',
        b'{"choices": [{"message": {"content": 5}}]}',
        b'{"choices": [{"message": {"tool_calls": {}}}]}',
        b'{"choices": [{"message": {"tool_calls": [{"function": {}}]}}]}',
    ],
)
def test_malformed_responses(fake_openai: FakeOpenAI, body: bytes) -> None:
    fake_openai.queue.append(Reply(body=body))
    with pytest.raises(LLMResponseError):
        make_client(fake_openai)[0].complete([{"role": "user", "content": "x"}])


def test_wrong_path_is_http_error(fake_openai: FakeOpenAI) -> None:
    client = OpenAICompatClient(
        base_url=fake_openai.base_url + "/falsch", model="m", retry=RetryPolicy(max_attempts=1)
    )
    with pytest.raises(LLMHTTPError) as info:
        client.complete([{"role": "user", "content": "x"}])
    assert info.value.status == 404


def test_configuration_validation() -> None:
    with pytest.raises(ValueError, match="http"):
        OpenAICompatClient(base_url="file:///etc/passwd", model="m")
    with pytest.raises(ValueError, match="model"):
        OpenAICompatClient(base_url="http://localhost:11434/v1", model="")
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)


def test_from_env() -> None:
    client = OpenAICompatClient.from_env(environ={"GW_LLM_MODEL": "llama3.1:8b"})
    assert client.model_name == "llama3.1:8b"
    assert "localhost:11434" in repr(client)
    with pytest.raises(ValueError, match="GW_LLM_MODEL"):
        OpenAICompatClient.from_env(environ={})
    remote = OpenAICompatClient.from_env(
        environ={
            "GW_LLM_MODEL": "m",
            "GW_LLM_BASE_URL": "https://example.invalid/v1",
            "GW_LLM_API_KEY": "k",
        }
    )
    assert "api_key=gesetzt" in repr(remote)


def test_httpx_transport(fake_openai: FakeOpenAI) -> None:
    pytest.importorskip("httpx")
    fake_openai.queue += [chat("über httpx"), Reply(body={}, delay_s=1.0)]
    client, _ = make_client(fake_openai, transport=HttpxTransport(trust_env=False))
    assert client.complete([{"role": "user", "content": "x"}]).content == "über httpx"
    slow, _ = make_client(
        fake_openai,
        transport=HttpxTransport(trust_env=False),
        timeout_s=0.2,
        retry=RetryPolicy(max_attempts=1),
    )
    with pytest.raises(LLMTransportError):
        slow.complete([{"role": "user", "content": "x"}])
