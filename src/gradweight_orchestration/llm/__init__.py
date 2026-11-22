"""Anbindung an Sprachmodelle: Client, Austauschformate, Skript-Modell."""

from __future__ import annotations

from .client import (
    HTTPResponse,
    HttpxTransport,
    OpenAICompatClient,
    RetryPolicy,
    Transport,
    UrllibTransport,
)
from .protocol import (
    AssistantTurn,
    JsonToolCalling,
    NativeToolCalling,
    ToolCallingMode,
    parse_json_turn,
)
from .scripted import ScriptContext, ScriptedLLM, ScriptedTurn, ScriptExhaustedError
from .types import (
    ChatModel,
    LLMError,
    LLMHTTPError,
    LLMResponse,
    LLMResponseError,
    LLMTransportError,
    Message,
    ProtocolError,
    ToolCall,
    ToolsNotSupportedError,
    Usage,
    estimate_tokens,
)

__all__ = [
    "AssistantTurn",
    "ChatModel",
    "HTTPResponse",
    "HttpxTransport",
    "JsonToolCalling",
    "LLMError",
    "LLMHTTPError",
    "LLMResponse",
    "LLMResponseError",
    "LLMTransportError",
    "Message",
    "NativeToolCalling",
    "OpenAICompatClient",
    "ProtocolError",
    "RetryPolicy",
    "ScriptContext",
    "ScriptExhaustedError",
    "ScriptedLLM",
    "ScriptedTurn",
    "ToolCall",
    "ToolCallingMode",
    "ToolsNotSupportedError",
    "Transport",
    "UrllibTransport",
    "Usage",
    "estimate_tokens",
    "parse_json_turn",
]
