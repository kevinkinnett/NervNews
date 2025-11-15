"""Tests for the LLM client debug logging behaviour."""

import logging

import pytest

from src.config.settings import LLMSettings
from src.llm.client import LLMClient, LLMClientError
from src.llm.prompts.base import JsonPromptTemplate


class _DummyClient:
    def chat(self, *, model, messages, options):  # pragma: no cover - simple stub
        return {"message": {"content": "not json"}}


class _DummyError(Exception):
    """Sentinel error used to simulate ollama failures."""


class _SchemaClient:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):  # pragma: no cover - simple stub
        self.calls.append(kwargs)
        return {"message": {"content": "{\"summary\": \"works\"}"}}


class _JSONClient:
    def chat(self, **kwargs):  # pragma: no cover - simple stub
        return {"message": {"content": "{\"result\": \"ok\"}"}}


def test_generate_structured_logs_debug_payload(monkeypatch, caplog) -> None:
    settings = LLMSettings(
        provider="ollama",
        model="dummy-model",
        base_url="http://localhost:11434",
        max_retries=1,
        debug_payloads=True,
    )
    client = LLMClient(settings)

    monkeypatch.setattr(client, "_get_client", lambda: (_DummyClient(), _DummyError))
    monkeypatch.setattr(
        client,
        "_build_debug_headers",
        lambda: {"Content-Type": "application/json", "Authorization": "Bearer secret"},
    )

    template = JsonPromptTemplate(
        name="debug-test",
        system_prompt="System",
        user_template="Explain {topic}",
        response_schema={"type": "object", "required": ["summary"]},
    )

    caplog.set_level(logging.DEBUG)

    with pytest.raises(LLMClientError) as excinfo:
        client.generate_structured(template, {"topic": "debugging"})

    debug_info = excinfo.value.debug_info
    assert debug_info is not None
    assert debug_info["url"].endswith("/api/chat")
    assert debug_info["payload"]["model"] == "dummy-model"
    assert "not json" in debug_info["response_body"]

    debug_messages = [
        record.message
        for record in caplog.records
        if record.levelno == logging.DEBUG and "LLM debug payload" in record.message
    ]
    assert debug_messages, "expected debug payload logs when flag is enabled"

    logged_payload = debug_messages[-1]
    assert "\"Authorization\": \"***\"" in logged_payload
    assert "not json" in logged_payload


def test_generate_structured_sends_schema(monkeypatch) -> None:
    settings = LLMSettings(
        provider="ollama",
        model="dummy-model",
        base_url="http://localhost:11434",
        max_retries=1,
    )
    client = LLMClient(settings)

    schema_client = _SchemaClient()
    monkeypatch.setattr(client, "_get_client", lambda: (schema_client, _DummyError))

    template = JsonPromptTemplate(
        name="schema-test",
        system_prompt="System",
        user_template="Explain {topic}",
        response_schema={"type": "object", "required": ["summary"]},
    )

    payload = client.generate_structured(template, {"topic": "testing"})

    assert payload == {"summary": "works"}

    assert schema_client.calls, "expected schema client to be invoked"
    kwargs = schema_client.calls[0]
    assert "format" in kwargs
    assert kwargs["format"] == template.response_format


def test_generate_structured_rejects_non_json_early(monkeypatch) -> None:
    settings = LLMSettings(
        provider="ollama",
        model="dummy-model",
        base_url="http://localhost:11434",
        max_retries=1,
    )
    client = LLMClient(settings)

    monkeypatch.setattr(client, "_get_client", lambda: (_DummyClient(), _DummyError))

    template = JsonPromptTemplate(
        name="reject-test",
        system_prompt="System",
        user_template="Explain {topic}",
        response_schema={"type": "object", "required": ["summary"]},
    )

    with pytest.raises(LLMClientError) as excinfo:
        client.generate_structured(template, {"topic": "debugging"})

    cause = excinfo.value.__cause__
    assert cause is not None
    assert "was not JSON" in str(cause)
    assert excinfo.value.debug_info["error"] == "LLM response was not JSON"


def test_generate_structured_logs_prompt_usage_and_warns(monkeypatch, caplog) -> None:
    settings = LLMSettings(
        provider="ollama",
        model="dummy-model",
        base_url="http://localhost:11434",
        context_window=10,
        max_retries=1,
    )
    client = LLMClient(settings)

    monkeypatch.setattr(client, "_get_client", lambda: (_JSONClient(), _DummyError))

    template = JsonPromptTemplate(
        name="oversize",
        system_prompt="System instructions.",
        user_template="Write a response about:\n{content}",
        response_schema={
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        },
    )

    caplog.set_level(logging.WARNING)

    payload = client.generate_structured(template, {"content": "x" * 400})

    assert payload == {"result": "ok"}

    warning_messages = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert warning_messages, "expected context window warning when prompt is oversized"
    oversize_records = [record for record in warning_messages if "exceeding context window" in record.message]
    assert oversize_records, "expected oversize prompt warning"
    record = oversize_records[-1]
    assert hasattr(record, "prompt_tokens")
    assert record.prompt_tokens > settings.context_window
