"""Mock HTTP checks that LangChain sends the approved OpenRouter controls."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from openrouter.errors import ServiceUnavailableResponseError

from evidence_rag.openrouter import build_structured_chat, structured_output_runnable
from evidence_rag.verification import load_settings

ROOT = Path(__file__).resolve().parents[1]


def _response(request: httpx.Request) -> httpx.Response:
    content = {
        "verdict": "SUPPORTS",
        "explanation": "The mocked passage supports the claim.",
        "sources": [{"page_title": "Mock Page", "sentence_ids": [7]}],
        "conflicting_evidence": False,
    }
    return httpx.Response(
        200,
        request=request,
        json={
            "id": "mock-completion",
            "provider": "DeepInfra",
            "model": "mistralai/mistral-small-3.2-24b-instruct",
            "object": "chat.completion",
            "system_fingerprint": None,
            "created": 1_794_312_000,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps(content)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 91, "completion_tokens": 23, "total_tokens": 114},
        },
    )


def test_chat_openrouter_request_contains_provider_privacy_and_json_schema_controls() -> None:
    settings = load_settings(ROOT / "config" / "verification.toml")
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _response(request)

    with httpx.Client(transport=httpx.MockTransport(handle), timeout=2) as client:
        chat, _sdk = build_structured_chat(
            api_key="synthetic-test-key",
            model=settings.model,
            provider_only=list(settings.provider_only),
            timeout_seconds=settings.request_timeout_seconds,
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
            http_client=client,
        )
        result = structured_output_runnable(chat).invoke(
            [
                SystemMessage(content="test system instruction"),
                HumanMessage(content='{"page_title":"Mock Page","sentence_ids":[7]}'),
            ]
        )

    assert len(seen) == 1
    body = json.loads(seen[0].content)
    assert body["model"] == settings.model
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 512
    assert body["provider"] == {
        "only": ["DeepInfra"],
        "allow_fallbacks": False,
        "zdr": True,
        "data_collection": "deny",
        "require_parameters": True,
    }
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert result["parsed"].verdict == "SUPPORTS"
    assert result["raw"].usage_metadata["input_tokens"] == 91
    assert result["raw"].usage_metadata["output_tokens"] == 23


def test_openrouter_builder_rejects_missing_key_and_non_deepinfra_route() -> None:
    settings = load_settings(ROOT / "config" / "verification.toml")
    with httpx.Client(transport=httpx.MockTransport(_response)) as client:
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            build_structured_chat(
                api_key="",
                model=settings.model,
                provider_only=list(settings.provider_only),
                timeout_seconds=30,
                temperature=0,
                max_output_tokens=512,
                http_client=client,
            )
        with pytest.raises(ValueError, match="DeepInfra"):
            build_structured_chat(
                api_key="synthetic-test-key",
                model=settings.model,
                provider_only=["another-provider"],
                timeout_seconds=30,
                temperature=0,
                max_output_tokens=512,
                http_client=client,
            )


def test_sdk_does_not_retry_transient_responses_itself() -> None:
    settings = load_settings(ROOT / "config" / "verification.toml")
    request_count = 0

    def fail_once(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            503,
            request=request,
            json={"error": {"code": 503, "message": "mock outage"}},
        )

    with httpx.Client(transport=httpx.MockTransport(fail_once), timeout=2) as client:
        chat, _sdk = build_structured_chat(
            api_key="synthetic-test-key",
            model=settings.model,
            provider_only=list(settings.provider_only),
            timeout_seconds=30,
            temperature=0,
            max_output_tokens=512,
            http_client=client,
        )
        with pytest.raises(ServiceUnavailableResponseError):
            structured_output_runnable(chat).invoke(
                [SystemMessage(content="system"), HumanMessage(content="claim and evidence")]
            )

    assert request_count == 1
