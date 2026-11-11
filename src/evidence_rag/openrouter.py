"""Thin, fail-closed LangChain boundary for the pinned OpenRouter endpoint."""

from __future__ import annotations

from typing import Annotated, Literal

import httpx
from langchain_openrouter import ChatOpenRouter
from openrouter import OpenRouter
from openrouter.utils.retries import BackoffStrategy, RetryConfig
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceReference(BaseModel):
    """Original FEVER source coordinates returned by the model."""

    model_config = ConfigDict(extra="forbid", strict=True)

    page_title: Annotated[str, Field(min_length=1)]
    sentence_ids: Annotated[list[Annotated[int, Field(ge=0)]], Field(min_length=1)]


class VerificationOutput(BaseModel):
    """Small structured contract; application code validates source membership."""

    model_config = ConfigDict(extra="forbid", strict=True)

    verdict: Literal["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    explanation: Annotated[str, Field(min_length=1, max_length=2_000)]
    sources: list[SourceReference]
    conflicting_evidence: bool

    @field_validator("explanation")
    @classmethod
    def explanation_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("explanation must contain non-whitespace text")
        return value.strip()


def build_structured_chat(
    *,
    api_key: str,
    model: str,
    provider_only: list[str],
    timeout_seconds: int,
    temperature: float,
    max_output_tokens: int,
    http_client: httpx.Client,
) -> tuple[ChatOpenRouter, OpenRouter]:
    """Build the exact SDK client whose outbound request is covered by tests."""
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required for live verification")
    if provider_only != ["DeepInfra"]:
        raise ValueError("verification is pinned to the DeepInfra provider only")
    if timeout_seconds < 1 or max_output_tokens < 1:
        raise ValueError("timeout and output token limit must be positive")

    no_sdk_retries = RetryConfig(
        strategy="none",
        backoff=BackoffStrategy(
            initial_interval=250,
            max_interval=250,
            exponent=1.0,
            max_elapsed_time=250,
            jitter_ms=0,
        ),
        retry_connection_errors=False,
    )
    sdk = OpenRouter(
        api_key=api_key,
        client=http_client,
        retry_config=no_sdk_retries,
        timeout_ms=timeout_seconds * 1_000,
    )
    chat = ChatOpenRouter(
        client=sdk,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_output_tokens,
        max_retries=0,
        openrouter_provider={
            "only": provider_only,
            "allow_fallbacks": False,
            "zdr": True,
            "data_collection": "deny",
            "require_parameters": True,
        },
    )
    return chat, sdk


def structured_output_runnable(chat: ChatOpenRouter):
    """Use OpenRouter's strict JSON Schema mode and retain provider usage metadata."""
    return chat.with_structured_output(
        VerificationOutput,
        method="json_schema",
        strict=True,
        include_raw=True,
    )
