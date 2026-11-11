"""Synthetic regressions for bounded evidence packing and response validation."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage

from evidence_rag import cli
from evidence_rag.models import RetrievalHit, SearchResponse
from evidence_rag.openrouter import SourceReference, VerificationOutput
from evidence_rag.verification import (
    SYSTEM_PROMPT,
    VerificationError,
    load_model_tokenizer,
    load_settings,
    pack_context,
    run_verification,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "verification.toml"
CORPUS_HASH = "a" * 64


class ByteTokenizer:
    def encode(self, text: str, *, disallowed_special: tuple[()] = ()) -> list[int]:
        del disallowed_special
        return list(text.encode("utf-8"))


class FakeRetriever:
    def __init__(self, method: str, hits: tuple[RetrievalHit, ...]) -> None:
        self.method = method
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        self.calls.append((query, top_k))
        return SearchResponse(self.method, query, 1.25, self.hits[:top_k])


class FakeRunnable:
    def __init__(self, values: list[Any]) -> None:
        self.values = values
        self.calls: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> Any:
        self.calls.append(messages)
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def _hit(
    passage_id: str = "p1",
    page_title: str = "Example Page",
    sentence_ids: tuple[int, ...] = (1,),
    text: str = "Synthetic source sentence.",
    rank: int = 1,
    method: str = "bm25",
) -> RetrievalHit:
    return RetrievalHit(
        passage_id=passage_id,
        page_title=page_title,
        sentence_ids=sentence_ids,
        score=1.0 / rank,
        text=text,
        rank=rank,
        retrieval_method=method,
    )


def _output(
    verdict: str = "SUPPORTS",
    *,
    sources: list[SourceReference] | None = None,
    conflict: bool = False,
) -> VerificationOutput:
    if sources is None:
        sources = [SourceReference(page_title="Example Page", sentence_ids=[1])]
    return VerificationOutput(
        verdict=verdict,
        explanation="The supplied synthetic passage warrants this decision.",
        sources=sources,
        conflicting_evidence=conflict,
    )


def _structured(output: Any, usage: dict[str, int] | None = None) -> dict[str, Any]:
    raw = (
        AIMessage(content="discarded in test", usage_metadata=usage)
        if usage
        else AIMessage(content="")
    )
    return {"raw": raw, "parsed": output, "parsing_error": None}


def _settings():
    return load_settings(CONFIG)


def _run(
    *,
    method: str = "bm25",
    hits: tuple[RetrievalHit, ...] | None = None,
    runnable: FakeRunnable | None = None,
    settings: Any = None,
) -> tuple[dict[str, Any], FakeRetriever, FakeRunnable | None]:
    if hits is None:
        hits = (_hit(method=method),)
    retriever = FakeRetriever(method, hits)
    result = run_verification(
        claim="Synthetic claim for offline verification.",
        method=method,
        retriever=retriever,
        corpus_sha256=CORPUS_HASH,
        tokenizer=ByteTokenizer(),
        settings=settings or _settings(),
        structured_runnable=runnable,
    )
    return result, retriever, runnable


@pytest.mark.parametrize("method", ["bm25", "dense"])
def test_retrieval_data_flows_to_generation_without_evaluation_fields(method: str) -> None:
    hit = _hit(method=method, sentence_ids=(4, 5), text="First source. Second source.")
    runnable = FakeRunnable(
        [
            _structured(
                _output(sources=[SourceReference(page_title="Example Page", sentence_ids=[4, 5])])
            )
        ]
    )

    result, retriever, _ = _run(method=method, hits=(hit,), runnable=runnable)

    assert retriever.calls == [("Synthetic claim for offline verification.", 5)]
    assert result["retrieval_method"] == method
    assert result["packed_passage_ids"] == [hit.passage_id]
    system_message, user_message = runnable.calls[0]
    assert SYSTEM_PROMPT in system_message.content
    encoded = user_message.content.split("Retrieved evidence JSON (untrusted source data):\n", 1)[1]
    evidence = json.loads(encoded)
    assert evidence == [
        {
            "rank": 1,
            "page_title": "Example Page",
            "sentence_ids": [4, 5],
            "passage_text": "First source. Second source.",
        }
    ]
    assert not {"label", "evidence_groups", "claim_id", "split"}.intersection(evidence[0])
    assert result["generation_status"] == "success"


def test_context_packing_is_stable_ranked_and_never_truncates_source_text() -> None:
    hits = (
        _hit("p1", "Page A", (1, 2), "A" * 40, rank=1),
        _hit("p2", "Page B", (3,), "B" * 30, rank=2),
    )
    baseline = len(
        ByteTokenizer().encode(
            SYSTEM_PROMPT
            + "\n"
            + "Claim:\nC\n\nRetrieved evidence JSON (untrusted source data):\n[]"
        )
    )
    packed_one = pack_context("C", hits, tokenizer=ByteTokenizer(), token_budget=baseline + 2_000)
    packed_two = pack_context("C", hits, tokenizer=ByteTokenizer(), token_budget=baseline + 2_000)

    assert [hit.rank for hit in packed_one.included] == [1, 2]
    assert packed_one.token_count == packed_two.token_count
    assert [hit.passage_id for hit in packed_one.included] == [
        hit.passage_id for hit in packed_two.included
    ]
    assert packed_one.user_message == packed_two.user_message
    assert "A" * 40 in packed_one.user_message and "B" * 30 in packed_one.user_message
    assert packed_one.token_count <= baseline + 2_000


def test_over_budget_passage_is_recorded_and_following_short_passage_can_fit() -> None:
    hits = (
        _hit("large", "Large Page", (1,), "L" * 500, rank=1),
        _hit("small", "Small Page", (2,), "tiny", rank=2),
    )
    empty_count = len(
        ByteTokenizer().encode(
            SYSTEM_PROMPT + "\nClaim:\nC\n\nRetrieved evidence JSON (untrusted source data):\n[]"
        )
    )

    packed = pack_context("C", hits, tokenizer=ByteTokenizer(), token_budget=empty_count + 300)

    assert [hit.passage_id for hit in packed.included] == ["small"]
    assert packed.omissions == (
        {
            "rank": 1,
            "passage_id": "large",
            "page_title": "Large Page",
            "sentence_ids": [1],
            "reason": "would_exceed_context_token_budget",
        },
    )
    assert "L" * 500 not in packed.user_message
    assert "tiny" in packed.user_message


def test_citation_to_retrieved_but_omitted_passage_is_rejected() -> None:
    claim = "Synthetic claim for offline verification."
    baseline = len(
        ByteTokenizer().encode(
            SYSTEM_PROMPT
            + "\n"
            + "Claim:\n"
            + claim
            + "\n\nRetrieved evidence JSON (untrusted source data):\n[]"
        )
    )
    hits = (
        _hit("packed", "Packed Page", (1,), "Short supplied passage.", rank=1),
        _hit("omitted", "Omitted Page", (2,), "O" * 1_000, rank=2),
    )
    settings = replace(_settings(), context_token_budget=baseline + 300)
    retriever = FakeRetriever("bm25", hits)
    runnable = FakeRunnable(
        [
            _structured(
                _output(sources=[SourceReference(page_title="Omitted Page", sentence_ids=[2])])
            )
        ]
    )

    result = run_verification(
        claim=claim,
        method="bm25",
        retriever=retriever,
        corpus_sha256=CORPUS_HASH,
        tokenizer=ByteTokenizer(),
        settings=settings,
        structured_runnable=runnable,
    )

    assert result["packed_passage_ids"] == ["packed"]
    assert result["context_omissions"][0]["passage_id"] == "omitted"
    assert result["generation_status"] == "operational_failure"
    assert result["operational_error"]["kind"] == "invalid_response"


@pytest.mark.parametrize("verdict", ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"])
def test_exact_verdict_names_and_valid_citations_are_accepted(verdict: str) -> None:
    sources = (
        []
        if verdict == "NOT ENOUGH INFO"
        else [SourceReference(page_title="Example Page", sentence_ids=[1])]
    )
    result, _, _ = _run(runnable=FakeRunnable([_structured(_output(verdict, sources=sources))]))

    assert result["verdict"] == verdict
    assert result["generation_status"] == "success"
    assert result["sources"] == [source.model_dump() for source in sources]
    rendered_texts = [source["text"] for source in result["rendered_sources"]]
    assert rendered_texts == (["Synthetic source sentence."] if sources else [])


def test_multi_page_multi_sentence_citations_render_original_passages() -> None:
    hits = (
        _hit("p1", "Example Page", (1, 2), "Source one and two.", rank=1),
        _hit("p2", "Other Page", (9,), "Other source nine.", rank=2),
    )
    output = _output(
        sources=[
            SourceReference(page_title="Example Page", sentence_ids=[1, 2]),
            SourceReference(page_title="Other Page", sentence_ids=[9]),
        ]
    )

    result, _, _ = _run(hits=hits, runnable=FakeRunnable([_structured(output)]))

    assert [source["page_title"] for source in result["sources"]] == ["Example Page", "Other Page"]
    assert [source["text"] for source in result["rendered_sources"]] == [
        "Source one and two.",
        "Other source nine.",
    ]


@pytest.mark.parametrize(
    ("output", "expected_kind"),
    [
        ({"verdict": "SUPPORTED"}, "invalid_response"),
        (
            _output(sources=[SourceReference(page_title="Invented Page", sentence_ids=[99])]),
            "invalid_response",
        ),
        (
            _output(
                "SUPPORTS",
                sources=[SourceReference(page_title="Example Page", sentence_ids=[1])],
                conflict=True,
            ),
            "invalid_response",
        ),
    ],
)
def test_invalid_schema_fabricated_citations_and_conflict_mismatch_fail_operationally(
    output: Any, expected_kind: str
) -> None:
    result, _, _ = _run(runnable=FakeRunnable([_structured(output)]))

    assert result["generation_status"] == "operational_failure"
    assert result["operational_error"]["kind"] == expected_kind
    assert "verdict" not in result
    assert len(result["generation_attempts"]) == 1


def test_malformed_structured_response_is_rejected_without_echoing_raw_response() -> None:
    secret_response = "synthetic malformed response body"
    result, _, _ = _run(
        runnable=FakeRunnable(
            [
                {
                    "raw": AIMessage(content=secret_response),
                    "parsed": None,
                    "parsing_error": ValueError(secret_response),
                }
            ]
        )
    )

    assert result["generation_status"] == "operational_failure"
    assert result["operational_error"]["kind"] == "invalid_response"
    assert secret_response not in json.dumps(result)


def test_duplicate_citation_sentence_ids_are_rejected() -> None:
    output = _output(sources=[SourceReference(page_title="Example Page", sentence_ids=[1, 1])])

    result, _, _ = _run(runnable=FakeRunnable([_structured(output)]))

    assert result["generation_status"] == "operational_failure"
    assert result["operational_error"]["kind"] == "invalid_response"


def test_conflicting_evidence_is_reported_as_not_enough_info() -> None:
    hits = (
        _hit("p1", "Example Page", (1,), "The claim is true.", rank=1),
        _hit("p2", "Other Page", (2,), "The claim is false.", rank=2),
    )
    output = _output(
        "NOT ENOUGH INFO",
        sources=[
            SourceReference(page_title="Example Page", sentence_ids=[1]),
            SourceReference(page_title="Other Page", sentence_ids=[2]),
        ],
        conflict=True,
    )

    result, _, _ = _run(hits=hits, runnable=FakeRunnable([_structured(output)]))

    assert result["verdict"] == "NOT ENOUGH INFO"
    assert result["conflicting_evidence"] is True
    assert len(result["sources"]) == 2


def test_conflict_without_citations_is_rejected() -> None:
    output = _output("NOT ENOUGH INFO", sources=[], conflict=True)

    result, _, _ = _run(runnable=FakeRunnable([_structured(output)]))

    assert result["generation_status"] == "operational_failure"
    assert result["operational_error"]["kind"] == "invalid_response"


def test_empty_evidence_returns_not_enough_info_without_provider_request() -> None:
    runnable = FakeRunnable([])

    result, retriever, _ = _run(hits=(), runnable=runnable)

    assert retriever.calls == [("Synthetic claim for offline verification.", 5)]
    assert runnable.calls == []
    assert result["verdict"] == "NOT ENOUGH INFO"
    assert result["generation_status"] == "not_requested_empty_context"
    assert result["generation_attempts"] == []
    assert "operational_error" not in result


def test_timeout_retries_once_records_attempts_and_uses_provider_token_cost() -> None:
    sleep_calls: list[float] = []
    runnable = FakeRunnable(
        [
            httpx.ReadTimeout("mock timeout"),
            _structured(_output(), {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}),
        ]
    )
    retriever = FakeRetriever("bm25", (_hit(),))
    result = run_verification(
        claim="Synthetic claim.",
        method="bm25",
        retriever=retriever,
        corpus_sha256=CORPUS_HASH,
        tokenizer=ByteTokenizer(),
        settings=_settings(),
        structured_runnable=runnable,
        sleep=sleep_calls.append,
    )

    assert len(result["generation_attempts"]) == 2
    assert [row["status"] for row in result["generation_attempts"]] == [
        "retryable_provider_error",
        "response_received",
    ]
    assert sleep_calls == [0.25]
    assert result["provider_token_usage"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "reported_for_all_attempts": False,
    }
    expected_upper_cost = (result["context_token_count"] * 2 * 0.075 + 2 * 512 * 0.20) / 1_000_000
    assert result["estimated_cost_usd"] == pytest.approx(expected_upper_cost)
    assert "upper estimate" in result["cost_estimate_basis"]


def test_exhausted_retry_is_operational_failure_not_abstention() -> None:
    runnable = FakeRunnable([httpx.ConnectTimeout("first"), httpx.ReadTimeout("second")])
    result, _, _ = _run(runnable=runnable)

    assert result["generation_status"] == "operational_failure"
    assert result["operational_error"]["kind"] == "provider_error"
    assert len(result["generation_attempts"]) == 2
    assert "verdict" not in result


def test_provider_usage_and_dated_pricing_are_recorded() -> None:
    result, _, _ = _run(
        runnable=FakeRunnable(
            [
                _structured(
                    _output(), {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
                )
            ]
        )
    )

    assert result["provider_token_usage"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "reported_for_all_attempts": True,
    }
    assert result["pricing_observed_on"] == "2026-11-11"
    assert result["pricing_source"].startswith("https://openrouter.ai/")
    assert result["estimated_cost_usd"] == pytest.approx(0.0000115)
    assert result["corpus_sha256"] == CORPUS_HASH
    assert len(result["configuration_sha256"]) == 64


def test_runtime_evidence_payload_has_no_gold_or_split_metadata() -> None:
    result, _, runnable = _run(runnable=FakeRunnable([_structured(_output())]))
    del result
    assert runnable is not None
    human_text = runnable.calls[0][1].content
    evidence_text = human_text.split("Retrieved evidence JSON (untrusted source data):\n", 1)[1]
    evidence = json.loads(evidence_text)
    assert set(evidence[0]) == {"rank", "page_title", "sentence_ids", "passage_text"}
    for forbidden in ("gold", "label", "evidence_groups", "claim_id", "split", "heldout"):
        assert forbidden not in evidence_text.casefold()


def test_model_tokenizer_hash_and_encoding_are_checked(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "mistral-small-3.2-24b" / "tekken.json"
    tokenizer_path.parent.mkdir()
    vocab = [
        {
            "rank": byte,
            "token_bytes": base64.b64encode(bytes([byte])).decode("ascii"),
        }
        for byte in range(256)
    ]
    token_bytes = json.dumps(
        {"config": {"pattern": "(?s).+"}, "vocab": vocab},
        separators=(",", ":"),
    ).encode()
    tokenizer_path.write_bytes(token_bytes)
    settings = replace(
        _settings(),
        tokenizer_relative_path="mistral-small-3.2-24b/tekken.json",
        tokenizer_sha256=hashlib.sha256(token_bytes).hexdigest(),
    )

    encoding = load_model_tokenizer(settings, tmp_path)
    assert len(encoding.encode("model tokenizer", disallowed_special=())) > 0

    tokenizer_path.write_bytes(token_bytes + b" ")
    with pytest.raises(VerificationError, match="SHA-256"):
        load_model_tokenizer(settings, tmp_path)


def test_cli_missing_key_is_operational_and_no_live_flag_never_reads_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def forbidden_dotenv_read(**kwargs: Any) -> None:
        raise AssertionError("a missing live/account confirmation must not read .env")

    monkeypatch.setattr(cli, "load_dotenv", forbidden_dotenv_read)
    account_confirmation_exit = cli.main(
        [
            "verify",
            "--claim",
            "Synthetic claim.",
            "--method",
            "bm25",
            "--live",
            "--config",
            str(CONFIG),
        ]
    )
    assert account_confirmation_exit == 2
    assert "--confirm-account-controls" in capsys.readouterr().err

    monkeypatch.setattr(cli, "load_dotenv", lambda **kwargs: None)
    missing_key_exit = cli.main(
        [
            "verify",
            "--claim",
            "Synthetic claim.",
            "--method",
            "bm25",
            "--live",
            "--confirm-account-controls",
            "--config",
            str(CONFIG),
            "--env-file",
            str(tmp_path / "absent.env"),
        ]
    )
    assert missing_key_exit == 2
    assert "OPENROUTER_API_KEY is missing" in capsys.readouterr().err

    def forbidden_dotenv_read_after_disable(**kwargs: Any) -> None:
        raise AssertionError("without --live, .env must not be read")

    monkeypatch.setattr(cli, "load_dotenv", forbidden_dotenv_read_after_disable)
    disabled_exit = cli.main(
        ["verify", "--claim", "Synthetic claim.", "--method", "bm25", "--config", str(CONFIG)]
    )
    assert disabled_exit == 2
    assert "--live" in capsys.readouterr().err


@pytest.mark.parametrize("method", ["bm25", "dense"])
def test_cli_selects_requested_existing_retriever_with_mocked_generation(
    method: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    retriever = FakeRetriever(method, (_hit(method=method),))

    class DenseStub(FakeRetriever):
        corpus_sha256 = CORPUS_HASH

        def __init__(self, *args: Any) -> None:
            del args
            calls.append("dense")
            super().__init__("dense", (_hit(method="dense"),))

        def close(self) -> None:
            calls.append("closed")

    class Bm25Stub:
        @classmethod
        def from_path(cls, path: Path) -> tuple[FakeRetriever, str]:
            del path
            calls.append("bm25")
            return retriever, CORPUS_HASH

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-do-not-use")
    monkeypatch.setattr(cli, "load_dotenv", lambda **kwargs: None)
    monkeypatch.setattr(cli, "load_model_tokenizer", lambda settings, path: ByteTokenizer())
    monkeypatch.setattr(
        cli,
        "create_live_runnable",
        lambda *args, **kwargs: (FakeRunnable([_structured(_output())]), None),
    )
    monkeypatch.setattr(cli, "BM25Retriever", Bm25Stub)
    monkeypatch.setattr(cli, "DenseRetriever", DenseStub)

    args = cli._parser().parse_args(
        [
            "verify",
            "--claim",
            "Synthetic claim.",
            "--method",
            method,
            "--live",
            "--confirm-account-controls",
            "--config",
            str(CONFIG),
            "--env-file",
            str(tmp_path / "absent.env"),
        ]
    )
    result = cli._verify(args)

    assert result["retrieval_method"] == method
    assert result["account_level_controls_confirmed_by_operator"] is True
    assert result["raw_prompt_response_logging"] is False
    assert calls[0] == method
    if method == "dense":
        assert calls == ["dense", "closed"]


def test_settings_fail_closed_if_provider_fallback_is_enabled(tmp_path: Path) -> None:
    contents = CONFIG.read_text(encoding="utf-8").replace(
        "allow_fallbacks = false", "allow_fallbacks = true"
    )
    config_path = tmp_path / "unsafe.toml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(VerificationError, match="fail-closed"):
        load_settings(config_path)


def test_settings_fail_closed_on_quoted_privacy_booleans(tmp_path: Path) -> None:
    contents = CONFIG.read_text(encoding="utf-8").replace("zdr = true", 'zdr = "true"')
    config_path = tmp_path / "wrong-types.toml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(VerificationError, match="fail-closed"):
        load_settings(config_path)
