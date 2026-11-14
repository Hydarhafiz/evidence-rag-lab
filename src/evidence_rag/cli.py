"""Command-line entry points for corpus preparation and retrieval."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import httpx
from dotenv import load_dotenv

from evidence_rag.corpus import CorpusError, prepare_corpus
from evidence_rag.embeddings import load_encoder
from evidence_rag.evaluation import run_development_smoke, run_retrieval_evaluation
from evidence_rag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    RetrievalError,
    build_dense_index,
)
from evidence_rag.verification import (
    VerificationError,
    create_live_runnable,
    load_model_tokenizer,
    load_settings,
    run_verification,
)

DEFAULT_TRAIN = Path("data/raw/train.jsonl")
DEFAULT_DEV = Path("data/raw/shared_task_dev.jsonl")
DEFAULT_WIKI = Path("data/raw/wiki-pages.zip")
DEFAULT_PASSAGES = Path("data/processed/passages.jsonl")
DEFAULT_EVALUATION = Path("data/processed/evaluation_samples_train.jsonl")
DEFAULT_HELD_OUT_EVALUATION = Path("data/processed/evaluation_samples_dev.jsonl")
DEFAULT_QDRANT = Path("data/indexes/qdrant")
DEFAULT_MODEL_CACHE = Path("models/hf")
DEFAULT_VERIFICATION_CONFIG = Path("config/verification.toml")
DEFAULT_EVALUATION_OUTPUT = Path("artifacts/generated/m2-f04")


def _add_model_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_MODEL_CACHE)


def _add_retrieval_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--passages", type=Path, default=DEFAULT_PASSAGES)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT)
    _add_model_paths(parser)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fever-rag")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="validate FEVER inputs and build the 5K corpus")
    prepare.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    prepare.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    prepare.add_argument("--wiki-zip", type=Path, default=DEFAULT_WIKI)
    prepare.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    prepare.add_argument("--target-passages", type=int, default=5_000)
    _add_model_paths(prepare)

    index = commands.add_parser("index", help="build the persistent local dense Qdrant index")
    index.add_argument("--passages", type=Path, default=DEFAULT_PASSAGES)
    index.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT)
    index.add_argument("--batch-size", type=int, default=64)
    _add_model_paths(index)

    search = commands.add_parser(
        "search", help="retrieve source passages using BM25 or dense search"
    )
    search.add_argument("--method", choices=("bm25", "dense"), required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=5)
    _add_retrieval_paths(search)

    smoke = commands.add_parser(
        "smoke",
        help=(
            "compare retrieval on the 30-claim development sample without opening held-out results"
        ),
    )
    smoke.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    smoke.add_argument("--top-k", type=int, default=5)
    _add_retrieval_paths(smoke)

    evaluate = commands.add_parser(
        "evaluate",
        help="compare BM25 and dense retrieval offline on the frozen evaluation sample",
    )
    evaluate.add_argument("--split", choices=("train", "dev"), default="train")
    evaluate.add_argument("--evaluation", type=Path)
    evaluate.add_argument("--freeze", type=Path)
    evaluate.add_argument("--confirm-freeze-sha256")
    evaluate.add_argument(
        "--confirm-heldout-phase-authorized",
        action="store_true",
        help="attest that the separate held-out phase approval was granted after freeze review",
    )
    evaluate.add_argument("--output", type=Path)
    _add_retrieval_paths(evaluate)

    verify = commands.add_parser(
        "verify",
        help="verify one claim against retrieved evidence (requires explicit --live)",
    )
    verify.add_argument("--claim", required=True)
    verify.add_argument("--method", choices=("bm25", "dense"), required=True)
    verify.add_argument("--live", action="store_true", help="authorize this bounded live request")
    verify.add_argument(
        "--confirm-account-controls",
        action="store_true",
        help="confirm OpenRouter ZDR, disabled logging, and a bounded spending cap are configured",
    )
    verify.add_argument("--passages", type=Path, default=DEFAULT_PASSAGES)
    verify.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT)
    _add_model_paths(verify)
    verify.add_argument("--config", type=Path, default=DEFAULT_VERIFICATION_CONFIG)
    verify.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser


def _prepare(args: argparse.Namespace) -> dict[str, object]:
    encoder = load_encoder(args.cache_dir)
    result = prepare_corpus(
        train_path=args.train,
        dev_path=args.dev,
        wiki_zip_path=args.wiki_zip,
        output_dir=args.output_dir,
        tokenizer=encoder.tokenizer,
        maximum_tokens=encoder.max_seq_length,
        target_passages=args.target_passages,
    )
    return {
        "elapsed_seconds": result.elapsed_seconds,
        "manifest": result.manifest,
    }


def _index(args: argparse.Namespace) -> dict[str, object]:
    bm25, bm25_hash = BM25Retriever.from_path(args.passages)
    result = build_dense_index(
        passages_path=args.passages,
        qdrant_path=args.qdrant_path,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
    )
    if result.corpus_sha256 != bm25_hash or result.passage_count != len(bm25.passages):
        raise RetrievalError("BM25 and Qdrant were built from different passage files")
    return {
        "passage_count": len(bm25.passages),
        "corpus_sha256": bm25_hash,
        "bm25_index_build_latency_ms": bm25.build_latency_ms,
        "dense_index": asdict(result),
    }


def _search(args: argparse.Namespace) -> dict[str, object]:
    if args.method == "bm25":
        retriever, _ = BM25Retriever.from_path(args.passages)
        response = retriever.search(args.query, args.top_k)
        index_build_ms = retriever.build_latency_ms
    else:
        retriever = DenseRetriever(args.passages, args.qdrant_path, args.cache_dir)
        try:
            response = retriever.search(args.query, args.top_k)
        finally:
            retriever.close()
        index_build_ms = None
    return {
        "index_build_latency_ms": index_build_ms,
        "response": asdict(response),
    }


def _smoke(args: argparse.Namespace) -> dict[str, object]:
    return run_development_smoke(
        passages_path=args.passages,
        evaluation_path=args.evaluation,
        qdrant_path=args.qdrant_path,
        cache_dir=args.cache_dir,
        top_k=args.top_k,
    )


def _evaluate(args: argparse.Namespace) -> dict[str, object]:
    split = args.split
    evaluation_path = args.evaluation
    if evaluation_path is None:
        evaluation_path = DEFAULT_EVALUATION if split == "train" else DEFAULT_HELD_OUT_EVALUATION
    report_path = args.output or DEFAULT_EVALUATION_OUTPUT / f"retrieval-evaluation-{split}.json"
    report = run_retrieval_evaluation(
        passages_path=args.passages,
        evaluation_path=evaluation_path,
        qdrant_path=args.qdrant_path,
        cache_dir=args.cache_dir,
        split=split,
        freeze_path=args.freeze,
        confirmed_freeze_sha256=args.confirm_freeze_sha256,
        held_out_phase_authorized=args.confirm_heldout_phase_authorized,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    methods = report["methods"]
    return {
        "split": split,
        "report_path": str(report_path),
        "dataset_identity": report["dataset_identity"],
        "paired_comparison": report["paired_comparison"],
        "methods": {
            method: {
                "startup": methods[method]["startup"],
                "latency": methods[method]["latency"],
                "ranking_stability": methods[method]["ranking_stability"],
                "metrics": methods[method]["metrics"],
            }
            for method in ("bm25", "dense")
        },
        "provider_calls": 0,
        "paid_inference": False,
    }


def _verify(args: argparse.Namespace) -> dict[str, object]:
    if not args.live:
        raise VerificationError(
            "live inference is disabled; pass --live to make up to two requests"
        )
    if not args.confirm_account_controls:
        raise VerificationError(
            "configure OpenRouter account privacy and spending controls, then pass "
            "--confirm-account-controls"
        )
    settings = load_settings(args.config)
    load_dotenv(dotenv_path=args.env_file, override=False)
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise VerificationError(
            "OPENROUTER_API_KEY is missing; set it in .env or the process environment"
        )
    tokenizer = load_model_tokenizer(settings, args.cache_dir)

    started = perf_counter()
    retriever = None
    try:
        if args.method == "bm25":
            retriever, corpus_hash = BM25Retriever.from_path(args.passages)
        else:
            retriever = DenseRetriever(args.passages, args.qdrant_path, args.cache_dir)
            corpus_hash = retriever.corpus_sha256
        with httpx.Client(timeout=settings.request_timeout_seconds) as http_client:
            runnable, _sdk = create_live_runnable(settings, api_key, http_client=http_client)
            result = run_verification(
                claim=args.claim,
                method=args.method,
                retriever=retriever,
                corpus_sha256=corpus_hash,
                tokenizer=tokenizer,
                settings=settings,
                structured_runnable=runnable,
                started_at=started,
            )
            result["account_level_controls_confirmed_by_operator"] = True
            result["raw_prompt_response_logging"] = settings.raw_prompt_response_logging
            return result
    finally:
        if isinstance(retriever, DenseRetriever):
            retriever.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {
        "prepare": _prepare,
        "index": _index,
        "search": _search,
        "smoke": _smoke,
        "evaluate": _evaluate,
        "verify": _verify,
    }
    try:
        result = handlers[args.command](args)
    except (CorpusError, RetrievalError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 2 if result.get("generation_status") == "operational_failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
