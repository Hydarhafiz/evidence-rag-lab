"""Safe FEVER source validation and deterministic local passage construction."""

from __future__ import annotations

import hashlib
import heapq
import json
import re
import stat
import sys
import unicodedata
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import cast
from uuid import NAMESPACE_URL, uuid5

from transformers import PreTrainedTokenizerBase

from evidence_rag.models import Passage

SELECTION_SEED = "FEVER-M1-F01-v1"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MODEL_DIMENSION = 384
MODEL_MAX_TOKENS = 512
WINDOW_SENTENCES = 4
WINDOW_STRIDE = 3
DEFAULT_TARGET_PASSAGES = 5_000

SOURCE_URLS = {
    "train.jsonl": "https://fever.ai/download/fever/train.jsonl",
    "shared_task_dev.jsonl": "https://fever.ai/download/fever/shared_task_dev.jsonl",
    "wiki-pages.zip": "https://fever.ai/download/fever/wiki-pages.zip",
}
SOURCE_HASHES = {
    "train.jsonl": "eba7e8f87076753f8494718b9a857827af7bf73e76c9e4b75420207d26e588b6",
    "shared_task_dev.jsonl": "e89865bfe1b4dd054e03dd57d7241a6fde24862905f31117cf0cd719f7c78df7",
    "wiki-pages.zip": "4b06d95da6adf7fe02d2796176c670dacccb21348da89cba4c50676ab99665f2",
}
EXPECTED_CLAIM_FIELDS = {"id", "claim", "label", "evidence", "verifiable"}
EXPECTED_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
EXPECTED_ARCHIVE_MEMBERS = 224
EXPECTED_JSONL_SHARDS = 109
EXPECTED_WIKI_PAGES = 5_416_537


class CorpusError(ValueError):
    """Raised when an accepted source or its expected structure is invalid."""


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    page_title: str
    sentence_id: int


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    split: str
    claim_id: int
    label: str
    claim: str
    evidence_groups: tuple[tuple[EvidenceReference, ...], ...]

    @property
    def references(self) -> tuple[EvidenceReference, ...]:
        return tuple(reference for group in self.evidence_groups for reference in group)

    def evaluation_payload(self) -> dict[str, object]:
        return {
            "split": self.split,
            "claim_id": self.claim_id,
            "label": self.label,
            "claim": self.claim,
            "evidence_groups": [
                [asdict(reference) for reference in group] for group in self.evidence_groups
            ],
        }


@dataclass(frozen=True, slots=True)
class SentenceRecord:
    sentence_id: int
    text: str


@dataclass(frozen=True, slots=True)
class WikiLineParse:
    records: tuple[SentenceRecord, ...]
    duplicate_ids: frozenset[int]
    nonnumeric_rows: int
    empty_text_rows: int


@dataclass(frozen=True, slots=True)
class SourceInventory:
    archive_members: int
    jsonl_shards: int
    wiki_pages: int
    missing_page_id_rows: int
    duplicate_sentence_ids: dict[str, frozenset[int]]
    found_references: frozenset[tuple[str, int]]
    nonnumeric_rows: int
    empty_text_rows: int

    @property
    def duplicate_page_ids(self) -> frozenset[str]:
        return frozenset(self.duplicate_sentence_ids)


@dataclass(frozen=True, slots=True)
class PassageBuildStats:
    base_windows: int
    token_split_windows: int
    split_sentences: int
    sentence_fragments: int
    maximum_token_length: int


@dataclass(frozen=True, slots=True)
class PreparationResult:
    manifest: dict[str, object]
    elapsed_seconds: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_hashes(
    train_path: Path,
    dev_path: Path,
    wiki_zip_path: Path,
) -> dict[str, str]:
    paths = {
        "train.jsonl": train_path,
        "shared_task_dev.jsonl": dev_path,
        "wiki-pages.zip": wiki_zip_path,
    }
    observed: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise CorpusError(f"required official source is missing: {path}")
        observed[name] = sha256_file(path)
        if observed[name] != SOURCE_HASHES[name]:
            raise CorpusError(
                f"source hash mismatch for {name}: expected {SOURCE_HASHES[name]}, "
                f"observed {observed[name]}"
            )
    return observed


def normalize_claim_text(claim: str) -> str:
    normalized = unicodedata.normalize("NFKC", claim).casefold()
    return " ".join(normalized.split())


def _json_object(raw: bytes, source_name: str, line_number: int) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise CorpusError(
            f"invalid JSON in {source_name} at line {line_number}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise CorpusError(f"expected a JSON object in {source_name} at line {line_number}")
    return cast(dict[str, object], value)


def _parse_claim_row(
    value: dict[str, object],
    split: str,
    line_number: int,
) -> ClaimRecord:
    if set(value) != EXPECTED_CLAIM_FIELDS:
        raise CorpusError(f"unexpected claim fields in {split} line {line_number}: {sorted(value)}")

    claim_id = value["id"]
    label = value["label"]
    claim = value["claim"]
    verifiable = value["verifiable"]
    evidence = value["evidence"]
    if isinstance(claim_id, bool) or not isinstance(claim_id, int):
        raise CorpusError(f"claim id must be an integer in {split} line {line_number}")
    if not isinstance(label, str) or label not in EXPECTED_LABELS:
        raise CorpusError(f"unknown FEVER label in {split} line {line_number}")
    if not isinstance(claim, str) or not claim.strip():
        raise CorpusError(f"claim text is empty or malformed in {split} line {line_number}")
    if not isinstance(verifiable, str) or verifiable not in {
        "VERIFIABLE",
        "NOT VERIFIABLE",
    }:
        raise CorpusError(f"invalid verifiable field in {split} line {line_number}")
    if (label == "NOT ENOUGH INFO") != (verifiable == "NOT VERIFIABLE"):
        raise CorpusError(f"label and verifiable field disagree in {split} line {line_number}")
    if not isinstance(evidence, list):
        raise CorpusError(f"evidence must be a list in {split} line {line_number}")

    groups: list[tuple[EvidenceReference, ...]] = []
    for group_index, raw_group in enumerate(evidence):
        if not isinstance(raw_group, list) or not raw_group:
            raise CorpusError(
                f"evidence group {group_index} is empty or malformed in {split} line {line_number}"
            )
        references: list[EvidenceReference] = []
        for item in raw_group:
            if not isinstance(item, list) or len(item) != 4:
                raise CorpusError(f"evidence tuple is malformed in {split} line {line_number}")
            annotation_id, evidence_id, page_title, sentence_id = item
            annotation_invalid = isinstance(annotation_id, bool) or not isinstance(
                annotation_id, int
            )
            evidence_invalid = evidence_id is not None and (
                isinstance(evidence_id, bool) or not isinstance(evidence_id, int)
            )
            if (
                annotation_invalid
                or evidence_invalid
                or (label != "NOT ENOUGH INFO" and evidence_id is None)
            ):
                raise CorpusError(
                    f"annotation identifiers are malformed in {split} line {line_number}"
                )
            if label == "NOT ENOUGH INFO":
                if page_title is not None or sentence_id is not None:
                    raise CorpusError(
                        "NEI evidence tuple contains a source reference "
                        f"in {split} line {line_number}"
                    )
                continue
            if not isinstance(page_title, str) or not page_title:
                raise CorpusError(
                    f"verifiable evidence has no exact page title in {split} line {line_number}"
                )
            if isinstance(sentence_id, bool) or not isinstance(sentence_id, int) or sentence_id < 0:
                raise CorpusError(
                    f"evidence sentence id is malformed in {split} line {line_number}"
                )
            references.append(EvidenceReference(page_title, sentence_id))
        groups.append(tuple(references))

    if label != "NOT ENOUGH INFO" and (not groups or any(not group for group in groups)):
        raise CorpusError(
            f"verifiable claim has an empty evidence group in {split} line {line_number}"
        )

    return ClaimRecord(
        split=split,
        claim_id=claim_id,
        label=label,
        claim=claim,
        evidence_groups=tuple(groups),
    )


def load_claims(path: Path, split: str) -> tuple[ClaimRecord, ...]:
    claims: list[ClaimRecord] = []
    seen_ids: set[int] = set()
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                raise CorpusError(f"blank JSONL row in {path.name} at line {line_number}")
            claim = _parse_claim_row(_json_object(raw, path.name, line_number), split, line_number)
            if claim.claim_id in seen_ids:
                raise CorpusError(f"duplicate claim id {claim.claim_id} in {path.name}")
            seen_ids.add(claim.claim_id)
            claims.append(claim)
    return tuple(claims)


def parse_wiki_lines(lines: str) -> WikiLineParse:
    found: set[int] = set()
    duplicate_ids: set[int] = set()
    rows: list[SentenceRecord] = []
    nonnumeric_rows = 0
    empty_text_rows = 0

    for row in lines.splitlines():
        fields = row.split("\t")
        raw_sentence_id = fields[0].strip() if fields else ""
        if not re.fullmatch(r"[0-9]+", raw_sentence_id):
            nonnumeric_rows += 1
            continue
        sentence_id = int(raw_sentence_id)
        if sentence_id in found:
            duplicate_ids.add(sentence_id)
        found.add(sentence_id)
        text = fields[1].strip() if len(fields) > 1 else ""
        if not text:
            empty_text_rows += 1
            continue
        rows.append(SentenceRecord(sentence_id, text))

    records = tuple(
        sorted(
            (record for record in rows if record.sentence_id not in duplicate_ids),
            key=lambda record: record.sentence_id,
        )
    )
    return WikiLineParse(
        records=records,
        duplicate_ids=frozenset(duplicate_ids),
        nonnumeric_rows=nonnumeric_rows,
        empty_text_rows=empty_text_rows,
    )


def _validate_zip_paths(archive: zipfile.ZipFile) -> tuple[zipfile.ZipInfo, ...]:
    infos = tuple(archive.infolist())
    if len(infos) != EXPECTED_ARCHIVE_MEMBERS:
        raise CorpusError(
            f"unexpected archive member count: expected {EXPECTED_ARCHIVE_MEMBERS}, "
            f"observed {len(infos)}"
        )
    names: set[str] = set()
    jsonl_infos: list[zipfile.ZipInfo] = []
    for info in infos:
        if info.filename in names:
            raise CorpusError(f"duplicate archive member name: {info.filename}")
        names.add(info.filename)
        member_path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if (
            member_path.is_absolute()
            or ".." in member_path.parts
            or "\\" in info.filename
            or (member_path.parts and ":" in member_path.parts[0])
            or stat.S_ISLNK(mode)
        ):
            raise CorpusError(f"unsafe archive member path: {info.filename}")
        if info.flag_bits & 1:
            raise CorpusError(f"encrypted archive member is not supported: {info.filename}")
        if member_path.parts and member_path.parts[0] == "wiki-pages":
            if info.is_dir():
                continue
            if not re.fullmatch(r"wiki-[0-9]{3}\.jsonl", member_path.name):
                raise CorpusError(f"unexpected Wikipedia archive member: {info.filename}")
            jsonl_infos.append(info)
        elif info.filename == "license.html" or (
            member_path.parts and member_path.parts[0] == "__MACOSX"
        ):
            continue
        elif not info.is_dir():
            raise CorpusError(f"unexpected FEVER archive member: {info.filename}")

    if len(jsonl_infos) != EXPECTED_JSONL_SHARDS:
        raise CorpusError(
            f"unexpected JSONL shard count: expected {EXPECTED_JSONL_SHARDS}, "
            f"observed {len(jsonl_infos)}"
        )
    return tuple(jsonl_infos)


def _parse_wiki_page(
    value: dict[str, object],
    member_name: str,
    line_number: int,
) -> tuple[str, str]:
    page_id = value.get("id")
    text = value.get("text")
    lines = value.get("lines")
    if not isinstance(page_id, str):
        raise CorpusError(f"malformed page id in {member_name} line {line_number}")
    if not isinstance(text, str) or not isinstance(lines, str):
        raise CorpusError(f"malformed page fields in {member_name} line {line_number}")
    return page_id, lines


def inventory_wikipedia_source(
    wiki_zip_path: Path,
    required_references: set[tuple[str, int]],
) -> SourceInventory:
    found_references: set[tuple[str, int]] = set()
    duplicate_sentence_ids: dict[str, frozenset[int]] = {}
    seen_page_ids: set[str] = set()
    nonnumeric_rows = 0
    empty_text_rows = 0
    page_count = 0
    missing_page_id_rows = 0

    try:
        archive = zipfile.ZipFile(wiki_zip_path)
    except (OSError, zipfile.BadZipFile) as error:
        raise CorpusError(f"cannot read FEVER Wikipedia archive: {error}") from error

    with archive:
        jsonl_infos = _validate_zip_paths(archive)
        required_by_page: dict[str, set[int]] = defaultdict(set)
        for page_title, sentence_id in required_references:
            required_by_page[page_title].add(sentence_id)

        for shard_index, info in enumerate(jsonl_infos, start=1):
            if shard_index == 1 or shard_index % 10 == 0 or shard_index == len(jsonl_infos):
                print(
                    f"FEVER source validation: {shard_index}/{len(jsonl_infos)} shards",
                    file=sys.stderr,
                    flush=True,
                )
            with archive.open(info, "r") as shard:
                for line_number, raw in enumerate(shard, start=1):
                    if not raw.strip():
                        raise CorpusError(
                            f"blank source row in {info.filename} at line {line_number}"
                        )
                    page_id, lines = _parse_wiki_page(
                        _json_object(raw, info.filename, line_number),
                        info.filename,
                        line_number,
                    )
                    page_count += 1
                    if not page_id:
                        missing_page_id_rows += 1
                        continue
                    if page_id in seen_page_ids:
                        raise CorpusError(f"duplicate Wikipedia page id in source: {page_id!r}")
                    seen_page_ids.add(page_id)

                    parsed = parse_wiki_lines(lines)
                    nonnumeric_rows += parsed.nonnumeric_rows
                    empty_text_rows += parsed.empty_text_rows
                    if parsed.duplicate_ids:
                        duplicate_sentence_ids[page_id] = parsed.duplicate_ids
                    required_ids = required_by_page.get(page_id, set())
                    found_references.update(
                        (page_id, record.sentence_id)
                        for record in parsed.records
                        if record.sentence_id in required_ids
                    )

    if page_count != EXPECTED_WIKI_PAGES:
        raise CorpusError(
            "unexpected Wikipedia page count: "
            f"expected {EXPECTED_WIKI_PAGES}, observed {page_count}"
        )
    if len(duplicate_sentence_ids) != 1:
        raise CorpusError(
            "accepted source contract expects one page with duplicate sentence identifiers; "
            f"observed {len(duplicate_sentence_ids)}"
        )

    return SourceInventory(
        archive_members=EXPECTED_ARCHIVE_MEMBERS,
        jsonl_shards=len(jsonl_infos),
        wiki_pages=page_count,
        missing_page_id_rows=missing_page_id_rows,
        duplicate_sentence_ids=duplicate_sentence_ids,
        found_references=frozenset(found_references),
        nonnumeric_rows=nonnumeric_rows,
        empty_text_rows=empty_text_rows,
    )


def _normalized_overlaps(
    train_claims: tuple[ClaimRecord, ...],
    dev_claims: tuple[ClaimRecord, ...],
) -> frozenset[str]:
    train_texts = {normalize_claim_text(claim.claim) for claim in train_claims}
    dev_texts = {normalize_claim_text(claim.claim) for claim in dev_claims}
    overlaps = frozenset(train_texts & dev_texts)
    if len(overlaps) != 3:
        raise CorpusError(
            "accepted FEVER contract expects three normalized train/dev overlaps, "
            f"observed {len(overlaps)}"
        )
    return overlaps


def select_evaluation_claims(
    train_claims: tuple[ClaimRecord, ...],
    dev_claims: tuple[ClaimRecord, ...],
    inventory: SourceInventory,
) -> tuple[ClaimRecord, ...]:
    excluded_texts = _normalized_overlaps(train_claims, dev_claims)
    duplicate_refs = {
        (page_title, sentence_id)
        for page_title, sentence_ids in inventory.duplicate_sentence_ids.items()
        for sentence_id in sentence_ids
    }
    selected: list[ClaimRecord] = []
    requested_counts = {"train": 10, "dev": 20}

    for split, claims in (("train", train_claims), ("dev", dev_claims)):
        for label in sorted(EXPECTED_LABELS):
            eligible = [
                claim
                for claim in claims
                if claim.label == label
                and normalize_claim_text(claim.claim) not in excluded_texts
                and all(
                    (reference.page_title, reference.sentence_id) in inventory.found_references
                    and (reference.page_title, reference.sentence_id) not in duplicate_refs
                    for reference in claim.references
                )
            ]
            eligible.sort(
                key=lambda claim: (
                    hashlib.sha256(
                        f"{SELECTION_SEED}|{split}|{claim.claim_id}".encode()
                    ).hexdigest(),
                    claim.claim_id,
                )
            )
            requested = requested_counts[split]
            if len(eligible) < requested:
                raise CorpusError(
                    f"not enough source-covered {split} {label} claims: "
                    f"need {requested}, found {len(eligible)}"
                )
            selected.extend(eligible[:requested])

    selected.sort(key=lambda claim: (claim.split, claim.label, claim.claim_id))
    return tuple(selected)


def token_length(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    input_ids = encoded["input_ids"]
    if not isinstance(input_ids, list):
        raise CorpusError("tokenizer returned an unexpected input_ids shape")
    return len(input_ids)


def _split_sentence(
    page_title: str,
    record: SentenceRecord,
    tokenizer: PreTrainedTokenizerBase,
    maximum_tokens: int,
) -> tuple[SentenceRecord, ...]:
    encoded = tokenizer(
        record.text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    token_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if not isinstance(token_ids, list) or not isinstance(offsets, list):
        raise CorpusError("tokenizer returned an unexpected sentence encoding")
    if not token_ids:
        if token_length(tokenizer, f"{page_title}\n{record.text}") <= maximum_tokens:
            return (record,)
        raise CorpusError(
            f"sentence {record.sentence_id} cannot be represented within the token limit"
        )
    prefix_tokens = tokenizer(page_title, add_special_tokens=True, truncation=False)["input_ids"]
    budget = maximum_tokens - len(prefix_tokens)
    if budget < 1:
        raise CorpusError(f"page title exceeds the encoder token limit: {page_title!r}")

    fragments: list[SentenceRecord] = []
    token_start = 0
    character_start = 0
    while token_start < len(token_ids):
        token_end = min(token_start + budget, len(token_ids))
        while token_end > token_start:
            character_end = offsets[token_end - 1][1]
            fragment_text = record.text[character_start:character_end]
            if token_length(tokenizer, f"{page_title}\n{fragment_text}") <= maximum_tokens:
                break
            token_end -= 1
        if token_end == token_start:
            raise CorpusError(
                f"tokenizer could not split sentence {record.sentence_id} under the limit"
            )
        character_end = offsets[token_end - 1][1]
        fragments.append(
            SentenceRecord(record.sentence_id, record.text[character_start:character_end])
        )
        token_start = token_end
        character_start = character_end

    if character_start < len(record.text):
        last = fragments[-1]
        last_text = last.text + record.text[character_start:]
        if token_length(tokenizer, f"{page_title}\n{last_text}") > maximum_tokens:
            raise CorpusError(
                f"sentence tail exceeds the encoder token limit: {record.sentence_id}"
            )
        fragments[-1] = SentenceRecord(record.sentence_id, last_text)
    return tuple(fragments)


def _passage_id(page_title: str, sentence_ids: tuple[int, ...], text: str) -> str:
    identity = json.dumps(
        [page_title, sentence_ids, text],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return str(uuid5(NAMESPACE_URL, identity))


def _make_passage(
    page_title: str,
    records: tuple[SentenceRecord, ...],
    source_sha256: str,
    tokenizer: PreTrainedTokenizerBase,
    maximum_tokens: int,
) -> Passage:
    text = f"{page_title}\n" + "\n".join(record.text for record in records)
    if token_length(tokenizer, text) > maximum_tokens:
        raise CorpusError(f"passage exceeds the configured token limit: {page_title!r}")
    sentence_ids = tuple(dict.fromkeys(record.sentence_id for record in records))
    return Passage(
        passage_id=_passage_id(page_title, sentence_ids, text),
        page_title=page_title,
        sentence_ids=sentence_ids,
        text=text,
        source_artifact="FEVER June 2017 wiki-pages.zip",
        source_sha256=source_sha256,
    )


def passages_for_window(
    page_title: str,
    records: tuple[SentenceRecord, ...],
    source_sha256: str,
    tokenizer: PreTrainedTokenizerBase,
    maximum_tokens: int,
) -> tuple[tuple[Passage, ...], int, int, int]:
    fragments_by_sentence = tuple(
        _split_sentence(page_title, record, tokenizer, maximum_tokens) for record in records
    )
    flattened = tuple(fragment for fragments in fragments_by_sentence for fragment in fragments)
    passages: list[Passage] = []
    current: list[SentenceRecord] = []
    for fragment in flattened:
        candidate = tuple(current + [fragment])
        candidate_text = f"{page_title}\n" + "\n".join(record.text for record in candidate)
        if current and token_length(tokenizer, candidate_text) > maximum_tokens:
            passages.append(
                _make_passage(
                    page_title,
                    tuple(current),
                    source_sha256,
                    tokenizer,
                    maximum_tokens,
                )
            )
            current = [fragment]
        else:
            current.append(fragment)
    if current:
        passages.append(
            _make_passage(
                page_title,
                tuple(current),
                source_sha256,
                tokenizer,
                maximum_tokens,
            )
        )

    split_sentences = sum(len(fragments) > 1 for fragments in fragments_by_sentence)
    sentence_fragments = sum(len(fragments) for fragments in fragments_by_sentence)
    maximum_length = max(token_length(tokenizer, passage.text) for passage in passages)
    return tuple(passages), split_sentences, sentence_fragments, maximum_length


def _page_windows(
    page_title: str,
    records: tuple[SentenceRecord, ...],
    source_sha256: str,
    tokenizer: PreTrainedTokenizerBase,
    maximum_tokens: int,
) -> tuple[tuple[Passage, ...], PassageBuildStats]:
    passages: list[Passage] = []
    token_split_windows = 0
    split_sentences = 0
    sentence_fragments = 0
    maximum_length = 0
    base_windows = 0
    for start in range(0, len(records), WINDOW_STRIDE):
        window = records[start : start + WINDOW_SENTENCES]
        if not window:
            continue
        base_windows += 1
        window_passages, split_count, fragment_count, window_maximum = passages_for_window(
            page_title,
            window,
            source_sha256,
            tokenizer,
            maximum_tokens,
        )
        if len(window_passages) > 1:
            token_split_windows += 1
        split_sentences += split_count
        sentence_fragments += fragment_count
        maximum_length = max(maximum_length, window_maximum)
        passages.extend(window_passages)
    return tuple(passages), PassageBuildStats(
        base_windows=base_windows,
        token_split_windows=token_split_windows,
        split_sentences=split_sentences,
        sentence_fragments=sentence_fragments,
        maximum_token_length=maximum_length,
    )


def _background_rank(page_title: str, first_sentence_id: int) -> str:
    value = f"{SELECTION_SEED}|background|{page_title}|{first_sentence_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _push_background_window(
    heap: list[tuple[int, bytes, int, str, int, tuple[SentenceRecord, ...]]],
    page_title: str,
    records: tuple[SentenceRecord, ...],
    capacity: int,
) -> None:
    first_sentence_id = records[0].sentence_id
    rank = _background_rank(page_title, first_sentence_id)
    rank_integer = int(rank, 16)
    page_bytes = page_title.encode("utf-8")
    reverse_page_bytes = bytes(255 - byte for byte in page_bytes)
    heap_key = (-rank_integer, reverse_page_bytes, -first_sentence_id)
    item = (*heap_key, page_title, first_sentence_id, records)
    if len(heap) < capacity:
        heapq.heappush(heap, item)
    elif heap_key > heap[0][:3]:
        heapq.heapreplace(heap, item)


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")
    temporary.replace(path)


def _build_passages_and_evaluation(
    wiki_zip_path: Path,
    selected_claims: tuple[ClaimRecord, ...],
    inventory: SourceInventory,
    tokenizer: PreTrainedTokenizerBase,
    maximum_tokens: int,
    target_passages: int,
    source_sha256: str,
) -> tuple[tuple[Passage, ...], dict[str, int], dict[str, int]]:
    evidence_pages = {
        reference.page_title for claim in selected_claims for reference in claim.references
    }
    if len(evidence_pages) > target_passages:
        raise CorpusError("selected evidence pages exceed the mini-corpus passage target")

    evidence_passages: list[Passage] = []
    evidence_window_count = 0
    split_windows = 0
    split_sentences = 0
    sentence_fragments = 0
    maximum_token_length = 0
    seen_page_ids: set[str] = set()
    background_heap: list[tuple[int, bytes, int, str, int, tuple[SentenceRecord, ...]]] = []
    candidate_window_count = 0
    nonnumeric_rows = 0
    empty_text_rows = 0
    source_record_count = 0

    with zipfile.ZipFile(wiki_zip_path) as archive:
        jsonl_infos = _validate_zip_paths(archive)
        for shard_index, info in enumerate(jsonl_infos, start=1):
            if shard_index == 1 or shard_index % 10 == 0 or shard_index == len(jsonl_infos):
                print(
                    f"FEVER passage build: {shard_index}/{len(jsonl_infos)} shards",
                    file=sys.stderr,
                    flush=True,
                )
            with archive.open(info, "r") as shard:
                for line_number, raw in enumerate(shard, start=1):
                    page_id, lines = _parse_wiki_page(
                        _json_object(raw, info.filename, line_number),
                        info.filename,
                        line_number,
                    )
                    source_record_count += 1
                    if not page_id:
                        continue
                    if page_id in seen_page_ids:
                        raise CorpusError(f"duplicate Wikipedia page id in source: {page_id!r}")
                    seen_page_ids.add(page_id)
                    parsed = parse_wiki_lines(lines)
                    nonnumeric_rows += parsed.nonnumeric_rows
                    empty_text_rows += parsed.empty_text_rows

                    if page_id in evidence_pages:
                        page_passages, page_stats = _page_windows(
                            page_id,
                            parsed.records,
                            source_sha256,
                            tokenizer,
                            maximum_tokens,
                        )
                        evidence_passages.extend(page_passages)
                        evidence_window_count += page_stats.base_windows
                        split_windows += page_stats.token_split_windows
                        split_sentences += page_stats.split_sentences
                        sentence_fragments += page_stats.sentence_fragments
                        maximum_token_length = max(
                            maximum_token_length, page_stats.maximum_token_length
                        )
                    elif page_id not in inventory.duplicate_page_ids:
                        for start in range(0, len(parsed.records), WINDOW_STRIDE):
                            window = parsed.records[start : start + WINDOW_SENTENCES]
                            if not window:
                                continue
                            candidate_window_count += 1
                            _push_background_window(
                                background_heap, page_id, window, target_passages
                            )

    if source_record_count != inventory.wiki_pages:
        raise CorpusError(
            "Wikipedia page count changed between source validation and passage build"
        )
    if len(seen_page_ids) != inventory.wiki_pages - inventory.missing_page_id_rows:
        raise CorpusError("unique Wikipedia page identifier count changed during corpus build")
    missing_evidence_pages = evidence_pages - seen_page_ids
    if missing_evidence_pages:
        raise CorpusError(
            f"selected evidence pages missing from source: {len(missing_evidence_pages)}"
        )

    background_needed = target_passages - len(evidence_passages)
    if background_needed < 0:
        raise CorpusError("evidence passages exceed the requested target")
    background_candidates = sorted(
        (
            (
                _background_rank(page_title, first_sentence_id),
                page_title,
                first_sentence_id,
                records,
            )
            for _, _, _, page_title, first_sentence_id, records in background_heap
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    background_passages: list[Passage] = []
    background_split_windows = 0
    background_split_sentences = 0
    background_sentence_fragments = 0
    for _, page_title, _, records in background_candidates:
        if len(background_passages) >= background_needed:
            break
        window_passages, split_count, fragment_count, window_maximum = passages_for_window(
            page_title,
            records,
            source_sha256,
            tokenizer,
            maximum_tokens,
        )
        if len(window_passages) > 1:
            background_split_windows += 1
        background_split_sentences += split_count
        background_sentence_fragments += fragment_count
        maximum_token_length = max(maximum_token_length, window_maximum)
        remaining = background_needed - len(background_passages)
        background_passages.extend(window_passages[:remaining])

    passages = tuple(evidence_passages + background_passages)
    if len(passages) != target_passages:
        raise CorpusError(
            f"mini-corpus target mismatch: requested {target_passages}, built {len(passages)}"
        )
    if len({passage.passage_id for passage in passages}) != len(passages):
        raise CorpusError("stable passage identifier collision in generated mini-corpus")

    evidence_pairs = {
        (reference.page_title, reference.sentence_id)
        for claim in selected_claims
        for reference in claim.references
    }
    corpus_pairs = {
        (passage.page_title, sentence_id)
        for passage in evidence_passages
        for sentence_id in passage.sentence_ids
    }
    uncovered_pairs = evidence_pairs - corpus_pairs
    if uncovered_pairs:
        raise CorpusError(
            f"selected evidence references are not represented: {len(uncovered_pairs)}"
        )

    passages = tuple(sorted(passages, key=lambda passage: passage.passage_id))
    selection_counts = {
        f"{split}_{label.lower().replace(' ', '_')}": sum(
            claim.split == split and claim.label == label for claim in selected_claims
        )
        for split in ("train", "dev")
        for label in sorted(EXPECTED_LABELS)
    }
    build_stats = {
        "evidence_pages": len(evidence_pages),
        "evidence_base_windows": evidence_window_count,
        "evidence_passages": len(evidence_passages),
        "background_candidates_seen": candidate_window_count,
        "background_passages": len(background_passages),
        "token_split_windows": split_windows + background_split_windows,
        "sentences_split_for_token_limit": split_sentences + background_split_sentences,
        "sentence_fragments": sentence_fragments + background_sentence_fragments,
        "maximum_token_length": maximum_token_length,
        "nonnumeric_wiki_rows": nonnumeric_rows,
        "empty_sentence_rows": empty_text_rows,
    }
    return passages, selection_counts, build_stats


def _atomic_write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_corpus(
    train_path: Path,
    dev_path: Path,
    wiki_zip_path: Path,
    output_dir: Path,
    tokenizer: PreTrainedTokenizerBase,
    maximum_tokens: int,
    target_passages: int = DEFAULT_TARGET_PASSAGES,
) -> PreparationResult:
    started = perf_counter()
    print("Checking official FEVER artifact hashes and claim schemas.", file=sys.stderr, flush=True)
    source_hashes = verify_source_hashes(train_path, dev_path, wiki_zip_path)
    train_claims = load_claims(train_path, "train")
    dev_claims = load_claims(dev_path, "dev")
    if len({claim.claim_id for claim in train_claims} & {claim.claim_id for claim in dev_claims}):
        raise CorpusError("claim ids overlap between the accepted train and development files")

    required_references = {
        (reference.page_title, reference.sentence_id)
        for claim in (*train_claims, *dev_claims)
        for reference in claim.references
    }
    print(
        "Validating Wikipedia page identifiers and evidence sentence references.",
        file=sys.stderr,
        flush=True,
    )
    inventory = inventory_wikipedia_source(wiki_zip_path, required_references)
    selected_claims = select_evaluation_claims(train_claims, dev_claims, inventory)
    tokenizer_limit = min(maximum_tokens, MODEL_MAX_TOKENS)
    if tokenizer_limit <= 0:
        raise CorpusError("encoder token limit must be positive")
    passages, selection_counts, build_stats = _build_passages_and_evaluation(
        wiki_zip_path,
        selected_claims,
        inventory,
        tokenizer,
        tokenizer_limit,
        target_passages,
        source_hashes["wiki-pages.zip"],
    )
    final_source_hashes = verify_source_hashes(train_path, dev_path, wiki_zip_path)
    if final_source_hashes != source_hashes:
        raise CorpusError("an accepted input changed while the corpus was being built")

    output_dir.mkdir(parents=True, exist_ok=True)
    passages_path = output_dir / "passages.jsonl"
    training_evaluation_path = output_dir / "evaluation_samples_train.jsonl"
    held_out_evaluation_path = output_dir / "evaluation_samples_dev.jsonl"
    manifest_path = output_dir / "corpus_manifest.json"
    _write_jsonl(passages_path, (passage.to_runtime_payload() for passage in passages))
    _write_jsonl(
        training_evaluation_path,
        (claim.evaluation_payload() for claim in selected_claims if claim.split == "train"),
    )
    _write_jsonl(
        held_out_evaluation_path,
        (claim.evaluation_payload() for claim in selected_claims if claim.split == "dev"),
    )
    obsolete_evaluation_path = output_dir / "evaluation_samples.jsonl"
    if obsolete_evaluation_path.exists():
        obsolete_evaluation_path.unlink()

    evidence_groups = [
        group for claim in selected_claims for group in claim.evidence_groups if group
    ]
    evidence_pairs = {
        (reference.page_title, reference.sentence_id)
        for claim in selected_claims
        for reference in claim.references
    }
    manifest: dict[str, object] = {
        "format_version": 1,
        "selection_seed": SELECTION_SEED,
        "sources": {
            name: {
                "url": SOURCE_URLS[name],
                "sha256": source_hashes[name],
            }
            for name in SOURCE_URLS
        },
        "source_structure": {
            "archive_members": inventory.archive_members,
            "jsonl_shards": inventory.jsonl_shards,
            "wiki_page_records": inventory.wiki_pages,
            "unique_wiki_page_ids": inventory.wiki_pages - inventory.missing_page_id_rows,
            "missing_page_id_rows": inventory.missing_page_id_rows,
            "duplicate_sentence_id_pages": len(inventory.duplicate_page_ids),
            "duplicate_sentence_ids": sum(
                len(sentence_ids) for sentence_ids in inventory.duplicate_sentence_ids.values()
            ),
            "nonnumeric_wiki_rows": inventory.nonnumeric_rows,
            "empty_sentence_rows": inventory.empty_text_rows,
        },
        "selection": {
            "train_claims": 30,
            "held_out_dev_claims": 60,
            "per_class_counts": selection_counts,
            "normalized_train_dev_text_overlaps_excluded": 3,
            "evidence_groups": len(evidence_groups),
            "unique_evidence_pages": len(
                {
                    reference.page_title
                    for claim in selected_claims
                    for reference in claim.references
                }
            ),
            "unique_page_sentence_pairs": len(evidence_pairs),
            "evidence_pairs_present_in_corpus": len(evidence_pairs),
            "coverage_fraction": 1.0,
            "gold_informed": True,
        },
        "passage_construction": {
            "target_passages": target_passages,
            "actual_passages": len(passages),
            "sentences_per_window": WINDOW_SENTENCES,
            "stride": WINDOW_STRIDE,
            "page_title_prefixed_to_text": True,
            "text_token_limit": tokenizer_limit,
            "text_truncation_count": 0,
            **build_stats,
        },
        "embedding": {
            "model": MODEL_NAME,
            "revision": MODEL_REVISION,
            "dimension": MODEL_DIMENSION,
            "tokenizer": "BAAI/bge-small-en-v1.5 tokenizer at the same immutable revision",
            "token_limit": MODEL_MAX_TOKENS,
        },
        "runtime_payload_fields": sorted(
            Passage.from_runtime_payload(passages[0].to_runtime_payload()).to_runtime_payload()
        ),
        "passages_sha256": sha256_file(passages_path),
        "note": (
            "Gold labels and evidence groups are stored in split-specific evaluation files, "
            "never in passages.jsonl."
        ),
    }
    _atomic_write_json(manifest_path, manifest)
    return PreparationResult(manifest=manifest, elapsed_seconds=perf_counter() - started)
