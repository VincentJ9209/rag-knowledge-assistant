"""Deterministic retrieval-evaluation utilities shared by scripts and notebooks."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
import json
import math
import random
import re
from typing import Any


GroundTruthRecord = Mapping[str, str]
SearchResult = Mapping[str, Any]
SearchFunction = Callable[[str, int], Sequence[SearchResult]]


def canonical_corpus_bytes(documents: Iterable[Mapping[str, Any]]) -> bytes:
    """Serialize a corpus deterministically after sorting documents by ID."""
    materialized = [dict(document) for document in documents]
    document_ids = [str(document.get("id", "")) for document in materialized]

    if any(not document_id for document_id in document_ids):
        raise ValueError("every corpus document must have a non-blank id")

    duplicate_ids = sorted(
        document_id
        for document_id, count in Counter(document_ids).items()
        if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"duplicate corpus document id: {duplicate_ids[0]}")

    ordered = sorted(materialized, key=lambda document: str(document["id"]))
    serialized = json.dumps(
        ordered,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{serialized}\n".encode("utf-8")


def corpus_sha256(documents: Iterable[Mapping[str, Any]]) -> str:
    """Return the SHA-256 digest of the canonical corpus representation."""
    return hashlib.sha256(canonical_corpus_bytes(documents)).hexdigest()


def normalize_question(question: str) -> str:
    """Normalize a question for case-insensitive duplicate detection."""
    return " ".join(re.sub(r"[^\w]+", " ", question.casefold()).split())


def validate_ground_truth(
    records: Iterable[GroundTruthRecord],
    valid_document_ids: set[str],
    questions_per_document: int = 5,
) -> None:
    """Raise ``ValueError`` when ground-truth rows violate the data contract."""
    if questions_per_document < 1:
        raise ValueError("questions_per_document must be at least 1")

    counts: Counter[str] = Counter()
    seen_questions: set[str] = set()

    for record in records:
        question = str(record.get("question", ""))
        document_id = str(record.get("document", ""))
        normalized_question = normalize_question(question)

        if not normalized_question:
            raise ValueError("blank question")
        if document_id not in valid_document_ids:
            raise ValueError(f"unknown document id: {document_id}")
        if normalized_question in seen_questions:
            raise ValueError(f"duplicate question: {normalized_question}")

        seen_questions.add(normalized_question)
        counts[document_id] += 1

    for document_id in sorted(valid_document_ids):
        actual = counts[document_id]
        if actual != questions_per_document:
            raise ValueError(
                f"{document_id} has {actual} questions; expected {questions_per_document}"
            )


def deterministic_split(
    records: Iterable[GroundTruthRecord],
    seed: int = 20260813,
    tuning_ratio: float = 0.70,
) -> list[dict[str, str]]:
    """Assign five questions per document to deterministic tuning/holdout sets."""
    if not 0.0 < tuning_ratio < 1.0:
        raise ValueError("tuning_ratio must be between 0 and 1")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        document_id = str(record.get("document", ""))
        question = str(record.get("question", ""))
        if not document_id or not normalize_question(question):
            raise ValueError("split records require non-blank question and document")
        grouped[document_id].append({"question": question, "document": document_id})

    if not grouped:
        return []
    if any(len(rows) != 5 for rows in grouped.values()):
        raise ValueError("deterministic split requires exactly 5 questions per document")

    document_ids = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(document_ids)

    total_questions = len(document_ids) * 5
    target_tuning = math.floor(total_questions * tuning_ratio)
    lower_tuning_per_document = max(1, min(4, math.floor(5 * tuning_ratio)))
    higher_tuning_per_document = lower_tuning_per_document + 1
    higher_count = target_tuning - lower_tuning_per_document * len(document_ids)

    if not 0 <= higher_count <= len(document_ids) or higher_tuning_per_document > 4:
        raise ValueError("tuning_ratio cannot preserve both splits for every document")

    split_records: list[dict[str, str]] = []
    for position, document_id in enumerate(document_ids):
        rows = sorted(
            grouped[document_id],
            key=lambda row: (normalize_question(row["question"]), row["question"]),
        )
        rng.shuffle(rows)
        tuning_count = (
            higher_tuning_per_document
            if position < higher_count
            else lower_tuning_per_document
        )
        for index, row in enumerate(rows):
            split_records.append(
                {**row, "split": "tuning" if index < tuning_count else "holdout"}
            )

    return split_records


def relevance_list(
    results: Sequence[SearchResult], expected_document_id: str
) -> list[bool]:
    """Convert ranked results to binary relevance labels."""
    return [str(result.get("id", "")) == expected_document_id for result in results]


def hit_rate_at_k(relevance: Sequence[bool], k: int = 5) -> float:
    """Return one when any relevant result occurs within the first ``k`` ranks."""
    if k < 1:
        raise ValueError("k must be at least 1")
    return float(any(relevance[:k]))


def reciprocal_rank(relevance: Sequence[bool], k: int = 5) -> float:
    """Return reciprocal rank for the first relevant result within ``k``."""
    if k < 1:
        raise ValueError("k must be at least 1")
    for rank, is_relevant in enumerate(relevance[:k], start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def evaluate(
    search_fn: SearchFunction,
    records: Iterable[GroundTruthRecord],
    k: int = 5,
) -> dict[str, float | int]:
    """Evaluate a retrieval function with Hit Rate@k and mean reciprocal rank."""
    materialized = list(records)
    if not materialized:
        raise ValueError("evaluation requires at least one ground-truth record")

    relevance = [
        relevance_list(
            search_fn(str(record["question"]), k),
            str(record["document"]),
        )
        for record in materialized
    ]

    return {
        f"hit_rate_at_{k}": sum(hit_rate_at_k(items, k) for items in relevance)
        / len(relevance),
        "mrr": sum(reciprocal_rank(items, k) for items in relevance) / len(relevance),
        "question_count": len(materialized),
    }
