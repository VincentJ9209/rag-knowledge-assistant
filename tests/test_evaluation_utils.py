from __future__ import annotations

import hashlib
import json

import pytest

from notebooks.evaluation_utils import (
    canonical_corpus_bytes,
    corpus_sha256,
    deterministic_split,
    evaluate,
    hit_rate_at_k,
    normalize_question,
    reciprocal_rank,
    relevance_list,
    validate_ground_truth,
)


def _questions(document_id: str) -> list[dict[str, str]]:
    return [
        {"question": f"Question {number} for {document_id}?", "document": document_id}
        for number in range(1, 6)
    ]


def test_canonical_corpus_hash_is_independent_of_document_and_key_order() -> None:
    first = [
        {"id": "b", "question": "Second", "answer": "B"},
        {"answer": "A", "question": "First", "id": "a"},
    ]
    second = [
        {"question": "First", "id": "a", "answer": "A"},
        {"question": "Second", "answer": "B", "id": "b"},
    ]

    expected_bytes = (
        '[{"answer":"A","id":"a","question":"First"},'
        '{"answer":"B","id":"b","question":"Second"}]\n'
    ).encode()

    assert canonical_corpus_bytes(first) == expected_bytes
    assert corpus_sha256(first) == hashlib.sha256(expected_bytes).hexdigest()
    assert corpus_sha256(first) == corpus_sha256(second)


@pytest.mark.parametrize(
    ("relevance", "k", "expected_hit_rate", "expected_mrr"),
    [
        ([False, True, False], 3, 1.0, 0.5),
        ([False, False, True], 2, 0.0, 0.0),
        ([], 5, 0.0, 0.0),
    ],
)
def test_hit_rate_and_mrr_use_only_the_requested_cutoff(
    relevance: list[bool],
    k: int,
    expected_hit_rate: float,
    expected_mrr: float,
) -> None:
    assert hit_rate_at_k(relevance, k) == expected_hit_rate
    assert reciprocal_rank(relevance, k) == expected_mrr


def test_relevance_list_matches_result_ids() -> None:
    results = [{"id": "other"}, {"id": "target"}, {"id": "later"}]

    assert relevance_list(results, "target") == [False, True, False]


def test_validate_ground_truth_rejects_unknown_document_id() -> None:
    records = [{"question": "Can I join late?", "document": "missing"}]

    with pytest.raises(ValueError, match="unknown document id: missing"):
        validate_ground_truth(records, {"known"}, questions_per_document=1)


def test_validate_ground_truth_rejects_normalized_duplicate_questions() -> None:
    records = [
        {"question": "  Can   I JOIN late? ", "document": "doc-1"},
        {"question": "can i join LATE?", "document": "doc-2"},
    ]

    with pytest.raises(ValueError, match="duplicate question: can i join late"):
        validate_ground_truth(records, {"doc-1", "doc-2"}, questions_per_document=1)


def test_validate_ground_truth_rejects_blank_question() -> None:
    with pytest.raises(ValueError, match="blank question"):
        validate_ground_truth(
            [{"question": " \n ", "document": "doc-1"}],
            {"doc-1"},
            questions_per_document=1,
        )


def test_validate_ground_truth_requires_expected_count_per_document() -> None:
    with pytest.raises(ValueError, match="doc-1 has 4 questions; expected 5"):
        validate_ground_truth(_questions("doc-1")[:-1], {"doc-1"})


def test_normalize_question_collapses_whitespace_and_ignores_case_and_punctuation() -> None:
    assert normalize_question("  What's\nNEW?  ") == "what s new"


def test_deterministic_split_matches_the_139_document_target() -> None:
    records = [record for number in range(139) for record in _questions(f"doc-{number:03d}")]

    first = deterministic_split(records, seed=20260813)
    second = deterministic_split(list(reversed(records)), seed=20260813)

    assert first == second
    assert sum(row["split"] == "tuning" for row in first) == 486
    assert sum(row["split"] == "holdout" for row in first) == 209

    splits_by_document = {
        document_id: {
            row["split"] for row in first if row["document"] == document_id
        }
        for document_id in {row["document"] for row in first}
    }
    assert all(splits == {"tuning", "holdout"} for splits in splits_by_document.values())


def test_evaluate_averages_hit_rate_and_reciprocal_rank() -> None:
    records = [
        {"question": "first", "document": "doc-a"},
        {"question": "second", "document": "doc-b"},
    ]

    def search_fn(question: str, k: int) -> list[dict[str, str]]:
        assert k == 5
        if question == "first":
            return [{"id": "doc-a"}, {"id": "other"}]
        return [{"id": "other"}, {"id": "doc-b"}]

    assert evaluate(search_fn, records) == {
        "hit_rate_at_5": 1.0,
        "mrr": 0.75,
        "question_count": 2,
    }


def test_canonical_corpus_bytes_rejects_duplicate_document_ids() -> None:
    duplicate = [{"id": "same"}, {"id": "same"}]

    with pytest.raises(ValueError, match="duplicate corpus document id: same"):
        canonical_corpus_bytes(duplicate)
