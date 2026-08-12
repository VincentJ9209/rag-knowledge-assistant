from __future__ import annotations

import csv

import pytest

from scripts.generate_ground_truth import (
    build_ground_truth,
    write_ground_truth,
)


def _document(document_id: str) -> dict[str, str]:
    return {
        "id": document_id,
        "course": "llm-zoomcamp",
        "section": "Course",
        "question": f"Original question for {document_id}?",
        "answer": f"Answer for {document_id}.",
    }


def _five_questions(document: dict[str, str]) -> list[str]:
    return [f"Student question {number} about {document['id']}?" for number in range(5)]


def test_build_ground_truth_generates_five_questions_and_both_splits_per_document() -> None:
    records = build_ground_truth(
        [_document("doc-a"), _document("doc-b")],
        _five_questions,
        workers=2,
        retries=0,
    )

    assert len(records) == 10
    assert {row["document"] for row in records} == {"doc-a", "doc-b"}
    for document_id in ("doc-a", "doc-b"):
        document_rows = [row for row in records if row["document"] == document_id]
        assert len(document_rows) == 5
        assert {row["split"] for row in document_rows} == {"tuning", "holdout"}


def test_build_ground_truth_retries_a_transient_generation_failure() -> None:
    attempts = 0

    def flaky_generator(document: dict[str, str]) -> list[str]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary API failure")
        return _five_questions(document)

    records = build_ground_truth(
        [_document("doc-a")],
        flaky_generator,
        workers=1,
        retries=1,
    )

    assert attempts == 2
    assert len(records) == 5


def test_build_ground_truth_rejects_a_non_five_question_response() -> None:
    def short_generator(document: dict[str, str]) -> list[str]:
        return _five_questions(document)[:-1]

    with pytest.raises(ValueError, match="doc-a generated 4 questions; expected 5"):
        build_ground_truth(
            [_document("doc-a")],
            short_generator,
            workers=1,
            retries=0,
        )


def test_write_ground_truth_creates_master_and_audit_csv(tmp_path) -> None:
    records = build_ground_truth(
        [_document("doc-a"), _document("doc-b")],
        _five_questions,
        workers=2,
        retries=0,
    )
    output = tmp_path / "ground_truth.csv"
    audit_output = tmp_path / "ground_truth_audit_sample.csv"

    write_ground_truth(records, output, audit_output, audit_size=4)

    with output.open(newline="", encoding="utf-8") as source:
        written = list(csv.DictReader(source))
    with audit_output.open(newline="", encoding="utf-8") as source:
        audit = list(csv.DictReader(source))

    assert written == records
    assert len(audit) == 4
    assert all(row in records for row in audit)
    assert audit == sorted(audit, key=lambda row: (row["document"], row["question"]))
