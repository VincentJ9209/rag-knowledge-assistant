"""Generate and validate deterministic RAG retrieval ground truth."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import random
import time
from typing import Any

from notebooks.evaluation_utils import deterministic_split, validate_ground_truth


Document = dict[str, Any]
QuestionGenerator = Callable[[Document], list[str]]


def _generate_with_retries(
    document: Document,
    question_generator: QuestionGenerator,
    retries: int,
    retry_delay_seconds: float,
) -> list[str]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            questions = question_generator(document)
            if len(questions) != 5:
                raise ValueError(
                    f"{document['id']} generated {len(questions)} questions; expected 5"
                )
            return questions
        except Exception as error:
            last_error = error
            if attempt == retries:
                raise
            if retry_delay_seconds:
                time.sleep(retry_delay_seconds * (2**attempt))

    raise RuntimeError("unreachable retry state") from last_error


def build_ground_truth(
    documents: Iterable[Document],
    question_generator: QuestionGenerator,
    *,
    workers: int = 4,
    retries: int = 2,
    retry_delay_seconds: float = 0.0,
) -> list[dict[str, str]]:
    """Generate five questions per document and assign deterministic splits."""
    materialized = [dict(document) for document in documents]
    if not 1 <= workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    if retries < 0:
        raise ValueError("retries cannot be negative")

    document_ids = [str(document.get("id", "")) for document in materialized]
    if any(not document_id for document_id in document_ids):
        raise ValueError("every document must have a non-blank id")
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("document ids must be unique")

    generated: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _generate_with_retries,
                document,
                question_generator,
                retries,
                retry_delay_seconds,
            ): str(document["id"])
            for document in materialized
        }
        for future in as_completed(futures):
            generated[futures[future]] = future.result()

    unsplit_records = [
        {"question": question.strip(), "document": document_id}
        for document_id in sorted(generated)
        for question in generated[document_id]
    ]
    valid_document_ids = set(document_ids)
    validate_ground_truth(unsplit_records, valid_document_ids)
    return deterministic_split(unsplit_records, seed=20260813, tuning_ratio=0.70)


def _audit_sample(
    records: list[dict[str, str]], audit_size: int
) -> list[dict[str, str]]:
    if audit_size < 1:
        raise ValueError("audit_size must be at least 1")
    sample_size = min(audit_size, len(records))
    sampled = random.Random(20260813).sample(records, sample_size)
    return sorted(sampled, key=lambda row: (row["document"], row["question"]))


def _write_csv_atomic(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["question", "document", "split"])
        writer.writeheader()
        writer.writerows(records)
    temporary_path.replace(path)


def write_ground_truth(
    records: list[dict[str, str]],
    output: Path,
    audit_output: Path,
    *,
    audit_size: int = 20,
) -> None:
    """Validate and atomically write the master and human-audit sample CSVs."""
    valid_document_ids = {record["document"] for record in records}
    validate_ground_truth(records, valid_document_ids)
    _write_csv_atomic(output, records)
    _write_csv_atomic(audit_output, _audit_sample(records, audit_size))


def _openai_question_generator(model: str) -> QuestionGenerator:
    from dotenv import load_dotenv
    from openai import OpenAI
    from pydantic import BaseModel, Field

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")

    class GeneratedQuestions(BaseModel):
        questions: list[str] = Field(min_length=5, max_length=5)

    client = OpenAI()

    def generate(document: Document) -> list[str]:
        response = client.responses.parse(
            model=model,
            instructions=(
                "Generate exactly five realistic student questions answerable only from "
                "the supplied FAQ document. Use natural student wording. Do not copy the "
                "original FAQ question verbatim. Return no answers or commentary."
            ),
            input=json.dumps(document, ensure_ascii=False),
            text_format=GeneratedQuestions,
        )
        if response.output_parsed is None:
            raise RuntimeError(f"model returned no parsed questions for {document['id']}")
        return response.output_parsed.questions

    return generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    documents = json.loads(args.corpus.read_text(encoding="utf-8"))
    audit_output = args.output.with_name("ground_truth_audit_sample.csv")
    try:
        records = build_ground_truth(
            documents,
            _openai_question_generator(args.model),
            workers=args.workers,
            retries=args.retries,
            retry_delay_seconds=1.0,
        )
        write_ground_truth(records, args.output, audit_output)
    except Exception as error:
        raise SystemExit(f"ground truth generation blocked: {error}") from error

    tuning_count = sum(record["split"] == "tuning" for record in records)
    holdout_count = len(records) - tuning_count
    print(f"ground_truth_rows={len(records)}")
    print(f"tuning_rows={tuning_count}")
    print(f"holdout_rows={holdout_count}")
    print(f"audit_sample_rows={min(20, len(records))}")


if __name__ == "__main__":
    main()
