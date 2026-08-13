"""Freeze a canonical evaluation corpus from an existing SQLiteSearch DB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
from typing import Any

from notebooks.evaluation_utils import canonical_corpus_bytes, corpus_sha256


REQUIRED_FIELDS = {"id", "course", "section", "question", "answer"}


def _read_documents(database: Path) -> list[dict[str, Any]]:
    if not database.is_file():
        raise FileNotFoundError(f"SQLite database not found: {database}")

    database_uri = f"file:{database.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)
    try:
        rows = connection.execute("select doc_json from docs order by id").fetchall()
    finally:
        connection.close()

    documents: list[dict[str, Any]] = []
    for (raw_document,) in rows:
        document = json.loads(raw_document)
        missing_fields = REQUIRED_FIELDS - document.keys()
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"corpus document is missing fields: {missing}")
        if document["course"] != "llm-zoomcamp":
            raise ValueError(f"unexpected course: {document['course']}")
        documents.append({field: document[field] for field in sorted(REQUIRED_FIELDS)})

    if not documents:
        raise ValueError("SQLite database contains no corpus documents")
    return documents


def freeze_corpus(
    database: Path,
    output_dir: Path,
    *,
    snapshot_date: str,
    corpus_source: str = "local_sqlitesearch:notebooks/faq.db",
) -> dict[str, Path]:
    """Export the SQLite corpus and its reproducibility metadata."""
    documents = _read_documents(database)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = output_dir / f"faq_llm_zoomcamp_{len(documents)}.json"
    metadata_path = output_dir / "evaluation_metadata.json"

    corpus_path.write_bytes(canonical_corpus_bytes(documents))
    metadata = {
        "corpus_document_count": len(documents),
        "corpus_sha256": corpus_sha256(documents),
        "corpus_source": corpus_source,
        "snapshot_date": snapshot_date,
        "generation_model": "gpt-5.4-mini",
        "generation_prompt_version": "v1",
        "questions_per_document": 5,
        "split_strategy": "stratified_by_document_question_family",
        "tuning_ratio_target": 0.70,
        "holdout_ratio_target": 0.30,
        "random_seed": 20260813,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return {"corpus_path": corpus_path, "metadata_path": metadata_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snapshot-date", required=True)
    args = parser.parse_args()

    result = freeze_corpus(
        args.database,
        args.output_dir,
        snapshot_date=args.snapshot_date,
    )
    documents = json.loads(result["corpus_path"].read_text(encoding="utf-8"))
    print(f"corpus_path={result['corpus_path']}")
    print(f"corpus_document_count={len(documents)}")
    print(f"corpus_sha256={corpus_sha256(documents)}")


if __name__ == "__main__":
    main()
