from __future__ import annotations

import json
import sqlite3

import pytest

from notebooks.evaluation_utils import corpus_sha256
from scripts.freeze_evaluation_corpus import freeze_corpus


def _create_database(path, documents: list[dict[str, str]]) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "create table docs (id integer primary key, doc_json text not null, "
        "vector_hash blob, course text)"
    )
    for document in documents:
        connection.execute(
            "insert into docs (doc_json, course) values (?, ?)",
            (json.dumps(document), document["course"]),
        )
    connection.commit()
    connection.close()


def test_freeze_corpus_exports_canonical_json_and_metadata(tmp_path) -> None:
    documents = [
        {
            "id": "doc-b",
            "course": "llm-zoomcamp",
            "section": "Setup",
            "question": "Second?",
            "answer": "B",
        },
        {
            "id": "doc-a",
            "course": "llm-zoomcamp",
            "section": "Setup",
            "question": "First?",
            "answer": "A",
        },
    ]
    database = tmp_path / "faq.db"
    output_dir = tmp_path / "evaluation"
    _create_database(database, documents)

    result = freeze_corpus(database, output_dir, snapshot_date="2026-08-13")

    corpus_path = output_dir / "faq_llm_zoomcamp_2.json"
    metadata_path = output_dir / "evaluation_metadata.json"
    exported = json.loads(corpus_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert [document["id"] for document in exported] == ["doc-a", "doc-b"]
    assert result == {"corpus_path": corpus_path, "metadata_path": metadata_path}
    assert metadata == {
        "corpus_document_count": 2,
        "corpus_sha256": corpus_sha256(documents),
        "corpus_source": "local_sqlitesearch:notebooks/faq.db",
        "snapshot_date": "2026-08-13",
        "generation_model": "gpt-5.4-mini",
        "generation_prompt_version": "v1",
        "questions_per_document": 5,
        "split_strategy": "stratified_by_document_question_family",
        "tuning_ratio_target": 0.7,
        "holdout_ratio_target": 0.3,
        "random_seed": 20260813,
    }


def test_freeze_corpus_rejects_non_llm_zoomcamp_documents(tmp_path) -> None:
    database = tmp_path / "faq.db"
    _create_database(
        database,
        [
            {
                "id": "other",
                "course": "ml-zoomcamp",
                "section": "General",
                "question": "Other?",
                "answer": "No",
            }
        ],
    )

    with pytest.raises(ValueError, match="unexpected course: ml-zoomcamp"):
        freeze_corpus(database, tmp_path / "evaluation", snapshot_date="2026-08-13")
