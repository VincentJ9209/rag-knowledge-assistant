from pathlib import Path

from scripts.evaluate_search import build_keyword_index, keyword_search


DOCUMENTS = [
    {"id": "target", "course": "llm-zoomcamp", "section": "Joining", "question": "Can I join late?", "answer": "Yes."},
    {"id": "other", "course": "llm-zoomcamp", "section": "Homework", "question": "How do I submit?", "answer": "Use the form."},
]


def test_keyword_search_indexes_frozen_documents_and_returns_original_ids(tmp_path: Path) -> None:
    index = build_keyword_index(DOCUMENTS, tmp_path / "evaluation.db")
    try:
        results = keyword_search(index, "join late", 5, question_boost=2.0, section_boost=0.5)
    finally:
        index.close()
    assert results[0]["id"] == "target"
    assert 1 <= len(results) <= 5
