import numpy as np
import pytest

from notebooks.vector_search import VectorSearchIndex, build_document_text, cosine_similarity


DOCUMENTS = [
    {"id": "timing", "course": "llm-zoomcamp", "section": "Course", "question": "Can I join late?", "answer": "Yes."},
    {"id": "homework", "course": "llm-zoomcamp", "section": "Homework", "question": "How to submit?", "answer": "Use the form."},
]


class FakeEmbedder:
    def embed_documents(self, texts):
        assert len(texts) == 2
        return np.array([[1.0, 0.0], [0.0, 1.0]])

    def embed_query(self, text):
        return np.array([0.9, 0.1])


def test_vector_search_ranks_and_preserves_original_document_fields() -> None:
    result = VectorSearchIndex(DOCUMENTS, FakeEmbedder()).search("course timing", k=1)[0]
    assert result["id"] == "timing"
    assert set(("id", "course", "section", "question", "answer")) <= result.keys()


def test_build_document_text_is_stable() -> None:
    assert build_document_text(DOCUMENTS[0]) == "Section: Course\nQuestion: Can I join late?\nAnswer: Yes."


def test_cosine_similarity_rejects_zero_vectors() -> None:
    with pytest.raises(ValueError, match="zero-length embedding"):
        cosine_similarity(np.array([0.0, 0.0]), np.array([[1.0, 0.0]]))
