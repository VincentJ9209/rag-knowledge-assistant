"""Small local vector-search baseline with injectable embeddings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def build_document_text(document: Mapping[str, Any]) -> str:
    return (
        f"Section: {document['section']}\n"
        f"Question: {document['question']}\n"
        f"Answer: {document['answer']}"
    )


def cosine_similarity(query_vector: np.ndarray, document_matrix: np.ndarray) -> np.ndarray:
    query = np.asarray(query_vector, dtype=float)
    documents = np.asarray(document_matrix, dtype=float)
    if query.ndim != 1 or documents.ndim != 2 or documents.shape[1] != query.shape[0]:
        raise ValueError("embedding dimensions do not match")
    query_norm = np.linalg.norm(query)
    document_norms = np.linalg.norm(documents, axis=1)
    if query_norm == 0 or np.any(document_norms == 0):
        raise ValueError("zero-length embedding")
    return (documents @ query) / (document_norms * query_norm)


class VectorSearchIndex:
    def __init__(self, documents: Sequence[Mapping[str, Any]], embedder: Embedder):
        self.documents = [dict(document) for document in documents]
        self.embedder = embedder
        texts = [build_document_text(document) for document in self.documents]
        self.document_embeddings = np.asarray(embedder.embed_documents(texts), dtype=float)
        if self.document_embeddings.shape[0] != len(self.documents):
            raise ValueError("embedder returned the wrong document count")

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if k < 1:
            raise ValueError("k must be at least 1")
        scores = cosine_similarity(self.embedder.embed_query(query), self.document_embeddings)
        ranked = np.argsort(-scores, kind="stable")[:k]
        return [dict(self.documents[index]) for index in ranked]


class OpenAIEmbedder:
    def __init__(self, client, model: str = "text-embedding-3-small", batch_size: int = 100):
        self.client = client
        self.model = model
        self.batch_size = batch_size

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            response = self.client.embeddings.create(
                model=self.model,
                input=list(texts[start : start + self.batch_size]),
            )
            vectors.extend(item.embedding for item in response.data)
        return np.asarray(vectors, dtype=float)

    def embed_query(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=[text])
        return np.asarray(response.data[0].embedding, dtype=float)
