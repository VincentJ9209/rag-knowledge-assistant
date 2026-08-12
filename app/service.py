"""Application service that adapts retrieval backends to the existing RAGBase."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from sqlitesearch import TextSearchIndex

from notebooks.rag_helper import RAGBase
from notebooks.vector_search import OpenAIEmbedder, VectorSearchIndex


class CredentialError(RuntimeError):
    pass


@dataclass
class AskResult:
    answer: str
    retrieval_backend: Literal["keyword", "vector"]
    sources: list[dict[str, str]]


class _VectorAdapter:
    def __init__(self, index: VectorSearchIndex):
        self.index = index

    def search(self, *, query, num_results=5, **_ignored):
        return self.index.search(query, k=num_results)


class RAGService:
    def __init__(
        self,
        corpus_path: Path = Path("data/evaluation/faq_llm_zoomcamp_139.json"),
        database_path: Path = Path(".runtime/faq.db"),
    ):
        self.corpus_path = corpus_path
        self.database_path = database_path
        self._documents = None
        self._keyword_index = None
        self._vector_index = None

    @property
    def documents(self):
        if self._documents is None:
            self._documents = json.loads(self.corpus_path.read_text(encoding="utf-8"))
        return self._documents

    def _get_keyword_index(self):
        if self._keyword_index is None:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            index = TextSearchIndex(
                text_fields=["question", "section", "answer"],
                keyword_fields=["course"],
                db_path=str(self.database_path),
            )
            if index.count() == 0:
                for document in self.documents:
                    index.add(document)
            self._keyword_index = index
        return self._keyword_index

    def _get_vector_index(self, client):
        if self._vector_index is None:
            self._vector_index = VectorSearchIndex(
                self.documents,
                OpenAIEmbedder(client, model="text-embedding-3-small"),
            )
        return self._vector_index

    def ask(self, question: str, retrieval_backend: Literal["keyword", "vector"]) -> AskResult:
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise CredentialError("OPENAI_API_KEY is required")
        client = OpenAI()
        if retrieval_backend == "keyword":
            index = self._get_keyword_index()
        else:
            index = _VectorAdapter(self._get_vector_index(client))
        rag = RAGBase(index=index, llm_client=client)
        sources = rag.search(question)
        answer = rag.llm(rag.build_prompt(question, sources))
        compact_sources = [
            {
                "document_id": str(document["id"]),
                "section": str(document["section"]),
                "question": str(document["question"]),
            }
            for document in sources
        ]
        return AskResult(answer, retrieval_backend, compact_sources)
