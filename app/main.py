"""FastAPI entrypoint."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.service import CredentialError, RAGService


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    retrieval_backend: Literal["keyword", "vector"] = "keyword"

    @classmethod
    def _strip_question(cls, value: str) -> str:
        return value.strip()


class Source(BaseModel):
    document_id: str
    section: str
    question: str


class AskResponse(BaseModel):
    answer: str
    retrieval_backend: Literal["keyword", "vector"]
    sources: list[Source]


def create_app(service_factory: Callable[[], object] | None = None) -> FastAPI:
    application = FastAPI(title="RAG Knowledge Assistant")
    factory = service_factory or RAGService

    @application.get("/health")
    def health():
        return {"status": "ok"}

    @application.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest):
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question must not be blank")
        try:
            result = factory().ask(question, request.retrieval_backend)
        except CredentialError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return result

    return application


app = create_app()
