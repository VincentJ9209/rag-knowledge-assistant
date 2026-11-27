"""FastAPI entrypoint."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.ollama import OllamaError
from app.service import (
    CredentialError, Provider, RAGService, RetrievalBackend, UnsupportedProviderBackend,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    retrieval_backend: RetrievalBackend = "keyword"
    provider: Provider = "openai"

    @classmethod
    def _strip_question(cls, value: str) -> str:
        return value.strip()


class Source(BaseModel):
    document_id: str
    section: str
    question: str


class AskResponse(BaseModel):
    answer: str
    retrieval_backend: RetrievalBackend
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
            service = factory()
            if request.provider == "ollama":
                return service.ask(question, request.retrieval_backend, provider="ollama")
            return service.ask(question, request.retrieval_backend)
        except CredentialError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except UnsupportedProviderBackend as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except OllamaError as error:
            raise HTTPException(status_code=error.status_code, detail=str(error)) from error

    return application


app = create_app()
