from fastapi.testclient import TestClient

from app.main import create_app
from app.service import AskResult, CredentialError


class FakeService:
    def ask(self, question, retrieval_backend):
        return AskResult(
            answer="You can still join.",
            retrieval_backend=retrieval_backend,
            sources=[{"document_id": "74eb249bbf", "section": "General", "question": "Can I join?"}],
        )


class MissingCredentialService:
    def ask(self, question, retrieval_backend):
        raise CredentialError("OPENAI_API_KEY is required")


def test_health_requires_no_credentials() -> None:
    response = TestClient(create_app(lambda: FakeService())).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_uses_injected_service_and_returns_compact_sources() -> None:
    response = TestClient(create_app(lambda: FakeService())).post(
        "/ask", json={"question": "Can I still join?"}
    )
    assert response.status_code == 200
    assert response.json()["retrieval_backend"] == "keyword"
    assert response.json()["sources"][0]["document_id"] == "74eb249bbf"


def test_ask_returns_controlled_503_for_missing_credentials() -> None:
    response = TestClient(create_app(lambda: MissingCredentialService())).post(
        "/ask", json={"question": "Can I still join?", "retrieval_backend": "vector"}
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "OPENAI_API_KEY is required"}


def test_ask_rejects_blank_questions_and_unknown_backends() -> None:
    client = TestClient(create_app(lambda: FakeService()))
    assert client.post("/ask", json={"question": "  "}).status_code == 422
    assert client.post("/ask", json={"question": "hi", "retrieval_backend": "hybrid"}).status_code == 422
