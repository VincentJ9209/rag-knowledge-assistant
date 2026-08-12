# Interview-Ready Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a reproducible RAG evaluation baseline, local NumPy vector retrieval, a testable FastAPI service, Docker packaging, CI, and evidence-based portfolio documentation by 2026-08-15.

**Architecture:** Freeze the existing 139-document SQLite corpus as canonical JSON, derive deterministic evaluation data from it, and run keyword/vector retrieval against the same holdout questions. Keep reusable behavior in small Python modules, inject external services in tests, expose the existing `RAGBase` through FastAPI, and package the verified service without secrets.

**Tech Stack:** Python 3.14, uv, SQLiteSearch, OpenAI SDK, NumPy, FastAPI, Uvicorn, pytest, httpx, Docker, GitHub Actions.

## Global Constraints

- Preserve `notebooks/ingest.py`, the SQLiteSearch backend, and `notebooks/rag_helper.py::RAGBase`.
- Keep the keyword baseline at question boost `2.0`, section boost `0.5`, course `llm-zoomcamp`, and top-k `5` until evaluation evidence supports a change.
- Use the frozen corpus for all evaluation and API bootstrap behavior; keep `.env`, SQLite runtime databases, and embedding caches ignored.
- Use TDD for production behavior: write a failing test, verify the expected failure, implement the minimum, then verify the focused and full suites.
- Do not add LangChain, LangGraph, hybrid search, reranking, external vector databases, monitoring platforms, Kubernetes, cloud deployment, or a second agent framework.
- Do not make real OpenAI calls in tests or CI, and do not claim metrics that were not freshly measured.
- Commit after each independently verified Sprint task; do not merge or push.

---

### Task 1: Frozen corpus and deterministic evaluation utilities

**Files:**
- Create: `data/evaluation/faq_llm_zoomcamp_139.json`
- Create: `data/evaluation/evaluation_metadata.json`
- Create: `data/evaluation/ground_truth.csv` only after successful API generation
- Create: `data/evaluation/ground_truth_audit_sample.csv` only after successful API generation
- Create: `notebooks/evaluation_utils.py`
- Create: `notebooks/07_evaluation_ground_truth.ipynb`
- Create: `tests/test_evaluation_utils.py`
- Create: `scripts/freeze_evaluation_corpus.py`
- Create: `scripts/generate_ground_truth.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `canonical_corpus_bytes(documents) -> bytes`, `corpus_sha256(documents) -> str`, `validate_ground_truth(records, valid_document_ids, questions_per_document=5) -> None`, `deterministic_split(records, seed=20260813, tuning_ratio=0.70) -> list[dict[str, str]]`, `relevance_list(results, expected_document_id) -> list[bool]`, `hit_rate_at_k(relevance, k=5) -> float`, `reciprocal_rank(relevance, k=5) -> float`, and `evaluate(search_fn, records, k=5) -> dict[str, float]`.
- Consumes: read-only `notebooks/faq.db` with exactly 139 JSON documents containing `id`, `course`, `section`, `question`, and `answer`.

- [ ] **Step 1: Write metric, validation, and deterministic-split tests**

```python
def test_metrics_rank_relevant_document():
    relevance = [False, True, False]
    assert hit_rate_at_k(relevance, 3) == 1.0
    assert reciprocal_rank(relevance, 3) == 0.5

def test_validation_rejects_unknown_document():
    with pytest.raises(ValueError, match="unknown document"):
        validate_ground_truth([{"question": "q", "document": "missing"}], {"known"}, 1)

def test_split_is_deterministic_and_represents_every_document():
    first = deterministic_split(five_questions_per_document, seed=20260813)
    second = deterministic_split(five_questions_per_document, seed=20260813)
    assert first == second
    assert all({"tuning", "holdout"} <= splits_for(doc) for doc in document_ids)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `uv run pytest tests/test_evaluation_utils.py -q`
Expected: collection fails because `notebooks.evaluation_utils` does not exist.

- [ ] **Step 3: Implement canonical hashing, QA, split, and metrics**

```python
def evaluate(search_fn, records, k=5):
    relevance = [relevance_list(search_fn(row["question"], k), row["document"]) for row in records]
    return {
        "hit_rate_at_5": sum(hit_rate_at_k(items, k) for items in relevance) / len(relevance),
        "mrr": sum(reciprocal_rank(items, k) for items in relevance) / len(relevance),
    }
```

- [ ] **Step 4: Verify GREEN and export the read-only 139-document corpus**

Run: `uv run pytest tests/test_evaluation_utils.py -q`
Expected: all focused tests pass.

Run: `uv run python scripts/freeze_evaluation_corpus.py --database notebooks/faq.db --output-dir data/evaluation --snapshot-date 2026-08-13`
Expected: 139 records, canonical SHA-256 printed, and metadata names the local SQLite source.

- [ ] **Step 5: Build the bounded-concurrency structured generation path and thin notebook**

```python
questions = client.responses.parse(
    model=model,
    input=messages,
    text_format=GeneratedQuestions,
).output_parsed.questions
```

The script must use one call per document, maximum concurrency `4`, bounded retries, exactly five questions per successful document, normalized duplicate checks, and atomic output only after full QA succeeds.

- [ ] **Step 6: Attempt fresh generation without fabricating fallback data**

Run: `uv run python scripts/generate_ground_truth.py --corpus data/evaluation/faq_llm_zoomcamp_139.json --output data/evaluation/ground_truth.csv --model gpt-5.4-mini --workers 4`
Expected: either 695 validated rows with the deterministic split and audit sample, or an exact credential/model/API blocker with no invented rows.

- [ ] **Step 7: Verify and commit Task 1**

Run: `uv run pytest tests/test_evaluation_utils.py -q`
Commit: `feat: freeze RAG evaluation corpus and utilities`

---

### Task 2: Keyword baseline, tuning, and holdout evaluation

**Files:**
- Create: `notebooks/08_search_evaluation.ipynb`
- Create: `scripts/evaluate_search.py`
- Create: `tests/test_keyword_evaluation.py`
- Create: `data/evaluation/search_evaluation_results.csv` after a successful measured run
- Modify: `notebooks/rag_helper.py` only if the evidence-backed candidate wins without holdout regression

**Interfaces:**
- Consumes: frozen corpus, validated ground truth, SQLiteSearch, and `evaluate()`.
- Produces: reproducible rows with backend, split, question boost, section boost, answer boost behavior, Hit Rate@5, and MRR.

- [ ] **Step 1: Test frozen-corpus indexing and exact baseline search arguments**

```python
def test_baseline_uses_existing_rag_weights():
    results = run_keyword_search(index, "Can I still join?", question_boost=2.0, section_boost=0.5, k=5)
    assert len(results) <= 5
```

- [ ] **Step 2: Verify RED, implement the temporary index/evaluator, and verify GREEN**

Run: `uv run pytest tests/test_keyword_evaluation.py -q`
Expected RED: missing evaluation script module.

Run after implementation: `uv run pytest tests/test_keyword_evaluation.py -q`
Expected GREEN: all focused tests pass without network access.

- [ ] **Step 3: Run the baseline and 6-10 candidate tuning grid**

Run: `uv run python scripts/evaluate_search.py --corpus data/evaluation/faq_llm_zoomcamp_139.json --ground-truth data/evaluation/ground_truth.csv --output data/evaluation/search_evaluation_results.csv`
Expected: tuning metrics for baseline and candidates, followed by exactly one holdout evaluation for the selected candidate.

- [ ] **Step 4: Preserve or update `RAGBase.search()` from evidence**

Keep `2.0/0.5` if the candidate trades metrics or regresses on holdout. If changing weights, first add a focused failing regression test for the chosen constants and commit the weight change separately.

- [ ] **Step 5: Verify and commit Task 2**

Run: `uv run pytest tests/test_keyword_evaluation.py tests/test_evaluation_utils.py -q`
Commit: `feat: benchmark keyword retrieval on frozen holdout`

---

### Task 3: Local NumPy vector retrieval

**Files:**
- Create: `notebooks/vector_search.py`
- Create: `scripts/evaluate_vector_search.py`
- Create: `tests/test_vector_search.py`
- Modify: `.gitignore`
- Modify: `data/evaluation/search_evaluation_results.csv` only after a successful measured run

**Interfaces:**
- Produces: `build_document_text(document) -> str`, `cosine_similarity(query_vector, document_matrix) -> numpy.ndarray`, and `VectorSearchIndex(documents, embedder).search(query, k=5) -> list[dict]`.
- Consumes: an injected embedder exposing `embed_documents(texts) -> ndarray` and `embed_query(text) -> ndarray`; production default model `text-embedding-3-small`.

- [ ] **Step 1: Write ranking, batching, metadata-preservation, and zero-vector tests**

```python
def test_vector_search_ranks_and_preserves_document_fields():
    index = VectorSearchIndex(documents, FakeEmbedder())
    result = index.search("course timing", k=1)[0]
    assert result["id"] == "doc-timing"
    assert set(("id", "course", "section", "question", "answer")) <= result.keys()
```

- [ ] **Step 2: Verify RED, implement the minimum NumPy search, and verify GREEN**

Run: `uv run pytest tests/test_vector_search.py -q`
Expected RED: missing `notebooks.vector_search`.

Run after implementation: `uv run pytest tests/test_vector_search.py -q`
Expected GREEN: all tests pass with fake embeddings and no network calls.

- [ ] **Step 3: Attempt the holdout benchmark with cache metadata**

Run: `uv run python scripts/evaluate_vector_search.py --corpus data/evaluation/faq_llm_zoomcamp_139.json --ground-truth data/evaluation/ground_truth.csv --model text-embedding-3-small --k 5`
Expected: measured vector Hit Rate@5/MRR or an exact credential/model/API blocker; cache stays ignored.

- [ ] **Step 4: Verify and commit Task 3**

Run: `uv run pytest tests/test_vector_search.py -q`
Commit: `feat: add local vector retrieval baseline`

---

### Task 4: FastAPI service around `RAGBase`

**Files:**
- Create: `app/__init__.py`
- Create: `app/main.py`
- Create: `app/service.py`
- Create: `tests/test_api.py`

**Interfaces:**
- Produces: `create_app(service_factory=None) -> FastAPI`, `GET /health`, `POST /ask`, `AskRequest(question: str, retrieval_backend: Literal["keyword", "vector"] = "keyword")`, and `AskResponse` with answer, backend, and compact sources.
- Consumes: the frozen corpus, runtime SQLite bootstrap, `RAGBase`, and optional vector search.

- [ ] **Step 1: Write API contract tests with an injected fake service**

```python
def test_health_requires_no_credentials(client):
    assert client.get("/health").json() == {"status": "ok"}

def test_ask_returns_sources(fake_client):
    response = fake_client.post("/ask", json={"question": "Can I join?"})
    assert response.status_code == 200
    assert response.json()["sources"][0]["document_id"] == "74eb249bbf"
```

- [ ] **Step 2: Verify RED, implement models/factory/service, and verify GREEN**

Run: `uv run pytest tests/test_api.py -q`
Expected RED: missing `app.main`.

Run after implementation: `uv run pytest tests/test_api.py -q`
Expected GREEN: health, fake ask, validation, and controlled 503 tests pass without OpenAI calls.

- [ ] **Step 3: Verify real application startup without credentials**

Run: `uv run uvicorn app.main:app --host 127.0.0.1 --port 8000`
Expected: process starts; `GET /health` returns HTTP 200; credential-dependent `/ask` returns a controlled HTTP 503.

- [ ] **Step 4: Commit Task 4**

Commit: `feat: expose RAG assistant through FastAPI`

---

### Task 5: Dependencies, complete tests, Docker, and CI

**Files:**
- Modify with uv: `pyproject.toml`
- Modify with uv: `uv.lock`
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `.github/workflows/ci.yml`
- Modify: `.gitignore`

**Interfaces:**
- Runtime dependencies: FastAPI, Uvicorn, NumPy, and existing required packages.
- Dev dependencies: pytest and httpx.

- [ ] **Step 1: Synchronize dependencies using uv**

Run: `uv add fastapi uvicorn numpy`
Run: `uv add --dev pytest httpx`
Run: `uv lock --check`
Expected: `pyproject.toml` and `uv.lock` agree.

- [ ] **Step 2: Run the complete test suite**

Run: `uv run pytest -q`
Expected: zero failures and no external OpenAI calls.

- [ ] **Step 3: Add a minimal secret-free container image**

```dockerfile
FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY app app
COPY notebooks notebooks
COPY data/evaluation data/evaluation
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 4: Add CI using `uv sync --frozen --dev` and `uv run pytest -q`**

```yaml
- uses: astral-sh/setup-uv@v6
- run: uv sync --frozen --dev
- run: uv run pytest -q
```

- [ ] **Step 5: Build and smoke-test Docker**

Run: `docker build -t rag-knowledge-assistant:interview-ready .`
Run: `docker run --rm -d --name rag-interview-ready -p 18000:8000 rag-knowledge-assistant:interview-ready`
Run: `Invoke-RestMethod http://127.0.0.1:18000/health`
Expected: `{"status":"ok"}` without secrets.

- [ ] **Step 6: Verify and commit Task 5**

Run: `uv run pytest -q`
Commit: `build: add reproducible tests Docker and CI`

---

### Task 6: Evidence-based README portfolio pass

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: fresh test count, keyword tuning/holdout results, vector benchmark if available, Docker build result, and `/health` smoke result.
- Produces: a three-minute interviewer view of the problem, architecture, verified capabilities, commands, evidence, and limitations.

- [ ] **Step 1: Capture fresh evidence before writing claims**

Run: `uv run pytest -q`
Run: `uv run python scripts/evaluate_search.py --corpus data/evaluation/faq_llm_zoomcamp_139.json --ground-truth data/evaluation/ground_truth.csv --output data/evaluation/search_evaluation_results.csv`
Run the vector benchmark only when credentials are available.

- [ ] **Step 2: Rewrite the first screen and verified architecture sections**

Include these distinct flows:

```text
Frozen FAQ Corpus -> Keyword OR Local NumPy Vector Retrieval -> Top-K Evidence -> Grounded Prompt -> OpenAI Response -> FastAPI
Ground Truth -> Deterministic Tuning / Holdout -> Hit Rate@5 + MRR -> Retrieval comparison
```

State unavailable vector metrics as incomplete, not as inferred values. Do not claim a vector database, human-audit pass, deployment, monitoring, hybrid search, or reranking.

- [ ] **Step 3: Verify README commands and commit Task 6**

Run: `uv run pytest -q`
Commit: `docs: present verified interview-ready RAG evidence`

---

### Task 7: Final verification gate

**Files:**
- Verify only; change files solely through a new failing-test-first correction cycle if a defect is found.

- [ ] **Step 1: Run the full required gate**

```powershell
git status --short
uv run pytest -q
uv run python scripts/evaluate_search.py --corpus data/evaluation/faq_llm_zoomcamp_139.json --ground-truth data/evaluation/ground_truth.csv --output data/evaluation/search_evaluation_results.csv
docker build -t rag-knowledge-assistant:interview-ready .
git log --oneline --decorate -n 10
```

- [ ] **Step 2: Start the container and verify `/health`**

Run the image on host port `18000`, request `/health`, record the HTTP/JSON result, and stop only that named container.

- [ ] **Step 3: Check requirements against artifacts and evidence**

Confirm all six capabilities are either implemented and verified or listed under genuine blockers with one recommended next action. Confirm the feature branch is not merged or pushed.
