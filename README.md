# RAG Knowledge Assistant

一個可重現、可評估、可透過 API 與 Docker 執行的 RAG 專案。系統將 DataTalks.Club 的 `139` 筆 LLM Zoomcamp FAQ 凍結為版本化語料，以相同 holdout set 比較 SQLite FTS keyword retrieval 與本機 NumPy vector retrieval，再把 Top-K evidence 交給 OpenAI 產生 grounded answer。

> **Verified on 2026-08-13:** `27 passed` · Keyword Hit Rate@5 `0.9713` · Vector Hit Rate@5 `0.9809` · Docker `/health` OK

## Interview-ready overview

### Problem

Notebook 型 RAG prototype 常缺少三件事：可重現的資料快照、獨立 holdout 評估，以及可部署的 application boundary。本專案保留既有 `RAGBase` 與 SQLiteSearch architecture，補齊 evaluation、semantic retrieval、FastAPI、pytest、Docker 與 CI。

### Verified capabilities

- 凍結 `139` 筆 LLM Zoomcamp FAQ，canonical corpus SHA-256：`678a3ac86505de79043d16a4297c2e7a053871b3cad81ab51148d9e43f3b381a`
- 使用 `gpt-5.4-mini` 產生並自動驗證 `695` 筆 retrieval questions
- Deterministic split：`486` tuning／`209` holdout；每一份文件在兩個 split 中都有代表題
- SQLiteSearch keyword baseline 與 `8` 組小型 tuning grid
- `text-embedding-3-small` + NumPy cosine similarity vector baseline，沒有外部 vector DB
- FastAPI `GET /health` 與 `POST /ask`，支援 `keyword`／`vector` backend
- Application 可在沒有 OpenAI key 時啟動；需要 credentials 的問答路徑會回傳受控 `503`
- `27` 項 pytest tests，不進行真實 OpenAI 呼叫
- Secret-free Docker image 與 GitHub Actions CI

## Architecture

```text
Frozen FAQ Corpus
        ↓
Persistent Keyword Retrieval OR Local NumPy Vector Retrieval
        ↓
Top-K Evidence
        ↓
Grounded Prompt
        ↓
OpenAI Response
        ↓
FastAPI
```

Evaluation 與 application lifecycle 分離：

```text
Ground Truth
    ↓
Deterministic Tuning / Holdout
    ↓
Hit Rate@5 + MRR
    ↓
Retrieval Backend Comparison
```

主要元件：

| Component | Responsibility |
|---|---|
| `notebooks/ingest.py` | DataTalks.Club FAQ loader |
| `notebooks/rag_helper.py::RAGBase` | Retrieval、context、prompt 與 generation coordinator |
| `notebooks/evaluation_utils.py` | Corpus hash、QA、split、Hit Rate@k、MRR 與 generic evaluator |
| `notebooks/vector_search.py` | Injectable embedding provider、batch document embeddings、cosine ranking |
| `app/service.py` | Frozen corpus bootstrap 與 retrieval backend adapter |
| `app/main.py` | Pydantic models、FastAPI factory 與 HTTP error boundary |

## Retrieval evidence

所有結果都使用相同的 frozen corpus、`k=5` 與 `209` 題 holdout set。

| Backend / decision | Split | Hit Rate@5 | MRR |
|---|---:|---:|---:|
| Keyword baseline，question `2.0`／section `0.5` | Tuning | `0.9650` | `0.8720` |
| Keyword baseline | Holdout | `0.9713` | `0.8561` |
| Selected keyword policy | Holdout | `0.9713` | `0.8561` |
| Vector，`text-embedding-3-small` | Holdout | `0.9809` | `0.9144` |

小型 tuning grid 沒有找到同時優於 baseline 的 keyword weights，因此保留 `RAGBase.search()` 原始設定：question boost `2.0`、section boost `0.5`、course filter `llm-zoomcamp`、top-k `5`。這避免為了沒有改善的參數變更而增加 regression risk。

Vector baseline 在這份 holdout 上較高，但它需要 embedding API、成本與 runtime cache；此結果只代表本次 frozen snapshot，不宣稱已達 production acceptance。

## Run locally

需求：Python `3.14+` 與 [uv](https://docs.astral.sh/uv/)。

```powershell
uv sync --frozen --dev
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health endpoint 不需要 secrets：

```powershell
curl.exe http://127.0.0.1:8000/health
```

預期輸出：

```json
{"status":"ok"}
```

要使用 `/ask`，請在 `.env` 設定：

```env
OPENAI_API_KEY="your_openai_api_key_here"
```

Keyword request：

```powershell
curl.exe -X POST http://127.0.0.1:8000/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"Can I still join after the course starts?","retrieval_backend":"keyword"}'
```

將 `retrieval_backend` 改成 `vector` 可使用 semantic retrieval。Response 會包含 `answer`、實際 backend，以及文件 ID、section、FAQ question 組成的 compact sources。

## Test and reproduce evaluation

```powershell
uv run pytest -q
```

重新執行 keyword evaluation：

```powershell
uv run python -m scripts.evaluate_search `
  --corpus data/evaluation/faq_llm_zoomcamp_139.json `
  --ground-truth data/evaluation/ground_truth.csv `
  --output data/evaluation/search_evaluation_results.csv
```

Vector benchmark 需要有效的 OpenAI key：

```powershell
uv run python -m scripts.evaluate_vector_search `
  --corpus data/evaluation/faq_llm_zoomcamp_139.json `
  --ground-truth data/evaluation/ground_truth.csv `
  --model text-embedding-3-small `
  --k 5
```

## Docker

```powershell
docker build -t rag-knowledge-assistant:interview-ready .
docker run --rm -p 8000:8000 rag-knowledge-assistant:interview-ready
curl.exe http://127.0.0.1:8000/health
```

`.dockerignore` 排除 `.env`、SQLite runtime DB、embedding cache、tests 與 local artifacts；image 不會內建 API key。

## Repository structure

```text
rag-knowledge-assistant/
├── app/                         # FastAPI and application service
├── data/evaluation/             # Frozen corpus, ground truth, measured results
├── notebooks/                   # Existing RAG plus thin evaluation notebooks
├── scripts/                     # Reproducible corpus generation and benchmarks
├── tests/                       # Offline pytest suite
├── .github/workflows/ci.yml
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

既有 Module 1 learning artifacts（persistent RAG、Function Calling、explicit Agentic Loop、ToyAIKit comparison 與 homework notebook）均保留，這個 Sprint 沒有重寫它們。

## Evaluation governance

- Ground Truth automated QA 檢查 blank questions、invalid document IDs、normalized duplicates，以及每文件恰好 `5` 題。
- Split 使用 seed `20260813`；139 文件的目標與實際分配為 `486／209`。
- `ground_truth_audit_sample.csv` 已準備 `20` 筆人工抽查樣本，但尚未宣告 human audit passed。
- Keyword candidate 只使用 tuning split 選擇，holdout 僅作一次 final comparison。
- README 數字來自 repository 內的 fresh-run result artifacts，不由 notebook presentation 反推。

## Limitations

- 本機 vector retrieval 是 NumPy matrix ranking，不是 vector database，也沒有 hybrid search 或 reranker。
- `/ask` 的 answer generation 仍需要 OpenAI credentials；vector backend 額外需要 embedding calls。
- Document embeddings 尚未做 production-grade persistent cache；service process 重啟後可能重新產生。
- Ground Truth 經 automated QA，但 20 筆 sample 尚待人工審核。
- 目前沒有 cloud deployment、monitoring platform、distributed serving 或 production load test。
- Metrics 僅適用於 `2026-08-13` frozen LLM Zoomcamp snapshot，不能外推到其他 corpus。

## Background

本專案源自 LLM Zoomcamp 2026 的 RAG／Agentic RAG 學習主線，工程演進如下：

```text
Basic RAG
→ Modular RAG
→ Persistent SQLite Retrieval
→ Function Calling
→ Explicit Agentic Loop
→ Framework Comparison
→ Frozen Evaluation + Holdout Metrics
→ Vector Baseline
→ FastAPI + Tests + Docker + CI
```
