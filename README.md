# RAG Knowledge Assistant

以 **Persistent Retrieval、Grounded Generation 與 LLM Tool Calling** 為核心的 RAG Knowledge Assistant。

> **Current Stage**
> ✅ Basic RAG · ✅ Modular RAG · ✅ Persistent SQLite RAG · ✅ Function Calling · ▶ Agentic Loop

---

## 專案概覽

RAG Knowledge Assistant 是一個以 **Retrieval-Augmented Generation (RAG)** 為核心的知識問答專案。

目前系統使用 DataTalks.Club FAQ 作為知識來源，將資料載入、檢索、Prompt 建構、LLM 生成與 Tool Calling 拆分成可重用的元件，並逐步從單次固定 RAG Pipeline 演進到可由 LLM 決定工具使用方式的 Agentic RAG 架構。

目前已完成的工程主線：

```text
Basic RAG
→ Modular RAG
→ Persistent SQLite Retrieval
→ Grounded Generation
→ Function Calling
→ Agentic Loop（進行中）
```

這個專案目前仍處於持續演進階段，因此 README 只呈現已完成並可由 Repository 驗證的能力，不提前宣告尚未實作的 Production features。

---

## Key Highlights

- 載入 **1,401 筆 DataTalks.Club FAQ documents**，並篩選 **139 筆 LLM Zoomcamp FAQ** 作為目前知識庫
- 使用 **MinSearch** 建立初始 keyword retrieval baseline
- 導入 **SQLiteSearch**，將 Retrieval 從記憶體索引升級為可跨 Kernel / Session 重用的 persistent index
- 將 FAQ ingestion 與 query lifecycle 分離，避免每次查詢都重新建立索引
- 建立可重用的 `RAGBase` abstraction，統一管理 retrieval、context、prompt 與 LLM generation
- 保留 **Top-K retrieval evidence**，可人工追蹤回答是否有對應知識來源
- 驗證 unknown-answer behavior，降低缺乏 evidence 時直接生成答案的風險
- 使用 **OpenAI Responses API Function Calling**
- 完成完整 single-tool round trip：

```text
User Question
→ LLM function_call
→ Python parses arguments
→ Local SQLite search
→ function_call_output
→ LLM final answer
```

- Notebook 已執行 fresh-kernel top-to-bottom verification，降低依賴舊記憶體狀態才能運作的風險

---

## Current Architecture

```mermaid
flowchart TD
    A[DataTalks.Club FAQ] --> B[Data Loading / Ingestion]
    B --> C[Filter: LLM Zoomcamp FAQ]
    C --> D[(SQLiteSearch<br/>faq.db)]
    D --> E[RAGBase.search]
    E --> F[Top-K FAQ Documents]
    F --> G[Context / Prompt Construction]
    G --> H[GPT-5.6]
    H --> I[Grounded Answer]

    J[User Question] --> K[OpenAI Function Calling]
    K -->|function_call: search| L[Local Python search()]
    L --> D
    F --> M[function_call_output]
    M --> H
```

### Persistent RAG Flow

```text
Knowledge Source
    ↓
Ingestion
    ↓
Persistent SQLite Index
    ↓
Query
    ↓
Top-K Retrieval
    ↓
Context
    ↓
Prompt
    ↓
LLM
    ↓
Grounded Answer
```

### Function Calling Flow

Function Calling 將部分 control flow 從固定 Python Pipeline 移交給 LLM：

```text
User Question
    ↓
LLM decides whether / how to call search
    ↓
function_call
    ↓
Application executes local tool
    ↓
function_call_output
    ↓
LLM uses retrieved evidence
    ↓
Final Answer
```

LLM 負責決定工具與 arguments；真正的 Python function、SQLite 與本機資料存取仍由 Application 執行。

---

## Repository Structure

```text
rag-knowledge-assistant/
├── .env.example
├── .gitignore
├── .python-version
├── README.md
├── main.py
├── pyproject.toml
├── uv.lock
└── notebooks/
    ├── 01_basic_rag.ipynb
    ├── 02_persistent_rag_ingest.ipynb
    ├── 03_persistent_rag_query.ipynb
    ├── 04_function_calling.ipynb
    ├── ingest.py
    └── rag_helper.py
```

### Learning / Engineering Milestones

| Artifact | Purpose |
|---|---|
| `01_basic_rag.ipynb` | 建立 Basic RAG：資料載入、搜尋、Prompt、LLM generation |
| `ingest.py` | 封裝 FAQ loading 與 MinSearch index construction |
| `rag_helper.py` | 提供可重用的 `RAGBase` |
| `02_persistent_rag_ingest.ipynb` | 建立並持久化 SQLite-backed FAQ index |
| `03_persistent_rag_query.ipynb` | 驗證 persistent retrieval、grounding 與 unknown-answer behavior |
| `04_function_calling.ipynb` | 實作 Function Calling single-tool round trip |

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.14+ |
| Environment / Dependency Management | uv |
| Notebook | Jupyter / VS Code |
| LLM API | OpenAI Responses API |
| Current Model Baseline | GPT-5.6 |
| Retrieval Baseline | MinSearch |
| Persistent Retrieval | SQLiteSearch / SQLite |
| Configuration | python-dotenv |
| Data Access | requests |
| Version Control | Git / GitHub |

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/VincentJ9209/rag-knowledge-assistant.git
cd rag-knowledge-assistant
```

### 2. Install dependencies

本專案使用 `uv` 管理 Python environment 與 dependencies：

```bash
uv sync
```

### 3. Configure the OpenAI API key

複製 `.env.example`：

```text
.env.example
→ .env
```

並設定：

```env
OPENAI_API_KEY="your_openai_api_key_here"
```

`.env` 已排除於 Git 追蹤之外，不應提交 API key。

### 4. Build the persistent knowledge index

開啟並由上而下執行：

```text
notebooks/02_persistent_rag_ingest.ipynb
```

此步驟會建立本機：

```text
notebooks/faq.db
```

`faq.db` 為 runtime artifact，已由 `.gitignore` 排除，不會提交至 GitHub。

### 5. Query the persistent RAG pipeline

執行：

```text
notebooks/03_persistent_rag_query.ipynb
```

此 Notebook 驗證：

- SQLite persistent retrieval
- `RAGBase` integration
- Top-K evidence inspection
- grounded generation
- unknown-answer behavior

### 6. Run the Function Calling workflow

執行：

```text
notebooks/04_function_calling.ipynb
```

可觀察完整 Tool Calling contract：

```text
Python search function
→ JSON tool schema
→ function_call
→ arguments parsing
→ local tool execution
→ function_call_output
→ final model response
```

---

## Design Decisions

### 1. Persistent Retrieval instead of rebuilding the index per query

初始 RAG 使用 in-memory MinSearch 方便快速理解 Retrieval 流程。

後續導入 SQLiteSearch，讓知識索引具備持久化能力：

```text
Ingestion lifecycle
≠
Query lifecycle
```

這使 query session 可以直接重新開啟既有 index，而不需要重新下載與 ingest 全部 FAQ。

### 2. Separate ingestion from querying

Persistent RAG 將兩種責任拆開：

```text
02_persistent_rag_ingest.ipynb
→ 建立 / 更新知識索引

03_persistent_rag_query.ipynb
→ 使用既有知識索引回答問題
```

這個切分更接近實際應用中「資料更新」與「線上查詢」不同生命週期的設計。

### 3. Reuse retrieval behavior during architecture changes

目前不因單一查詢結果任意調整 retrieval weights。

Retrieval tuning 將在建立 evaluation framework 後，以可比較的 evidence 與 metrics 作為依據。

### 4. Keep tool execution inside the application

在 Function Calling 中：

```text
LLM → decides what tool to request
Application → validates and executes the tool
LLM → consumes tool output
```

模型不直接操作 SQLite 或本機 Python runtime，使工具權限與執行責任維持在 Application layer。

---

## Current Stage

| Stage | Status |
|---|---|
| Basic RAG | ✅ Completed |
| Modular RAG | ✅ Completed |
| Persistent SQLite RAG | ✅ Completed |
| Grounding / Evidence Inspection | ✅ Completed |
| Unknown-answer Verification | ✅ Completed |
| Function Calling | ✅ Completed |
| Agentic Loop | ▶ Current |
| Automated RAG Evaluation | ○ Planned |
| Application / API Layer | ○ Planned |
| Automated Tests | ○ Planned |
| Docker / CI/CD | ○ Planned |
| Deployment | ○ Planned |

---

## Known Limitations

目前版本主要是 learning-to-engineering transition 階段，因此仍有以下限制：

- 主要執行入口仍以 Notebook 為主
- 尚未建立 automated RAG evaluation dataset 與 metrics
- 尚未建立完整 Agentic Loop
- 尚未提供 FastAPI / application service layer
- 尚未建立完整 automated test suite
- 尚未加入 Docker、CI/CD 與 cloud deployment
- SQLiteSearch 適合目前規模與 persistent retrieval learning stage，後續若知識量與查詢需求增加，需重新評估 search backend

這些項目會依照後續 milestone 逐步加入，而不是在現階段提前包裝為既有功能。

---

## Roadmap

下一階段預計依序推進：

1. **Agentic Loop**
   - 讓 LLM 可根據 tool result 決定是否再次搜尋
   - 支援 query reformulation / repeated retrieval
   - 建立明確 stop condition

2. **RAG Evaluation**
   - 建立 evaluation questions
   - 評估 retrieval quality
   - 評估 grounded answer quality
   - 使用 evaluation evidence 再進行 retrieval tuning

3. **Application Architecture**
   - 從 Notebook prototype 抽離 application code
   - 建立清楚的 package boundaries
   - 加入 validation、error handling 與 logging

4. **API / Testing**
   - FastAPI service layer
   - automated tests
   - reproducible API behavior

5. **Delivery / Operations**
   - Docker
   - CI/CD
   - deployment
   - observability

---

## Learning Source

本專案以 **LLM Zoomcamp 2026** 作為系統化學習基礎，並在課程內容之上持續整理與延伸工程實作。

目前重點並非單純複製教學 Notebook，而是逐步建立：

- 可重用的 RAG abstraction
- persistent retrieval lifecycle
- reproducible project workflow
- evidence-based retrieval / generation inspection
- Function Calling 與 Agentic RAG control flow
- 後續可延伸到 API、Evaluation、Testing 與 Deployment 的作品集架構

---

## Project Status

目前專案已建立可運作的 Persistent RAG + Function Calling baseline，下一個工程 milestone 為 **Agentic Loop**。

README 將隨專案 milestone 持續更新；待 Evaluation、Application Layer 與 Deployment 完成後，再整理為正式 portfolio release。
