# RAG Knowledge Assistant

以 **Persistent Retrieval、Grounded Generation、LLM Tool Calling 與 Agentic Loop** 為核心的 RAG Knowledge Assistant。

> **Current Stage**
> ✅ Basic RAG · ✅ Modular RAG · ✅ Persistent SQLite RAG · ✅ Function Calling · ✅ Agentic Loop

---

## 專案概覽

RAG Knowledge Assistant 是一個以 **Retrieval-Augmented Generation (RAG)** 為核心的知識問答專案。

目前系統使用 DataTalks.Club FAQ 作為知識來源，將資料載入、檢索、Prompt 建構、LLM 生成、Tool Calling 與 Agent control flow 拆分成可理解、可驗證且可逐步擴充的元件。

專案已從單次固定 RAG Pipeline 演進到可由 LLM 根據目前 evidence 決定是否使用搜尋工具、如何改寫 query、是否需要再次搜尋，以及何時停止工具呼叫並產生最終回答的 Agentic RAG baseline。

目前已完成的工程主線：

```text
Basic RAG
→ Modular RAG
→ Persistent SQLite Retrieval
→ Grounded Generation
→ Function Calling
→ Agentic Loop
```

README 只呈現目前已完成並可由 Repository 驗證的能力，不提前宣告尚未實作的 Production features。

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

- 進一步將單次 Function Calling 擴充為 **Agentic Loop**
- 由模型自行決定：
  - 是否呼叫 `search`
  - 使用什麼 search query
  - 是否需要再次搜尋
  - 何時停止 Tool Calling
- 保留完整 Agent execution history：
  - user input
  - function call
  - function call output
  - final message
- 加入 `max_iterations` safety guard，避免無限制 Tool Calling
- 驗證 Agent 最終回答可由既有 Top-K evidence 支撐
- Notebook 已執行 fresh-kernel top-to-bottom verification，降低依賴舊記憶體狀態才能運作的風險

---

## Current Architecture

```mermaid
flowchart TD
    A[DataTalks.Club FAQ] --> B[Data Loading / Ingestion]
    B --> C[Filter: LLM Zoomcamp FAQ]
    C --> D[(SQLiteSearch<br/>faq.db)]

    U[User Question] --> AG[Agent / GPT-5.6]
    AG -->|function_call: search| S[Local Python search()]
    S --> D
    D --> R[RAGBase.search]
    R --> K[Top-K FAQ Evidence]
    K --> O[function_call_output]
    O --> AG

    AG -->|needs more evidence| S
    AG -->|enough evidence| F[Final Grounded Answer]
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

### Agentic Loop Flow

Agentic Loop 將 Function Calling 從固定一次 round trip 擴充為可重複 decision cycle：

```text
User Question
    ↓
LLM Decision
    ↓
┌──────────────────────────────┐
│ function_call requested?     │
└──────────────────────────────┘
        │ Yes
        ↓
Execute local search()
        ↓
function_call_output
        ↓
Update agent history
        ↓
LLM Decision
        │
        ├── search again
        │      ↓
        │    repeat
        │
        └── final message
               ↓
              STOP
```

目前 Agent history 保留完整的 execution trajectory，讓模型可以根據先前的工具呼叫與 retrieval evidence 決定下一步。

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
    ├── 05_agentic_loop.ipynb
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
| `05_agentic_loop.ipynb` | 實作 Agentic RAG loop、state management、stop condition 與 grounding audit |

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
| Agent Tool Interface | OpenAI Function Calling |
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

### 7. Run the Agentic RAG workflow

執行：

```text
notebooks/05_agentic_loop.ipynb
```

此 Notebook 從手動 Function Calling trajectory 開始，逐步建立：

```text
Instructions
+
Tools
+
Message History
↓
Agent Decision
↓
Tool Execution
↓
Observation
↓
Repeated Decision
↓
Stop Condition
```

最後封裝為：

```python
run_agent(user_question, max_iterations=5)
```

Agent 可以在執行期間自行決定是否再次搜尋，或在 evidence 足夠時停止 Tool Calling 並回答。

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
LLM
→ decides what tool to request

Application
→ validates and executes the tool

LLM
→ consumes tool output
```

模型不直接操作 SQLite 或本機 Python runtime，使工具權限與執行責任維持在 Application layer。

### 5. Preserve explicit agent state

Agent Loop 不只保存一般 user / assistant messages，也保留：

```text
function_call
function_call_output
message
```

完整 execution history 讓模型能根據先前 action 與 observation 決定下一步，也為後續 debugging、logging、evaluation 與 observability 提供可追蹤基礎。

### 6. Use an explicit termination guard

Agent 可能重複要求 Tool Calling，因此 `run_agent()` 使用有限 iteration：

```text
max_iterations = 5
```

當模型持續要求工具而超過限制時，由 Application 主動中止流程。

這個 guard 避免依賴模型自行保證一定停止。

---

## Agentic RAG Verification

目前 Agentic Loop 已完成以下實際驗證：

```text
User Question
↓
GPT-5.6
↓
function_call: search
↓
SQLiteSearch retrieves 5 documents
↓
function_call_output
↓
GPT-5.6
↓
final message
↓
STOP
```

其中一次 fresh-run trajectory：

```text
Agent iterations: 2
Function calls: 1
Function outputs: 1
Final messages: 1
```

Agent 的完整 history：

```text
user
→ function_call
→ function_call_output
→ message
```

此外，最終回答中的主要 claims 也經人工對照 Top-K FAQ evidence：

- 課程開始後仍可加入
- Certificate 需在 project submission 開放期間完成符合條件的 Capstone
- Homework submission form 關閉後不接受 late submission
- Missing homework 不影響 certificate eligibility

這些資訊均能在 retrieved FAQ evidence 中找到對應支持。

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
| Agentic Loop | ✅ Completed |
| Agent Trajectory Inspection | ✅ Completed |
| Grounding Evidence Audit | ✅ Completed |
| Automated RAG Evaluation | ○ Planned |
| Application / API Layer | ○ Planned |
| Automated Tests | ○ Planned |
| Docker / CI/CD | ○ Planned |
| Deployment | ○ Planned |

---

## Known Limitations

目前版本仍處於 learning-to-engineering transition 階段，因此有以下限制：

- 主要執行入口仍以 Notebook 為主
- 目前 Agent 只有一個 `search` tool
- 尚未建立 automated RAG evaluation dataset 與 metrics
- 尚未對 multi-step / multi-search Agent trajectories 建立系統化測試集
- 尚未提供 FastAPI / application service layer
- 尚未建立完整 automated test suite
- 尚未加入 structured logging / tracing / observability
- 尚未加入 Docker、CI/CD 與 cloud deployment
- SQLiteSearch 適合目前規模與 persistent retrieval learning stage；後續若知識量與查詢需求增加，需要重新評估 search backend

這些項目會依照後續 milestone 逐步加入，而不是在現階段提前包裝為既有功能。

---

## Roadmap

### 1. Agent Framework Exploration

在理解原生 Agentic Loop control flow 後，再比較較高階的 agent abstractions / frameworks：

- Tool registration
- Agent loop abstraction
- state management
- framework trade-offs
- abstraction cost vs. implementation simplicity

### 2. RAG Evaluation

建立可重複的 evaluation framework：

- evaluation questions
- retrieval quality
- Top-K relevance
- grounded answer quality
- unsupported claim detection
- Agent tool-use behavior

Retrieval tuning 將在這個階段以 evaluation evidence 作為依據。

### 3. Application Architecture

將目前 Notebook prototype 逐步抽離為 application code：

- clear package boundaries
- configuration management
- validation
- error handling
- logging
- reusable Agent / RAG services

### 4. API / Testing

建立可被其他應用使用的 service layer：

- FastAPI
- request / response schemas
- automated tests
- agent behavior regression tests
- reproducible API behavior

### 5. Delivery / Operations

補齊 application delivery 能力：

- Docker
- CI/CD
- deployment
- observability
- runtime configuration

---

## Learning Source

本專案以 **LLM Zoomcamp 2026** 作為系統化學習基礎，並在課程內容之上持續整理與延伸工程實作。

目前重點不是單純複製教學 Notebook，而是逐步建立：

- reusable RAG abstraction
- persistent retrieval lifecycle
- reproducible project workflow
- evidence-based retrieval / generation inspection
- OpenAI Function Calling
- explicit Agentic RAG control flow
- Agent state / trajectory inspection
- grounding evidence audit
- 後續可延伸至 Evaluation、API、Testing 與 Deployment 的作品集架構

---

## Project Status

目前專案已建立可運作並完成 fresh-kernel verification 的：

```text
Persistent RAG
+
Function Calling
+
Agentic Loop
```

目前 Agent 可以：

```text
Receive Question
→ Decide to Search
→ Generate Search Query
→ Execute Local Retrieval
→ Consume Retrieved Evidence
→ Decide Whether to Continue
→ Stop and Produce a Grounded Answer
```

下一階段將先理解較高階 Agent abstraction / framework 如何封裝目前已經手動實作的 control flow，再進一步推進 Evaluation 與 application architecture。

README 將隨專案 milestone 持續更新；待 Evaluation、Application Layer、Testing 與 Deployment 等工程能力成熟後，再整理為正式 portfolio release。