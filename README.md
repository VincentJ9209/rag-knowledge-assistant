# RAG Knowledge Assistant

以 **Persistent Retrieval、Grounded Generation、LLM Tool Calling、Agentic Loop 與 Framework Abstraction** 為核心的 RAG Knowledge Assistant。

> **Current Stage**
> ✅ Basic RAG · ✅ Modular RAG · ✅ Persistent SQLite RAG · ✅ Function Calling · ✅ Agentic Loop · ✅ ToyAIKit Comparison · ✅ Module 1 E2E Validation

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
→ Explicit Agentic Loop
→ ToyAIKit Framework Comparison
→ Module 1 End-to-End Validation
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
- 使用 **ToyAIKit** 對照手寫 `run_agent()`，驗證 framework 對 tool registration、function dispatch、message history 與 repeated model/tool orchestration 的 abstraction boundary
- 完成 framework-selection checkpoint：保留 explicit Agent Loop 作為可讀、可除錯的 baseline；除非新 framework 能解決明確的 capability、integration 或 maintainability 問題，否則不額外增加 dependency
- 完成獨立的 **Module 1 Agentic RAG Homework** end-to-end validation：固定課程 commit `8c1834d` 載入 **72 lesson pages**，MinSearch Top-1 命中 `01-agentic-rag/lessons/14-agentic-loop.md`
- Chunking 設定為 `size=2000, step=1000`，產生 **295 chunks**；相同 RAG query 的 input tokens 從 **7,135 降至 2,318**，約 **3.08× fewer**
- 將 chunk search 暴露為 Agent tool；fresh-kernel verification 中 Agent 自主完成 **4 次 search calls** 後產生最終回答

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
├── homework/
│   └── 01_agentic_rag_homework.ipynb
└── notebooks/
    ├── 01_basic_rag.ipynb
    ├── 02_persistent_rag_ingest.ipynb
    ├── 03_persistent_rag_query.ipynb
    ├── 04_function_calling.ipynb
    ├── 05_agentic_loop.ipynb
    ├── 06_toyaikit.ipynb
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
| `06_toyaikit.ipynb` | 對照 explicit Agent Loop 與 ToyAIKit framework abstraction，驗證 tool registration、dispatch、history 與 orchestration |
| `homework/01_agentic_rag_homework.ipynb` | Module 1 end-to-end validation：lesson ingestion、MinSearch、RAG token usage、chunking 與 multi-search agent |

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
| Agent Framework Comparison | ToyAIKit |
| Course Dataset Loader / Chunking | gitsource |
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

### 8. Compare the ToyAIKit framework workflow

執行：

```text
notebooks/06_toyaikit.ipynb
```

此 Notebook 將前一階段手寫的 explicit `run_agent()` 與 ToyAIKit framework 進行對照，驗證：

- tool registration 與 schema generation
- function dispatch
- message / trajectory management
- repeated model-tool orchestration
- token / usage reporting

ToyAIKit 可以減少 Agent Loop 的 orchestration boilerplate，但 application 仍負責 retrieval behavior、tool permissions、instructions、grounding 與 evaluation。

### 9. Reproduce the Module 1 integration homework

執行：

```text
homework/01_agentic_rag_homework.ipynb
```

此 Homework 刻意與目前 persistent FAQ baseline 分離，使用固定課程 commit `8c1834d` 與 `gpt-5.4-mini` 進行可重現的 end-to-end integration validation。

Fresh-kernel 驗證結果：

```text
Lesson pages: 72
Chunking: size=2000, step=1000
Chunks: 295
Full-page RAG input tokens: 7,135
Chunked RAG input tokens: 2,318
Reduction: 3.08x fewer
Fresh-run Agent search calls: 4
```

其中 `Fresh-run Agent search calls: 4` 是本次 fresh-kernel 執行的實測 trajectory；Agent 的 tool-call 次數屬於 model-driven behavior，可能因不同執行而變動，因此 `4` 不應視為固定 deterministic output。

這些結果證明 chunking 能在該 Homework query 中降低輸入 context / token usage，並驗證 Agent 可以自主進行多次 retrieval。這個實驗目前不取代主要的 SQLite FAQ architecture，也不在缺少 systematic evaluation 的情況下直接視為 retrieval quality 已提升。

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

### 7. Preserve the explicit Agent Loop baseline

導入 ToyAIKit 後，仍保留 `run_agent()` 與手寫 Agent Loop 作為可讀、可除錯的 reference implementation。

Framework 可以封裝 tool registration、dispatch、message history 與 repeated orchestration，但 application 仍負責 retrieval behavior、tool permissions、instructions、grounding、evaluation 與 termination policy。

保留 explicit baseline 的目的，是讓 framework abstraction 出現異常時，仍能回到清楚的 control flow 進行比較、除錯與 regression analysis。

### 8. Adopt frameworks only for a concrete capability need

完成 ToyAIKit comparison 後，不再為了增加 framework 數量而導入第二套 Agent framework。

新的 framework / dependency 必須解決明確問題，例如：

- capability gap
- integration requirement
- maintainability
- observability
- deployment constraint

若現有 explicit loop 或 ToyAIKit 已能滿足需求，優先維持較小的 dependency surface 與較低的 abstraction cost。

### 9. Keep homework experiments isolated until evaluation

`homework/01_agentic_rag_homework.ipynb` 使用不同的資料 schema、固定課程 commit、`gpt-5.4-mini` 與 chunked lesson corpus，因此維持為獨立 integration checkpoint，不直接覆寫目前 FAQ-specific `RAGBase` 與 SQLite retrieval baseline。

Homework 已量測到 chunking 將該 query 的 input tokens 從 7,135 降至 2,318，約 3.08x fewer；這是 prompt-size / token-use evidence，不等同於 retrieval quality 或 answer quality 已提升。

因此，在建立 systematic evaluation dataset 與 metrics 前，不根據單一 Homework 結果調整主線 retrieval weights、chunking strategy 或 production architecture。

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
| ToyAIKit Framework Comparison | ✅ Completed |
| Framework Selection Checkpoint | ✅ Completed |
| Module 1 Agentic RAG Homework | ✅ Completed |
| Chunking Token-Use Measurement | ✅ Completed |
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

### 1. RAG Evaluation

下一個工程 milestone 是建立可重複的 evaluation framework，先用可比較 evidence 驗證 retrieval 與 generation，再決定是否調整 weights、chunking 或 prompt strategy。

預計包含：

- evaluation question set
- retrieval relevance / Top-K quality
- grounded answer quality
- unsupported claim detection
- Agent tool-use behavior
- regression checks across explicit loop and ToyAIKit

目前不因單一 Homework query 的 token reduction 直接調整主線 retrieval architecture。

### 2. Application Architecture

將目前 Notebook prototype 逐步抽離為可維護的 application code：

- clear package boundaries
- configuration management
- validation
- error handling
- logging
- reusable RAG / Agent services

### 3. API / Testing

建立可被其他應用呼叫的 service layer 與 automated verification：

- FastAPI
- request / response schemas
- automated tests
- RAG regression tests
- Agent behavior regression tests
- reproducible API behavior

### 4. Delivery / Operations

補齊 application delivery 與 runtime operations：

- Docker
- CI/CD
- deployment
- observability
- runtime configuration

Framework adoption 之後仍遵循 capability-driven 原則；除非新的需求明確需要，否則不為增加技術數量導入第二套 Agent framework。

---

## Learning Source

本專案以 **LLM Zoomcamp 2026** 作為系統化學習基礎，並在課程內容之上持續整理與延伸工程實作。

目前 Module 1 Agentic RAG 已完成從 Basic RAG、persistent retrieval、Function Calling、explicit Agentic Loop 到 ToyAIKit framework comparison 與 end-to-end Homework validation 的完整學習主線。

目前重點是把課程概念轉化為可驗證的工程能力，包括：

- reusable RAG abstraction
- persistent retrieval lifecycle
- reproducible project workflow
- evidence-based retrieval / generation inspection
- OpenAI Function Calling
- explicit Agentic RAG control flow
- Agent state / trajectory inspection
- grounding evidence audit
- ToyAIKit framework comparison
- framework adoption criteria
- fixed-snapshot end-to-end homework validation
- evaluation-first engineering roadmap

Module 1 Homework 使用固定課程 commit `8c1834d` 驗證 72 lesson pages、295 chunks、RAG token usage 與 multi-search Agent behavior；這些量測結果保留為獨立 evidence，不在 systematic evaluation 前直接改寫主線 retrieval architecture。

---

## Project Status

目前專案已建立並完成 fresh-kernel verification 的：

```text
Persistent RAG
+
Function Calling
+
Explicit Agentic Loop
+
ToyAIKit Framework Comparison
+
Module 1 End-to-End Validation
```

目前系統已能：

```text
Receive Question
→ Decide Whether to Search
→ Generate Search Query
→ Execute Local Retrieval
→ Consume Retrieved Evidence
→ Decide Whether to Search Again
→ Stop and Produce a Grounded Answer
```

ToyAIKit comparison 已驗證 framework 可以封裝 tool registration、dispatch、message history 與 repeated orchestration，同時保留 application 對 retrieval、permissions、instructions、grounding、evaluation 與 termination policy 的責任。

Module 1 Homework 另以固定 source snapshot 驗證：

```text
Lesson pages: 72
Chunks: 295
Full-page RAG input tokens: 7,135
Chunked RAG input tokens: 2,318
Reduction: 3.08x fewer
Fresh-run Agent search calls: 4
```

上述 Homework 結果屬於 integration / token-use evidence，不代表主線 retrieval quality 已經提升，也不直接取代目前的 persistent SQLite FAQ architecture。

下一個工程 milestone 是 **RAG Evaluation**：建立可重複的 evaluation question set、retrieval relevance / Top-K metrics、grounded answer quality、unsupported claim detection 與 Agent behavior regression checks，再以量測結果決定 retrieval tuning、chunking 或 prompt strategy 是否需要調整。

README 將持續隨可驗證 milestone 更新；後續再逐步推進 Application Architecture、API / Testing、Docker、CI/CD、Deployment 與 Observability。
