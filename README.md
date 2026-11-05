# AgentFlow-AI

Multi-agent AI content generation and evaluation platform.

A user supplies a topic and a target platform (LinkedIn first). A stateful
LangGraph workflow then researches the topic, plans content angles, writes
posts, reviews them, and verifies factual claims — with a bounded
Writer ↔ Editor feedback loop rather than a linear prompt chain.

The project's real subject is **agent orchestration**: state management,
conditional routing, feedback loops, observability, and evaluation.

## Status

**All seven phases complete.** A Retriever fetches and ranks real page
passages; a Researcher grounded in that evidence, a Planner that assigns each
post a distinct angle, and a Writer ↔ Editor ↔ Fact Checker loop sharing one
hard revision budget run behind it. Runs stream live to a React Flow view of
the graph, are kept in a run history with cross-run metrics and cost
estimates, and the whole pipeline can be measured against a single-agent
baseline. Gemini and Groq are both supported as LLM providers.

| Phase | Scope | State |
|-------|-------|-------|
| 1 | FastAPI + React foundation, config, health check | ✅ done |
| 2 | LangGraph MVP: Writer ↔ Editor loop, max 3 revisions | ✅ done |
| 3 | Real research, Content Planner, Fact Checker, source tracking | ✅ done |
| 4 | SSE streaming, React Flow visualisation, live activity log, results panel | ✅ done |
| 5 | Evidence retrieval: page fetch, chunking, embeddings, grounding guardrails | ✅ done |
| 6 | Single-agent baseline vs multi-agent evaluation harness | ✅ done |
| 7 | Cost estimates, run history, cross-run metrics | ✅ done |

## Stack

**Backend** — Python 3.11, FastAPI, Pydantic v2, pydantic-settings, LangGraph
**LLM / search** — Gemini or Groq for generation, Gemini embeddings, Tavily search
**Frontend** — React 19, TypeScript 6, Vite 8, Tailwind CSS v4, shadcn/ui, React Flow
**Storage** — SQLite (standard library) for run history; live runs stay in memory. Swapping in PostgreSQL touches `app/services/run_store.py` alone.

## Requirements

- Python 3.11+
- Node.js 20.19+ (Vite 8 requirement)

## Setup

### Backend

```powershell
python -m venv backend/.venv
backend\.venv\Scripts\Activate.ps1
pip install -r backend/requirements-dev.txt
copy backend\.env.example backend\.env
```

Run it from the `backend/` directory:

```powershell
uvicorn app.main:app --reload --port 8000
```

> **Single worker only.** Phase 4 holds live workflow runs and their SSE queues
> in process memory, so a second worker would route a client to a process that
> has never heard of its run.

### Frontend

```powershell
cd frontend
npm install
copy .env.example .env
npm run dev
```

- Frontend: http://localhost:5173
- API docs: http://localhost:8000/docs

### Running a workflow

```powershell
curl -X POST http://localhost:8000/api/workflows/run `
  -H "Content-Type: application/json" `
  -d '{"topic":"How AI is changing software engineering","platform":"linkedin","tone":"professional","number_of_posts":2}'
```

This synchronous endpoint blocks until the run finishes. The UI instead uses
the streaming API: `POST /api/workflows` queues a run and returns its id,
`GET /api/workflows/{id}/events` streams execution events over SSE (replaying
any the client missed), and `GET /api/workflows/{id}` returns the state and,
once finished, the result.

Returns HTTP 200 with the generated posts, the Editor's verdict, the revision
count and per-agent metrics. A run that executes but fails also returns 200,
carrying `workflow_status` and structured `errors`; non-2xx is reserved for
invalid input (422) and unusable configuration (503).

Every result carries a `cost` estimate at list price (see
[Cost estimates](#cost-estimates)).

### Run history and metrics

Runs started through the streaming API are written to SQLite
(`backend/data/agentflow.db`) when they finish, events included. The UI's
**History** page lists them and shows aggregates across all of them.

- `GET /api/workflows?limit=50&offset=0` — finished runs, newest first
- `GET /api/metrics/summary` — run outcomes, approval rate, mean quality,
  revisions, duration, tokens and cost, plus per-agent execution counts,
  failure rates, latency and tokens

A finished run stays readable after the in-memory registry evicts it or the
server restarts: `GET /api/workflows/{id}` and its event stream fall back to
history, so the execution page replays the full graph. Averages of quality,
revisions and approval count only runs that reached a verdict; a run that
crashed is not scored as zero.

### Evaluating against a single agent

```powershell
cd backend; .venv\Scripts\python.exe -m app.evaluation --limit 3
```

For each topic in `backend/evals/topics.json`, the multi-agent workflow runs
as normal, then a single-agent baseline writes from the **same** sources and
retrieved excerpts in one LLM call. Both outputs are scored the same way:

- **Quality** — a neutral rubric judge (clarity, structure, engagement,
  relevance, tone, distinctness, overall) that is not told which system wrote
  the posts. It is deliberately not the workflow's Editor, which the
  multi-agent output was revised to satisfy.
- **Grounding** — the Fact Checker re-checks both outputs against the raw
  search snippets and excerpts, not against either system's own notes.
- **Cost** — LLM calls, tokens, generation latency and list-price cost,
  excluding the shared search and the evaluators' own calls.

Results go to `backend/evals/results/` as a Markdown report and the full JSON.
Options: `--posts N`, `--tone`, `--topics FILE`, `--pause SECONDS` (default 20,
to let per-minute quotas refill between topics), `--out DIR`.

Each topic spends one Tavily search, a full workflow run, one baseline call
and four evaluator calls. The judge is the same model as the systems under
test and may favour its own style; treat small differences on a handful of
topics as inconclusive.

### Tests

```powershell
cd backend; .venv\Scripts\python.exe -m pytest tests/
```

442 tests, no network calls and no API key required - the LLM and search
providers are stubbed throughout.

## Configuration

All configuration is environment-driven. `backend/app/core/config.py` is the
only module that reads the environment; everything else depends on the
validated `Settings` object.

| Variable | Default | Purpose |
|----------|---------|---------|
| `APP_ENV` | `development` | Switches log format and health strictness |
| `LOG_LEVEL` | `INFO` | Root log level |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed origins |
| `LLM_PROVIDER` | `gemini` | `gemini` or `groq` |
| `GEMINI_API_KEY` | — | [aistudio.google.com](https://aistudio.google.com/apikey), free tier |
| `GEMINI_MODEL` | `gemini-3.6-flash` | `gemini-2.5-flash` is closed to new API keys |
| `GROQ_API_KEY` | — | [console.groq.com](https://console.groq.com), free tier, no card |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Must support strict JSON schema mode; Llama models on Groq do not |
| `GROQ_MAX_COMPLETION_TOKENS` | `4096` | Prompt + this cap must stay under Groq's 8,000 tokens/minute |
| `TAVILY_API_KEY` | — | [app.tavily.com](https://app.tavily.com), 1,000 free credits/month, no card |
| `TAVILY_MAX_RESULTS` | `5` | Sources fetched per run |
| `TAVILY_SEARCH_DEPTH` | `basic` | `basic` or `advanced` |
| `RETRIEVAL_ENABLED` | `true` | Off falls back to search snippets; needs `GEMINI_API_KEY` even with Groq |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Always Gemini: Groq has no embeddings API |
| `EMBEDDING_DIMENSIONS` | `768` | Truncated from the native 3072 |
| `RETRIEVAL_CHUNK_SIZE` | `1200` | Characters per chunk (~300 tokens) |
| `RETRIEVAL_CHUNK_OVERLAP` | `150` | Must be smaller than the chunk size |
| `RETRIEVAL_TOP_K` | `8` | Passages handed to the agents |
| `RETRIEVAL_MIN_SIMILARITY` | `0.5` | Below this a passage is treated as off-topic |
| `RETRIEVAL_MAX_CHUNKS_PER_SOURCE` | `3` | Stops one page filling the evidence budget |
| `RETRIEVAL_MAX_CHUNKS_PER_RUN` | `90` | Embedding cap per run; free tier allows 100 passages/minute |
| `MAX_REVISIONS` | `3` | Hard cap on Writer ↔ Editor cycles |
| `LLM_TIMEOUT_SECONDS` | `60` | Per-call provider timeout |
| `LLM_MAX_RETRIES` | `2` | Retries on malformed structured output |
| `LLM_INPUT_PRICE_PER_MTOK` | — | USD per million prompt tokens; overrides the price table |
| `LLM_OUTPUT_PRICE_PER_MTOK` | — | USD per million output tokens; set both or neither |
| `RUN_STORE_ENABLED` | `true` | Record finished runs for history and metrics |
| `RUN_STORE_PATH` | `backend/data/agentflow.db` | SQLite file for run history |

The server starts without any keys; `/api/health` then reports
`llm_configured: false`. Running a workflow needs the key for the selected
`LLM_PROVIDER` plus `TAVILY_API_KEY`.

### Cost estimates

Costs are estimates at list price, not a bill: on a free tier the real charge
is zero. They answer what a run would cost on a paid plan, which is what makes
the single-agent and multi-agent systems comparable in money as well as
tokens. Reasoning tokens are priced as output, as both providers bill them.

The built-in table in `backend/app/services/pricing.py` (checked 2026-11, from
published third-party listings; verify against the provider before relying
on it):

| Model | Input $/M | Output $/M |
|-------|-----------|------------|
| `openai/gpt-oss-120b` (Groq) | 0.15 | 0.60 |
| `gemini-3.6-flash` | 1.50 | 7.50 (standard rate; an introductory discount may apply) |

Any other model is reported as unpriced rather than guessed at. Set
`LLM_INPUT_PRICE_PER_MTOK` and `LLM_OUTPUT_PRICE_PER_MTOK` to price against
your own rates.

### Secrets

Provider keys live in `backend/.env` and never leave the server. Every
`VITE_`-prefixed frontend variable is compiled into the JavaScript bundle and
is publicly readable, so the frontend holds only `VITE_API_BASE_URL`. All LLM
calls route through FastAPI for this reason.

## Design decisions

**A source can only come from a real search result.** URLs never appear in any
LLM response schema. The Researcher cites sources by index into a list it was
handed, and indexes that do not resolve are dropped. Fabricated citations are
not discouraged by prompting - they are structurally impossible.

**Factual cleanliness outranks editorial score when choosing the output.** A
draft verified free of unsupported claims beats a higher-scoring draft that
carries them. An *unchecked* draft is not clean: absence of verification is not
evidence of correctness, so it ranks with the unclean ones.

**A run never claims verification it does not have.** `completed_unverified` is
a distinct terminal status from `completed_with_errors` - nothing crashed, but
the claims are unsupported or unchecked - and every run returns a `warnings`
list a client must display alongside the posts.

**Both critics share one revision budget.** An Editor rejection and a Fact
Checker rejection draw on the same `revision_count`. Separate allowances would
let the two loops alternate and multiply worst-case token spend.

**Graph state is a `TypedDict`, not a Pydantic model.** LangGraph nodes return
partial updates merged by reducers; a `BaseModel` root fights that. Pydantic is
used for the payloads *inside* state and for the HTTP boundary, which is where
structured-output validation belongs.

**The revision cap lives in routing code, not the prompt.** A model told it has
three revisions will not reliably respect it; a conditional edge reading
`revision_count` will.

**Agents depend on a service layer, never a provider SDK.** `LLMProvider` is a
Protocol with per-provider adapters that normalise token usage into one
`LLMResponse`. Swapping providers touches one adapter — and comparable token
accounting is what makes the Phase 6 evaluation possible at all.

**Metrics and events are a wrapper, not per-agent code.** Every node is
registered through `run_agent_with_metrics`, so timing, error capture and event
emission cannot be forgotten by an individual agent.

**The evaluation controls for luck.** Both systems write from one search, so a
better result cannot come from a better set of sources. Neither evaluator knows
which system it is scoring, and neither is a component of the system under
test. A score an evaluator failed to produce is left out, never counted as
zero.

**History is an addition, never a dependency.** The in-memory registry is still
the live path. If the SQLite file cannot be opened, runs carry on and history
is simply off; a failed history write is logged, not surfaced as a failed run.

## Layout

```
backend/app/
  core/       config, structured logging, domain errors
  schemas/    HTTP request/response contracts
  models/     domain objects used inside graph state
  routes/     API endpoints
  services/   LLM, embedding and search providers, retrieval, metrics, pricing,
              run registry, run store
  agents/     agent implementations and their prompts
  graph/      LangGraph state, nodes, routing
  evaluation/ single-agent baseline, blind judge, runner, report, CLI

backend/evals/
  topics.json the evaluation topic set
  results/    evaluation reports (gitignored)

frontend/src/
  components/ UI, including generated shadcn/ui primitives
  hooks/      data-fetching and SSE hooks
  pages/      create, execution and history pages
  services/   typed API client
  types/      hand-maintained mirrors of backend schemas
```

`frontend/src/types/` is maintained by hand. When a backend schema changes,
both sides change in the same commit — there is no codegen step by design.

## Checks

```powershell
cd backend; .venv\Scripts\python.exe -m ruff check app
cd frontend; npm run lint; npm run build
```
