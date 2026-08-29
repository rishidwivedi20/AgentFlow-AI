# OrchestrAI

Multi-agent AI content generation and evaluation platform.

A user supplies a topic and a target platform (LinkedIn first). A stateful
LangGraph workflow then researches the topic, plans content angles, writes
posts, reviews them, and verifies factual claims — with a bounded
Writer ↔ Editor feedback loop rather than a linear prompt chain.

The project's real subject is **agent orchestration**: state management,
conditional routing, feedback loops, observability, and evaluation.

## Status

**Phase 3 (full agent pipeline) — complete.** Five agents run behind
`POST /api/workflows/run`: a Researcher grounded in real web search, a Planner
that assigns each post a distinct angle, and a Writer ↔ Editor ↔ Fact Checker
loop sharing one hard revision budget. Verified end to end against the live
Gemini and Tavily APIs.

| Phase | Scope | State |
|-------|-------|-------|
| 1 | FastAPI + React foundation, config, health check | ✅ done |
| 2 | LangGraph MVP: Writer ↔ Editor loop, max 3 revisions | ✅ done |
| 3 | Real research, Content Planner, Fact Checker, source tracking | ✅ done |
| 4 | SSE streaming, React Flow visualisation, live activity log | next |
| 5 | Latency, token usage, revision and workflow metrics | planned |
| 6 | Single-agent baseline vs multi-agent comparison | planned |

## Stack

**Backend** — Python 3.11, FastAPI, Pydantic v2, pydantic-settings, LangGraph (Phase 2+)
**Frontend** — React 19, TypeScript 6, Vite 8, Tailwind CSS v4, shadcn/ui, React Flow (Phase 4)
**Database** — none by design in the MVP; the architecture keeps PostgreSQL a later, contained addition.

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

Returns HTTP 200 with the generated posts, the Editor's verdict, the revision
count and per-agent metrics. A run that executes but fails also returns 200,
carrying `workflow_status` and structured `errors`; non-2xx is reserved for
invalid input (422) and unusable configuration (503).

### Tests

```powershell
cd backend; .venv\Scripts\python.exe -m pytest tests/
```

190 tests, no network calls and no API key required - both the LLM and search
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
| `TAVILY_API_KEY` | — | [app.tavily.com](https://app.tavily.com), 1,000 free credits/month, no card |
| `TAVILY_MAX_RESULTS` | `5` | Sources fetched per run |
| `GROQ_API_KEY` | — | [console.groq.com](https://console.groq.com), free tier |
| `MAX_REVISIONS` | `3` | Hard cap on Writer ↔ Editor cycles |
| `LLM_TIMEOUT_SECONDS` | `60` | Per-call provider timeout |
| `LLM_MAX_RETRIES` | `2` | Retries on malformed structured output |

No provider key is required for Phase 1; `/api/health` reports
`llm_configured: false`, which is the expected state.

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

## Layout

```
backend/app/
  core/       config, structured logging, domain errors
  schemas/    HTTP request/response contracts
  models/     domain objects used inside graph state
  routes/     API endpoints
  services/   LLM abstraction, metrics, run registry
  agents/     agent implementations (Phase 2+)
  graph/      LangGraph state, nodes, routing (Phase 2+)

frontend/src/
  components/ UI, including generated shadcn/ui primitives
  hooks/      data-fetching and SSE hooks
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
