# MVP Backend

This folder contains a separate MVP backend path that keeps the main `src/` implementation untouched.

The MVP backend will automatically read `mvp/.env` if that file exists.

## MVP Goals

- use an API LLM for the Master Brain
- use an API LLM for routing, query decomposition, and reranking
- keep the same overall workflow shape as the main backend
- use GCS-backed catalog/images as the default data source
- use SerpApi Google Shopping for external web product results

## MVP Design Overview

The MVP keeps the main `src/` backend untouched and concentrates the experimental serving path inside `mvp/`. The design goal is to preserve the same high-level compound-agent flow while making it easier to swap in API models, iterate on orchestration, and test product-search behaviors without retraining local models first.

### Turn Analysis And Routing

Every user turn starts with a router pass in `router.py`. That pass classifies the turn, decomposes the request into structured retrieval fields, and can optionally infer lightweight user preferences in the same LLM call. This keeps preference inference on the same latency budget as routing instead of paying for a second extraction request.

### Retrieval And Result Fusion

`agent.py` orchestrates local retrieval from SQLite plus LanceDB, optional Google Shopping web retrieval through SerpApi, and product normalization into a single response shape for the frontend. The current-turn structured filters always take precedence, and the reranker can optionally see a compact preference context so long-term tastes help sort candidates without overriding the user’s explicit request.

### Worksheet State

When worksheets are enabled, the MVP can keep session-scoped search state across turns. That lets the backend ask for missing constraints, hold onto compare candidates, and continue the same shopping task instead of treating each message as a fully independent search.

### Agent Acts

When agent acts are enabled, the final response can be generated through structured output acts such as recommend, compare, or report. This gives the response layer tighter formatting control while still using the same retrieval and routing backbone underneath.

### Preference Memory

Preference memory is intentionally lightweight. Turn-level preferences inferred during routing are merged into a session profile in Redis, and `POST /api/session/finalize` can persist the merged profile into SQLite for longer-term reuse by nickname or user identity. The preference layer is designed to influence reranking and continuity, not to replace explicit query constraints from the current turn.

## How To Use

This is the current repo-level path for running and testing the MVP backend.

### 1. Configure the MVP

From the repo root, copy [`mvp/.env.example`](.env.example) to `mvp/.env` and fill in your real keys.

At minimum, set:

- `OPENAI_API_KEY`
- `SERPAPI_API_KEY` if you want web search

Feature flags you will likely want to toggle while testing:

- `MVP_USE_WORKSHEETS=true` to keep multi-turn worksheet state
- `MVP_USE_AGENT_ACTS=true` to use grounded response acts
- `MVP_USE_PREFERENCE_INFERENCE=true` to infer turn-level preferences
- `MVP_USE_PREFERENCE_RERANKING=true` to let reranking reuse stored preferences
- `MVP_EMIT_WORKSHEET_EVENTS=false` if you want the worksheet logic active in the backend without showing the worksheet panel on the frontend

### 2. Start Redis If Memory Is Enabled

If `MVP_USE_MEMORY=true`, start Redis before launching the backend. The MVP also uses Redis-backed session state for short-horizon memory and preference/session caching when available.

### 3. Run The MVP Backend

From the repo root:

```powershell
& "C:\Users\Caleb\miniconda3\envs\commerce-agent\python.exe" -m uvicorn mvp.api:app --host 127.0.0.1 --port 8011
```

Quick backend check:

```powershell
curl http://127.0.0.1:8011/health
```

### 4. Run The Frontend

In a separate terminal:

```powershell
cd frontend
npm install
npm run dev
```

Then open `http://localhost:3000`.

### 5. Use The MVP In The UI

The current frontend supports:

- general shopping chat
- text-based search
- image-based search
- local vs local-plus-web search toggle
- nickname-based identity

Typical test flow:

1. Set a nickname so the backend can associate memory and finalized preferences with a stable user ID.
2. Ask for products with text, for example `recommend me some black hoodies under $80`.
3. Try follow-up constraints, comparison requests, or image uploads.
4. Toggle web search on if you want local-plus-web product results instead of catalog-only results.

### 6. Finalize A Session When You Want To Persist It

At the end of a conversation, call:

```powershell
curl -X POST http://127.0.0.1:8011/api/session/finalize `
  -H "Content-Type: application/json" `
  -d "{\"session_id\":\"YOUR_SESSION_ID\",\"user_id\":\"YOUR_USER_ID\"}"
```

That finalize step is where the MVP currently:

- merges session preferences into the durable SQLite preference DB
- appends a local conversation record to `data/evaluation/conversation_recordings.jsonl`
- uploads the preference DB to GCS under `preference/`
- appends the finalized conversation record to GCS under `evaluations/recording.jsonl`

### 7. Run Offline Evaluation On Recorded Conversations

After conversations have been recorded, you can evaluate them locally or from GCS:

```powershell
& "C:\Users\Caleb\miniconda3\envs\commerce-agent\python.exe" -m mvp.evaluation.evaluator `
  --input gs://web-agent-data-caleb-2026/evaluations/recording.jsonl `
  --mode both `
  --output mvp/evaluation/evaluation_report.json
```

Use `--mode python` for deterministic rubric scoring, `--mode llm` for API-based critic scoring, or `--mode both` to generate both views in the same report.

## MVP Limitations

- **Limited context awareness**: the current MVP is still weak at conversational carryover when the next turn depends on implied references instead of explicit restatement. For example, if the user first says `recommend me some jeans` and then follows with `recommend me some blue ones`, the system may not reliably resolve what `ones` refers to.
- **Worksheet structure is for the backend, not the user**: the worksheet is meant to help the router, retrieval stack, and response generation stay structured around catalog-supported fields, but user turns should still remain permissive, especially when web search may surface products with looser or different schemas.
- **Small local catalog**: the local product database is still relatively small, so retrieval quality and variety are constrained by coverage, not just model quality. some data does not have image to it.
- **Latency is still heavier than the long-term target**: right now the MVP can spend expensive model time on routing, reranking, and other small subtasks that do not necessarily need a larger API model.
  

## Planned Improvements

- **Recent-turn caching in Redis**: cache roughly the most recent 10 rounds so the system can better resolve short-horizon follow-ups and references before falling back to longer-term memory.
- **Periodic conversation summaries into LanceDB**: after each 10-round block, write a summary of the current conversation state into LanceDB so the agent can recover shopping context and preferences beyond the immediate Redis window.
- **Tokenizer-aware memory compression**: use tokenizer-bounded summarization so these longer-term memory writes stay compact, retrieval-friendly, and cheap enough to maintain over time.
- **Broader catalog coverage**: expand the local database so local retrieval has better recall before leaning on web search.
- **Smaller specialized models through SGLang**: allocate cheaper tasks such as routing, query decomposition, reranking, and similar helper work to tiny open-source models served through SGLang, while reserving the larger model budget for synthesis-heavy turns.

## Key Files

- [`api.py`](api.py): FastAPI entrypoint for the MVP backend
- [`agent.py`](agent.py): MVP orchestrator
- [`router.py`](router.py): API-LLM router and reranker

## Environment Variables

Start by copying [`mvp/.env.example`](.env.example) to `mvp/.env` and filling in your real keys.

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional, defaults to OpenAI)
- `MVP_MASTER_BRAIN_MODEL` (optional, default: `gpt-4o-mini`)
- `MVP_ROUTER_MODEL` (optional, default: `gpt-4o-mini`)
- `MVP_RERANKER_MODEL` (optional, default: same as router)
- `MVP_USE_WORKSHEETS` (optional: `true` or `false`; default: `false`)
- `MVP_USE_AGENT_ACTS` (optional: `true` or `false`; default: `false`)
- `MVP_ACT_MODE` (optional: `dynamic` or `hardcoded`; default: `dynamic` when acts are enabled)
- `SERPAPI_API_KEY`
- `SERPAPI_MOCK_RESULTS_PATH` for mock web-search testing
- `SERPAPI_LOCATION`, `SERPAPI_GL`, `SERPAPI_HL` (optional)
- `MVP_WEB_NUM_RESULTS` (optional, default: `1`)
- `MVP_USE_MEMORY` (optional, default: `true`)
- `MVP_USE_PREFERENCE_INFERENCE` (optional, default: `false`)
- `MVP_USE_PREFERENCE_RERANKING` (optional, default: `false`)
- `MVP_PREFERENCE_REDIS_TTL_SECONDS` (optional, default: `3600`)
- `MVP_USER_PREFERENCES_DB_PATH` (optional, default: `data/processed/user_preferences.db`)
- `MVP_GCS_CATALOG_DB_URL` (optional, defaults to the GCS `catalog.db` path)
- `MVP_GCS_LANCEDB_PUBLIC_PREFIX` (optional, defaults to the public `data/processed/lancedb/` prefix in the same bucket)
- `MVP_GCS_LANCEDB_MANIFEST_URL` (optional, overrides prefix listing with an explicit JSON manifest of LanceDB files)

## Run The MVP Backend

```powershell
& "C:\Users\Caleb\miniconda3\envs\commerce-agent\python.exe" -m uvicorn mvp.api:app --host 127.0.0.1 --port 8011
```

## Docker On A GCP L4 VM

This is the recommended direction once you move the MVP off your laptop.

Current first-pass stack:

- `mvp-backend`: FastAPI app, Gemini/OpenAI-compatible routing + generation, local embedders, LanceDB hydrate logic
- `redis`: short-term session memory

Files:

- [`../Dockerfile.mvp`](../Dockerfile.mvp)
- [`../docker-compose.mvp.yml`](../docker-compose.mvp.yml)

Bring it up on the VM:

```bash
docker compose -f docker-compose.mvp.yml up --build -d
```

View logs:

```bash
docker compose -f docker-compose.mvp.yml logs -f mvp-backend
```

Stop it:

```bash
docker compose -f docker-compose.mvp.yml down
```

Notes:

- `mvp/.env` is injected at runtime through Compose and is excluded from the Docker build context
- `./data` is mounted into the container, so hydrated `catalog.db` and LanceDB caches persist on the VM
- Hugging Face and Torch caches are stored in Docker volumes to avoid repeated model downloads
- this first pass keeps embedders in the backend container; later we can split out `vision`, `handyman`, and OSS LLM services if you want cleaner isolation
- for A/B testing on the VM, flip `MVP_USE_WORKSHEETS` and `MVP_USE_AGENT_ACTS` in `mvp/.env`, then restart the MVP backend container

## Run With Mock Web Search

```powershell
$env:OPENAI_API_KEY="your-openai-key"
$env:SERPAPI_MOCK_RESULTS_PATH="scripts/test/mock_serpapi_google_shopping.json"
& "C:\Users\Caleb\miniconda3\envs\commerce-agent\python.exe" -m uvicorn mvp.api:app --host 127.0.0.1 --port 8011
```

## Notes

- the MVP backend can download `catalog.db` from GCS automatically if the local SQLite file is missing
- the MVP backend can also hydrate the local LanceDB cache from GCS automatically when `data/processed/lancedb` is missing
- product images are served as public GCS URLs in MVP responses
- Redis is still required if `MVP_USE_MEMORY=true`
- when preference inference is enabled, extracted turn-level preferences are cached in Redis under the active session
- when session finalization is called, merged durable preferences are stored in `user_preferences.db`
- the main backend in `src/` is unchanged

## Preference Memory MVP

The MVP can now do a lightweight preference pass without changing worksheet state:

- when preference inference is enabled, the router runs a single turn-analysis LLM call that returns intent, decomposition fields, and inferred preferences together
- the backend merges those inferred preferences into the session store before continuing the rest of the turn
- extracted preferences are merged into a session-scoped Redis cache
- search reranking can inject a compact preference context block into the reranker prompt
- calling `POST /api/session/finalize` persists the merged session profile into SQLite and clears the session cache

The reranker keeps current-turn constraints above stored preferences. If the user normally likes red but asks for black shirts right now, the reranker prompt explicitly tells the model to follow the current query.

## LanceDB Manifest Format

If public bucket-prefix listing is not available, point `MVP_GCS_LANCEDB_MANIFEST_URL` at a JSON file shaped like this:

```json
{
  "files": [
    {
      "path": "product_vectors.lance/_versions/18446744073709551614.manifest",
      "url": "https://storage.googleapis.com/web-agent-data-caleb-2026/data/processed/lancedb/product_vectors.lance/_versions/18446744073709551614.manifest"
    }
  ]
}
```
