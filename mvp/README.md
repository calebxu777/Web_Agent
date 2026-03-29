# MVP Backend

This folder contains a separate MVP backend path that keeps the main `src/` implementation untouched.

The MVP backend will automatically read `mvp/.env` if that file exists.

## MVP Goals

- use an API LLM for the Master Brain
- use an API LLM for routing, query decomposition, and reranking
- keep the same overall workflow shape as the main backend
- use GCS-backed catalog/images as the default data source
- use SerpApi Google Shopping for external web product results

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
- `SERPAPI_API_KEY`
- `SERPAPI_MOCK_RESULTS_PATH` for mock web-search testing
- `SERPAPI_LOCATION`, `SERPAPI_GL`, `SERPAPI_HL` (optional)
- `MVP_USE_MEMORY` (optional, default: `true`)
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
- the main backend in `src/` is unchanged

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
