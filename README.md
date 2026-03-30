# Compound AI Commerce Agent

A compound AI commerce assistant built around a dual-model workflow, hybrid retrieval, and a lightweight evaluation-friendly serving setup. The MVP prioritizes fast iteration with a hosted API LLM for generation, while keeping the system ready to compare against cluster-hosted open-source models later.

Live frontend: [web-agent-pi.vercel.app](https://web-agent-pi.vercel.app/)

Frontend setup and deployment notes: [frontend/README.md](frontend/README.md)

## MVP Backend Status

The production-ready backend path right now is the MVP implementation. For setup, environment variables, and deployment details, refer to [mvp/README.md](mvp/README.md).

The main backend in `src/` is still in active testing and model-training/comparison work, so it should be treated as experimental for now.

## What It Offers

- Visit the live frontend: [web-agent-pi.vercel.app](https://web-agent-pi.vercel.app/)
- Ask general shopping or product-related questions in chat
- Run text-based product search
- Run image-based product search by uploading a product photo
- Toggle between local catalog search and local + web search
- Set a nickname so the system can remember your preferences across sessions

## Workflow

```text
User / Frontend
  -> Next.js UI + SSE
  -> FastAPI Backend
  -> Handyman
     intent routing + query decomposition + reranking
```

<table>
  <tr>
    <th align="left" width="33%">General Talk</th>
    <th align="left" width="34%">Text-Based Search</th>
    <th align="left" width="33%">Image-Based Search</th>
  </tr>
  <tr>
    <td valign="top">
      <strong>Route</strong><br>
      <code>general_talk</code><br><br>
      <strong>Flow</strong><br>
      Memory recall
    </td>
    <td valign="top">
      <strong>Route</strong><br>
      <code>text_search</code><br><br>
      <strong>Flow</strong><br>
      Local catalog retrieval<br>
      <em>SQLite + LanceDB</em><br><br>
      Web product search<br>
      <em>SerpApi Google Shopping</em><br><br>
      Fuse ranked local + web candidates
    </td>
    <td valign="top">
      <strong>Route</strong><br>
      <code>image_search</code><br><br>
      <strong>Flow</strong><br>
      Image understanding<br><br>
      Local visual retrieval<br><br>
      Optional web product search
    </td>
  </tr>
</table>

```text
All routes
  -> Memory
     Redis episodic + LanceDB semantic/conversational
  -> Master Brain
     MVP: API LLM | Later: post-trained OSS comparison
  -> Streamed response + product cards
```

## Architecture Highlights

- **Frontend**: Next.js chat app with SSE streaming and a state-aware status pipeline
- **Backend**: FastAPI orchestration layer for routing, retrieval, memory, and streaming
- **Handyman Model**: small model for intent classification, query decomposition, and reranking
- **Master Brain**: primary synthesis model for final responses and recommendation writing
- **Worksheet Orchestration (MVP)**: session-scoped worksheet state for multi-turn product search, clarification, and compare flows
- **Preference Memory (MVP)**: single-pass turn analysis that infers user preferences during routing, caches them per session, and reuses them during reranking and session finalization
- **Evaluation Pipeline (MVP)**: conversation recordings, inferred preferences, and finalized preference snapshots can be stored locally and synced to GCS, supporting offline quality review now and future DPO-style data creation for testing open-source replacements for the hosted LLM path
- **Grounded Agent Acts (Optional MVP)**: structured response acts for report, recommend, and compare generation, enabled through MVP env switches
- **Catalog RAG Retrieval**: SQLite-backed metadata filtering plus LanceDB vector search over the product catalog
- **Web Search**: SerpApi Google Shopping for structured external product results
- **Memory Tiers**: Redis for short-term session memory and LanceDB for longer-term semantic/conversational memory

## Current MVP Direction

- **Master Brain for MVP**: use an API LLM backend first for faster iteration and demos
- **Open-source comparison path**: serve an open-source LLM through SGLang on a cluster and compare quality/latency against the API LLM path
- **Evaluation-to-training loop**: use the MVP evaluation pipeline to collect conversation traces, preferences, and critique signals that can later be curated into DPO-style training data for open-source Master Brain experiments
- **Handyman role**: use the Handyman model not only for routing and query decomposition, but also as the reranker over local and web candidate products
- **Web search path**: use SerpApi Google Shopping instead of generic page scraping so web results arrive in a more structured product format

## Potential Optimizations

- **RadixAttention (KV Caching)**: cache product-description context on the GPU so follow-up turns can reuse prior context instead of recomputing it from scratch
- **EAGLE-3 Speculative Decoding**: let the smaller Handyman model draft tokens for the larger Master Brain, while the Master Brain verifies them in parallel for significantly higher throughput without changing the final output quality

## Datasets Used

- **Amazon (2023/2024)**: the main commerce dataset for product titles, descriptions, technical specs, reviews, and Rufus-style product metadata
- **H&M Fashion**: the main fashion-image dataset for higher-quality product photography and finer-grained visual search on texture, silhouette, and pattern

Together, these datasets let the repo combine:

- strong product metadata and reviews for reasoning-heavy text search
- stronger fashion imagery for visual retrieval and image-based search

## Data And Embedding Flow

The main offline pipeline is:

1. Download raw datasets
2. Clean and enrich the raw product records
3. Normalize products into a unified SQLite catalog
4. Generate semantic embeddings with `BGE-M3`
5. Generate visual embeddings with `DINOv2`
6. Store vectors in LanceDB for retrieval

The important scripts are:

- `scripts/data_processing/00_download_raw_datasets.py`: download the raw Amazon and H&M source data
- `scripts/data_processing/00_clean_and_enrich.py`: clean and enrich raw product records before indexing
- `scripts/create_db/01_normalize_catalog.py`: build the canonical SQLite catalog
- `scripts/create_db/02_extract_embeddings.py`: create semantic and visual embeddings and write them into LanceDB

## Repo Guide

### `scripts/`

- `scripts/create_db/`: catalog normalization and embedding/index build steps
- `scripts/data_processing/`: raw dataset download and preprocessing
- `scripts/deploy/start_production.sh`: deployment helper for serving flows
- `scripts/evaluation/`: question generation, evaluation runs, and quality reporting
- `scripts/post_training/`: SFT, DPO, Handyman LoRA data generation, training, and quantization
- `scripts/SGLang/`: launch helpers for cluster or post-training SGLang serving
- `scripts/test/`: data checks and retrieval smoke tests, including mock SerpApi fixtures

### `mvp/`

- `mvp/api.py`: separate FastAPI entrypoint for the MVP backend
- `mvp/agent.py`: MVP orchestration logic, retrieval fusion, product normalization, and streaming workflow
- `mvp/router.py`: API-LLM intent detection, query decomposition, and reranking
- `mvp/README.md`: MVP-specific setup and deployment notes
- `mvp/.env.example`: example environment variables for API-provider MVP runs

## Project Structure

- `config/` - central application configuration
- `scripts/` - data prep, evaluation, deployment, and test utilities
- `src/` - backend orchestration, retrieval, memory, and model client logic
- `frontend/` - Next.js chat application

## Running the Application

### Backend Setup

1. Set up the Python environment
2. Start Redis for episodic memory
3. Prepare local catalog databases and embeddings
4. Start the FastAPI backend
5. Point the frontend to the backend port you are using

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Testing

You can run lightweight offline tests without starting the backend server.

These tests cover core MVP behavior such as:

- general-talk, text-search, and image-search route detection
- local product image normalization
- local category-filter sanitizing
- keyword-based ranking boosts
- web-result cap configuration

Run them from the repo root:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Backend Modes

- **Local / Mock**: local FastAPI app, local retrieval, mock SerpApi results for safe testing
- **Cluster-tunneled**: local FastAPI app, cluster-hosted SGLang model servers tunneled to localhost
- **API-provider**: local FastAPI app, external API LLM backend for the Master Brain path

## Frontend Features

- Text chat with streamed responses
- Product cards for local and web results
- Optional web-search toggle
- Image upload flow for visual search
- Nickname-based session identity
