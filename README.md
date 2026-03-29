# Compound AI Commerce Agent

A compound AI commerce assistant built around a dual-model workflow, hybrid retrieval, and a lightweight evaluation-friendly serving setup. The MVP prioritizes fast iteration with a hosted API LLM for generation, while keeping the system ready to compare against cluster-hosted open-source models later.

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
- **Local Retrieval**: SQLite for structured metadata filters plus LanceDB for vector search
- **Web Search**: SerpApi Google Shopping for structured external product results
- **Memory Tiers**: Redis for short-term session memory and LanceDB for longer-term semantic/conversational memory

## Current MVP Direction

- **Master Brain for MVP**: use an API LLM backend first for faster iteration and demos
- **Open-source comparison path**: serve an open-source LLM through SGLang on a cluster and compare quality/latency against the API LLM path
- **Handyman role**: use the Handyman model not only for routing and query decomposition, but also as the reranker over local and web candidate products
- **Web search path**: use SerpApi Google Shopping instead of generic page scraping so web results arrive in a more structured product format

## Potential Optimizations

- **RadixAttention (KV Caching)**: cache product-description context on the GPU so follow-up turns can reuse prior context instead of recomputing it from scratch
- **EAGLE-3 Speculative Decoding**: let the smaller Handyman model draft tokens for the larger Master Brain, while the Master Brain verifies them in parallel for significantly higher throughput without changing the final output quality

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
