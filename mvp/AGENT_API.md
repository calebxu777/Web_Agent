# MVP Agent API

This document describes the current HTTP API surface for the MVP backend implemented in `mvp/api.py`.

It is written as a handoff reference for reviewers, collaborators, and interview discussions. The API is functional and documented, but it is still an MVP surface rather than a versioned production contract.

## Overview

The MVP backend exposes:

- a streaming chat endpoint for shopping assistance
- nickname-based lightweight identity
- end-of-session finalization for preference persistence and evaluation logging
- test/feedback ingestion endpoints for product capture workflows

## API Status

- Stage: MVP / prototype
- Versioning: currently unversioned
- Auth: none in the current MVP
- Transport: HTTP
- Primary response modes:
  - `application/json` for standard endpoints
  - `text/event-stream` for `POST /api/chat`

## Base URL

Use the base URL for the environment where the backend is deployed.

Examples:

```text
Local development: http://127.0.0.1:8011
Deployed environment: https://<mvp-backend-domain>
```

Notes:

- The exact deployed host should be supplied separately per environment.
- The repository documentation intentionally does not hardcode a live VM host or public IP.
- If FastAPI docs are exposed in the target environment, the following are also available:

```text
/docs
/openapi.json
```

## Conventions

- Request body format: JSON unless otherwise stated
- Time fields: Unix timestamps in seconds
- Identity model:
  - `user_id`: stable identity, usually a nickname
  - `session_id`: short-horizon conversation identifier used for Redis-backed session memory

## Endpoint Summary

| Method | Path | Purpose | Response Type |
| --- | --- | --- | --- |
| `GET` | `/health` | Liveness check | JSON |
| `POST` | `/api/chat` | Main shopping assistant endpoint | SSE |
| `POST` | `/api/session/finalize` | Persist end-of-session artifacts | JSON |
| `POST` | `/api/nickname` | Create or reuse nickname | JSON |
| `GET` | `/api/nickname/{name}/check` | Check nickname availability/existence | JSON |
| `POST` | `/api/ingest` | Store an ingested product in the MVP test DB | JSON |
| `GET` | `/api/ingest/verify` | Inspect stored ingested products | JSON |
| `DELETE` | `/api/ingest/cleanup` | Remove all ingested test products | JSON |
| `DELETE` | `/api/ingest/{product_id}` | Remove one ingested product | JSON |

## 1. Health

### `GET /health`

Simple liveness check for the backend process.

Status:

- `200 OK`

Response:

```json
{
  "status": "ok",
  "mode": "mvp"
}
```

## 2. Chat

### `POST /api/chat`

Primary shopping assistant endpoint. This is the main conversational API for the MVP.

Content type:

- Request: `application/json`
- Response: `text/event-stream`

Status:

- `200 OK` for a successfully opened stream

Request body:

```json
{
  "message": "recommend some black hoodies under 80",
  "hasImage": false,
  "imageBase64": null,
  "webSearch": false,
  "session_id": "session_demo_001",
  "user_id": "demo_user"
}
```

Fields:

- `message`: user text for the current turn
- `hasImage`: whether an image is attached
- `imageBase64`: optional base64-encoded image payload for image search
- `webSearch`: enables local-plus-web retrieval for the current turn
- `session_id`: conversation/session key for Redis-backed short-term memory and worksheet continuity
- `user_id`: stable user identity, typically the chosen nickname

Behavior notes:

- If `user_id` is omitted, the backend falls back to `anon_<session_id>`.
- If `session_id` is omitted, FastAPI generates one. For multi-turn continuity, clients should send a stable `session_id`.
- In the current web app, the Next.js proxy now keeps a browser-session cookie-backed `session_id` so the frontend can reuse Redis-backed short-horizon memory across turns.

### Streaming format

The endpoint streams Server-Sent Events of the form:

```text
data: {"type":"status","stage":"intent_detection","message":"Understanding your request..."}
```

The stream terminates with:

```text
data: [DONE]
```

### SSE event types

#### `status`

Pipeline progress update.

Example:

```json
{
  "type": "status",
  "stage": "reranking",
  "message": "Ranking products by relevance..."
}
```

Common stages:

- `cold_start`
- `intent_detection`
- `decomposing_query`
- `sourcing_local`
- `sourcing_web`
- `reranking`
- `recalling_memory`
- `generating`
- `preparing_compare`
- `analyzing_image`

#### `products`

Structured product payload for the client.

Text search example:

```json
{
  "type": "products",
  "items": [
    {
      "id": "amz_B0B25BDPVP",
      "product_id": "amz_B0B25BDPVP",
      "title": "Columbia Men's CSC Basic Logo Ii Hoodie",
      "price": 60.0,
      "brand": "Columbia",
      "merchant": "Columbia",
      "image": "https://storage.googleapis.com/<bucket>/amazon/amz_B0B25BDPVP.jpg",
      "image_url": "https://storage.googleapis.com/<bucket>/amazon/amz_B0B25BDPVP.jpg",
      "image_urls": "https://storage.googleapis.com/<bucket>/amazon/amz_B0B25BDPVP.jpg"
    }
  ],
  "tags": ["hoodie", "simple"],
  "filters": {
    "price_max": 80,
    "color": "black"
  }
}
```

Compare-mode example:

```json
{
  "type": "products",
  "items": [...],
  "tags": ["price", "reviews"],
  "mode": "compare"
}
```

Image-search example:

```json
{
  "type": "products",
  "items": [...],
  "tags": ["blue", "graphic tee"],
  "caption": "blue graphic short sleeve t-shirt"
}
```

#### `token`

Streaming assistant text token.

Example:

```json
{
  "type": "token",
  "content": "Here are three strong options"
}
```

#### `error`

Structured stream error if the backend fails after the stream has already started.

Example:

```json
{
  "type": "error",
  "message": "Server Error: ..."
}
```

#### `done`

Internal stream-completion event emitted before the final SSE terminator.

Example:

```json
{
  "type": "done"
}
```

#### `worksheet_state`

Optional worksheet snapshot used for the MVP UI when worksheet events are enabled.

This should be treated as MVP-internal/optional behavior rather than a long-term client contract.

Example:

```json
{
  "type": "worksheet_state",
  "worksheet": {
    "name": "product_search",
    "status": "active",
    "values": {
      "product_type": "hoodie",
      "color": "black",
      "price_max": 80,
      "rewritten_query": "black hoodie under $80"
    },
    "missing_required_fields": [],
    "result_counts": {
      "local_count": 10,
      "web_count": 0,
      "reranked_count": 5
    },
    "last_updated_at": 1774891742.18
  }
}
```

#### `debug`

Optional timing metadata, only emitted when debug metadata is enabled.

This is diagnostic output, not a stable client contract.

Example:

```json
{
  "type": "debug",
  "timings": {
    "decomposition": 1.21,
    "local_retrieval": 0.93,
    "generation": 5.84
  }
}
```

## 3. Session Finalization

### `POST /api/session/finalize`

Finalizes a session and persists end-of-session artifacts.

Content type:

- Request: `application/json`
- Response: `application/json`

Status:

- `200 OK`

Request body:

```json
{
  "session_id": "session_demo_001",
  "user_id": "demo_user"
}
```

Current MVP behavior:

- merges Redis session preferences into the durable SQLite preference database
- appends a local evaluation conversation record
- uploads the preference database to GCS
- appends the finalized conversation record to GCS
- clears session-scoped worksheet and conversation artifacts

Example response:

```json
{
  "status": "finalized",
  "user_id": "demo_user",
  "session_id": "session_demo_001",
  "preferences": {
    "color": ["black"],
    "style": ["simple"]
  },
  "updated_at": 1774892939.92,
  "evaluation_record_path": "data/evaluation/conversation_recordings.jsonl",
  "evaluation_gcs_uri": "gs://<bucket>/evaluations/recording.jsonl",
  "evaluation_gcs_error": null,
  "preference_db_gcs_uri": "gs://<bucket>/preference/user_preferences.db",
  "preference_db_gcs_error": null
}
```

Notes:

- The GCS artifact fields are useful operational outputs in the current MVP, but they are implementation-facing rather than core chat-contract fields.

## 4. Nickname Identity

### `POST /api/nickname`

Creates or reuses a nickname.

Content type:

- Request: `application/json`
- Response: `application/json`

Status:

- `200 OK`
- `400 Bad Request` for invalid nickname length

Request body:

```json
{
  "nickname": "demo_user"
}
```

Example success response:

```json
{
  "status": "created",
  "nickname": "demo_user",
  "message": "Welcome, demo_user! I'll remember your preferences."
}
```

Possible `status` values:

- `created`
- `welcome_back`

Validation:

- nickname length must be between 2 and 30 characters

### `GET /api/nickname/{name}/check`

Checks whether a nickname already exists.

Status:

- `200 OK`

Example response:

```json
{
  "nickname": "demo_user",
  "exists": true,
  "message": "This nickname is already taken - you'll be welcomed back!"
}
```

## 5. Ingestion Feedback/Test API

These endpoints support the MVP product-ingestion feedback path used during experimentation.

### `POST /api/ingest`

Stores a product in the MVP ingest test database.

Content type:

- Request: `application/json`
- Response: `application/json`

Status:

- `200 OK`

Request body:

```json
{
  "product_data": {
    "title": "Sample Product",
    "description": "Sample description",
    "price": 19.99,
    "brand": "Sample Brand",
    "category": "Clothing",
    "image_url": "https://merchant.example/product.jpg",
    "url": "https://merchant.example/item"
  }
}
```

Example response:

```json
{
  "status": "ingested",
  "product_id": "web_ab12cd34ef56",
  "title": "Sample Product"
}
```

### `GET /api/ingest/verify`

Returns the currently stored ingested test products.

Status:

- `200 OK`

Example response:

```json
{
  "count": 2,
  "products": [
    {
      "product_id": "web_ab12cd34ef56",
      "title": "Sample Product"
    }
  ]
}
```

### `DELETE /api/ingest/cleanup`

Deletes all ingested test products.

Status:

- `200 OK`

Example response:

```json
{
  "status": "cleaned",
  "deleted_count": 2
}
```

### `DELETE /api/ingest/{product_id}`

Deletes a single ingested product by ID.

Status:

- `200 OK`

Example response:

```json
{
  "status": "deleted",
  "product_id": "web_ab12cd34ef56"
}
```

If the product does not exist:

```json
{
  "status": "not_found",
  "product_id": "web_ab12cd34ef56"
}
```

## 6. Typical Usage Flow

Normal session flow:

1. `POST /api/nickname`
2. `POST /api/chat` one or more times
3. `POST /api/session/finalize`

Stateless smoke test flow:

1. `GET /health`
2. `POST /api/chat`

Feedback/ingestion flow:

1. `POST /api/ingest`
2. `GET /api/ingest/verify`
3. `DELETE /api/ingest/{product_id}` or `DELETE /api/ingest/cleanup`

## 7. Current Caveats

- The API is MVP-stage and currently unversioned.
- `/api/chat` is streaming-first, so clients must support SSE parsing.
- `session_id` continuity matters for Redis-backed short-horizon memory.
- `user_id` continuity matters for durable preference reuse.
- `worksheet_state` and `debug` are optional/internal MVP events and should not be treated as guaranteed long-term contract fields.
- The current MVP has no auth or rate-limiting layer in front of these endpoints.
