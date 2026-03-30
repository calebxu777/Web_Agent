# MVP Agent API

This document describes the current HTTP API surface for the MVP backend in `mvp/api.py`.

## Base URL

Local development:

```text
http://127.0.0.1:8011
```

VM deployment:

```text
http://<your-vm-host>:8011
```

The backend is a FastAPI app, so interactive docs are also available at:

```text
/docs
/openapi.json
```

## Conventions

- Transport: HTTP
- Auth: none in the current MVP
- Request body format: JSON unless noted otherwise
- Response format: JSON for standard endpoints, Server-Sent Events for `/api/chat`
- Identity model:
  - `user_id` is the stable identity, usually the nickname
  - `session_id` is the short-horizon conversation ID used for Redis-backed session memory

## 1. Health

### `GET /health`

Simple backend liveness check.

Response:

```json
{
  "status": "ok",
  "mode": "mvp"
}
```

## 2. Chat

### `POST /api/chat`

Primary agent endpoint. This is the main MVP shopping assistant API.

Request body:

```json
{
  "message": "recommend me some black hoodies under 80",
  "hasImage": false,
  "imageBase64": null,
  "webSearch": false,
  "session_id": "session-123",
  "user_id": "caleb"
}
```

Fields:

- `message`: user text
- `hasImage`: whether an image is attached
- `imageBase64`: optional base64 image payload for image search
- `webSearch`: whether web retrieval should be enabled for this turn
- `session_id`: session key for short-term Redis memory and worksheet continuity
- `user_id`: stable user identity, typically the nickname

Notes:

- If `user_id` is omitted, the backend falls back to `anon_<session_id>`.
- If `session_id` is omitted, FastAPI generates one, but multi-turn continuity works best when a stable `session_id` is sent.
- In the current frontend deployment, the Next proxy now keeps a browser-session cookie-backed `session_id`.

### Streaming response format

`/api/chat` returns Server-Sent Events.

Each event is streamed like:

```text
data: {"type":"status","stage":"intent_detection","message":"Understanding your request..."}

```

The stream ends with:

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

Structured product payload for the frontend.

Text-search example:

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
      "image": "https://storage.googleapis.com/...",
      "image_url": "https://storage.googleapis.com/...",
      "image_urls": "https://storage.googleapis.com/..."
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

#### `worksheet_state`

Optional backend worksheet state snapshot. This is only emitted when worksheet events are enabled.

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

#### `token`

Streaming assistant text token.

Example:

```json
{
  "type": "token",
  "content": "Here are three great options"
}
```

#### `debug`

Optional timing metadata.

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

#### `error`

Streaming error object if the backend fails mid-stream.

Example:

```json
{
  "type": "error",
  "message": "Server Error: ..."
}
```

#### `done`

The backend also emits a JSON event before the SSE sentinel:

```json
{
  "type": "done"
}
```

The final stream terminator is still:

```text
data: [DONE]
```

## 3. Session Finalization

### `POST /api/session/finalize`

Persists end-of-session artifacts.

Request body:

```json
{
  "session_id": "session-123",
  "user_id": "caleb"
}
```

What it currently does:

- merges Redis session preferences into the durable SQLite preference DB
- appends a local evaluation conversation record
- uploads the preference DB to GCS
- appends the finalized conversation record to GCS
- clears session-scoped worksheet and conversation artifacts

Example response:

```json
{
  "status": "finalized",
  "user_id": "caleb",
  "session_id": "session-123",
  "preferences": {
    "color": ["black"],
    "style": ["simple"]
  },
  "updated_at": 1774892939.92,
  "evaluation_record_path": "data/evaluation/conversation_recordings.jsonl",
  "evaluation_gcs_uri": "gs://web-agent-data-caleb-2026/evaluations/recording.jsonl",
  "evaluation_gcs_error": null,
  "preference_db_gcs_uri": "gs://web-agent-data-caleb-2026/preference/user_preferences.db",
  "preference_db_gcs_error": null
}
```

## 4. Nickname Identity

### `POST /api/nickname`

Creates or reuses a nickname.

Request body:

```json
{
  "nickname": "caleb"
}
```

Example response:

```json
{
  "status": "created",
  "nickname": "caleb",
  "message": "Welcome, caleb! I'll remember your preferences."
}
```

Possible `status` values:

- `created`
- `welcome_back`

Validation:

- nickname length must be 2-30 characters

### `GET /api/nickname/{name}/check`

Checks whether a nickname already exists.

Example response:

```json
{
  "nickname": "caleb",
  "exists": true,
  "message": "This nickname is already taken - you'll be welcomed back!"
}
```

## 5. Ingestion Feedback/Test API

These endpoints support the MVP ingestion feedback path for web products.

### `POST /api/ingest`

Request body:

```json
{
  "product_data": {
    "title": "Sample Product",
    "description": "Sample description",
    "price": 19.99,
    "brand": "Sample Brand",
    "category": "Clothing",
    "image_url": "https://...",
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

Returns all currently stored ingested test products.

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

Deletes all test-ingested products.

Example response:

```json
{
  "status": "cleaned",
  "deleted_count": 2
}
```

### `DELETE /api/ingest/{product_id}`

Deletes one ingested product by ID.

Example response:

```json
{
  "status": "deleted",
  "product_id": "web_ab12cd34ef56"
}
```

If it does not exist:

```json
{
  "status": "not_found",
  "product_id": "web_ab12cd34ef56"
}
```

## 6. Typical API Flow

For a normal MVP session:

1. `POST /api/nickname`
2. `POST /api/chat` one or more times
3. `POST /api/session/finalize`

For a stateless smoke test:

1. `GET /health`
2. `POST /api/chat`

For feedback/ingestion testing:

1. `POST /api/ingest`
2. `GET /api/ingest/verify`
3. `DELETE /api/ingest/{product_id}` or `DELETE /api/ingest/cleanup`

## 7. Current MVP API Caveats

- `/api/chat` is streaming-first, so clients must support SSE-style parsing.
- `session_id` continuity matters for Redis-backed short-horizon memory.
- `user_id` continuity matters for durable preference reuse.
- `worksheet_state` is an optional internal-state event, not a guaranteed public contract for all future clients.
- the MVP currently has no auth or rate limiting layer in front of these endpoints.
