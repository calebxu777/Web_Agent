"""
api.py - MVP FastAPI backend
============================
Separate MVP entrypoint that keeps the current backend intact while using an
API-LLM-based Master Brain/router stack.
"""

import base64
import json
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from mvp.agent import MVPCommerceAgent, MVPConfig


agent = None
config = {}
nickname_db = None
ingest_db = None


def resolve_env_flag(raw_value: str | None, default: bool = False) -> bool:
    if raw_value is None or not raw_value.strip():
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_mvp_act_mode(raw_value: str | None, use_agent_acts: bool = False) -> str:
    if not use_agent_acts:
        return "off"

    value = (raw_value or "").strip().lower()
    if value in {"dynamic", "hardcoded"}:
        return value
    return "dynamic"


def load_env_file(env_path: Path):
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


class NicknameDB:
    def __init__(self, db_path: str = "data/nicknames.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nicknames (
                nickname    TEXT PRIMARY KEY COLLATE NOCASE,
                created_at  REAL,
                last_seen   REAL
            )
            """
        )
        self.conn.commit()

    def get_or_create(self, nickname: str) -> dict:
        row = self.conn.execute(
            "SELECT * FROM nicknames WHERE nickname = ? COLLATE NOCASE",
            (nickname,),
        ).fetchone()
        now = time.time()
        if row:
            self.conn.execute(
                "UPDATE nicknames SET last_seen = ? WHERE nickname = ? COLLATE NOCASE",
                (now, nickname),
            )
            self.conn.commit()
            return {"status": "welcome_back", "nickname": dict(row)["nickname"]}

        self.conn.execute(
            "INSERT INTO nicknames (nickname, created_at, last_seen) VALUES (?, ?, ?)",
            (nickname, now, now),
        )
        self.conn.commit()
        return {"status": "created", "nickname": nickname}

    def check_exists(self, nickname: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM nicknames WHERE nickname = ? COLLATE NOCASE",
            (nickname,),
        ).fetchone()
        return row is not None


class IngestTestDB:
    def __init__(self, db_path: str = "data/ingest_test.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingested_products (
                product_id    TEXT PRIMARY KEY,
                title         TEXT,
                description   TEXT,
                price         REAL,
                brand         TEXT,
                category      TEXT,
                image_url     TEXT,
                source_url    TEXT,
                ingested_at   REAL
            )
            """
        )
        self.conn.commit()

    def insert(self, product: dict) -> str:
        import hashlib

        url = product.get("url", product.get("source_url", ""))
        pid = f"web_{hashlib.sha256(url.encode()).hexdigest()[:12]}"
        now = time.time()
        self.conn.execute(
            """
            INSERT OR REPLACE INTO ingested_products
            (product_id, title, description, price, brand, category, image_url, source_url, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pid,
                product.get("title", ""),
                product.get("description", ""),
                product.get("price"),
                product.get("brand", ""),
                product.get("category", ""),
                product.get("image", product.get("image_url", "")),
                url,
                now,
            ),
        )
        self.conn.commit()
        return pid

    def list_all(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM ingested_products").fetchall()
        return [dict(r) for r in rows]

    def delete(self, product_id: str) -> bool:
        cur = self.conn.execute("DELETE FROM ingested_products WHERE product_id = ?", (product_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def clear(self) -> int:
        cur = self.conn.execute("DELETE FROM ingested_products")
        self.conn.commit()
        return cur.rowcount


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, config, nickname_db, ingest_db
    print("[MVP FastAPI] Starting Server...")

    load_env_file(Path("mvp/.env"))

    config_path = Path("config/settings.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    gcs_public_url = config.get("image_storage", {}).get("gcs", {}).get(
        "public_url_prefix",
        "https://storage.googleapis.com/web-agent-data-caleb-2026",
    )
    use_worksheets = resolve_env_flag(os.environ.get("MVP_USE_WORKSHEETS"), default=False)
    use_agent_acts = resolve_env_flag(os.environ.get("MVP_USE_AGENT_ACTS"), default=False)
    use_preference_inference = resolve_env_flag(
        os.environ.get("MVP_USE_PREFERENCE_INFERENCE"),
        default=False,
    )
    use_preference_reranking = resolve_env_flag(
        os.environ.get("MVP_USE_PREFERENCE_RERANKING"),
        default=False,
    )

    agent_config = MVPConfig(
        master_brain_model_name=os.environ.get("MVP_MASTER_BRAIN_MODEL", "gpt-4o-mini"),
        router_model_name=os.environ.get("MVP_ROUTER_MODEL", "gpt-4o-mini"),
        reranker_model_name=os.environ.get(
            "MVP_RERANKER_MODEL",
            os.environ.get("MVP_ROUTER_MODEL", "gpt-4o-mini"),
        ),
        api_base=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        use_florence=os.environ.get("MVP_USE_FLORENCE", "").lower() in {"1", "true", "yes"},
        use_web_search=bool(
            os.environ.get("SERPAPI_API_KEY", "").strip()
            or os.environ.get("SERPAPI_MOCK_RESULTS_PATH", "").strip()
        ),
        use_visual_verifier=os.environ.get("MVP_USE_VISUAL_VERIFIER", "").lower() in {"1", "true", "yes"},
        serpapi_api_key=os.environ.get("SERPAPI_API_KEY", ""),
        serpapi_api_base=os.environ.get("SERPAPI_API_BASE", "https://serpapi.com"),
        serpapi_location=os.environ.get("SERPAPI_LOCATION", ""),
        serpapi_gl=os.environ.get("SERPAPI_GL", "us"),
        serpapi_hl=os.environ.get("SERPAPI_HL", "en"),
        serpapi_mock_results_path=os.environ.get("SERPAPI_MOCK_RESULTS_PATH", ""),
        web_num_results=max(1, int(os.environ.get("MVP_WEB_NUM_RESULTS", "1"))),
        use_memory=os.environ.get("MVP_USE_MEMORY", "true").lower() not in {"0", "false", "no"},
        use_preference_inference=use_preference_inference,
        use_preference_reranking=use_preference_reranking,
        preference_redis_ttl_seconds=int(os.environ.get("MVP_PREFERENCE_REDIS_TTL_SECONDS", "3600")),
        user_preferences_db_path=os.environ.get(
            "MVP_USER_PREFERENCES_DB_PATH",
            "data/processed/user_preferences.db",
        ),
        image_storage_provider="gcs",
        gcs_public_url=gcs_public_url,
        log_timing=True,
        catalog_db_url=os.environ.get(
            "MVP_GCS_CATALOG_DB_URL",
            f"{gcs_public_url.rstrip('/')}/metadata/catalog.db",
        ),
        lancedb_public_prefix=os.environ.get(
            "MVP_GCS_LANCEDB_PUBLIC_PREFIX",
            f"{gcs_public_url.rstrip('/')}/data/processed/lancedb",
        ),
        lancedb_manifest_url=os.environ.get("MVP_GCS_LANCEDB_MANIFEST_URL", ""),
        use_worksheets=use_worksheets,
        use_agent_acts=use_agent_acts,
        act_mode=resolve_mvp_act_mode(
            os.environ.get("MVP_ACT_MODE"),
            use_agent_acts=use_agent_acts,
        ),
    )

    agent = MVPCommerceAgent(config=config, agent_config=agent_config)
    nickname_db = NicknameDB()
    ingest_db = IngestTestDB()
    print("[MVP FastAPI] NicknameDB + IngestTestDB ready")

    yield
    print("[MVP FastAPI] Shutting down...")


app = FastAPI(title="Commerce Agent MVP API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = ""
    hasImage: bool = False
    imageBase64: str | None = None
    webSearch: bool = False
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""


class NicknameRequest(BaseModel):
    nickname: str


class IngestRequest(BaseModel):
    product_data: dict


class SessionFinalizeRequest(BaseModel):
    session_id: str
    user_id: str = ""


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "mode": "mvp"})


@app.post("/api/nickname")
async def set_nickname(req: NicknameRequest):
    name = req.nickname.strip()
    if not name or len(name) < 2 or len(name) > 30:
        return JSONResponse({"error": "Nickname must be 2-30 characters"}, status_code=400)

    result = nickname_db.get_or_create(name)
    messages = {
        "created": f"Welcome, {result['nickname']}! I'll remember your preferences.",
        "welcome_back": f"Welcome back, {result['nickname']}!",
    }
    return JSONResponse({**result, "message": messages[result["status"]]})


@app.get("/api/nickname/{name}/check")
async def check_nickname(name: str):
    exists = nickname_db.check_exists(name.strip())
    return JSONResponse(
        {
            "nickname": name.strip(),
            "exists": exists,
            "message": "This nickname is already taken - you'll be welcomed back!" if exists else "Available!",
        }
    )


@app.post("/api/ingest")
async def ingest_product(req: IngestRequest):
    product = req.product_data
    pid = ingest_db.insert(product)
    return JSONResponse({"status": "ingested", "product_id": pid, "title": product.get("title", "")})


@app.get("/api/ingest/verify")
async def verify_ingested():
    products = ingest_db.list_all()
    return JSONResponse({"count": len(products), "products": products})


@app.delete("/api/ingest/cleanup")
async def cleanup_ingested():
    count = ingest_db.clear()
    return JSONResponse({"status": "cleaned", "deleted_count": count})


@app.delete("/api/ingest/{product_id}")
async def delete_ingested(product_id: str):
    deleted = ingest_db.delete(product_id)
    return JSONResponse({"status": "deleted" if deleted else "not_found", "product_id": product_id})


@app.post("/api/session/finalize")
async def finalize_session(req: SessionFinalizeRequest):
    user_id = req.user_id.strip() if req.user_id else f"anon_{req.session_id}"
    result = await agent.finalize_session(user_id=user_id, session_id=req.session_id)
    return JSONResponse(result)


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    user_id = req.user_id.strip() if req.user_id else f"anon_{req.session_id}"

    image_bytes = None
    if req.hasImage and req.imageBase64:
        try:
            clean_b64 = req.imageBase64.split(",")[-1]
            image_bytes = base64.b64decode(clean_b64)
        except Exception as e:
            print(f"Failed to decode base64 image: {e}")

    async def sse_generator():
        try:
            async for chunk in agent.handle_message(
                user_id=user_id,
                session_id=req.session_id,
                message=req.message,
                image_bytes=image_bytes,
                web_search_enabled=req.webSearch,
            ):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            import traceback

            traceback.print_exc()
            error_json = json.dumps({"type": "error", "message": f"Server Error: {str(e)}"})
            yield f"data: {error_json}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mvp.api:app", host="0.0.0.0", port=8011, reload=True)
