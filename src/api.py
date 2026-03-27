"""
api.py — FastAPI Backend for the Commerce Agent
=================================================
Serves the SSE streaming endpoint, nickname system, and ingestor.
Supports mock_mode for local laptop testing without GPU/Redis/LanceDB.
"""

import base64
import json
import os
import sqlite3
import time
import uuid
import yaml
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agent import CommerceAgent, AgentConfig

# Global Agent State
agent = None
config = {}
nickname_db = None  # SQLite connection for nicknames


# -----------------------------------------------------------------
# Environment-based overrides
# -----------------------------------------------------------------
MOCK_MODE = os.environ.get("MOCK_MODE", "1") != "0"


# -----------------------------------------------------------------
# Nickname Database (lightweight SQLite — separate from catalog)
# -----------------------------------------------------------------
class NicknameDB:
    """Simple SQLite store for nickname-based user identity."""

    def __init__(self, db_path: str = "data/nicknames.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS nicknames (
                nickname    TEXT PRIMARY KEY COLLATE NOCASE,
                created_at  REAL,
                last_seen   REAL
            )
        """)
        self.conn.commit()

    def get_or_create(self, nickname: str) -> dict:
        """Returns (status, nickname). Status is 'created' or 'welcome_back'."""
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
        else:
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


# -----------------------------------------------------------------
# Ingestor Test DB (lightweight SQLite for mock ingest testing)
# -----------------------------------------------------------------
class IngestTestDB:
    """Tiny SQLite to test the ingestor pipeline without the full catalog."""

    def __init__(self, db_path: str = "data/ingest_test.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("""
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
        """)
        self.conn.commit()

    def insert(self, product: dict) -> str:
        import hashlib
        url = product.get("url", product.get("source_url", ""))
        pid = f"web_{hashlib.sha256(url.encode()).hexdigest()[:12]}"
        now = time.time()
        self.conn.execute("""
            INSERT OR REPLACE INTO ingested_products
            (product_id, title, description, price, brand, category, image_url, source_url, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pid,
            product.get("title", ""),
            product.get("description", ""),
            product.get("price"),
            product.get("brand", ""),
            product.get("category", ""),
            product.get("image", product.get("image_url", "")),
            url,
            now,
        ))
        self.conn.commit()
        return pid

    def list_all(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM ingested_products").fetchall()
        return [dict(r) for r in rows]

    def delete(self, product_id: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM ingested_products WHERE product_id = ?", (product_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def clear(self) -> int:
        cur = self.conn.execute("DELETE FROM ingested_products")
        self.conn.commit()
        return cur.rowcount


# Global instances
ingest_db = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hook for startup/shutdown."""
    global agent, config, nickname_db, ingest_db
    print("[FastAPI] Starting Server...")

    config_path = Path("config/settings.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config at {config_path.absolute()}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Read the image path and provider from settings
    provider = config.get("image_storage", {}).get("provider", "local")
    local_path = config.get("image_storage", {}).get("local", {}).get("base_path", "")

    # Create the Agent configuration for MVP
    ac = AgentConfig(
        master_brain_model_name=config["models"]["master_brain"]["model_id"],
        handyman_model_name=config["models"]["handyman"]["model_id"],
        use_florence=False,
        use_visual_verifier=False,
        use_web_search=False,
        image_storage_provider=provider,
        local_image_base_path=local_path,
        log_timing=True,
        mock_mode=MOCK_MODE,
    )

    print(f"[FastAPI] mock_mode={ac.mock_mode}")
    agent = CommerceAgent(config=config, agent_config=ac)

    # Initialize lightweight DBs (always available, even in mock mode)
    nickname_db = NicknameDB()
    ingest_db = IngestTestDB()
    print("[FastAPI] NicknameDB + IngestTestDB ready")

    yield
    print("[FastAPI] Shutting down...")


# Initialize the App
app = FastAPI(title="Commerce Agent API", lifespan=lifespan)

# Allow the Next.js frontend (port 3000) to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------
# Toggle Integration: Serving Local Imagery
# -----------------------------------------------------------------
try:
    with open(Path("config/settings.yaml"), "r", encoding="utf-8") as _f:
        _temp_cfg = yaml.safe_load(_f)
        if _temp_cfg.get("image_storage", {}).get("provider") == "local":
            _base_path = _temp_cfg["image_storage"]["local"]["base_path"]
            if Path(_base_path).exists():
                print(f"[FastAPI] Mounting local images at /api/images from {_base_path}")
                app.mount("/api/images", StaticFiles(directory=_base_path), name="images")
            else:
                print(f"[FastAPI] Local image path {_base_path} not found — skipping mount (OK in mock mode)")
except Exception as e:
    print(f"[FastAPI] Image mount skipped: {e}")


# -----------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = ""
    hasImage: bool = False
    imageBase64: str | None = None
    webSearch: bool = False
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""  # Empty = anonymous, set from nickname on frontend


class NicknameRequest(BaseModel):
    nickname: str


class IngestRequest(BaseModel):
    product_data: dict


# -----------------------------------------------------------------
# Health Check
# -----------------------------------------------------------------
@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "mock_mode": MOCK_MODE})


# -----------------------------------------------------------------
# Nickname Endpoints
# -----------------------------------------------------------------
@app.post("/api/nickname")
async def set_nickname(req: NicknameRequest):
    """Register or 'log in' with a nickname."""
    name = req.nickname.strip()
    if not name or len(name) < 2 or len(name) > 30:
        return JSONResponse(
            {"error": "Nickname must be 2-30 characters"},
            status_code=400,
        )

    result = nickname_db.get_or_create(name)
    messages = {
        "created": f"Welcome, {result['nickname']}! I'll remember your preferences.",
        "welcome_back": f"Welcome back, {result['nickname']}!",
    }
    return JSONResponse({
        **result,
        "message": messages[result["status"]],
    })


@app.get("/api/nickname/{name}/check")
async def check_nickname(name: str):
    """Check if a nickname already exists."""
    exists = nickname_db.check_exists(name.strip())
    return JSONResponse({
        "nickname": name.strip(),
        "exists": exists,
        "message": "This nickname is already taken — you'll be welcomed back!" if exists else "Available!",
    })


# -----------------------------------------------------------------
# Ingestor Endpoints
# -----------------------------------------------------------------
@app.post("/api/ingest")
async def ingest_product(req: IngestRequest):
    """
    Ingest a web product into the catalog.
    In mock mode: writes to a lightweight test SQLite DB.
    In production: would call the full ingestor pipeline.
    """
    product = req.product_data
    print(f"[Ingest] Received: {product.get('title', 'N/A')[:60]}")

    pid = ingest_db.insert(product)
    print(f"[Ingest] ✅ Inserted as {pid}")

    return JSONResponse({
        "status": "ingested",
        "product_id": pid,
        "title": product.get("title", ""),
    })


@app.get("/api/ingest/verify")
async def verify_ingested():
    """List all ingested products (for testing)."""
    products = ingest_db.list_all()
    return JSONResponse({
        "count": len(products),
        "products": products,
    })


@app.delete("/api/ingest/cleanup")
async def cleanup_ingested():
    """Delete all test-ingested products."""
    count = ingest_db.clear()
    return JSONResponse({
        "status": "cleaned",
        "deleted_count": count,
    })


@app.delete("/api/ingest/{product_id}")
async def delete_ingested(product_id: str):
    """Delete a specific ingested product."""
    deleted = ingest_db.delete(product_id)
    return JSONResponse({
        "status": "deleted" if deleted else "not_found",
        "product_id": product_id,
    })


# -----------------------------------------------------------------
# Core Chat Streaming Endpoint
# -----------------------------------------------------------------
@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Main communication pathway between Next.js and the CommerceAgent.
    We convert the Agent's raw asyncio generator straight into Server-Sent Events.
    """
    # Resolve user_id: nickname if set, otherwise anonymous
    user_id = req.user_id.strip() if req.user_id else f"anon_{req.session_id}"

    # Process the base64 string into bytes for the DINOv2 / Handyman VLM
    image_bytes = None
    if req.hasImage and req.imageBase64:
        try:
            clean_b64 = req.imageBase64.split(",")[-1]
            image_bytes = base64.b64decode(clean_b64)
        except Exception as e:
            print(f"Failed to decode base64 image: {e}")

    # The generator yields raw json strings from the Agent pipeline
    async def sse_generator():
        try:
            async for chunk in agent.handle_message(
                user_id=user_id,
                session_id=req.session_id,
                message=req.message,
                image_bytes=image_bytes,
                web_search_enabled=req.webSearch
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
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
