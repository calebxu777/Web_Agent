import base64
import json
import uuid
import yaml
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agent import CommerceAgent, AgentConfig

# Global Agent State
agent = None
config = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hook for startup/shutdown."""
    global agent, config
    print("[FastAPI] Starting Server...")
    
    config_path = Path("config/settings.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config at {config_path.absolute()}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # Read the image path and provider from settings
    provider = config.get("image_storage", {}).get("provider", "local")
    local_path = config.get("image_storage", {}).get("local", {}).get("base_path", "")

    # Create the Agent configuration exactly reflecting the MVP setup
    # Note: We enforce the "local" vs "gcs" toggle here
    ac = AgentConfig(
        master_brain_model_name=config["models"]["master_brain"]["model_id"],
        handyman_model_name=config["models"]["handyman"]["model_id"],
        use_florence=False, 
        use_visual_verifier=False, 
        use_web_search=False,
        image_storage_provider=provider,
        local_image_base_path=local_path,
        log_timing=True,
        mock_mode=True # Enabled specifically for local laptop UI testing
    )
    
    agent = CommerceAgent(config=config, agent_config=ac)
    # The agent lazily initializes on its very first inference request
    
    yield
    print("[FastAPI] Shutting down...")


# Initialize the App
app = FastAPI(title="Commerce Agent API", lifespan=lifespan)

# Allow the Next.js frontend (port 3000) to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------
# Toggle Integration: Serving Local Local Imagery
# -----------------------------------------------------------------
# If provider == "local", the agent returns image URLs like:
# "/api/images/hm/hm_123.jpg"
# We intercept that here and serve the file from your C: drive.
# -----------------------------------------------------------------
with open(Path("config/settings.yaml"), "r", encoding="utf-8") as _f:
    _temp_cfg = yaml.safe_load(_f)
    if _temp_cfg.get("image_storage", {}).get("provider") == "local":
        _base_path = _temp_cfg["image_storage"]["local"]["base_path"]
        if Path(_base_path).exists():
            print(f"[FastAPI] Mounting local images at /api/images from {_base_path}")
            app.mount("/api/images", StaticFiles(directory=_base_path), name="images")
        else:
            print(f"[FastAPI WARNING] Local image path {_base_path} not found!")


# -----------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = ""
    hasImage: bool = False
    imageBase64: str | None = None  # Added explicitly for you to handle image uploads
    webSearch: bool = False
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "demo_user"


# -----------------------------------------------------------------
# Core Chat Streaming Endpoint
# -----------------------------------------------------------------
@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Main communication pathway between Next.js and the CommerceAgent.
    We convert the Agent's raw asyncio generator straight into Server-Sent Events.
    """
    
    # Process the base64 string into bytes for the DINOv2 / Handyman VLM
    image_bytes = None
    if req.hasImage and req.imageBase64:
        try:
            # Strip standard HTML base64 prefix if present (e.g. data:image/jpeg;base64,...)
            clean_b64 = req.imageBase64.split(",")[-1]
            image_bytes = base64.b64decode(clean_b64)
        except Exception as e:
            print(f"Failed to decode base64 image: {e}")
            
    # The generator yields raw json strings from the Agent pipeline
    async def sse_generator():
        try:
            async for chunk in agent.handle_message(
                user_id=req.user_id,
                session_id=req.session_id,
                message=req.message,
                image_bytes=image_bytes,
                web_search_enabled=req.webSearch
            ):
                # The browser expects strict SSE syntax: "data: {JSON}\n\n"
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
            "X-Accel-Buffering": "no" # Stop Nginx from buffering if deployed anywhere later
        }
    )

if __name__ == "__main__":
    import uvicorn
    # Typically launched as: uvicorn src.api:app --reload --port 8000
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
