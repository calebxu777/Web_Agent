# Compound AI Commerce Agent

A high-performance Compound AI System designed to replicate and exceed "2026 Rufus-class" commerce capabilities. Features a tiered model architecture (Qwen 3.5), multimodal vector search (DINOv2 + BGE-M3), and EAGLE-2 speculative decoding.

## Architecture Highlights
- **Router (0.8B Handyman)**: Intent classification, query decomposition
- **Master Brain (9B Synthesis)**: High-EQ generation, contextual recommendation
- **Memory Tiers**: Redis (Episodic), LanceDB (Semantic + Conversational)
- **Frontend**: Next.js App using SSE streaming to mimic Gemini-style Apple-inspired aesthetic with state-aware loading pipelines.

## Project Structure
- `config/` - Central application configuration
- `scripts/` - SFT, DPO, Quantization and Data Normalization workflows
- `src/` - Backend model logic, routers, retrievers
- `frontend/` - Next.js Chat Application

## Running the Application
### Backend Setup
1. Instantiate environment via `source env.sh --install`
2. Prepare databases: `python scripts/01_normalize_catalog.py` & `02_extract_embeddings.py`
3. Launch Inference servers: `python scripts/08_launch_sglang.py`

### Frontend Application
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000)

## Frontend Features
- Upload product images to invoke visual search
- Apple-grade typography and glass-morphism aesthetic
- Progress tracking pipeline indicating server-state (Cold Start -> Image Analyzers -> Generation)
