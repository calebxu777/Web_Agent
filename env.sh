#!/bin/bash
# ============================================================
# Compound AI Commerce Agent — Environment Setup
# ============================================================
# Usage:
#   source env.sh              # Set up environment
#   source env.sh --install    # Full install (first time)
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="commerce-agent"

# ---- Colors ----
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🛒 Commerce Agent Environment Setup${NC}"

# ---- Conda environment ----
if command -v conda &> /dev/null; then
    if ! conda env list | grep -q "$ENV_NAME"; then
        echo -e "${GREEN}Creating conda environment: $ENV_NAME${NC}"
        conda create -n "$ENV_NAME" python=3.11 -y
    fi
    conda activate "$ENV_NAME"
else
    # Fallback to venv
    if [ ! -d "$PROJECT_ROOT/.venv" ]; then
        echo -e "${GREEN}Creating venv...${NC}"
        python3.11 -m venv "$PROJECT_ROOT/.venv"
    fi
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# ---- CUDA (PACE ICE cluster) ----
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
fi

# ---- Module loads for PACE ICE ----
if command -v module &> /dev/null; then
    module load cuda/12.1 2>/dev/null || true
    module load anaconda3 2>/dev/null || true
fi

# ---- Project paths ----
export PROJECT_ROOT="$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CONFIG_PATH="$PROJECT_ROOT/config/settings.yaml"

# ---- GCP credentials ----
# Set this to your service account key path
# export GOOGLE_APPLICATION_CREDENTIALS="$PROJECT_ROOT/config/gcp-key.json"

# ---- Full install (first time) ----
if [ "$1" = "--install" ]; then
    echo -e "${GREEN}Installing PyTorch with CUDA 12.1...${NC}"
    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
    
    echo -e "${GREEN}Installing remaining dependencies...${NC}"
    pip install --upgrade pip
    pip install -r "$PROJECT_ROOT/requirements.txt"

    echo -e "${GREEN}Creating directory structure...${NC}"
    mkdir -p "$PROJECT_ROOT/data/raw/amazon"
    mkdir -p "$PROJECT_ROOT/data/raw/hm"
    mkdir -p "$PROJECT_ROOT/data/raw/lvis"
    mkdir -p "$PROJECT_ROOT/data/raw/laion"
    mkdir -p "$PROJECT_ROOT/data/processed"
    mkdir -p "$PROJECT_ROOT/data/training"
    mkdir -p "$PROJECT_ROOT/models/sft_adapter"
    mkdir -p "$PROJECT_ROOT/models/dpo_adapter"
    mkdir -p "$PROJECT_ROOT/models/quantized"

    echo -e "${GREEN}✅ Installation complete!${NC}"
fi

echo -e "${GREEN}✅ Environment ready. Project root: $PROJECT_ROOT${NC}"
