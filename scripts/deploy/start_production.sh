#!/bin/bash
# scripts/deploy/start_production.sh
# Starts SGLang servers and FastAPI backend in a tmux session on the GPU VM.

echo "Starting Production Environment..."

# Ensure tmux is installed
if ! command -v tmux &> /dev/null
then
    echo "tmux could not be found. Installing..."
    sudo apt-get update && sudo apt-get install -y tmux
fi

# Kill existing session if any
tmux kill-session -t commerce-agent 2>/dev/null

# Create a new tmux session detached
tmux new-session -d -s commerce-agent -n "models"

# Pane 0 (Models): Launch SGLang models
tmux send-keys -t commerce-agent:models "python scripts/SGLang/mvp_sglang_launcher.py" C-m

# Create a second window for FastAPI
tmux new-window -t commerce-agent:1 -n "api"

# Pane 1 (API): Wait for a few seconds to let SGLang initialize, then start FastAPI
tmux send-keys -t commerce-agent:api "echo 'Waiting for models to start...'; sleep 10; uvicorn src.api:app --host 0.0.0.0 --port 8000" C-m

echo "All services are starting!"
echo "To view the models starting:   tmux attach-session -t commerce-agent:0"
echo "To view the FastAPI logs:      tmux attach-session -t commerce-agent:1"
echo "To detach from tmux, press Ctrl+B then D."
