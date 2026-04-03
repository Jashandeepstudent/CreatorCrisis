#!/bin/bash
# start.sh — Launch both FastAPI (port 8000) and Gradio UI (port 7860)
# Used as the Docker CMD for HF Spaces deployment.

set -e

echo "── CreatorCrisisEnv starting ────────────────────────────"
echo "   FastAPI API  → http://localhost:8000/docs"
echo "   Gradio UI    → http://localhost:7860"
echo "─────────────────────────────────────────────────────────"

# Start FastAPI in background
uvicorn server.api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    --no-access-log &

API_PID=$!
echo "FastAPI started (PID $API_PID)"

# Wait briefly for API to be ready before UI starts
sleep 2

# Start Gradio in foreground (HF Spaces requires this to stay alive on 7860)
python ui.py

# If Gradio exits, kill the API too
kill $API_PID 2>/dev/null || true