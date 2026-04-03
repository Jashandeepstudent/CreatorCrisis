#!/bin/bash
# start.sh — Launch FastAPI on port 7860 (HF Spaces public port)
# HF Spaces only exposes port 7860 externally.
# FastAPI serves both the OpenEnv API and a redirect to /docs.

set -e

echo "── CreatorCrisisEnv starting ────────────────────────────"
echo "   FastAPI API  → http://0.0.0.0:7860/docs"
echo "   Health       → http://0.0.0.0:7860/health"
echo "   Reset        → http://0.0.0.0:7860/reset"
echo "─────────────────────────────────────────────────────────"

# Run FastAPI on 7860 — the only port HF Spaces exposes publicly
exec uvicorn server.api:app \
    --host 0.0.0.0 \
    --port 7860 \
    --log-level info
