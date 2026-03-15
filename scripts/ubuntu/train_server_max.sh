#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://127.0.0.1:8000}"

curl -sS -X POST "${API_URL}/api/train/start" \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "classification",
    "training_profile": "server_max",
    "model_name": "catboost",
    "save_as_best": true
  }'

echo
echo "Server-max training triggered. Poll status at ${API_URL}/api/train/status"
