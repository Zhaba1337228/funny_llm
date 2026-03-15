#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://127.0.0.1:8000}"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is not installed."
  echo "Install it with:"
  echo "  apt-get update && apt-get install -y curl"
  exit 1
fi

for _ in {1..30}; do
  if curl -fsS "${API_URL}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! curl -fsS "${API_URL}/api/health" >/dev/null 2>&1; then
  echo "Backend is not reachable at ${API_URL}."
  echo "Start the native stack first:"
  echo "  bash scripts/ubuntu/run_native_stack.sh"
  exit 1
fi

curl -sS -X POST "${API_URL}/api/train/start" \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "classification",
    "training_profile": "server_max",
    "model_name": "catboost",
    "save_as_best": true
  }'

echo
echo "Native server-max training triggered. Poll status at ${API_URL}/api/train/status"
