#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f .run/backend.pid ]] && kill -0 "$(cat .run/backend.pid)" >/dev/null 2>&1; then
  kill "$(cat .run/backend.pid)" >/dev/null 2>&1 || true
  rm -f .run/backend.pid
  echo "Backend stopped."
else
  echo "Backend is not running."
fi

if command -v systemctl >/dev/null 2>&1; then
  systemctl stop nginx >/dev/null 2>&1 || true
fi

echo "Native stack stopped."
