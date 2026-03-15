#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or not in PATH."
  echo "Run this first:"
  echo "  sudo bash scripts/ubuntu/bootstrap_ubuntu_24_04_gpu.sh"
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not available."
  echo "Try:"
  echo "  sudo systemctl enable --now docker"
  exit 1
fi

export DEFAULT_TRAINING_PROFILE="${DEFAULT_TRAINING_PROFILE:-server_max}"
export MAX_CPU_WORKERS="${MAX_CPU_WORKERS:-38}"
export TORCH_DATA_LOADER_WORKERS="${TORCH_DATA_LOADER_WORKERS:-16}"
export TORCH_EVAL_BATCH_SIZE="${TORCH_EVAL_BATCH_SIZE:-131072}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-38}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-38}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-38}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"

docker compose up --build -d

echo
echo "Stack started."
echo "Frontend: http://localhost/"
echo "Backend:  http://localhost:8000/docs"
