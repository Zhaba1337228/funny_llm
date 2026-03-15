#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script as root:"
  echo "  sudo bash scripts/ubuntu/run_native_stack.sh"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is not installed."
  echo "Run this first:"
  echo "  bash scripts/ubuntu/bootstrap_native_ubuntu_24_04.sh"
  exit 1
fi

if ! command -v node >/dev/null 2>&1; then
  echo "node is not installed."
  echo "Run this first:"
  echo "  bash scripts/ubuntu/bootstrap_native_ubuntu_24_04.sh"
  exit 1
fi

if ! command -v nginx >/dev/null 2>&1; then
  echo "nginx is not installed."
  echo "Run this first:"
  echo "  bash scripts/ubuntu/bootstrap_native_ubuntu_24_04.sh"
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

mkdir -p logs .run

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.server.txt
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio

pushd frontend >/dev/null
npm ci
npm run build
popd >/dev/null

if [[ -f .run/backend.pid ]] && kill -0 "$(cat .run/backend.pid)" >/dev/null 2>&1; then
  kill "$(cat .run/backend.pid)" >/dev/null 2>&1 || true
  sleep 2
fi

nohup .venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 > logs/backend.log 2>&1 &
echo $! > .run/backend.pid

NGINX_CONF_TMP=".run/funny_llm.nginx.conf"
sed "s#__ROOT_DIR__#${ROOT_DIR//\#/\\#}#g" scripts/ubuntu/funny_llm.nginx.conf > "${NGINX_CONF_TMP}"
cp "${NGINX_CONF_TMP}" /etc/nginx/sites-available/funny_llm
ln -sfn /etc/nginx/sites-available/funny_llm /etc/nginx/sites-enabled/funny_llm
rm -f /etc/nginx/sites-enabled/default

nginx -t
systemctl enable --now nginx
systemctl restart nginx

for _ in {1..60}; do
  if curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
  echo "Backend failed to start. Check logs/backend.log"
  exit 1
fi

echo
echo "Native stack started."
echo "Web UI:       http://$(hostname -I | awk '{print $1}')/"
echo "Backend docs: http://$(hostname -I | awk '{print $1}'):8000/docs"
echo "Backend log:  ${ROOT_DIR}/logs/backend.log"
