#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script as root:"
  echo "  sudo bash scripts/ubuntu/bootstrap_native_ubuntu_24_04.sh"
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  gnupg \
  build-essential \
  python3 \
  python3-dev \
  python3-pip \
  python3-venv \
  nginx

if ! command -v node >/dev/null 2>&1 || ! node -v | grep -Eq '^v(20|21|22|23|24)\.'; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y nodejs
fi

echo
echo "Native runtime dependencies are installed."
echo "python: $(python3 --version)"
echo "node:   $(node --version)"
echo "npm:    $(npm --version)"
echo "nginx:  $(nginx -v 2>&1)"
