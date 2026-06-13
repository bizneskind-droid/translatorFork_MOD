#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v uv &>/dev/null; then
    echo "[!] uv не найден. Установи: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "[*] Синхронизируем зависимости..."
uv sync

echo "[*] Запускаем..."
exec uv run python main.py "$@"
