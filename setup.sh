#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Keep all generated files inside this project.
export UV_CACHE_DIR="$PWD/work/uv-cache"
export PIP_CACHE_DIR="$PWD/work/pip-cache"
export IPYTHONDIR="$PWD/work/ipython"
export JUPYTER_CONFIG_DIR="$PWD/work/jupyter/config"
export JUPYTER_DATA_DIR="$PWD/.venv/share/jupyter"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$IPYTHONDIR" "$JUPYTER_CONFIG_DIR"
if command -v uv >/dev/null 2>&1; then
  uv venv --python python3.12 --allow-existing .venv
  uv pip sync --python .venv/bin/python requirements.lock.txt
else
  python3.12 -m venv .venv
  .venv/bin/python -m pip install -r requirements.lock.txt
fi
.venv/bin/python -m ipykernel install --prefix "$PWD/.venv" --name python3 --display-name "DML ACR (Python 3.12)"
printf '%s\n' '설치 완료. ./start_jupyter.sh 로 JupyterLab을 실행하세요.'
