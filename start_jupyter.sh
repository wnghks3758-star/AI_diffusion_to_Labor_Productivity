#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export JUPYTER_CONFIG_DIR="$PWD/work/jupyter/config"
export JUPYTER_DATA_DIR="$PWD/.venv/share/jupyter"
export JUPYTER_RUNTIME_DIR="$PWD/work/jupyter/runtime"
export IPYTHONDIR="$PWD/work/ipython"
export MPLCONFIGDIR="$PWD/work/matplotlib"
export XDG_CACHE_HOME="$PWD/work/cache"
mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_RUNTIME_DIR" "$IPYTHONDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
exec .venv/bin/jupyter lab --ServerApp.root_dir="$PWD" --no-browser "$@"
