#!/usr/bin/env bash
# Lumen launcher for macOS and Linux.
#
#   scripts/run_lumen.sh [lumen options] [images...]      e.g.  scripts/run_lumen.sh --port 9000 cells.tif
#
# First run: creates .venv in the project folder (python3 -m venv) and installs Lumen into it.
# Later runs: just starts Lumen.  Environment variables:
#   PYTHON      interpreter used to create the venv (default: python3; needs 3.11+)
#   LUMEN_VENV  virtual environment to use (default: <project>/.venv)
#   LUMEN_HOME  Lumen's per-user folder for recent files, plugins and the sample cache (default: ~/.lumen)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${LUMEN_VENV:-$ROOT/.venv}"
PY="${PYTHON:-python3}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "Creating a virtual environment in $VENV ..."
  "$PY" -m venv "$VENV"
fi

if ! "$VENV/bin/python" -c "import lumen" >/dev/null 2>&1; then
  echo "Installing Lumen into $VENV ..."
  if "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install -e "$ROOT"
  elif command -v uv >/dev/null 2>&1; then
    uv pip install --python "$VENV/bin/python" -e "$ROOT"
  else
    echo "error: neither pip nor uv is available to install Lumen into $VENV" >&2
    exit 1
  fi
fi

exec "$VENV/bin/python" -m lumen "$@"
