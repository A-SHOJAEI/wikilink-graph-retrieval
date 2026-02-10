#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-python3}"
GET_PIP_URL_DEFAULT="https://bootstrap.pypa.io/get-pip.py"
GET_PIP_URL="${GET_PIP_URL:-$GET_PIP_URL_DEFAULT}"
GET_PIP_SHA256="${GET_PIP_SHA256:-}"

REQ_FILE="${REQ_FILE:-requirements.txt}"

if [[ ! -x "$(command -v "$PYTHON")" ]]; then
  echo "ERROR: '$PYTHON' not found on PATH." >&2
  exit 1
fi

mkdir -p .cache

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON" -m venv --without-pip "$VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: venv python not found at '$VENV_PY'." >&2
  exit 1
fi

GET_PIP_PATH=".cache/get-pip.py"

if [[ ! -x "$VENV_DIR/bin/pip" ]]; then
  # Download get-pip.py using stdlib only (no curl/wget dependency).
  "$VENV_PY" - <<'PY' "$GET_PIP_URL" "$GET_PIP_PATH"
import os
import sys
import urllib.request

url = sys.argv[1]
out_path = sys.argv[2]
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with urllib.request.urlopen(url) as r:
    data = r.read()

tmp = out_path + ".tmp"
with open(tmp, "wb") as f:
    f.write(data)
os.replace(tmp, out_path)

print(f"Downloaded get-pip.py ({len(data)} bytes) from {url} to {out_path}")
PY

  if [[ -n "$GET_PIP_SHA256" ]]; then
    ACTUAL_SHA256="$("$VENV_PY" - <<'PY' "$GET_PIP_PATH"
import hashlib, sys
p = sys.argv[1]
h = hashlib.sha256()
with open(p, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
print(h.hexdigest())
PY
)"
    if [[ "$ACTUAL_SHA256" != "$GET_PIP_SHA256" ]]; then
      echo "ERROR: get-pip.py sha256 mismatch." >&2
      echo "Expected: $GET_PIP_SHA256" >&2
      echo "Actual:   $ACTUAL_SHA256" >&2
      exit 1
    fi
  fi

  # Install pip into the venv (ensurepip may be missing on the host).
  "$VENV_PY" "$GET_PIP_PATH" "pip==24.2" "setuptools==75.3.0" "wheel==0.44.0"
fi

"$VENV_DIR/bin/pip" install --disable-pip-version-check --no-input -r "$REQ_FILE"

echo "Venv ready at '$VENV_DIR'."
