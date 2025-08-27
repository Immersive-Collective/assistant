#!/usr/bin/env bash
# install.sh — set up venv, install deps, and (optionally) fetch a GGUF from Hugging Face
# Usage:
#   ./install.sh                      # install only
#   DOWNLOAD=1 ./install.sh           # install + download GGUF to src/models
#   HF_REPO=owner/repo HF_FILE=foo.gguf DOWNLOAD=1 ./install.sh
#   HUGGING_FACE_HUB_TOKEN=xxxx DOWNLOAD=1 ./install.sh

set -Eeuo pipefail

log() { printf "\033[1;34m[install]\033[0m %s\n" "$*"; }
warn(){ printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; }

# --- project root ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then :; else ROOT="$SCRIPT_DIR"; fi
cd "$ROOT"

# --- defaults / env knobs ---
VENV="$ROOT/.venv"
REQ="$ROOT/requirements.txt"
APP_PATH="src/app.py"
MODELS_DIR="$ROOT/src/models"
DOWNLOAD="${DOWNLOAD:-0}"
HF_REPO="${HF_REPO:-bartowski/Meta-Llama-3.1-8B-Instruct-GGUF}"
HF_FILE="${HF_FILE:-Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# --- sanity checks ---
command -v python3 >/dev/null 2>&1 || { err "python3 not found"; exit 1; }

# --- venv ---
if [[ ! -d "$VENV" ]]; then
  log "Creating virtualenv at $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"
log "Python: $(python -V)"
log "Pip:    $(pip -V)"

# --- base deps ---
pip install --upgrade pip setuptools wheel

if [[ -f "$REQ" ]]; then
  log "Installing from requirements.txt"
  pip install -r "$REQ"
else
  log "No requirements.txt found — installing minimal set"
  cat > "$REQ" <<'EOF'
Flask>=3.0
Flask-Limiter>=3.7
llama-cpp-python>=0.3.0
huggingface_hub>=0.23
hf_transfer>=0.1.6
EOF
  pip install -r "$REQ"
fi

# --- optional: Apple Silicon Metal-accelerated llama-cpp-python ---
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  log "Detected macOS arm64 — installing Metal wheel for llama-cpp-python (best effort)"
  pip install --upgrade "llama-cpp-python" \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal || true
fi

# --- optional download from Hugging Face ---
if [[ "$DOWNLOAD" == "1" ]]; then
  mkdir -p "$MODELS_DIR"
  log "Preparing to download GGUF from Hugging Face"
  # try non-interactive login if token present
  if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    log "Using HUGGING_FACE_HUB_TOKEN for login"
    huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN" --add-to-git-credential false || {
      warn "Token login failed; ensure the token is valid and the model license is accepted."
    }
  else
    # not failing hard; user may already be logged in or model is public
    huggingface-cli whoami >/dev/null 2>&1 || warn "Not logged in to Hugging Face. If the model is gated, run: huggingface-cli login"
  fi

  log "Downloading: repo=$HF_REPO file=$HF_FILE → $MODELS_DIR"
  # Prefer CLI download (no symlinks, real file on disk)
  if ! huggingface-cli download "$HF_REPO" "$HF_FILE" --local-dir "$MODELS_DIR" --local-dir-use-symlinks False; then
    warn "CLI download failed; trying Python fallback (hf_hub_download)"
    python - <<PY || { err "Download failed. Accept the license on the model page and try again."; exit 1; }
from huggingface_hub import hf_hub_download
import os, sys
repo = os.environ.get("HF_REPO")
fname = os.environ.get("HF_FILE")
out_dir = os.environ.get("MODELS_DIR")
p = hf_hub_download(repo_id=repo, filename=fname)
dst = os.path.join(out_dir, os.path.basename(p))
import shutil; shutil.copy2(p, dst)
print(dst)
PY
  fi

  if [[ -f "$MODELS_DIR/$HF_FILE" ]]; then
    log "Downloaded: $MODELS_DIR/$HF_FILE"
  else
    err "Model file not found after download attempt: $MODELS_DIR/$HF_FILE"
    exit 1
  fi
fi

# --- final instructions ---
echo
log "Setup complete."
echo
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  export FLASK_APP=$APP_PATH"
echo "  export FLASK_DEBUG=1"
if [[ -f "$MODELS_DIR/$HF_FILE" ]]; then
  echo "  # Model found in src/models — auto-discovery will pick it up"
else
  echo "  # Provide a GGUF model (any one of these):"
  echo "  #   1) Place a .gguf under src/models/"
  echo "  #   2) Set LLAMA_GGUF to an absolute file path:"
  echo "  #        export LLAMA_GGUF=\"/abs/path/to/model.gguf\""
  echo "  #   3) Point to a directory of GGUF files:"
  echo "  #        export LLAMA_MODELS_DIR=\"/abs/path/to/models\""
fi
echo "  flask run"
echo
echo "Hugging Face quick download (examples):"
echo "  # fast path to src/models (requires license acceptance and login)"
echo "  DOWNLOAD=1 HF_REPO=$HF_REPO HF_FILE=$HF_FILE ./install.sh"
echo
echo "If /llama/health shows 503:"
echo "  - Ensure a .gguf exists in src/models or set LLAMA_GGUF / LLAMA_MODELS_DIR."
echo "  - If gated, accept the license on the model page and 'huggingface-cli login'."
