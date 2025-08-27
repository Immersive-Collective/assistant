# Assistant — Local Llama Chat (Flask + llama.cpp)

Minimal, fast, **in-process** assistant UI with streaming responses.  
Backed by `llama.cpp` via `llama-cpp-python`. Works with a local **GGUF** model.

---

## Directory layout

Either location works for models (auto-discovery checks both):

```

assistant/
├─ src/
│  ├─ app.py
│  └─ templates/llama/assistant.html
├─ models/           # <- GGUF here (repo root)
└─ src/models/       # <- or here (beside app.py)

````

---

## Quick start (one command)

```bash
git clone git@github.com:Immersive-Collective/assistant.git
cd assistant
chmod +x install.sh
./install.sh
````

Run:

```bash
source .venv/bin/activate
export FLASK_APP=src/app.py
export FLASK_DEBUG=1
flask run
# open http://127.0.0.1:5000/assistant
```

---

## Get a GGUF model

### A) Use `install.sh` to download from Hugging Face (preferred)

```bash
# defaults: bartowski/Meta-Llama-3.1-8B-Instruct-GGUF  Q4_K_M file
DOWNLOAD=1 ./install.sh

# customize:
DOWNLOAD=1 HF_REPO="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF" \
HF_FILE="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
./install.sh
```

Requirements for gated models:

```bash
source .venv/bin/activate
pip install -U huggingface_hub hf_transfer
huggingface-cli login      # paste token
# accept the model license on the model page (once)
```

### B) Point to an existing file or directory

```bash
# Exact file
export LLAMA_GGUF="/absolute/path/to/your-model.gguf"

# Or a directory containing .gguf files
export LLAMA_MODELS_DIR="/absolute/path/to/models"
```

### C) Symlink into this repo

```bash
mkdir -p models
ln -s /absolute/path/to/your-model.gguf models/your-model.gguf
```

Verify model resolution:

```
http://127.0.0.1:5000/llama/health
# expect 200 with "resolved_exists": true and a model name
```

---

## Manual setup (if not using install.sh)

### 1) Create & activate a virtualenv

```bash
cd assistant
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell
python -m pip install --upgrade pip
```

### 2) Install requirements

```bash
pip install -r requirements.txt
```

If missing, create a minimal `requirements.txt`:

```bash
cat > requirements.txt << 'EOF'
Flask>=3.0
Flask-Limiter>=3.7
llama-cpp-python>=0.3.0
huggingface_hub>=0.23
hf_transfer>=0.1.6
EOF
pip install -r requirements.txt
```

> macOS (Apple Silicon) Metal acceleration:

```bash
pip install --upgrade "llama-cpp-python" \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
# or build from source:
# CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

### 3) Run

```bash
export FLASK_APP=src/app.py
export FLASK_DEBUG=1
flask run
```

Open `http://127.0.0.1:5000/assistant`.

---

## Endpoints

* `GET /assistant` — chat UI (Jinja template `templates/llama/assistant.html`)

* `POST /assistant` — simple streaming generation
  **Body**

  ```json
  {
    "messages": [
      {"role":"user","content":"Hello"}
    ],
    "max_tokens": 1024,
    "temperature": 0.6,
    "top_p": 0.9
    }
  ```

  **cURL**

  ```bash
  curl -N -s http://127.0.0.1:5000/assistant \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Write a haiku."}]}'
  ```

* `GET /llama/health` — model status/config (no forced load)

* `POST /llama` — lower-level streaming endpoint (same `messages` schema)

* `POST /llama/reset` — clear loaded model pointer (set `ADMIN_TOKEN` to gate)

---

## Environment knobs

| Variable             | Description                                        | Default                |
| -------------------- | -------------------------------------------------- | ---------------------- |
| `LLAMA_GGUF`         | Absolute path to a `.gguf` model file              | —                      |
| `LLAMA_MODELS_DIR`   | Directory to search for `.gguf` files              | —                      |
| `LLAMA_N_CTX`        | Context window (tokens)                            | 2048                   |
| `LLAMA_N_BATCH`      | Batch size                                         | 128                    |
| `LLAMA_N_GPU_LAYERS` | Layers offloaded to GPU/Metal (0=CPU)              | 10                     |
| `LLAMA_SYSTEM_HINT`  | System prompt seed                                 | “concise and helpful…” |
| `ADMIN_TOKEN`        | Required header `X-Admin-Token` for `/llama/reset` | —                      |

---

## UI notes

* Mobile-friendly chat; iOS Safari zoom disabled (16px inputs, `100dvh`).
* Advanced controls (temperature/top-p/max tokens) can be toggled:

  ```python
  app.config["ASSISTANT_SHOW_TECH"] = True
  ```

---

## Troubleshooting

* **`/llama/health` = 503**
  Model not found. Ensure a `.gguf` exists under `models/` or `src/models/`, or set `LLAMA_GGUF` / `LLAMA_MODELS_DIR`. For gated models, accept the license and `huggingface-cli login`.

* **`context_overflow` or 500 on long pastes**
  The prompt (history + input) exceeded `LLAMA_N_CTX`. Trim old turns on the client (keep last 6–12), reduce input size, or raise `LLAMA_N_CTX` (uses more RAM).

* **Rate limit warnings**
  Dev default uses in-memory limiter. For production, configure a persistent storage backend per Flask-Limiter docs.

* **Metal/CUDA acceleration**
  Install the appropriate `llama-cpp-python` wheel and set `LLAMA_N_GPU_LAYERS > 0`.

---

## License

See [LICENSE](LICENSE).

