# app.py
import os
import glob
import logging
from functools import wraps
from threading import Lock
from time import monotonic
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, Response, redirect, url_for

# ----------------------------- App -----------------------------

app = Flask(__name__, template_folder="templates")
app.config.setdefault("ASSISTANT_SHOW_TECH", False)
app.config.setdefault("TEMPLATES_AUTO_RELOAD", True)

logging.basicConfig(level=logging.INFO)
log = app.logger

# Optional login guard: if flask_login not installed, make it a no-op
try:
    from flask_login import login_required  # noqa
except Exception:  # pragma: no cover
    def login_required(f):
        return f

# Optional rate limiter: if not installed, make decorators no-ops
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(get_remote_address, app=app, default_limits=[])
except Exception:  # pragma: no cover
    class _NoLimiter:
        def limit(self, *_args, **_kwargs):
            def deco(f): return f
            return deco
    limiter = _NoLimiter()

# Simple no-cache decorator + proxy streaming hint
def nocache(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        resp = f(*args, **kwargs)
        if hasattr(resp, "headers"):
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            # Help Nginx/Proxies not buffer SSE-like streams
            resp.headers["X-Accel-Buffering"] = "no"
            resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp
    return wrapper

def is_admin():
    token = os.getenv("ADMIN_TOKEN", "").strip()
    if not token:
        return True
    return request.headers.get("X-Admin-Token", "") == token

# ------------------------- Llama loader ------------------------

try:
    from llama_cpp import Llama
    _LLAMA_IMPORT_ERR = None
    _LLAMA_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    Llama = None
    _LLAMA_IMPORT_ERR = f"{type(_e).__name__}: {_e}"
    _LLAMA_AVAILABLE = False

PREFERRED_MODELS = [
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "Qwen_Qwen3-4B-Instruct-2507-Q5_K_M.gguf",
]

DEFAULT_CFG = {
    "n_ctx": int(os.getenv("LLAMA_N_CTX", 4096)),
    "n_batch": int(os.getenv("LLAMA_N_BATCH", 128)),
    "n_gpu_layers": int(os.getenv("LLAMA_N_GPU_LAYERS", 10)),  # >0 to offload on Metal/CUDA
    "use_mlock": False,
    "verbose": False,
}

SYSTEM_HINT = os.getenv(
    "LLAMA_SYSTEM_HINT",
    "You are concise and helpful. Keep answers short unless asked otherwise."
)

_llm = None
_llm_load_lock = Lock()
_RESOLVED_PATH = None
_RESOLVED_MODE = None  # "env" / "hardcoded" / "discovered" / "missing"

def _candidate_dirs():
    try:
        base = app.root_path
    except Exception:
        base = os.getcwd()
    return [
        os.getenv("LLAMA_MODELS_DIR") or os.path.join(base, "models"),
        os.path.join(os.getcwd(), "models"),
    ]

def _discover_models():
    paths = []
    for d in _candidate_dirs():
        if d and os.path.isdir(d):
            paths.extend(sorted(glob.glob(os.path.join(d, "*.gguf"))))
    return paths

def _resolve_model_path():
    env_path = os.getenv("LLAMA_GGUF", "").strip()
    if env_path:
        if not os.path.isabs(env_path):
            try:
                base = app.root_path
            except Exception:
                base = os.getcwd()
            env_path = os.path.normpath(os.path.join(base, env_path))
        return env_path, "env"

    for d in _candidate_dirs():
        if not d or not os.path.isdir(d):
            continue
        for name in PREFERRED_MODELS:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p, "hardcoded"

    found = _discover_models()
    if found:
        return found[0], "discovered"
    return None, "missing"

def _get_llm():
    global _llm, _RESOLVED_PATH, _RESOLVED_MODE
    if not _LLAMA_AVAILABLE:
        log.error(f"[LLAMA] llama_cpp import failed: {_LLAMA_IMPORT_ERR}")
        raise RuntimeError(f"llama_cpp not available: {_LLAMA_IMPORT_ERR}")

    with _llm_load_lock:
        if _llm is None:
            _RESOLVED_PATH, _RESOLVED_MODE = _resolve_model_path()
            if not _RESOLVED_PATH or not os.path.exists(_RESOLVED_PATH):
                candidates = _discover_models()
                log.error("[LLAMA] Model file not found", extra={
                    "resolved": _RESOLVED_PATH,
                    "mode": _RESOLVED_MODE,
                    "candidates": candidates
                })
                raise RuntimeError(
                    f"Model file not found. Resolved='{_RESOLVED_PATH}' mode='{_RESOLVED_MODE}'. "
                    f"Candidates={candidates}"
                )

            cfg = {"model_path": _RESOLVED_PATH, **DEFAULT_CFG}
            log.info("[LLAMA] Loading model", extra={
                "model_path": _RESOLVED_PATH,
                "mode": _RESOLVED_MODE,
                "cfg": {**cfg, "model_path": os.path.basename(_RESOLVED_PATH)}
            })
            t0 = monotonic()
            _llm = Llama(**cfg)
            dt = monotonic() - t0
            log.info("[LLAMA] Model loaded", extra={"seconds": round(dt, 3)})
    return _llm

# --------------------------- Routes ---------------------------

@app.get("/")
@nocache
def index():
    return redirect(url_for("assistant_ui"))

@app.get("/llama/health")
@nocache
def llama_health():
    env_raw = os.getenv("LLAMA_GGUF", "").strip() or None
    disc = _discover_models()

    global _RESOLVED_PATH, _RESOLVED_MODE
    if _llm is not None and _RESOLVED_PATH:
        resolved, mode = _RESOLVED_PATH, _RESOLVED_MODE
    else:
        resolved, mode = _resolve_model_path()

    exists = bool(resolved and os.path.exists(resolved))
    return jsonify({
        "import_ok": _LLAMA_AVAILABLE,
        "import_error": _LLAMA_IMPORT_ERR,
        "resolved_mode": mode,
        "resolved_path": resolved,
        "resolved_exists": exists,
        "env_path": env_raw,
        "discovered": disc,
        "model_loaded": _llm is not None,
        "cfg": {
            **DEFAULT_CFG,
            "model_path": os.path.basename(resolved) if resolved else None
        },
        "hint": "Set LLAMA_GGUF or place a *.gguf under ./models",
    }), (200 if _LLAMA_AVAILABLE and exists else 503)


@app.post("/llama/switch")
@nocache
def llama_switch():
    if not is_admin():
        return jsonify({"error": "Forbidden"}), 403

    data = request.get_json(silent=True) or {}
    model_path = data.get("model_path")
    if not model_path or not os.path.exists(model_path):
        return jsonify({"error": "Invalid model_path"}), 400

    global _llm, _RESOLVED_PATH, _RESOLVED_MODE
    with _llm_load_lock:
        _llm = None
        _RESOLVED_PATH = model_path
        _RESOLVED_MODE = "manual"
        cfg = {"model_path": _RESOLVED_PATH, **DEFAULT_CFG}
        log.info(f"[LLAMA] Switching to model: {os.path.basename(_RESOLVED_PATH)}")
        t0 = monotonic()
        _llm = Llama(**cfg)
        dt = monotonic() - t0
        log.info(f"[LLAMA] Active model = {os.path.basename(_RESOLVED_PATH)} (mode={_RESOLVED_MODE}), loaded in {dt:.2f}s")

    return jsonify({
        "status": "switched",
        "model": os.path.basename(_RESOLVED_PATH)
    })



@app.post("/llama/reset")
@nocache
def llama_reset():
    if not is_admin():
        return jsonify({"error": "Forbidden"}), 403
    global _llm, _RESOLVED_PATH, _RESOLVED_MODE
    with _llm_load_lock:
        _llm = None
        _RESOLVED_PATH = None
        _RESOLVED_MODE = None
    log.warning("[LLAMA] Model pointer cleared by admin")
    return jsonify({"status": "reset"}), 200

@app.post("/llama")
@login_required
@nocache
@limiter.limit("20/minute")
def llama_generate():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages") or []
    if not messages:
        return jsonify({"error": "Missing 'messages'."}), 400

    convo = f"System: {SYSTEM_HINT}\n\n"
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if role == "user":
            convo += f"User: {content}\n"
        elif role == "assistant":
            convo += f"Assistant: {content}\n"
    convo += "Assistant: "

    max_tokens  = int(data.get("max_tokens", 256))
    temperature = float(data.get("temperature", 0.6))
    top_p       = float(data.get("top_p", 0.9))

    def stream_plain():
        llm = _get_llm()
        for out in llm(
            prompt=convo,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["User:", "System:", "Assistant:"],
            stream=True,
        ):
            text = (out.get("choices") or [{}])[0].get("text", "")
            if text:
                yield text

    resp = Response(stream_plain(), mimetype="text/plain")
    resp.headers["X-Accel-Buffering"] = "no"
    return resp

# -------------------- Simple Assistant UI/API -----------------

@app.get("/assistant")
@login_required
@nocache
def assistant_ui():
    try:
        return render_template(
            "llama/assistant.html",
            show_tech=bool(app.config.get("ASSISTANT_SHOW_TECH", False)),
        )
    except Exception:
        log.exception("[ASSISTANT] Failed to render llama/assistant.html")
        return ("Template error", 500)

@app.post("/assistant")
@login_required
@nocache
@limiter.limit("30/minute")
def assistant_generate():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages") or []
    if not messages:
        return jsonify({"error": "Missing 'messages'."}), 400

    def build_convo(msgs):
        s = f"System: {SYSTEM_HINT}\n\n"
        for m in msgs:
            role = m.get("role", "user")
            c = (m.get("content") or "").strip()
            if not c:
                continue
            if role == "user":
                s += f"User: {c}\n"
            elif role == "assistant":
                s += f"Assistant: {c}\n"
            else:
                s += f"{role.capitalize()}: {c}\n"
        return s + "Assistant: "

    llm = _get_llm()
    try:
        n_ctx = int(getattr(llm, "context_params").n_ctx)  # llama-cpp >= 0.2.76
    except Exception:
        n_ctx = DEFAULT_CFG.get("n_ctx", 2048)

    SAFE_MARGIN = 8
    MIN_GEN = 32
    trimmed = list(messages[-12:])  # keep at most the last 12 turns

    while True:
        convo = build_convo(trimmed)
        used = len(llm.tokenize(convo.encode("utf-8"), add_bos=False))
        head = n_ctx - SAFE_MARGIN - used
        if head >= MIN_GEN or len(trimmed) <= 1:
            break
        trimmed.pop(0)

    convo = build_convo(trimmed)
    used = len(llm.tokenize(convo.encode("utf-8"), add_bos=False))
    head = n_ctx - SAFE_MARGIN - used
    if head < MIN_GEN:
        return jsonify({"error": "context_overflow",
                        "detail": f"Prompt too large for n_ctx={n_ctx}. Reduce input."}), 413

    req_max = int(data.get("max_tokens", 1024))
    max_tokens = max(MIN_GEN, min(req_max, head))
    temperature = float(data.get("temperature", 0.6))
    top_p = float(data.get("top_p", 0.9))

    def stream_plain():
        try:
            for out in llm(
                prompt=convo,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["User:", "System:", "Assistant:"],
                stream=True,
            ):
                text = (out.get("choices") or [{}])[0].get("text", "")
                if text:
                    yield text
        except Exception:
            log.exception("[ASSISTANT] Generation failed")
            yield "\n[error]\n"

    resp = Response(stream_plain(), mimetype="text/plain")
    resp.headers["X-Accel-Buffering"] = "no"
    return resp



########################################################################
# CREATOR
########################################################################

@app.get("/creator")
@login_required
@nocache
def creator_ui():
    try:
        return render_template(
            "llama/creator.html",
            show_tech=bool(app.config.get("ASSISTANT_SHOW_TECH", False)),
        )
    except Exception:
        log.exception("[CREATOR] Failed to render llama/creator.html")
        return ("Template error", 500)

@app.post("/creator")
@login_required
@nocache
@limiter.limit("30/minute")
def creator_generate():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages") or []
    if not messages:
        return jsonify({"error": "Missing 'messages'."}), 400

    def build_convo(msgs):
        s = f"System: {SYSTEM_HINT}\n\n"
        for m in msgs:
            role = m.get("role", "user")
            c = (m.get("content") or "").strip()
            if not c:
                continue
            if role == "user":
                s += f"User: {c}\n"
            elif role == "assistant":
                s += f"Assistant: {c}\n"
            else:
                s += f"{role.capitalize()}: {c}\n"
        return s + "Assistant: "

    llm = _get_llm()
    try:
        n_ctx = int(getattr(llm, "context_params").n_ctx)
    except Exception:
        n_ctx = DEFAULT_CFG.get("n_ctx", 2048)

    SAFE_MARGIN = 8
    MIN_GEN = 32
    trimmed = list(messages[-12:])

    while True:
        convo = build_convo(trimmed)
        used = len(llm.tokenize(convo.encode("utf-8"), add_bos=False))
        head = n_ctx - SAFE_MARGIN - used
        if head >= MIN_GEN or len(trimmed) <= 1:
            break
        trimmed.pop(0)

    convo = build_convo(trimmed)
    used = len(llm.tokenize(convo.encode("utf-8"), add_bos=False))
    head = n_ctx - SAFE_MARGIN - used
    if head < MIN_GEN:
        return jsonify({"error": "context_overflow",
                        "detail": f"Prompt too large for n_ctx={n_ctx}. Reduce input."}), 413

    req_max = int(data.get("max_tokens", 1024))
    max_tokens = max(MIN_GEN, min(req_max, head))
    temperature = float(data.get("temperature", 0.6))
    top_p = float(data.get("top_p", 0.9))

    def stream_plain():
        try:
            for out in llm(
                prompt=convo,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["User:", "System:", "Assistant:"],
                stream=True,
            ):
                text = (out.get("choices") or [{}])[0].get("text", "")
                if text:
                    yield text
        except Exception:
            log.exception("[CREATOR] Generation failed")
            yield "\n[error]\n"

    resp = Response(stream_plain(), mimetype="text/plain")
    resp.headers["X-Accel-Buffering"] = "no"
    return resp



# --------------------------- Main -----------------------------

if __name__ == "__main__":
    # Development runner (Gunicorn recommended for deploy)
    host = os.getenv("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_RUN_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    # Avoid reloader to prevent semaphore warnings on macOS/Python 3.13
    app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)
