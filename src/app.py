# app.py
import os
import glob
import logging
from functools import wraps
from threading import Lock
from time import monotonic
from uuid import uuid4
import mimetypes

mimetypes.add_type("image/svg+xml", ".svg")

from flask import Flask, jsonify, render_template, request, Response, redirect, url_for

# ─────────────────────── Optional Google GenAI (Gemini) ───────────────────────
try:
    from google import genai
    from google.genai import types as genai_types
    from google.genai.errors import ClientError
    _GENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    genai = None
    genai_types = None
    ClientError = Exception
    _GENAI_AVAILABLE = False

_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-image-preview",
]

_gemini_client = None
_genai_lock = Lock()

# ─────────────────────────────── Flask App ───────────────────────────────
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

# No-cache decorator
def nocache(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        resp = f(*args, **kwargs)
        if hasattr(resp, "headers"):
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["X-Accel-Buffering"] = "no"   # for SSE-like streaming
            resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp
    return wrapper

def is_admin():
    token = os.getenv("ADMIN_TOKEN", "").strip()
    if not token:
        return True
    return request.headers.get("X-Admin-Token", "") == token

# Static/gen dir for inline assets (Gemini image parts)
STATIC_DIR = app.static_folder or os.path.join(app.root_path, "static")
GEN_DIR = os.path.join(STATIC_DIR, "gen")
os.makedirs(GEN_DIR, exist_ok=True)



# ------------------------- Gemini backend ----------------------


CREATOR_SYS_INST = (
    "You are concise. When asked for HTML/CSS/JS, respond with a SINGLE "
    "```html``` fenced block containing a complete, valid document. "
    "Do not include any explanations, apologies, or text outside the fence."
)

_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-image-preview",
]

# Capabilities per model (so we don't request IMAGE where it's not allowed)
_GEMINI_MODEL_CAPS = {
    "gemini-2.5-flash": {"image": False},              # text only to avoid regional image errors
    "gemini-2.5-flash-image-preview": {"image": True}, # can serve inline images
}

def _gemini_modalities_for(name: str):
    caps = _GEMINI_MODEL_CAPS.get(name, {})
    return ["TEXT", "IMAGE"] if caps.get("image") else ["TEXT"]

_gemini_client = None
_genai_lock = Lock()

STATIC_DIR = app.static_folder or os.path.join(app.root_path, "static")
GEN_DIR = os.path.join(STATIC_DIR, "gen")
os.makedirs(GEN_DIR, exist_ok=True)

def _get_gemini():
    if not _GENAI_AVAILABLE:
        raise RuntimeError("google-genai not installed. pip install google-genai")
    global _gemini_client
    with _genai_lock:
        if _gemini_client is None:
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not set")
            _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client

def _gemini_contents_from_messages(msgs):
    out = []
    for m in msgs:
        role = m.get("role", "user")
        text = (m.get("content") or "").strip()
        if not text:
            continue
        gr = "user" if role == "user" else "model"
        out.append(genai_types.Content(
            role=gr,
            parts=[genai_types.Part.from_text(text=text)]
        ))
    return out

def _save_inline_asset(mime_type: str, data: bytes) -> str:
    ext = mimetypes.guess_extension(mime_type) or ".bin"
    name = f"{uuid4().hex}{ext}"
    path = os.path.join(GEN_DIR, name)
    with open(path, "wb") as f:
        f.write(data)
    return url_for("static", filename=f"gen/{name}", _external=False)



def _stream_gemini(model_name: str, messages, system_instruction: str | None = None):
    client = _get_gemini()
    contents = _gemini_contents_from_messages(messages)
    config = genai_types.GenerateContentConfig(
        response_modalities=_gemini_modalities_for(model_name),
        **({"system_instruction": system_instruction} if system_instruction else {})
    )

    try:
        for chunk in client.models.generate_content_stream(
            model=model_name, contents=contents, config=config
        ):
            # Prefer the top-level streaming text
            if getattr(chunk, "text", None):
                yield chunk.text

            cand = (getattr(chunk, "candidates", None) or [None])[0]
            if not cand or not getattr(cand, "content", None) or not getattr(cand.content, "parts", None):
                continue

            # Handle only inline images from parts; DO NOT emit part.text
            for part in cand.content.parts:
                if getattr(part, "inline_data", None) and part.inline_data.data:
                    try:
                        img_url = _save_inline_asset(part.inline_data.mime_type, part.inline_data.data)
                        yield f'\n```html\n<img src="{img_url}" alt="Generated image"/>\n```\n'
                    except Exception:
                        log.exception("[GEMINI] saving inline image failed")

    except ClientError as e:
        code, status, msg = _err_info(e)

        # 1) Quota / rate-limit -> retry TEXT-only on a cheaper model
        if _is_quota(e, msg, status, code):
            log.warning("[GEMINI] Quota hit (%s / %s). Retrying TEXT-only on gemini-2.5-flash", code, status)
            yield "\n[error] Gemini quota hit for the selected model; retrying on text-only.\n"
            try:
                for t in _stream_gemini_text_only(messages, model_name="gemini-2.5-flash"):
                    yield t
                return
            except Exception:
                log.exception("[GEMINI] TEXT-only retry also failed")

            yield "\n[error] Gemini quota still exceeded on text-only. Please switch to a local model or try another API project.\n"
            return

        # 2) Regional / permission issues for IMAGE -> retry TEXT-only
        if (code in (400, 403)) and any(s in (msg or "").lower() for s in [
            "image generation is not available",
            "image generation is not allowed",
            "image generation not available",
            "failed_precondition",
            "permission_denied",
        ]):
            log.warning("[GEMINI] IMAGE not allowed on %s; retrying TEXT-only", model_name)
            try:
                for t in _stream_gemini_text_only(messages, model_name="gemini-2.5-flash"):
                    yield t
                return
            except Exception:
                log.exception("[GEMINI] TEXT-only retry failed")
            yield "\n[error] No text-only fallback available (permission/region).\n"
            return

        # 3) All other client errors -> short message
        log.exception("[GEMINI] API error")
        yield f"\n[error] Gemini error: {msg}\n"




def _stream_gemini_text_only(messages, model_name="gemini-2.5-flash"):
    client = _get_gemini()
    contents = _gemini_contents_from_messages(messages)
    cfg = genai_types.GenerateContentConfig(response_modalities=["TEXT"])
    for chunk in client.models.generate_content_stream(model=model_name, contents=contents, config=cfg):
        if getattr(chunk, "text", None):
            yield chunk.text
        else:
            cand = (chunk.candidates or [None])[0]
            if cand and cand.content and cand.content.parts:
                for part in cand.content.parts:
                    if getattr(part, "text", None):
                        yield part.text






# ───────────────────────────── Llama loader ─────────────────────────────
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
    "n_gpu_layers": int(os.getenv("LLAMA_N_GPU_LAYERS", 10)),
    "use_mlock": False,
    "verbose": False,
}

SYSTEM_HINT = os.getenv(
    "LLAMA_SYSTEM_HINT",
    "You are concise and helpful. Keep answers short unless asked otherwise."
)

_llm = None
_llm_load_lock = Lock()
_RESOLVED_PATH = None        # can be local *.gguf or a Gemini model name
_RESOLVED_MODE = None        # "env" / "hardcoded" / "discovered" / "manual" / "gemini" / "missing"

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
    global _llm, _RESOLVED_PATH, _RESOLVED_MODE  # ← move global to the top
    # guard: never try to load Llama while in Gemini mode
    if _RESOLVED_MODE == "gemini":
        raise RuntimeError("Llama not active while Gemini backend selected")
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

# ─────────────────────────────── Routes ───────────────────────────────
@app.get("/")
@nocache
def index():
    return redirect(url_for("creator_ui"))

@app.get("/favicon.ico")
def favicon():
    return redirect(url_for('static', filename='favicon.ico'))



@app.get("/llama/health")
@nocache
def llama_health():
    env_raw = os.getenv("LLAMA_GGUF", "").strip() or None
    disc = _discover_models()

    # ✅ Prefer the explicit selection, even when _llm is None (Gemini mode)
    global _RESOLVED_PATH, _RESOLVED_MODE
    if _RESOLVED_PATH is not None:
        resolved, mode = _RESOLVED_PATH, _RESOLVED_MODE
    else:
        resolved, mode = _resolve_model_path()

    selected_is_gemini = bool(resolved and (resolved in _GEMINI_MODELS or mode == "gemini"))
    exists = bool(resolved and (os.path.exists(resolved) or resolved in _GEMINI_MODELS))

    # Consider Gemini "ok" even if llama_cpp isn't present
    ok = bool(exists and (_LLAMA_AVAILABLE or selected_is_gemini))

    return jsonify({
        "ok": ok,
        "import_ok": _LLAMA_AVAILABLE,
        "import_error": _LLAMA_IMPORT_ERR,
        "resolved_mode": mode,
        "resolved_path": resolved,
        "resolved_exists": exists,
        "selected_is_gemini": selected_is_gemini,
        "env_path": env_raw,
        "discovered": disc,
        "remotes": _GEMINI_MODELS,
        "model_loaded": _llm is not None,
        "cfg": { **DEFAULT_CFG, "model_path": os.path.basename(resolved) if resolved else None },
        "hint": "Set LLAMA_GGUF or place a *.gguf under ./models; for Gemini set GEMINI_API_KEY",
    }), (200 if ok else 503)



# @app.get("/llama/health")
# @nocache
# def llama_health():
#     # prefer explicit selection if set, even when _llm is None (Gemini mode)
#     global _RESOLVED_PATH, _RESOLVED_MODE
#     if _RESOLVED_PATH is not None:
#         resolved, mode = _RESOLVED_PATH, _RESOLVED_MODE
#     else:
#         resolved, mode = _resolve_model_path()

#     disc = _discover_models()
#     # local file existence
#     local_exists = bool(resolved and os.path.exists(resolved))
#     # is selection a Gemini remote?
#     selected_is_gemini = bool(resolved in _GEMINI_MODELS or mode == "gemini")
#     # Gemini availability (client + API key present)
#     gemini_ok = bool(_GENAI_AVAILABLE and os.getenv("GEMINI_API_KEY", "").strip())
#     # overall OK
#     ok = (selected_is_gemini and gemini_ok) or (local_exists and _LLAMA_AVAILABLE)

#     payload = {
#         "ok": ok,
#         "import_ok": _LLAMA_AVAILABLE,
#         "gemini_ok": gemini_ok,
#         "resolved_mode": mode,
#         "resolved_path": resolved,
#         "resolved_exists": (local_exists or (resolved in _GEMINI_MODELS)),
#         "selected_is_gemini": selected_is_gemini,
#         "env_path": os.getenv("LLAMA_GGUF", "").strip() or None,
#         "discovered": disc,
#         "remotes": _GEMINI_MODELS,
#         "model_loaded": _llm is not None,
#         "cfg": {**DEFAULT_CFG, "model_path": os.path.basename(resolved) if resolved else None},
#         "hint": "Set LLAMA_GGUF or place a *.gguf under ./models; set GEMINI_API_KEY for Gemini.",
#     }
#     return jsonify(payload), (200 if ok else 503)

@app.post("/llama/switch")
@nocache
def llama_switch():
    if not is_admin():
        return jsonify({"error": "Forbidden"}), 403

    data = request.get_json(silent=True) or {}
    model_path = data.get("model_path")

    global _llm, _RESOLVED_PATH, _RESOLVED_MODE
    with _llm_load_lock:
        # Gemini selection (no Llama load)
        if model_path in _GEMINI_MODELS:
            _llm = None
            _RESOLVED_PATH = model_path
            _RESOLVED_MODE = "gemini"
            log.info(f"[BACKEND] Using Gemini model: {_RESOLVED_PATH}")
            return jsonify({"status": "switched", "model": _RESOLVED_PATH})

        # Local model selection
        if not model_path or not os.path.exists(model_path):
            return jsonify({"error": "Invalid model_path"}), 400

        _RESOLVED_PATH = model_path
        _RESOLVED_MODE = "manual"
        _llm = None
        cfg = {"model_path": _RESOLVED_PATH, **DEFAULT_CFG}
        log.info(f"[LLAMA] Switching to model: {os.path.basename(_RESOLVED_PATH)}")
        t0 = monotonic()
        _llm = Llama(**cfg)
        dt = monotonic() - t0
        log.info(f"[LLAMA] Active model = {os.path.basename(_RESOLVED_PATH)} (mode={_RESOLVED_MODE}), loaded in {dt:.2f}s")

    return jsonify({"status": "switched", "model": os.path.basename(_RESOLVED_PATH)})

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

# ───────────────────── Assistant (chat-only) ─────────────────────
@app.post("/assistant")
@login_required
@nocache
@limiter.limit("30/minute")
def assistant_generate():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages") or []
    if not messages:
        return jsonify({"error": "Missing 'messages'."}), 400

    # Gemini branch
    if _RESOLVED_MODE == "gemini" and (_RESOLVED_PATH in _GEMINI_MODELS):
        def stream_plain():
            try:
                for chunk in _stream_gemini(_RESOLVED_PATH, messages[-12:]):
                    if chunk:
                        yield chunk
            except Exception:
                log.exception("[ASSISTANT] Gemini generation failed")
                yield "\n[error]\n"
        resp = Response(stream_plain(), mimetype="text/plain")
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    # Llama branch
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
            log.exception("[ASSISTANT] Generation failed")
            yield "\n[error]\n"

    resp = Response(stream_plain(), mimetype="text/plain")
    resp.headers["X-Accel-Buffering"] = "no"
    return resp

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

# ───────────────────── Creator (chat + HTML preview) ─────────────────────
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

    # Gemini branch
    if _RESOLVED_MODE == "gemini" and (_RESOLVED_PATH in _GEMINI_MODELS):
        def stream_plain():
            try:
                for chunk in _stream_gemini(
                    _RESOLVED_PATH,
                    messages[-12:],
                    system_instruction=CREATOR_SYS_INST,   # ← add this
                ):
                    if chunk:
                        yield chunk
            except Exception:
                log.exception("[CREATOR] Gemini generation failed")
                yield "\n[error]\n"
        resp = Response(stream_plain(), mimetype="text/plain")
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    # Llama branch
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

# ─────────────────────────────── Main ───────────────────────────────
if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_RUN_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)
