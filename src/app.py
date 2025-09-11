# app.py
# (Remove any API keys from source. Set GEMINI_API_KEY, ADMIN_TOKEN, and LLAMA_GGUF via environment variables.)

import os
import glob
import logging
from functools import wraps
from threading import Lock
from time import monotonic
from uuid import uuid4
import mimetypes
from io import BytesIO
import base64
import re
import httpx

from typing import Tuple, Optional, Dict, Any

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

# NOTE: we will discover remote Gemini models dynamically
_GEMINI_MODELS_STATIC = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-image-preview",
]
# dynamic mapping populated at runtime: name -> {"image": bool, "raw": <model object/dict>}
_GEMINI_REMOTES: Dict[str, Dict[str, Any]] = {}
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


# ----------------- helpers -----------------

def _err_info(exc: Exception) -> Tuple[Optional[int], Optional[str], str]:
    """
    Normalize exception -> (code, status, message).
    Works with google.genai.errors.ClientError and with generic exceptions.
    """
    try:
        code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        status = getattr(exc, "status", None)
        response_json = getattr(exc, "response_json", None) or getattr(exc, "response", None)
        msg = None

        if hasattr(response_json, "json"):
            try:
                j = response_json.json()
            except Exception:
                j = None
        else:
            j = response_json

        if isinstance(j, dict):
            msg = j.get("error", {}).get("message") or j.get("message") or j.get("error_description")
            code = code or j.get("error", {}).get("code") or j.get("code")
            status = status or j.get("status")
        if not msg:
            msg = str(exc)

        try:
            code = int(code) if code is not None else None
        except Exception:
            code = None

        return code, status or None, msg
    except Exception as e:
        return None, None, str(exc or e)


def _is_quota(exc: Exception, msg: str | None, status: str | None, code: Optional[int]) -> bool:
    if code == 429:
        return True
    s = (str(status or "")).lower()
    if "quota" in s or "resource_exhausted" in s or "rate" in s:
        return True
    if msg:
        ml = msg.lower()
        if any(x in ml for x in ("quota", "exceed", "resource_exhausted", "rate limit", "rate_limit", "too many requests")):
            return True
    if exc.__class__.__name__.lower().startswith("clienterror") and code is None and msg and "quota" in msg.lower():
        return True
    return False


def _parse_retry_delay(exc: Exception) -> Optional[float]:
    try:
        text = str(exc)
        m = re.search(r'retryDelay["\']?\s*[:=]\s*["\']?(\d+)(s)?', text, re.IGNORECASE)
        if m:
            return float(m.group(1))
        m2 = re.search(r'Retry-After[:\s]+(\d+)', text, re.IGNORECASE)
        if m2:
            return float(m2.group(1))
    except Exception:
        pass
    return None





# ─────────────────────── OpenAI (Responses API) ───────────────────────
# 🧷 Place near the other optional imports (below the Gemini try/except)
try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    _OpenAI = None
    _OPENAI_AVAILABLE = False

_openai_client = None
# OpenAI models you want available in the selector (text-only via Responses API)
_OPENAI_MODELS_STATIC = [
    # GPT-5 family
    "gpt-5", "gpt-5-mini", "gpt-5-nano",
    # GPT-4.1 family
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    # o-series
    "o3", "o4-mini",
    # 4o family
    "gpt-4o", "gpt-4o-mini",
]



# after the OpenAI block (right below _OPENAI_MODELS_STATIC)
_OPENAI_REMOTES: Dict[str, Dict[str, Any]] = {}

def _refresh_openai_models():
    """
    Populate a simple registry like _GEMINI_REMOTES but for OpenAI.
    You can extend via env: OPENAI_MODELS="gpt-4o-mini,gpt-4o,gpt-4.1-mini"
    """
    _OPENAI_REMOTES.clear()
    names = list(_OPENAI_MODELS_STATIC)
    extra = [s.strip() for s in os.getenv("OPENAI_MODELS", "").split(",") if s.strip()]
    for n in extra:
        if n not in names:
            names.append(n)
    for n in names:
        norm = _normalize_openai_model(n)  # -> "openai/<id>"
        _OPENAI_REMOTES[norm] = {"image": False, "via": "openai.responses"}

def _get_openai():
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("openai python SDK not installed. pip install openai")
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        # 🔒 guard: catch common mis-pastes
        if api_key.startswith(("org-", "org_", "proj_", "project_")):
            raise RuntimeError("OPENAI_API_KEY looks like an org/project id. Use a secret key that starts with 'sk-'. Set OPENAI_ORG separately.")
        _openai_client = _OpenAI(api_key=api_key, organization=os.getenv("OPENAI_ORG") or None)
    return _openai_client


def _is_openai_model(name: str | None) -> bool:
    if not name:
        return False
    low = str(name).lower()
    return low.startswith("openai:") or low.startswith("openai/") or (low in _OPENAI_MODELS_STATIC)

def _normalize_openai_model(name: str) -> str:
    """
    Accepts 'openai:gpt-4o-mini', 'openai/gpt-4o-mini', or 'gpt-4o-mini'.
    Returns normalized 'openai/gpt-4o-mini'.
    """
    if not name:
        raise RuntimeError("OpenAI model name is empty")
    low = name.lower()
    if low.startswith("openai:"):
        return "openai/" + name.split(":", 1)[1]
    if low.startswith("openai/"):
        return name
    # plain model id
    return f"openai/{name}"

def _openai_model_id(resolved_path: str | None) -> str:
    """
    Extracts the raw OpenAI model id from normalized 'openai/<model>'.
    Falls back to a safe default if needed.
    """
    if not resolved_path:
        return _OPENAI_MODELS_STATIC[0]
    low = resolved_path.lower()
    if low.startswith("openai/"):
        return resolved_path.split("/", 1)[1]
    return resolved_path  # trust caller

def _to_openai_messages(msgs, system_instruction: str | None = None):
    """
    Convert your internal messages into OpenAI Responses 'messages' format.
    """
    out = []
    if system_instruction:
        out.append({"role": "system", "content": system_instruction})
    for m in msgs:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        out.append({"role": ("assistant" if role == "assistant" else "user"), "content": content})
    return out

def _stream_openai(messages, system_instruction: str | None = None):
    """
    Stream text via OpenAI Responses API.
    """
    client = _get_openai()
    model = _openai_model_id(_RESOLVED_PATH)
    try:
        # Build a Responses-compatible input payload (supports system + chat turns)
        payload = _to_openai_messages(messages, system_instruction)

        # Streaming Responses API (yields incremental text deltas)
        with client.responses.stream(
            model=model,
            input=payload,
        ) as stream:
            for event in stream:
                # Forward incremental text deltas to the client UI.
                if getattr(event, "type", "") == "response.output_text.delta":
                    chunk = getattr(event, "delta", "")
                    if chunk:
                        yield chunk

            # Ensure the stream is fully finalized (surfaces server-side errors)
            _ = stream.get_final_response()

    except Exception as e:
        # Keep parity with your other backends: surface a concise error line
        err = str(e)
        yield f"\n[error] OpenAI error: {err}\n"
        return









# ------------------------- Gemini backend ----------------------

# --- Imagen gating (skip when project isn't billed) ---
_IMAGEN_ALLOWED = True  # flipped to False once we detect a billing-gate error

def _is_imagen_billing_error(code: Optional[int], msg: Optional[str]) -> bool:
    text = (msg or "").lower()
    return (code in (400, 403)) and (
        "only accessible to billed users" in text
        or "billing" in text
        or "permission" in text and "billing" in text
    )

# ───────── rate-limit scratchpad (last-seen headers) ─────────
_LAST_RATELIMIT: Dict[str, Any] = {}

def _parse_rate_limit_headers(hdrs) -> Dict[str, Any]:
    """
    Pulls common rate-limit headers into a flat dict (case-insensitive).
    Safe to call with None or any mapping-like headers object.
    """
    out: Dict[str, Any] = {}
    if not hdrs:
        return out
    try:
        low = {str(k).lower(): v for k, v in dict(hdrs).items()}
    except Exception:
        try:
            # httpx.Headers-like
            low = {str(k).lower(): hdrs.get(k) for k in hdrs.keys()}
        except Exception:
            return out

    keys = [
        "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests",
        "x-ratelimit-limit-tokens",   "x-ratelimit-remaining-tokens",   "x-ratelimit-reset-tokens",
        "x-ratelimit-limit",          "x-ratelimit-remaining",          "x-ratelimit-reset",
        "ratelimit-limit",            "ratelimit-remaining",            "ratelimit-reset",
    ]
    for k in keys:
        if k in low:
            out[k] = low[k]
    return out

def _update_last_ratelimit(hdrs) -> None:
    try:
        rl = _parse_rate_limit_headers(hdrs)
        if rl:
            _LAST_RATELIMIT.clear()
            _LAST_RATELIMIT.update(rl)
    except Exception:
        pass




CREATOR_SYS_INST = (
    "You are concise. When asked for HTML/CSS/JS, respond with a SINGLE "
    "```html``` fenced block containing a complete, valid document. "
    "Do not include any explanations, apologies, or text outside the fence."
)

# Capabilities per model name we discover (populated at init-time)
_GEMINI_MODEL_CAPS: Dict[str, Dict[str, bool]] = {}

def _gemini_modalities_for(name: str):
    caps = _GEMINI_MODEL_CAPS.get(name, {})
    return ["TEXT", "IMAGE"] if caps.get("image") else ["TEXT"]


def _image_candidates_ordered() -> list[str]:
    """
    Return image-capable Gemini model names, normalized to 'models/...'.
    Put the currently selected model first (if image-capable).
    """
    def _norm(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        return name if str(name).startswith("models/") else f"models/{name}"

    cands: list[str] = []
    seen = set()

    def add(name: Optional[str]):
        if not name:
            return
        norm = _norm(name)
        if not norm or norm in seen:
            return
        meta = _GEMINI_REMOTES.get(norm) or _GEMINI_REMOTES.get(name)
        if meta and meta.get("image") and meta.get("via") in ("generate_content", "images"):
            cands.append(norm)
            seen.add(norm)

    # Selected first
    add(_RESOLVED_PATH)

    # Then all known remotes
    for n in list(_GEMINI_REMOTES.keys()):
        add(n)

    return cands



def _normalize_model_path(name: str) -> str:
    # Accept both "gemini-2.5-flash" and "models/gemini-2.5-flash"
    if not name:
        raise RuntimeError("model name is empty")
    return name if name.startswith("models/") else f"models/{name}"

def _to_rest_contents(messages):
    out = []
    for m in messages:
        role = m.get("role", "user")
        text = m.get("content")
        if not isinstance(text, str) or not text.strip():
            # REST path here is TEXT-only; skip non-text parts
            continue
        # Gemini REST expects role "user" or "model"
        rest_role = "model" if role == "assistant" else "user"
        out.append({"role": rest_role, "parts": [{"text": text}]})
    return out

def _rest_generate_content_stream(model_name: str, messages, system_instruction: str | None = None):
    """
    Minimal REST call to:
      POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
    This returns one chunk (non-streaming). We keep a generator interface for the caller.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set for REST call")

    model_path = _normalize_model_path(model_name)
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent"

    payload: Dict[str, Any] = {
        "contents": _to_rest_contents(messages),
    }
    if system_instruction:
        payload["system_instruction"] = {
            "role": "system",
            "parts": [{"text": system_instruction}]
        }

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
    }

    with httpx.Client(timeout=60) as client:
        r = client.post(url, headers=headers, json=payload)

        # record last-seen ratelimit headers (success or error)
        _update_last_ratelimit(getattr(r, "headers", None))

        if r.status_code >= 400:
            # surface the backend error message
            try:
                j = r.json()
                msg = j.get("error", {}).get("message") or j.get("message") or r.text
            except Exception:
                msg = r.text
            raise ClientError(msg) if isinstance(ClientError, type) else RuntimeError(msg)

        j = r.json()
        # Concatenate all text parts from first candidate (common case)
        out = []
        for cand in (j.get("candidates") or []):
            content = cand.get("content") or {}
            for part in (content.get("parts") or []):
                if "text" in part and part["text"]:
                    out.append(part["text"])
        text = "".join(out).strip()
        if text:
            yield text




def _get_gemini():
    """
    Initialize and return a genai client. Also refresh remote model list once.
    """
    if not _GENAI_AVAILABLE:
        raise RuntimeError("google-genai not installed. pip install google-genai")
    global _gemini_client
    with _genai_lock:
        if _gemini_client is None:
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not set")
            _gemini_client = genai.Client(api_key=api_key)
            # refresh remotes once on client init
            try:
                _refresh_gemini_models(_gemini_client)
            except Exception:
                log.exception("[GEMINI] failed to refresh remote model list during client init")
    return _gemini_client



def _imagen_predict_rest(model: str, prompt: str, sample_count: int = 1):
    """
    Direct REST call to Imagen :predict endpoint.
    POST https://generativelanguage.googleapis.com/v1beta/models/{model}:predict
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:predict"
    payload = {"instances": [{"prompt": prompt}], "parameters": {"sampleCount": int(sample_count)}}
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    with httpx.Client(timeout=60) as client:
        r = client.post(url, headers=headers, json=payload)
        _update_last_ratelimit(getattr(r, "headers", None))
        if r.status_code >= 400:
            j = None
            msg = r.text
            try:
                j = r.json()
                msg = j.get("error", {}).get("message") or j.get("message") or msg
            except Exception:
                pass
            if isinstance(ClientError, type):
                raise ClientError(r.status_code, j if isinstance(j, dict) else {}, r)
            raise RuntimeError(msg)
        return r.json()




def _refresh_gemini_models(client):
    """
    Populate _GEMINI_REMOTES and _GEMINI_MODEL_CAPS using the remote API.
    Tries multiple list methods because SDK surfaces can differ.
    Detects still-image support and which API to use:
      - via == "generate_content"  (Gemini image-preview via generateContent)
      - via == "images"            (Imagen/Images API, or REST :predict)
    Skips video-only models (e.g., VEO).
    """
    global _GEMINI_REMOTES, _GEMINI_MODEL_CAPS
    _GEMINI_REMOTES = {}
    _GEMINI_MODEL_CAPS = {}

    candidates = []

    # list defensively
    try:
        if getattr(client, "models", None) and getattr(client.models, "list", None):
            res = client.models.list()
            if isinstance(res, (list, tuple)):
                candidates = list(res)
            elif hasattr(res, "models"):
                candidates = list(res.models)
            else:
                try:
                    candidates = list(res)
                except Exception:
                    candidates = []
        elif hasattr(client, "list_models"):
            res = client.list_models()
            if isinstance(res, (list, tuple)):
                candidates = list(res)
            elif hasattr(res, "models"):
                candidates = list(res.models)
            else:
                try:
                    candidates = list(res)
                except Exception:
                    candidates = []
    except Exception as e:
        log.exception("[GEMINI] list models failed via models.list / list_models: %s", e)
        candidates = []

    def _record(name: str, image_support: bool, via: Optional[str], raw):
        # Normalize to "models/..." for consistency everywhere
        name = name if str(name).startswith("models/") else f"models/{name}"
        _GEMINI_REMOTES[name] = {"image": bool(image_support), "via": via, "raw": raw}
        _GEMINI_MODEL_CAPS[name] = {"image": bool(image_support)}

    # static fallback if nothing listed
    if not candidates:
        for n in _GEMINI_MODELS_STATIC:
            lname = n.lower()
            if "imagen" in lname:
                _record(n, True, "images", None)
            elif "gemini" in lname and "image" in lname:
                _record(n, True, "generate_content", None)
            else:
                _record(n, False, None, None)
        log.info("[GEMINI] fallback to static remotes: %s", list(_GEMINI_REMOTES.keys()))
        return

    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple, set)):
            return list(x)
        return [x]

    for m in candidates:
        try:
            if isinstance(m, str):
                name = m
                meta = None
            else:
                name = (
                    getattr(m, "name", None)
                    or getattr(m, "model", None)
                    or (m.get("name") if isinstance(m, dict) else None)
                    or getattr(m, "id", None)
                    or (m.get("id") if isinstance(m, dict) else None)
                ) or str(m)
                meta = getattr(m, "metadata", None) or (m.get("metadata") if isinstance(m, dict) else None)

            methods = (
                _as_list(getattr(m, "supported_methods", None))
                or _as_list(getattr(m, "methods", None))
                or (_as_list(meta.get("supportedMethods")) if isinstance(meta, dict) else [])
            )
            modalities = (
                _as_list(getattr(m, "supported_modalities", None))
                or (_as_list(meta.get("supportedModalities")) if isinstance(meta, dict) else [])
            )

            lname = (name or "").lower()
            image_support, via = False, None

            # ⬇️ IMPORTANT: force Imagen family to 'images' path FIRST
            if "imagen" in lname:
                image_support, via = True, "images"

            # video-only? skip for still images
            elif "veo" in lname or any(str(x).upper() == "VIDEO" for x in modalities):
                image_support, via = False, None

            # IMAGE modality advertised -> use generateContent path for Gemini-only models
            elif any(str(x).upper() == "IMAGE" for x in modalities):
                image_support, via = True, "generate_content"

            # images API methods → images path
            elif any(
                "generateimage" in str(x).lower()
                or "images.generate" in str(x).lower()
                or "generate_images" in str(x).lower()
                for x in methods
            ):
                image_support, via = True, "images"

            # name heuristics for older previews
            elif "image-preview" in lname or ("gemini" in lname and "image" in lname):
                image_support, via = True, "generate_content"

            _record(name, image_support, via, m)
        except Exception:
            log.exception("[GEMINI] failed to parse model entry: %s", m)

    log.info("[GEMINI] discovered remotes: %s", list(_GEMINI_REMOTES.keys()))




# Accept text content and image payloads (path / base64 / bytes)
def _gemini_contents_from_messages(msgs):
    out = []
    for m in msgs:
        role = m.get("role", "user")
        raw = m.get("content")

        # try image formats first
        try:
            if isinstance(raw, dict):
                if raw.get("image_path"):
                    path = raw["image_path"]
                    try:
                        from PIL import Image
                        img = Image.open(path)
                        out.append(img)
                        continue
                    except Exception:
                        log.exception("[GEMINI] failed to open image_path %s", path)
                if raw.get("image_b64"):
                    b64 = raw["image_b64"]
                    try:
                        from PIL import Image
                        data = base64.b64decode(b64)
                        img = Image.open(BytesIO(data))
                        out.append(img)
                        continue
                    except Exception:
                        log.exception("[GEMINI] failed to decode image_b64")
            if isinstance(raw, (bytes, bytearray)):
                try:
                    from PIL import Image
                    img = Image.open(BytesIO(raw))
                    out.append(img)
                    continue
                except Exception:
                    log.exception("[GEMINI] failed to open bytes image")
        except Exception:
            log.exception("[GEMINI] unexpected error while processing potential image message")

        # fallback to text content
        text = (raw or "").strip() if isinstance(raw, str) or raw is None else None
        if not text:
            continue
        gr = "user" if role == "user" else "model"
        out.append(genai_types.Content(
            role=gr,
            parts=[genai_types.Part.from_text(text=text)]
        ))
    return out


def _imagen_generate_call(client, model: str, prompt: str):
    """
    Call Imagen/Images depending on the installed google-genai SDK shape.
    Tries SDK entry points, then falls back to the REST :predict endpoint.
    """
    # SDK (varies by version)
    if hasattr(client, "images") and hasattr(client.images, "generate"):
        try:
            return client.images.generate(model=model, prompt=prompt)
        except Exception:
            pass
    if hasattr(client, "models") and hasattr(client.models, "generate_images"):
        try:
            return client.models.generate_images(model=model, prompt=prompt)
        except Exception:
            pass
    if hasattr(client, "images") and hasattr(client.images, "generate_image"):
        try:
            return client.images.generate_image(model=model, prompt=prompt)
        except Exception:
            pass

    # REST :predict (works for imagen-4.x-*)
    try:
        return _imagen_predict_rest(model, prompt, sample_count=1)
    except Exception as e:
        # Final fallback (rare): some preview image models answered via generate_content
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            try:
                return client.models.generate_content(
                    model=model,
                    contents=[genai_types.Content(
                        role="user",
                        parts=[genai_types.Part.from_text(text=prompt)]
                    )],
                    config=genai_types.GenerateContentConfig(response_modalities=["IMAGE"])
                )
            except Exception:
                pass
        raise e



def _generate_gemini_preview_still(prompt: str, model: str = "models/gemini-2.5-flash-image-preview") -> str:
    """
    Generate an inline image via Gemini *image-preview* models using the SDK's
    generate_content (returns inline_data). Produces a full ```html fenced doc.
    """
    client = _get_gemini()
    contents = [genai_types.Content(
        role="user",
        parts=[genai_types.Part.from_text(text=prompt)]
    )]
    cfg = genai_types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])

    resp = client.models.generate_content(model=model, contents=contents, config=cfg)

    # pull first inline image
    cand = (getattr(resp, "candidates", None) or [None])[0]
    if cand and getattr(cand, "content", None):
        for part in (getattr(cand.content, "parts", []) or []):
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                mime = getattr(inline, "mime_type", None) or "image/png"
                url = _save_inline_asset(mime, inline.data)
                return f"""```html
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Generated Image</title>
<style>:root{{color-scheme:dark}}html,body{{margin:0;height:100%;background:#0b0b0b}}.wrap{{min-height:100%;display:grid;place-items:center;padding:2rem}}img{{max-width:95vw;max-height:90vh;border-radius:16px}}</style>
</head><body><div class="wrap"><img src="{url}" alt="Generated image"/></div></body></html>
```"""
    # if no image returned, surface any text the model produced
    if getattr(resp, "text", None):
        return resp.text
    raise RuntimeError("Gemini preview model returned no image")



def _extract_image_bytes(resp) -> tuple[bytes, str]:
    """
    Normalize different response shapes to (bytes, mime_type).
    Handles:
      - SDK objects with .bytes/.data / .inline_data
      - SDK generate_content responses with inline image parts
      - REST :predict dicts with predictions[].bytesBase64Encoded
      - Lists/iterables of any of the above
    """
    # REST :predict JSON shape
    if isinstance(resp, dict) and "predictions" in resp:
        preds = resp.get("predictions") or []
        for p in preds:
            if not isinstance(p, dict):
                continue
            b64 = p.get("bytesBase64Encoded") or p.get("b64_data") or p.get("b64Data")
            if b64:
                try:
                    data = base64.b64decode(b64)
                    mime = p.get("mimeType") or p.get("mime_type") or "image/png"
                    return data, mime
                except Exception:
                    pass

    seq = None
    if isinstance(resp, (list, tuple)):
        seq = resp
    else:
        try:
            if hasattr(resp, "__iter__") and not isinstance(resp, (bytes, bytearray, str)):
                seq = list(resp)
        except Exception:
            seq = None
    if seq is None:
        seq = [resp]

    for obj in seq:
        for attr in ("bytes", "data"):
            data = getattr(obj, attr, None)
            if data:
                mime = getattr(obj, "mime_type", None) or getattr(obj, "mime", None) or "image/png"
                return data, mime
        inline = getattr(obj, "inline_data", None)
        if inline is not None:
            data = getattr(inline, "data", None)
            if data:
                mime = getattr(inline, "mime_type", None) or "image/png"
                return data, mime
        image = getattr(obj, "image", None)
        if image is not None:
            for attr in ("bytes", "data"):
                data = getattr(image, attr, None)
                if data:
                    mime = getattr(image, "mime_type", None) or getattr(image, "mime", None) or "image/png"
                    return data, mime
        b64 = getattr(obj, "b64_data", None) or getattr(obj, "b64Data", None)
        if b64:
            return base64.b64decode(b64), (getattr(obj, "mime_type", None) or "image/png")

    for obj in seq:
        cand = getattr(obj, "candidates", None)
        if cand:
            first = cand[0]
            parts = getattr(getattr(first, "content", None), "parts", []) or []
            for part in parts:
                inline = getattr(part, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    return inline.data, (getattr(inline, "mime_type", None) or "image/png")

    raise RuntimeError("Image API returned no bytes")







def _generate_image_best_effort(prompt: str) -> str:
    """
    Try image-capable remotes with smart ordering and gating:
      - Prefer Gemini *image-preview* (generateContent) first (free-tier friendly).
      - If Imagen returns a "billed users only" 400/403, disable Imagen for this process
        and future attempts (_IMAGEN_ALLOWED=False) to avoid repeated 400s.
      - On 429 from preview, surface a clean retry hint using Retry-After / retryDelay.
    Returns a full ```html fenced document on success.
    """
    global _IMAGEN_ALLOWED   # <-- move this to the top of the function

    client = _get_gemini()
    all_candidates = _image_candidates_ordered()
    if not all_candidates:
        raise RuntimeError("No image-capable remote models available")

    # Classify and order: preview first, then images (unless gated off)
    preview_models = []
    imagen_models = []
    for name in all_candidates:
        meta = _GEMINI_REMOTES.get(name, {})
        if meta.get("via") == "generate_content":     # e.g., gemini-2.5-flash-image-preview
            preview_models.append(name)
        elif meta.get("via") == "images":             # e.g., imagen-4.*
            imagen_models.append(name)

    ordered = preview_models + (imagen_models if _IMAGEN_ALLOWED else [])

    last_err: Optional[Exception] = None

    for name in ordered:
        meta = _GEMINI_REMOTES.get(name, {})
        via = meta.get("via")

        try:
            ...
        except ClientError as e:
            code, status, msg = _err_info(e)

            # If Imagen is gated by billing, disable future Imagen tries and continue to preview.
            if meta.get("via") == "images" and _is_imagen_billing_error(code, msg):
                _IMAGEN_ALLOWED = False      # <-- keep this assignment, no inner 'global' needed
                last_err = e
                continue
            ...


            # Rate limit (common for preview); surface clean guidance
            if code == 429:
                delay = _parse_retry_delay(e)
                hint = f" Rate limit; retry after ~{int(delay)}s." if delay else " Rate limit; please retry shortly."
                raise RuntimeError(f"Image generation is temporarily rate-limited.{hint}")

            # 404 on generateContent for an Imagen model (not supported) → just try next
            if code == 404:
                last_err = e
                continue

            # Other errors → try next candidate
            last_err = e
            continue

        except Exception as e:
            last_err = e
            continue

    # If we got here, all candidates failed
    raise last_err or RuntimeError("Image generation failed (no usable remote)")







def _save_inline_asset(mime_type: str, data: bytes) -> str:
    ext = mimetypes.guess_extension(mime_type) or ".bin"
    name = f"{uuid4().hex}{ext}"
    path = os.path.join(GEN_DIR, name)
    with open(path, "wb") as f:
        f.write(data)
    return url_for("static", filename=f"gen/{name}", _external=False)


def _generate_imagen_still(prompt: str, model: str = "imagen-4.0-fast-generate-001") -> str:
    """
    Generate a still image via Imagen/Images, save under /static/gen, and return
    a COMPLETE ```html fenced document for the Creator preview.
    """
    client = _get_gemini()
    resp = _imagen_generate_call(client, model, prompt)
    data, mime = _extract_image_bytes(resp)
    url = _save_inline_asset(mime, data)
    return f"""```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Generated Image</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{ color-scheme: dark; }}
  html,body {{ margin:0; height:100%; background:#0b0b0b; }}
  .wrap {{ min-height:100%; display:grid; place-items:center; padding:2rem; }}
  img {{ max-width:95vw; max-height:90vh; border-radius:16px; }}
</style>
</head>
<body>
  <div class="wrap">
    <img src="{url}" alt="Generated image"/>
  </div>
</body>
</html>
```"""


def _process_candidate_parts(cand):
    """
    Generator yielding strings for pieces found in candidate.content.parts:
     - yields part.text when present
     - yields fenced html <img ...> blocks for inline images
    """
    if not cand or not getattr(cand, "content", None):
        return
    parts = getattr(cand.content, "parts", []) or []
    for part in parts:
        if getattr(part, "text", None):
            yield part.text
        if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
            try:
                mime = getattr(part.inline_data, "mime_type", "image/png")
                data = part.inline_data.data
                url = _save_inline_asset(mime, data)
                yield f'\n```html\n<img src="{url}" alt="Generated image"/>\n```\n'
            except Exception:
                log.exception("[GEMINI] saving inline image failed")
                continue


def _pick_text_fallback() -> str:
    """
    Pick a safe text-capable Gemini model when the selected remote is images-only.
    Prefer fast, widely-available models.
    """
    prefs = [
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-1.5-flash",
        "models/gemini-2.5-pro",
    ]
    for name in prefs:
        if name in _GEMINI_REMOTES or name in _GEMINI_MODELS_STATIC:
            return name
    return "models/gemini-2.5-flash"







def _stream_gemini(model_name: str, messages, system_instruction: str | None = None):
    """
    Streams text/images from Gemini.

    IMPORTANT:
    - If the currently selected remote is images-only (via == "images", e.g. imagen-4.*)
      and the request is TEXT-ONLY, we transparently swap to a text-capable fallback
      (e.g., models/gemini-2.5-flash) so normal chat keeps working.
    - We also treat 404 Not Found from generateContent as a signal to retry via REST,
      and if the model was images-only we retry with the text fallback.
    """
    force_rest = os.getenv("GEMINI_FORCE_REST", "0") == "1"
    can_sdk = _GENAI_AVAILABLE and bool(os.getenv("GEMINI_API_KEY", "").strip())

    # TEXT-only? (no dict/bytes images in messages)
    text_only = all(isinstance(m.get("content"), str) for m in messages)

    # Normalize and inspect the selected remote's capabilities
    norm_name = _normalize_model_path(model_name)
    meta = _GEMINI_REMOTES.get(norm_name, {})
    via = meta.get("via")

    # If the selected remote is images-only and the request is text-only,
    # swap to a text-capable fallback (so "hi" works while an Imagen model is selected).
    target_model = norm_name
    if via == "images" and text_only:
        target_model = _pick_text_fallback()

    # Helper to run the REST single-chunk fallback
    def _rest_fallback():
        for chunk in _rest_generate_content_stream(target_model, messages, system_instruction):
            yield chunk

    # ───────── Path 1: forced REST or no SDK ─────────
    if force_rest or not can_sdk:
        yield from _rest_fallback()
        return

    # ───────── Path 2: SDK streaming first, then REST fallback ─────────
    client = _get_gemini()
    contents = _gemini_contents_from_messages(messages)
    config = genai_types.GenerateContentConfig(
        response_modalities=_gemini_modalities_for(target_model),
        **({"system_instruction": system_instruction} if system_instruction else {})
    )

    try:
        for chunk in client.models.generate_content_stream(
            model=target_model, contents=contents, config=config
        ):
            if getattr(chunk, "text", None):
                yield chunk.text

            cand = (getattr(chunk, "candidates", None) or [None])[0]
            if cand:
                for piece in _process_candidate_parts(cand):
                    if piece:
                        yield piece
        return

    except ClientError as e:
        # Try to capture rate-limit headers if the SDK exposed them
        try:
            _update_last_ratelimit(getattr(getattr(e, "response", None), "headers", None))
        except Exception:
            pass

        code, status, msg = _err_info(e)

        # Typical quota/perm OR "method not found" (404) from generateContent → REST fallback
        if _is_quota(e, msg, status, code) or (text_only and code in (400, 403, 404, 429)):
            try:
                yield from _rest_fallback()
                return
            except Exception:
                log.exception("[GEMINI] REST fallback failed")

        # Surface the error (kept concise for UI)
        yield f"\n[error] Gemini error: {msg}\n"
        return

    except Exception as e:
        log.exception("[GEMINI] Unexpected streaming error: %s", e)
        if text_only:
            try:
                yield from _rest_fallback()
                return
            except Exception:
                log.exception("[GEMINI] REST fallback failed after unexpected error")
        yield "\n[error] Gemini unexpected error (see server logs).\n"
        return




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
_RESOLVED_PATH = None
_RESOLVED_MODE = None

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




# ─────────────────────── llama/health tweaks ───────────────────────
# 🧷 Inside llama_health(), add OpenAI awareness so health doesn’t show red when OpenAI is active.

@app.get("/llama/health")
@nocache
def llama_health():
    env_raw = os.getenv("LLAMA_GGUF", "").strip() or None
    disc = _discover_models()

    global _RESOLVED_PATH, _RESOLVED_MODE
    if _RESOLVED_PATH is not None:
        resolved, mode = _RESOLVED_PATH, _RESOLVED_MODE
    else:
        resolved, mode = _resolve_model_path()

    selected_is_gemini = bool(resolved and (resolved in _GEMINI_REMOTES or mode == "gemini" or resolved in _GEMINI_MODELS_STATIC))
    selected_is_openai = bool(mode == "openai")
    exists = bool(resolved and (os.path.exists(resolved) or resolved in _GEMINI_REMOTES or resolved in _GEMINI_MODELS_STATIC or selected_is_openai))

    gemini_ok = _GENAI_AVAILABLE and bool(os.getenv("GEMINI_API_KEY", "").strip())
    openai_ok = _OPENAI_AVAILABLE and bool(os.getenv("OPENAI_API_KEY", "").strip())
    ok = bool(
        (selected_is_gemini and gemini_ok)
        or (selected_is_openai and openai_ok)
        or (os.path.exists(resolved) if resolved and not (selected_is_gemini or selected_is_openai) else False)
        or selected_is_gemini
        or selected_is_openai
    )

    current_remote = None
    if selected_is_gemini:
        meta = _GEMINI_REMOTES.get(resolved) or {}
        current_remote = {"name": resolved, "image": bool(meta.get("image")), "via": meta.get("via")}
    elif selected_is_openai:
        current_remote = {"name": resolved, "image": False, "via": "openai.responses"}

    return jsonify({
        "ok": ok,
        "import_ok": _LLAMA_AVAILABLE,
        "import_error": _LLAMA_IMPORT_ERR,
        "resolved_mode": mode,
        "resolved_path": resolved,
        "resolved_exists": exists,
        "selected_is_gemini": selected_is_gemini,
        "selected_is_openai": selected_is_openai,   # NEW
        "env_path": env_raw,
        "discovered": disc,

       "remotes": (list(_GEMINI_REMOTES.keys()) + list(_OPENAI_REMOTES.keys()))
                   or (_GEMINI_MODELS_STATIC + [f"openai/{m}" for m in _OPENAI_MODELS_STATIC]),
        "remotes_details": {
            **{k: {"image": v.get("image"), "via": v.get("via")} for k, v in _GEMINI_REMOTES.items()},
            **{k: {"image": v.get("image"), "via": v.get("via")} for k, v in _OPENAI_REMOTES.items()},
        },
        

        "current_remote": current_remote,
        "model_loaded": _llm is not None,
        "cfg": { **DEFAULT_CFG, "model_path": os.path.basename(resolved) if resolved and os.path.isabs(resolved) else resolved },
        "ratelimit": dict(_LAST_RATELIMIT),
        "hint": "Set LLAMA_GGUF or place a *.gguf under ./models; for Gemini set GEMINI_API_KEY; for OpenAI set OPENAI_API_KEY",
    }), (200 if ok else 503)




# ─────────────────────── /llama/switch extension ───────────────────────
# 🧷 Inside the existing llama_switch() just before the "Local model selection" branch.

@app.post("/llama/switch")
@nocache
def llama_switch():
    if not is_admin():
        return jsonify({"error": "Forbidden"}), 403

    data = request.get_json(silent=True) or {}
    model_path = data.get("model_path")

    global _llm, _RESOLVED_PATH, _RESOLVED_MODE
    with _llm_load_lock:

        # NEW: OpenAI remote selection (accept registry keys or plain ids)
        if _is_openai_model(model_path) or model_path in _OPENAI_REMOTES:
            _llm = None
            _RESOLVED_PATH = model_path if model_path.startswith("openai/") else _normalize_openai_model(model_path)
            _RESOLVED_MODE = "openai"
            log.info(f"[BACKEND] Using OpenAI model: {_RESOLVED_PATH}")
            return jsonify({"status": "switched", "model": _RESOLVED_PATH, "image": False, "via": "openai.responses"})


        # Gemini remote selection
        if model_path in _GEMINI_REMOTES or model_path in _GEMINI_MODELS_STATIC:
            _llm = None
            _RESOLVED_PATH = model_path
            _RESOLVED_MODE = "gemini"
            log.info(f"[BACKEND] Using Gemini model: {_RESOLVED_PATH}")
            try:
                client = _get_gemini()
                _refresh_gemini_models(client)
            except Exception:
                log.exception("[BACKEND] re-init Gemini client failed")
            meta = _GEMINI_REMOTES.get(model_path, {})
            return jsonify({"status": "switched", "model": _RESOLVED_PATH, "image": bool(meta.get("image")), "via": meta.get("via")})

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

    return jsonify({"status": "switched", "model": os.path.basename(_RESOLVED_PATH), "image": False, "via": None})


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

    # NEW: OpenAI branch
    if _RESOLVED_MODE == "openai":
        def stream_plain():
            try:
                for chunk in _stream_openai(messages[-12:]):
                    if chunk:
                        yield chunk
            except Exception:
                log.exception("[ASSISTANT] OpenAI generation failed")
                yield "\n[error]\n"
        resp = Response(stream_plain(), mimetype="text/plain")
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    # (Gemini branch remains next, then Llama branch)
    if _RESOLVED_MODE == "gemini" and (_RESOLVED_PATH in _GEMINI_REMOTES or _RESOLVED_PATH in _GEMINI_MODELS_STATIC):

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


# --- Image intent detection (place near your other helpers) ---
_IMG_VERBS = ("generate", "make", "create", "render", "draw", "paint", "produce")
_IMG_NOUNS = ("image", "picture", "photo", "pic", "art", "illustration", "png", "jpg", "jpeg", "gif")

def _detect_image_request(text: str) -> bool:
    if not text:
        return False
    t = text.lower().strip()

    # single-word cues like "image" or "picture"
    if t in _IMG_NOUNS:
        return True

    # verb + image noun anywhere
    if any(v in t for v in _IMG_VERBS) and any(n in t for n in _IMG_NOUNS):
        return True

    # common "image/picture/photo of ..." phrasing
    if re.search(r"\b(image|picture|photo|illustration|art)\s+of\b", t):
        return True

    # short "subject + photo/image" phrasing (e.g., "cat photo")
    if re.search(r"\b(\w+)\s+(photo|image|picture|illustration)\b", t):
        return True

    return False


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


# ─────────────────────── /creator (OpenAI branch) ───────────────────────
# 🧷 Add an OpenAI path parallel to your Gemini/Llama logic.
#     - OpenAI path is text-only; if wants_image is True, show the same helpful error.

@app.post("/creator")
@login_required
@nocache
@limiter.limit("30/minute")
def creator_generate():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages") or []
    if not messages:
        return jsonify({"error": "Missing 'messages'."}), 400

    last_text = (messages[-1].get("content") or "")

    wants_image = bool(re.search(
        r"^\s*(generate|make|create|render)\b.*\b(image|picture|photo)\b|^\s*(an?\s+)?(image|picture|photo)\s+of\b",
        last_text,
        re.IGNORECASE,
    ))

    # Prefer currently selected Imagen model if selected, otherwise a fast default
    active_image_model = "imagen-4.0-fast-generate-001"
    try:
        sel = (_RESOLVED_PATH or "").lower()
        if "imagen" in sel:
            active_image_model = _RESOLVED_PATH
    except Exception:
        pass

    # NEW: OpenAI branch (text-only creator; uses your CREATOR_SYS_INST)
    if _RESOLVED_MODE == "openai":
        if wants_image:
            return Response(
                "\n[error] Image generation requires an image-capable backend. "
                "Switch to a Gemini Imagen model (e.g., /llama/switch to 'imagen-4.0-fast-generate-001').\n",
                mimetype="text/plain",
            )
        def stream_plain():
            try:
                for chunk in _stream_openai(messages[-12:], system_instruction=CREATOR_SYS_INST):
                    if chunk:
                        yield chunk
            except Exception:
                log.exception("[CREATOR] OpenAI generation failed")
                yield "\n[error]\n"
        resp = Response(stream_plain(), mimetype="text/plain")
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    # (Gemini branch unchanged)
    if _RESOLVED_MODE == "gemini" and (_RESOLVED_PATH in _GEMINI_REMOTES or _RESOLVED_PATH in _GEMINI_MODELS_STATIC):
        if wants_image:
            try:
                html_doc = _generate_image_best_effort(last_text)
                return Response((c for c in [html_doc]), mimetype="text/plain")
            except Exception as e:
                code, status, msg = _err_info(e)
                return Response(
                    f"\n[error] Image generation failed ({code or status or 'error'}): {msg}\n",
                    mimetype="text/plain",
                )

        # Normal (non-image) Creator flow
        def stream_plain():
            try:
                for chunk in _stream_gemini(
                    _RESOLVED_PATH,
                    messages[-12:],
                    system_instruction=CREATOR_SYS_INST,
                ):
                    if chunk:
                        yield chunk
            except Exception:
                log.exception("[CREATOR] Gemini generation failed")
                yield "\n[error]\n"

        resp = Response(stream_plain(), mimetype="text/plain")
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    # ───────────── Llama branch ─────────────
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

    if wants_image:
        return Response(
            "\n[error] Image generation requires an image-capable backend. "
            "Switch to a Gemini Imagen model (e.g., POST /llama/switch with 'imagen-4.0-fast-generate-001').\n",
            mimetype="text/plain",
        )

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
        return jsonify({
            "error": "context_overflow",
            "detail": f"Prompt too large for n_ctx={n_ctx}. Reduce input."
        }), 413

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





# Extra endpoint to re-list remote Gemini models on demand
@app.post("/llama/refresh_remotes")
@nocache
def refresh_remotes():
    if not is_admin():
        return jsonify({"error": "Forbidden"}), 403
    try:
        # Gemini
        client = _get_gemini()
        _refresh_gemini_models(client)
    except Exception:
        log.exception("[GEMINI] refresh_remotes failed (Gemini)")

    try:
        # OpenAI (static + OPENAI_MODELS env)
        _refresh_openai_models()
    except Exception:
        log.exception("[OPENAI] refresh_remotes failed (OpenAI)")

    all_remotes = list(_GEMINI_REMOTES.keys()) + list(_OPENAI_REMOTES.keys())
    return jsonify({"status": "ok", "remotes": all_remotes})



# ─────────────────────────────── Main ───────────────────────────────
if __name__ == "__main__":
    # existing Gemini warm-up...
    if _GENAI_AVAILABLE and os.getenv("GEMINI_API_KEY", "").strip():
        try:
            _get_gemini()
        except Exception:
            log.exception("[GEMINI] initial client init failed")

    # NEW: prime OpenAI registry
    try:
        _refresh_openai_models()
    except Exception:
        log.exception("[OPENAI] initial registry init failed")

    host = os.getenv("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_RUN_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)
