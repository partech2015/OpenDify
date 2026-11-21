import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import secrets
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx
import ujson
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, UJSONResponse
from starlette.background import BackgroundTask

# 加载环境变量（优先于日志配置）
load_dotenv()

# 日志配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").strip().upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.WARNING), format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
for lib in ["httpx", "httpcore", "uvicorn.access"]:
    logging.getLogger(lib).setLevel(logging.ERROR)

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

def _env_first(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if value:
            return value
    return default

VALID_API_KEYS = [key.strip() for key in os.getenv("VALID_API_KEYS", "").split(",") if key.strip()]
VALID_API_KEYS_SET = frozenset(VALID_API_KEYS)
DIFY_API_BASE = (os.getenv("DIFY_API_BASE", "https://api.dify.ai/v1") or "").rstrip("/")
TIMEOUT = float(os.getenv("TIMEOUT", "180.0") or "180.0")
AUTH_MODE = os.getenv("AUTH_MODE", "required").strip().lower()  # required|disabled
try:
    CONVERSATION_MEMORY_MODE = int(os.getenv("CONVERSATION_MEMORY_MODE", "1").strip() or "1")
except Exception:
    CONVERSATION_MEMORY_MODE = 1

# 性能优化常量
CONNECTION_POOL_SIZE = 100
CONNECTION_TIMEOUT = TIMEOUT
KEEPALIVE_TIMEOUT = 30.0
TTL_APP_CACHE_SECONDS = 1800.0
TTL_FILE_CACHE_SECONDS = 900.0
MAX_FILE_CACHE_ENTRIES = 128

# 工具与非流式配置
TOOL_SUPPORT = _env_flag("TOOL_SUPPORT", True)
TOOL_BUFFER_LIMIT = int(os.getenv("TOOL_BUFFER_LIMIT", "2000000") or "2000000")
DIFY_FORCE_STREAMING = _env_flag("DIFY_FORCE_STREAMING", True)
MAX_FILES_PER_REQUEST = int(os.getenv("MAX_FILES_PER_REQUEST", "10") or "10")

OPEN_TOOL_TAG = "<opendify_tool_call>"
CLOSE_TOOL_TAG = "</opendify_tool_call>"

def fast_uuid() -> str:
    """快速生成UUID hex"""
    return secrets.token_hex(16)

def _parse_model_map(raw: Optional[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not raw or not isinstance(raw, str):
        return mapping
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" in item:
            left, right = item.split(":", 1)
        elif "=" in item:
            left, right = item.split("=", 1)
        else:
            continue
        src = left.strip()
        dst = right.strip()
        if src and dst:
            mapping[src] = dst
    return mapping

def _normalize_openai_chat_completions_url(value: Optional[str]) -> str:
    url = (value or "").strip()
    if not url:
        return ""
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"

def _anthropic_error_payload(message: str, *, type: str) -> Dict[str, Any]:
    return {"type": "error", "error": {"type": type, "message": message}}

def _is_anthropic_compat_request(request: Request) -> bool:
    path = request.url.path or ""
    if path.startswith("/v1/messages") or path.startswith("/anthropic/v1/messages"):
        return True
    if path.startswith("/v1/models") and (request.headers.get("anthropic-version") or request.headers.get("anthropic-beta")):
        return True
    if path.startswith("/anthropic/v1/models") and (request.headers.get("anthropic-version") or request.headers.get("anthropic-beta")):
        return True
    return False

def _anthropic_error_type_from_status(status_code: int, *, openai_code: Optional[str] = None) -> str:
    if status_code == 401 or openai_code == "invalid_api_key":
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 404:
        return "not_found_error"
    if status_code == 429:
        return "rate_limit_error"
    if status_code >= 500:
        return "api_error"
    return "invalid_request_error"

def _estimate_token_count_from_text(text: str) -> int:
    if not text:
        return 0
    try:
        cjk = 0
        total = 0
        for ch in text:
            total += 1
            if "\u4e00" <= ch <= "\u9fff":
                cjk += 1
        if total and (cjk / total) >= 0.3:
            return total
        return max(1, int(round(len(text.encode("utf-8")) / 4)))
    except Exception:
        return max(1, int(round(len(text) / 4)))

def _extract_text_from_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
        return " ".join(parts)
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text
        return _extract_text_from_content([value])
    return str(value)

def _claude_request_to_text_for_token_count(req: Dict[str, Any]) -> str:
    parts: List[str] = []

    system = req.get("system")
    system_text = _extract_text_from_content(system)
    if system_text.strip():
        parts.append(system_text.strip())

    messages = req.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or "user"
            content = msg.get("content")
            if isinstance(content, str):
                if content.strip():
                    parts.append(f"[{role}]: {content.strip()}")
                continue

            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        text = block.get("text")
                        if isinstance(text, str) and text.strip():
                            parts.append(text.strip())
                    elif btype == "tool_use":
                        name = block.get("name")
                        if isinstance(name, str) and name.strip():
                            parts.append(name.strip())
                        input_obj = block.get("input")
                        if isinstance(input_obj, dict) and input_obj:
                            parts.append(ujson.dumps(input_obj, ensure_ascii=False))
                    elif btype == "tool_result":
                        tr_content = _extract_text_from_content(block.get("content"))
                        if tr_content.strip():
                            parts.append(tr_content.strip())
                continue

            other = _extract_text_from_content(content)
            if other.strip():
                parts.append(f"[{role}]: {other.strip()}")

    tools = req.get("tools")
    if isinstance(tools, list) and tools:
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = tool.get("name")
            if isinstance(name, str) and name.strip():
                parts.append(name.strip())
            desc = tool.get("description")
            if isinstance(desc, str) and desc.strip():
                parts.append(desc.strip())
            schema = tool.get("input_schema")
            if isinstance(schema, dict) and schema:
                parts.append(ujson.dumps(schema, ensure_ascii=False))

    return "\n".join(parts)

def _openai_error_payload(
    message: str,
    *,
    type: str = "invalid_request_error",
    code: Optional[str] = None,
    param: Optional[str] = None,
) -> Dict[str, Any]:
    return {"error": {"message": message, "type": type, "param": param, "code": code}}

class OpenAIHTTPError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        type: str = "invalid_request_error",
        code: Optional[str] = None,
        param: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.type = type
        self.code = code
        self.param = param

    def payload(self) -> Dict[str, Any]:
        return _openai_error_payload(self.message, type=self.type, code=self.code, param=self.param)

class DifyModelManager:
    """管理Dify模型与API密钥映射及文件缓存"""

    def __init__(self):
        self.api_keys: List[str] = []
        self.name_to_api_key: Dict[str, str] = {}
        self._app_cache: Dict[str, Tuple[str, float]] = {}
        self._file_cache: OrderedDict[Tuple[str, str, str], Tuple[str, float]] = OrderedDict()
        self._refresh_lock = asyncio.Lock()

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout=CONNECTION_TIMEOUT, connect=10.0),
            limits=httpx.Limits(
                max_keepalive_connections=CONNECTION_POOL_SIZE,
                max_connections=CONNECTION_POOL_SIZE,
                keepalive_expiry=KEEPALIVE_TIMEOUT,
            ),
            http2=True,
            verify=_env_flag("DIFY_SSL_VERIFY", True),
        )
        self.load_api_keys()

    def load_api_keys(self):
        keys_str = os.getenv("DIFY_API_KEYS", "")
        self.api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]

    async def fetch_app_info(self, api_key: str) -> Optional[str]:
        try:
            now = time.monotonic()
            if api_key in self._app_cache:
                cached_name, cached_time = self._app_cache[api_key]
                if now - cached_time < TTL_APP_CACHE_SECONDS:
                    return cached_name

            headers = {"Authorization": f"Bearer {api_key}"}
            rsp = await self._client.get(
                f"{DIFY_API_BASE}/info",
                headers=headers,
                params={"user": "default_user"},
                timeout=10.0,
            )

            if rsp.status_code == 200:
                app_info = ujson.loads(rsp.content)
                app_name = app_info.get("name", "Unknown App")
                self._app_cache[api_key] = (app_name, now)
                return app_name
            logger.warning(f"Failed to fetch Dify app info (status={rsp.status_code}) for key {api_key[:8]}...")
            return None
        except Exception as e:
            logger.warning(f"Error fetching Dify app info: {e}")
            return None

    async def refresh_model_info(self):
        async with self._refresh_lock:
            tasks = [self.fetch_app_info(key) for key in self.api_keys]
            names = await asyncio.gather(*tasks, return_exceptions=True)

            new_map: Dict[str, str] = {}
            for key, name in zip(self.api_keys, names):
                if isinstance(name, str) and name:
                    if name in new_map:
                        logger.warning(f"Duplicate Dify app name '{name}'; later API key overrides earlier one")
                    new_map[name] = key

            # 仅当刷新成功或未配置 key 时替换，避免短暂断网期间原子清空导致并发 404
            if new_map or not self.api_keys:
                self.name_to_api_key = new_map

    async def ensure_loaded(self):
        if not self.name_to_api_key and self.api_keys:
            await self.refresh_model_info()

    def get_api_key(self, model: str) -> Optional[str]:
        return self.name_to_api_key.get(model)

    def get_available_models(self) -> List[Dict[str, Any]]:
        timestamp = int(time.time())
        return [
            {"id": name, "object": "model", "created": timestamp, "owned_by": "dify"}
            for name in self.name_to_api_key.keys()
        ]

    async def resolve_image_file(
        self, dify_key: Optional[str], image_url: str, user_id: str
    ) -> Optional[Dict[str, Any]]:
        if not image_url or not isinstance(image_url, str):
            return None
        url = image_url.strip()
        if not url:
            return None

        if url.startswith("http://") or url.startswith("https://"):
            return {"type": "image", "transfer_method": "remote_url", "url": url}

        if not dify_key:
            return None

        mime_type = "image/png"
        ext = ".png"
        b64_payload = url

        if url.startswith("data:"):
            if "," not in url:
                return None
            header, b64_payload = url.split(",", 1)
            mime_match = re.match(r"data:(image/[a-zA-Z0-9.+-]+)", header, re.IGNORECASE)
            if mime_match:
                mime_type = mime_match.group(1).lower()
                if mime_type in ("image/jpeg", "image/jpg"):
                    mime_type = "image/jpeg"
                    ext = ".jpg"
                elif mime_type == "image/png":
                    ext = ".png"
                elif mime_type == "image/webp":
                    ext = ".webp"
                elif mime_type == "image/gif":
                    ext = ".gif"
                else:
                    ext = ".png"

        try:
            img_bytes = base64.b64decode(b64_payload, validate=False)
        except Exception as e:
            logger.warning(f"Failed to decode base64 image: {e}")
            return None

        if not img_bytes:
            return None

        img_hash = hashlib.sha256(img_bytes).hexdigest()
        cache_key = (dify_key, user_id, img_hash)
        now = time.monotonic()

        cached = self._file_cache.get(cache_key)
        if cached:
            cached_id, cached_ts = cached
            if now - cached_ts < TTL_FILE_CACHE_SECONDS:
                self._file_cache.move_to_end(cache_key)
                return {"type": "image", "transfer_method": "local_file", "upload_file_id": cached_id}
            self._file_cache.pop(cache_key, None)

        try:
            headers = {"Authorization": f"Bearer {dify_key}"}
            files = {"file": (f"image{ext}", img_bytes, mime_type)}
            data = {"user": user_id}
            rsp = await self._client.post(
                f"{DIFY_API_BASE}/files/upload",
                headers=headers,
                files=files,
                data=data,
                timeout=30.0,
            )
            if rsp.status_code in (200, 201):
                file_info = ujson.loads(rsp.content)
                file_id = file_info.get("id")
                if file_id:
                    self._file_cache[cache_key] = (str(file_id), now)
                    while len(self._file_cache) > MAX_FILE_CACHE_ENTRIES:
                        self._file_cache.popitem(last=False)
                    return {"type": "image", "transfer_method": "local_file", "upload_file_id": str(file_id)}
            logger.warning(f"Dify /files/upload failed (status={rsp.status_code}): {rsp.text[:300]}")
        except Exception as e:
            logger.warning(f"Error uploading image to Dify: {e}")

        return None

    async def close(self):
        await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client

model_manager = DifyModelManager()

anthropic_client = httpx.AsyncClient(
    timeout=httpx.Timeout(timeout=CONNECTION_TIMEOUT, connect=10.0),
    limits=httpx.Limits(
        max_keepalive_connections=CONNECTION_POOL_SIZE,
        max_connections=CONNECTION_POOL_SIZE,
        keepalive_expiry=KEEPALIVE_TIMEOUT,
    ),
    http2=True,
    verify=_env_flag("ANTHROPIC_SSL_VERIFY", True),
)

openai_upstream_client = httpx.AsyncClient(
    timeout=httpx.Timeout(timeout=CONNECTION_TIMEOUT, connect=10.0),
    limits=httpx.Limits(
        max_keepalive_connections=CONNECTION_POOL_SIZE,
        max_connections=CONNECTION_POOL_SIZE,
        keepalive_expiry=KEEPALIVE_TIMEOUT,
    ),
    http2=True,
    verify=_env_flag("UPSTREAM_OPENAI_SSL_VERIFY", True),
)

def _rfc3339_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

async def _resolve_claude_model_to_dify(model: str) -> Tuple[str, str]:
    await model_manager.ensure_loaded()

    dify_key = model_manager.get_api_key(model)
    if dify_key:
        return model, dify_key

    model_map = _parse_model_map(_env_first("CLAUDE_MODEL_MAP", default=""))
    mapped = model_map.get(model)
    if mapped:
        mapped_key = model_manager.get_api_key(mapped)
        if mapped_key:
            return mapped, mapped_key

    default_dify_model = _env_first("CLAUDE_DEFAULT_MODEL", default=None)
    if default_dify_model:
        default_key = model_manager.get_api_key(default_dify_model)
        if default_key:
            logger.warning(f"Claude model '{model}' not found, fallback to CLAUDE_DEFAULT_MODEL '{default_dify_model}'")
            return default_dify_model, default_key

    available = list(model_manager.name_to_api_key.keys())
    if available:
        fallback_model = available[0]
        fallback_key = model_manager.get_api_key(fallback_model)
        if fallback_key:
            logger.warning(f"Claude model '{model}' not found, fallback to first Dify model '{fallback_model}'")
            return fallback_model, fallback_key

    raise OpenAIHTTPError(
        404,
        f"The model '{model}' does not exist",
        type="invalid_request_error",
        code="model_not_found",
        param="model",
    )

async def _anthropic_models_page(request: Request) -> Dict[str, Any]:
    await model_manager.ensure_loaded()

    model_map = _parse_model_map(_env_first("CLAUDE_MODEL_MAP", default=""))
    model_ids = list(model_map.keys()) if model_map else list(model_manager.name_to_api_key.keys())
    if not model_ids:
        upstream_model = _env_first("UPSTREAM_OPENAI_MODEL", "model", default=None)
        if upstream_model:
            model_ids = [upstream_model.strip()]

    limit_raw = request.query_params.get("limit")
    before_id = request.query_params.get("before_id")
    after_id = request.query_params.get("after_id")

    limit: Optional[int] = None
    if isinstance(limit_raw, str) and limit_raw.strip():
        try:
            limit_val = int(limit_raw)
            if limit_val > 0:
                limit = limit_val
        except Exception:
            limit = None

    start = 0
    end = len(model_ids)
    if isinstance(after_id, str) and after_id in model_ids:
        start = model_ids.index(after_id) + 1
    if isinstance(before_id, str) and before_id in model_ids:
        end = model_ids.index(before_id)
    if end < start:
        end = start

    window = model_ids[start:end]
    if limit is not None:
        window = window[:limit]

    has_more = (start + len(window)) < end
    first_id = window[0] if window else None
    last_id = window[-1] if window else None

    created_at = _rfc3339_now()
    data = [{"id": mid, "type": "model", "display_name": mid, "created_at": created_at} for mid in window]
    return {"data": data, "has_more": bool(has_more), "first_id": first_id, "last_id": last_id}

async def _anthropic_model_info(model_id: str) -> Dict[str, Any]:
    await model_manager.ensure_loaded()

    model_map = _parse_model_map(_env_first("CLAUDE_MODEL_MAP", default=""))
    upstream_model = _env_first("UPSTREAM_OPENAI_MODEL", "model", default=None)
    exists = (
        (model_id in model_map)
        or (model_id in model_manager.name_to_api_key)
        or (bool(upstream_model) and model_id == upstream_model.strip())
    )
    if not exists:
        raise OpenAIHTTPError(
            404,
            f"The model '{model_id}' does not exist",
            type="invalid_request_error",
            code="model_not_found",
            param="model",
        )

    return {"id": model_id, "type": "model", "display_name": model_id, "created_at": _rfc3339_now()}

@asynccontextmanager
async def lifespan(_: FastAPI):
    if not VALID_API_KEYS and AUTH_MODE != "disabled":
        logger.warning("VALID_API_KEYS not configured")
    await model_manager.refresh_model_info()
    yield
    await anthropic_client.aclose()
    await openai_upstream_client.aclose()
    await model_manager.close()

app = FastAPI(title="Dify to OpenAI API Proxy", docs_url=None, redoc_url=None, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,
)

@app.exception_handler(OpenAIHTTPError)
async def _openai_http_error_handler(request: Request, exc: OpenAIHTTPError) -> UJSONResponse:
    if _is_anthropic_compat_request(request):
        err_type = _anthropic_error_type_from_status(exc.status_code, openai_code=exc.code)
        return UJSONResponse(status_code=exc.status_code, content=_anthropic_error_payload(exc.message, type=err_type))
    return UJSONResponse(status_code=exc.status_code, content=exc.payload())

@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, exc: RequestValidationError) -> UJSONResponse:
    if _is_anthropic_compat_request(request):
        return UJSONResponse(
            status_code=400,
            content=_anthropic_error_payload("Invalid request parameters", type="invalid_request_error"),
        )
    return UJSONResponse(
        status_code=400,
        content=_openai_error_payload(
            "Invalid request parameters",
            type="invalid_request_error",
            code="invalid_request_error",
            param=None,
        ),
    )

@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException) -> UJSONResponse:
    detail = exc.detail

    if _is_anthropic_compat_request(request):
        message = "Request failed"
        openai_code: Optional[str] = None

        if isinstance(detail, dict):
            if isinstance(detail.get("error"), dict):
                err_obj = detail["error"]
                message = err_obj.get("message") or message
                openai_code = err_obj.get("code")
            elif isinstance(detail.get("message"), str):
                message = detail.get("message") or message

        if detail is not None and not isinstance(detail, dict):
            message = str(detail)

        err_type = _anthropic_error_type_from_status(exc.status_code, openai_code=openai_code)
        return UJSONResponse(status_code=exc.status_code, content=_anthropic_error_payload(message, type=err_type))

    if isinstance(detail, dict) and "error" in detail and isinstance(detail["error"], dict):
        return UJSONResponse(status_code=exc.status_code, content=detail)

    message = str(detail) if detail is not None else "Request failed"
    err_type = "invalid_request_error" if exc.status_code < 500 else "server_error"
    return UJSONResponse(status_code=exc.status_code, content=_openai_error_payload(message, type=err_type, code=None, param=None))

async def verify_api_key(request: Request) -> str:
    if AUTH_MODE == "disabled":
        return ""
    auth_header = request.headers.get("Authorization")
    x_api_key = request.headers.get("X-API-Key")

    key: Optional[str] = None
    if isinstance(auth_header, str) and auth_header.strip():
        stripped = auth_header.strip()
        if stripped.lower().startswith("bearer "):
            key = stripped[7:].strip()
        else:
            key = stripped
    elif isinstance(x_api_key, str) and x_api_key.strip():
        key = x_api_key.strip()

    if not key:
        raise OpenAIHTTPError(
            401,
            "Invalid Authorization header",
            type="invalid_request_error",
            code="invalid_api_key",
            param=None,
        )

    if key not in VALID_API_KEYS_SET:
        raise OpenAIHTTPError(
            401,
            "Invalid API key",
            type="invalid_request_error",
            code="invalid_api_key",
            param=None,
        )
    return key

def generate_tool_prompt(tools: List[Dict[str, Any]]) -> str:
    """生成保留完整参数Schema的工具定义提示"""
    if not tools:
        return ""

    tool_definitions: List[str] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue

        function_spec = tool.get("function") or {}
        if not isinstance(function_spec, dict):
            continue
        function_name = (function_spec.get("name") or "").strip()
        if not function_name:
            continue

        function_description = (function_spec.get("description") or "").strip()
        parameters = function_spec.get("parameters")
        if not isinstance(parameters, dict) or not parameters:
            parameters = {"type": "object", "properties": {}}

        params_json = ujson.dumps(parameters, ensure_ascii=False)
        tool_info = [
            f"## {function_name}",
            f"**Purpose**: {function_description}" if function_description else "**Purpose**: (none)",
            f"**Parameters JSON Schema**: {params_json}",
        ]
        tool_definitions.append("\n".join(tool_info))

    if not tool_definitions:
        return ""

    return "# AVAILABLE FUNCTIONS\n\n" + "\n\n---\n\n".join(tool_definitions)

def generate_tool_protocol_footer(tool_choice: Any) -> str:
    """将工具调用输出规范放在请求末尾，提高长上下文下的指令遵循率"""
    lines = [
        "# TOOL CALLING PROTOCOL",
        f"If you decide to call one or more functions from AVAILABLE FUNCTIONS, wrap EACH call in `{OPEN_TOOL_TAG}` and `{CLOSE_TOOL_TAG}` tags:",
        OPEN_TOOL_TAG,
        '{"name": "function_name", "arguments": {"param": "value"}}',
        CLOSE_TOOL_TAG,
        "Rules:",
        "1. `name` MUST exactly match a function name from AVAILABLE FUNCTIONS.",
        "2. `arguments` MUST be a valid JSON object matching the function's Parameters JSON Schema (do NOT double-escape JSON into a string).",
        f"3. To invoke multiple tools in parallel, output multiple `{OPEN_TOOL_TAG}...{CLOSE_TOOL_TAG}` blocks.",
        f"4. If no function call is needed, answer the user directly WITHOUT `{OPEN_TOOL_TAG}` tags.",
    ]
    if tool_choice == "required":
        lines.append(f"[IMPORTANT]: You MUST invoke at least one function in this turn using `{OPEN_TOOL_TAG}...{CLOSE_TOOL_TAG}`.")
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        fname = ((tool_choice.get("function") or {}).get("name") or "").strip()
        if fname:
            lines.append(f"[IMPORTANT]: You MUST invoke `{fname}` in this turn using `{OPEN_TOOL_TAG}...{CLOSE_TOOL_TAG}`.")
    return "\n".join(lines)

def _extract_content_text_and_images(content: Any) -> Tuple[str, List[str]]:
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if isinstance(content, list):
        text_parts: List[str] = []
        image_urls: List[str] = []
        for item in content:
            if isinstance(item, str):
                if item:
                    text_parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype in ("text", "input_text"):
                txt = item.get("text")
                if isinstance(txt, str) and txt:
                    text_parts.append(txt)
            elif itype == "image_url":
                img_obj = item.get("image_url")
                if isinstance(img_obj, dict):
                    url = img_obj.get("url")
                elif isinstance(img_obj, str):
                    url = img_obj
                else:
                    url = None
                if isinstance(url, str) and url.strip():
                    image_urls.append(url.strip())
                    text_parts.append("[Image Attached]")
            elif itype == "input_image":
                url = item.get("image_url") or item.get("url")
                if isinstance(url, str) and url.strip():
                    image_urls.append(url.strip())
                    text_parts.append("[Image Attached]")
            elif isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        return "\n".join(text_parts), image_urls
    if isinstance(content, dict):
        return _extract_content_text_and_images([content])
    return str(content), []

async def transform_openai_to_dify(
    openai_request: Dict[str, Any],
    conversation_id: Optional[str],
    dify_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """将OpenAI请求转换为Dify请求（完整支持多模态图片、developer角色与工具协议）"""
    messages = openai_request.get("messages", [])
    if not messages:
        return None

    user_id = str(openai_request.get("user") or "default_user").strip() or "default_user"

    tools = openai_request.get("tools") or []
    if not tools and isinstance(openai_request.get("functions"), list):
        tools = [{"type": "function", "function": f} for f in openai_request.get("functions") if isinstance(f, dict)]
    tool_choice = openai_request.get("tool_choice", "auto")

    if CONVERSATION_MEMORY_MODE == 2 and conversation_id:
        last_assistant_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            if (messages[idx] or {}).get("role") == "assistant":
                last_assistant_idx = idx
                break
        system_messages = [m for m in messages if (m or {}).get("role") in ("system", "developer")]
        delta_messages = messages[last_assistant_idx + 1:] if last_assistant_idx >= 0 else messages
        messages_for_query = system_messages + [
            m for m in delta_messages if (m or {}).get("role") not in ("system", "developer")
        ]
    else:
        messages_for_query = messages

    query_parts: List[str] = []
    collected_image_urls: List[str] = []

    for msg in messages_for_query:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content_text, img_urls = _extract_content_text_and_images(msg.get("content"))
        if img_urls:
            collected_image_urls.extend(img_urls)

        if role in ("system", "developer"):
            if content_text.strip():
                query_parts.append(f"[System]: {content_text}")
        elif role == "user":
            query_parts.append(f"[User]: {content_text}")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if not tool_calls and isinstance(msg.get("function_call"), dict):
                tool_calls = [{"type": "function", "function": msg["function_call"]}]

            assistant_segments: List[str] = []
            if content_text.strip():
                assistant_segments.append(content_text)

            if isinstance(tool_calls, list) and tool_calls:
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    func = tc.get("function") or {}
                    if not isinstance(func, dict):
                        continue
                    fname = (func.get("name") or "").strip()
                    fargs = func.get("arguments")
                    parsed_args: Any = {}
                    if isinstance(fargs, str) and fargs.strip():
                        try:
                            parsed_args = ujson.loads(fargs)
                        except Exception:
                            parsed_args = fargs
                    elif isinstance(fargs, dict):
                        parsed_args = fargs
                    call_json = ujson.dumps({"name": fname, "arguments": parsed_args}, ensure_ascii=False)
                    assistant_segments.append(f"{OPEN_TOOL_TAG}\n{call_json}\n{CLOSE_TOOL_TAG}")

            if assistant_segments:
                query_parts.append(f"[Assistant]: " + "\n".join(assistant_segments))
        elif role == "tool":
            tool_name = msg.get("name") or "tool"
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id:
                query_parts.append(f"[Tool '{tool_name}' (tool_call_id={tool_call_id}) Result]: {content_text}")
            else:
                query_parts.append(f"[Tool '{tool_name}' Result]: {content_text}")
        else:
            if content_text.strip():
                query_parts.append(f"[{role or 'user'}]: {content_text}")

    has_active_tools = bool(tools and TOOL_SUPPORT and tool_choice != "none")
    if has_active_tools:
        tools_prompt = generate_tool_prompt(tools)
        if tools_prompt:
            # 将函数列表插在首条System之后（或最前），把调用协议放在末尾靠近最新提问
            insert_idx = 0
            while insert_idx < len(query_parts) and query_parts[insert_idx].startswith("[System]:"):
                insert_idx += 1
            query_parts.insert(insert_idx, tools_prompt)
            query_parts.append(generate_tool_protocol_footer(tool_choice))

    user_query = "\n\n".join(part for part in query_parts if part)

    inputs_obj = openai_request.get("inputs")
    if not isinstance(inputs_obj, dict):
        inputs_obj = {}

    dify_request: Dict[str, Any] = {
        "inputs": inputs_obj,
        "query": user_query,
        "response_mode": "streaming" if openai_request.get("stream", False) else "blocking",
        "user": user_id,
    }

    if conversation_id:
        dify_request["conversation_id"] = conversation_id

    if collected_image_urls:
        # 若历史图片过多，优先保留最近的 MAX_FILES_PER_REQUEST 张
        recent_urls = collected_image_urls[-MAX_FILES_PER_REQUEST:] if MAX_FILES_PER_REQUEST > 0 else []
        if recent_urls:
            resolved = await asyncio.gather(
                *(model_manager.resolve_image_file(dify_key, u, user_id) for u in recent_urls),
                return_exceptions=True,
            )
            attached_files = [item for item in resolved if isinstance(item, dict)]
            if attached_files:
                dify_request["files"] = attached_files

    return dify_request

def _normalize_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        if "function" in tc and isinstance(tc["function"], dict):
            func = tc["function"]
            name = (func.get("name") or "").strip()
            args = func.get("arguments")
            tc_type = tc.get("type") or "function"
            call_id = tc.get("id") or f"call_{fast_uuid()}"
        elif "name" in tc:
            name = (tc.get("name") or "").strip()
            args = tc.get("arguments")
            tc_type = "function"
            call_id = tc.get("id") or f"call_{fast_uuid()}"
        else:
            continue

        if not name:
            continue

        if args is None:
            args_str = "{}"
        elif isinstance(args, str):
            args_str = args
        else:
            args_str = ujson.dumps(args, ensure_ascii=False)

        normalized.append(
            {
                "id": call_id,
                "type": tc_type,
                "function": {"name": name, "arguments": args_str},
            }
        )
    return normalized

def _strip_optional_code_fence(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        nl_idx = s.find("\n")
        if nl_idx != -1:
            s = s[nl_idx + 1 :]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()

def _parse_tool_call_payload(raw_inner: str) -> Optional[List[Dict[str, Any]]]:
    """解析 <opendify_tool_call> 内部的 JSON（兼容单对象、tool_calls包裹对象、数组）"""
    cleaned = _strip_optional_code_fence(raw_inner)
    if not cleaned:
        return None

    parsed: Any = None
    try:
        parsed = ujson.loads(cleaned)
    except Exception:
        # 若JSON后带了多余文本，尝试用标准库 raw_decode 截取首个完整JSON值
        start_idx = -1
        for i, ch in enumerate(cleaned):
            if ch in "{[":
                start_idx = i
                break
        if start_idx == -1:
            return None
        try:
            parsed, _ = json.JSONDecoder().raw_decode(cleaned, start_idx)
        except Exception:
            return None

    candidates: List[Dict[str, Any]] = []
    if isinstance(parsed, dict):
        if isinstance(parsed.get("tool_calls"), list):
            candidates = [x for x in parsed["tool_calls"] if isinstance(x, dict)]
        elif "name" in parsed or "function" in parsed:
            candidates = [parsed]
    elif isinstance(parsed, list):
        candidates = [x for x in parsed if isinstance(x, dict)]

    normalized = _normalize_tool_calls(candidates)
    return normalized if normalized else None

def _find_complete_close_tag(buf: str) -> Optional[Tuple[str, str]]:
    """
    在 buf 中查找与开头内容匹配的 </opendify_tool_call> 结束标签。
    追踪 JSON 字符串状态与大括号深度，防止参数字符串内包含反引号或 </opendify_tool_call> 导致误截断。
    """
    if CLOSE_TOOL_TAG not in buf:
        return None

    n = len(buf)
    i = 0
    while i < n and buf[i].isspace():
        i += 1
    if i >= n:
        return None

    # 跳过可选的 ```json 行
    if buf.startswith("```", i):
        nl = buf.find("\n", i)
        if nl == -1:
            return None
        i = nl + 1
        while i < n and buf[i].isspace():
            i += 1
        if i >= n:
            return None

    if buf[i] in "{[":
        in_str = False
        escape = False
        depth = 0
        j = i
        json_end = -1
        while j < n:
            ch = buf[j]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if buf.startswith(CLOSE_TOOL_TAG, j):
                    return buf[:j], buf[j + len(CLOSE_TOOL_TAG) :]
                if ch == '"':
                    in_str = True
                elif ch in "{[":
                    depth += 1
                elif ch in "}]":
                    depth -= 1
                    if depth == 0:
                        json_end = j
                        break
            j += 1

        if json_end != -1:
            close_idx = buf.find(CLOSE_TOOL_TAG, json_end + 1)
            if close_idx != -1:
                return buf[:close_idx], buf[close_idx + len(CLOSE_TOOL_TAG) :]
            return None
        return None

    close_idx = buf.find(CLOSE_TOOL_TAG, i)
    if close_idx != -1:
        return buf[:close_idx], buf[close_idx + len(CLOSE_TOOL_TAG) :]
    return None

def _find_opendify_tool_call_spans(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    pos = 0
    while pos < len(text):
        start = text.find(OPEN_TOOL_TAG, pos)
        if start == -1:
            break
        inner_start = start + len(OPEN_TOOL_TAG)
        res = _find_complete_close_tag(text[inner_start:])
        if res is None:
            break
        inner, _ = res
        end = inner_start + len(inner) + len(CLOSE_TOOL_TAG)
        spans.append((start, end, inner))
        pos = end
    return spans

def _find_legacy_fenced_tool_spans(text: str) -> List[Tuple[int, int, List[Dict[str, Any]]]]:
    """无 [^`]+ 缺陷的旧格式 ```json {"tool_calls": ...} ``` 兜底提取"""
    spans: List[Tuple[int, int, List[Dict[str, Any]]]] = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"```(?:json)?\s*\{", text, re.IGNORECASE):
        fence_start = match.start()
        brace_pos = match.end() - 1
        try:
            obj, json_end = decoder.raw_decode(text, brace_pos)
        except Exception:
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("tool_calls"), list):
            continue
        rest = text[json_end:]
        ws_len = len(rest) - len(rest.lstrip())
        after_ws = json_end + ws_len
        if text.startswith("```", after_ws):
            normalized = _normalize_tool_calls(obj["tool_calls"])
            if normalized:
                spans.append((fence_start, after_ws + 3, normalized))
    return spans

def extract_tool_invocations(text: str) -> Optional[List[Dict[str, Any]]]:
    """从完整响应文本中提取工具调用（优先新标签，兜底旧围栏JSON）"""
    if not text:
        return None

    scannable = text[:TOOL_BUFFER_LIMIT] if TOOL_BUFFER_LIMIT > 0 else text

    all_calls: List[Dict[str, Any]] = []
    for _, _, inner in _find_opendify_tool_call_spans(scannable):
        parsed = _parse_tool_call_payload(inner)
        if parsed:
            all_calls.extend(parsed)

    if all_calls:
        return all_calls

    for _, _, calls in _find_legacy_fenced_tool_spans(scannable):
        all_calls.extend(calls)

    return all_calls if all_calls else None

def remove_tool_json_content(text: str) -> str:
    """从响应文本中移除已提取的工具调用块"""
    if not text:
        return ""

    spans: List[Tuple[int, int]] = [(s, e) for s, e, inner in _find_opendify_tool_call_spans(text) if _parse_tool_call_payload(inner)]
    for s, e, _ in _find_legacy_fenced_tool_spans(text):
        spans.append((s, e))

    if not spans:
        return text.strip()

    spans.sort()
    pieces: List[str] = []
    cursor = 0
    for s, e in spans:
        if s >= cursor:
            pieces.append(text[cursor:s])
            cursor = e
    pieces.append(text[cursor:])
    return "".join(pieces).strip()

class _OpenDifyStreamParser:
    """流式增量解析器：阻止 <opendify_tool_call> 原文泄露到客户端 content 中"""

    def __init__(self):
        self.text_buf: str = ""
        self.in_tag: bool = False
        self.tag_buf: str = ""

    def feed(self, chunk: str) -> List[Tuple[str, Any]]:
        events: List[Tuple[str, Any]] = []
        pending = chunk
        while pending:
            if not self.in_tag:
                self.text_buf += pending
                pending = ""

                idx = self.text_buf.find(OPEN_TOOL_TAG)
                if idx != -1:
                    if idx > 0:
                        events.append(("text", self.text_buf[:idx]))
                    self.in_tag = True
                    pending = self.text_buf[idx + len(OPEN_TOOL_TAG) :]
                    self.text_buf = ""
                else:
                    max_prefix = min(len(self.text_buf), len(OPEN_TOOL_TAG) - 1)
                    matched_k = 0
                    for k in range(max_prefix, 0, -1):
                        if self.text_buf.endswith(OPEN_TOOL_TAG[:k]):
                            matched_k = k
                            break
                    if matched_k > 0:
                        safe = self.text_buf[:-matched_k]
                        if safe:
                            events.append(("text", safe))
                        self.text_buf = self.text_buf[-matched_k:]
                    else:
                        if self.text_buf:
                            events.append(("text", self.text_buf))
                            self.text_buf = ""
            else:
                self.tag_buf += pending
                pending = ""

                if TOOL_BUFFER_LIMIT > 0 and len(self.tag_buf) > TOOL_BUFFER_LIMIT:
                    events.append(("text", OPEN_TOOL_TAG + self.tag_buf))
                    self.tag_buf = ""
                    self.in_tag = False
                    break

                res = _find_complete_close_tag(self.tag_buf)
                if res is not None:
                    inner, remainder = res
                    parsed = _parse_tool_call_payload(inner)
                    if parsed:
                        events.append(("tool_calls", parsed))
                    else:
                        events.append(("text", OPEN_TOOL_TAG + inner + CLOSE_TOOL_TAG))
                    self.tag_buf = ""
                    self.in_tag = False
                    pending = remainder

        return events

    def finalize(self) -> List[Tuple[str, Any]]:
        events: List[Tuple[str, Any]] = []
        if self.in_tag:
            close_idx = self.tag_buf.rfind(CLOSE_TOOL_TAG)
            if close_idx != -1:
                inner = self.tag_buf[:close_idx]
                remainder = self.tag_buf[close_idx + len(CLOSE_TOOL_TAG) :]
                parsed = _parse_tool_call_payload(inner)
                if parsed:
                    events.append(("tool_calls", parsed))
                    if remainder:
                        events.append(("text", remainder))
                else:
                    events.append(("text", OPEN_TOOL_TAG + self.tag_buf))
            else:
                events.append(("text", OPEN_TOOL_TAG + self.tag_buf))
            self.tag_buf = ""
            self.in_tag = False

        if self.text_buf:
            events.append(("text", self.text_buf))
            self.text_buf = ""

        return events

def _to_unix_timestamp(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        raw = value.strip()
        if raw.isdigit():
            return int(raw)
        try:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except Exception:
            return int(time.time())
    return int(time.time())

def _ensure_chatcmpl_id(value: Any) -> str:
    if not value:
        return f"chatcmpl-{fast_uuid()}"
    s = str(value)
    if s.startswith("chatcmpl-"):
        return s
    return f"chatcmpl-{s}"

def _ensure_resp_id(value: Any) -> str:
    if not value:
        return f"resp_{fast_uuid()}"
    s = str(value)
    if s.startswith("resp_"):
        return s
    if s.startswith("chatcmpl-"):
        s = s[len("chatcmpl-") :]
    return f"resp_{s}"

def _ensure_anthropic_msg_id(value: Any) -> str:
    if not value:
        return f"msg_{fast_uuid()}"
    s = str(value)
    if s.startswith("msg_"):
        return s
    if s.startswith("chatcmpl-"):
        s = s[len("chatcmpl-") :]
    return f"msg_{s}"

def _responses_input_to_chat_messages(input_value: Any) -> List[Dict[str, Any]]:
    if input_value is None:
        return []
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]

    if isinstance(input_value, dict):
        if isinstance(input_value.get("role"), str):
            return [
                {
                    "role": input_value.get("role") or "user",
                    "content": _extract_text_from_content(input_value.get("content")),
                }
            ]
        return [{"role": "user", "content": _extract_text_from_content([input_value])}]

    if isinstance(input_value, list):
        looks_like_messages = any(isinstance(it, dict) and isinstance(it.get("role"), str) for it in input_value)
        if looks_like_messages:
            messages: List[Dict[str, Any]] = []
            for it in input_value:
                if not isinstance(it, dict):
                    continue
                role = it.get("role") or "user"
                content = it.get("content")
                if isinstance(content, list):
                    messages.append({"role": role, "content": content})
                else:
                    messages.append({"role": role, "content": _extract_text_from_content(content)})
            return messages
        return [{"role": "user", "content": input_value}]

    return [{"role": "user", "content": str(input_value)}]

def transform_dify_to_openai_response(dify_response: Dict[str, Any], model: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """将Dify响应转换为OpenAI标准格式"""
    answer = dify_response.get("answer", "")
    tool_calls: List[Dict[str, Any]] = []

    if answer and TOOL_SUPPORT:
        extracted_tools = extract_tool_invocations(answer)
        if extracted_tools:
            tool_calls = _normalize_tool_calls(extracted_tools)
            answer = remove_tool_json_content(answer)

    if "tool_calls" in dify_response and not tool_calls:
        tool_calls = _normalize_tool_calls(list(dify_response.get("tool_calls") or []))

    message_content: Dict[str, Any] = {
        "role": "assistant",
        "content": answer.strip() if isinstance(answer, str) and answer.strip() else None,
    }

    if tool_calls:
        message_content["tool_calls"] = tool_calls

    usage = (dify_response.get("metadata") or {}).get("usage") if isinstance(dify_response.get("metadata"), dict) else None
    if not isinstance(usage, dict):
        usage = {}

    openai_response = {
        "id": _ensure_chatcmpl_id(dify_response.get("message_id")),
        "object": "chat.completion",
        "created": _to_unix_timestamp(dify_response.get("created_at")),
        "model": model,
        "system_fingerprint": "fp_dify",
        "service_tier": None,
        "choices": [
            {
                "index": 0,
                "message": message_content,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }

    conversation_id = dify_response.get("conversation_id")
    return openai_response, conversation_id

async def transform_openai_responses_to_dify(
    responses_request: Dict[str, Any],
    conversation_id: Optional[str],
    dify_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    model = responses_request.get("model")
    if not isinstance(model, str) or not model.strip():
        return None

    messages: List[Dict[str, Any]] = []

    instructions = responses_request.get("instructions")
    instructions_text = _extract_text_from_content(instructions)
    if instructions_text.strip():
        messages.append({"role": "system", "content": instructions_text})

    input_value = responses_request.get("input")
    messages.extend(_responses_input_to_chat_messages(input_value))

    openai_like: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": bool(responses_request.get("stream", False)),
    }

    if "tools" in responses_request:
        openai_like["tools"] = responses_request.get("tools") or []
    if "tool_choice" in responses_request:
        openai_like["tool_choice"] = responses_request.get("tool_choice")

    user = responses_request.get("user")
    if isinstance(user, str) and user.strip():
        openai_like["user"] = user
    elif isinstance(responses_request.get("metadata"), dict):
        meta_user = (responses_request.get("metadata") or {}).get("user_id")
        if meta_user is not None:
            openai_like["user"] = str(meta_user)

    return await transform_openai_to_dify(openai_like, conversation_id, dify_key=dify_key)

def transform_dify_to_openai_responses(dify_response: Dict[str, Any], model: str) -> Tuple[Dict[str, Any], Optional[str]]:
    answer = dify_response.get("answer", "")
    tool_calls: List[Dict[str, Any]] = []

    if answer and TOOL_SUPPORT:
        extracted_tools = extract_tool_invocations(answer)
        if extracted_tools:
            tool_calls = _normalize_tool_calls(extracted_tools)
            answer = remove_tool_json_content(answer)

    if "tool_calls" in dify_response and not tool_calls:
        tool_calls = _normalize_tool_calls(list(dify_response.get("tool_calls") or []))

    output: List[Dict[str, Any]] = []
    for tc in tool_calls:
        func = tc.get("function") or {}
        call_id = tc.get("id") or f"call_{fast_uuid()}"
        output.append(
            {
                "id": call_id,
                "type": "function_call",
                "call_id": call_id,
                "name": (func or {}).get("name") or "",
                "arguments": (func or {}).get("arguments") or "",
                "status": "completed",
            }
        )

    answer_text = answer.strip() if isinstance(answer, str) else str(answer)
    if answer_text:
        output.append(
            {
                "id": f"msg_{fast_uuid()}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": answer_text, "annotations": []}],
            }
        )

    usage = (dify_response.get("metadata") or {}).get("usage") if isinstance(dify_response.get("metadata"), dict) else None
    if not isinstance(usage, dict):
        usage = {}

    if not output:
        output.append(
            {
                "id": f"msg_{fast_uuid()}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "", "annotations": []}],
            }
        )

    resp = {
        "id": _ensure_resp_id(dify_response.get("message_id")),
        "object": "response",
        "created_at": _to_unix_timestamp(dify_response.get("created_at")),
        "model": model,
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": usage.get("completion_tokens", 0),
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": usage.get("total_tokens", 0),
        },
        "error": None,
    }

    conversation_id = dify_response.get("conversation_id")
    return resp, conversation_id

def _claude_tools_to_openai_tools(tools: Any) -> List[Dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    converted: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        parameters = tool.get("input_schema") or tool.get("parameters") or {}
        if not isinstance(parameters, dict) or not parameters:
            parameters = {"type": "object", "properties": {}}
        elif parameters.get("type") is None:
            parameters = dict(parameters)
            parameters["type"] = "object"
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description") or "",
                    "parameters": parameters,
                },
            }
        )
    return converted

def _claude_message_to_openai_messages(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    role = msg.get("role") or "user"
    content = msg.get("content")

    if isinstance(content, str):
        return [{"role": role, "content": content}]

    if not isinstance(content, list):
        return [{"role": role, "content": _extract_text_from_content(content)}]

    content_parts: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    tool_uses: List[Dict[str, Any]] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            txt = block.get("text")
            if isinstance(txt, str) and txt.strip():
                content_parts.append({"type": "text", "text": txt})
        elif btype == "image":
            source = block.get("source") or {}
            if isinstance(source, dict):
                stype = source.get("type")
                if stype == "base64" and source.get("data"):
                    media_type = source.get("media_type") or "image/png"
                    content_parts.append(
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{source['data']}"}}
                    )
                elif stype == "url" and source.get("url"):
                    content_parts.append({"type": "image_url", "image_url": {"url": source["url"]}})
        elif btype == "tool_result":
            tool_results.append(block)
        elif btype == "tool_use":
            tool_uses.append(block)

    primary_content: Any
    if any(p.get("type") == "image_url" for p in content_parts):
        primary_content = content_parts
    else:
        primary_content = " ".join(p.get("text", "") for p in content_parts).strip()

    primary: Dict[str, Any] = {"role": role, "content": primary_content}

    if role == "assistant" and tool_uses:
        tool_calls: List[Dict[str, Any]] = []
        for tu in tool_uses:
            call_id = tu.get("id") or f"call_{fast_uuid()}"
            name = tu.get("name") or ""
            raw_input = tu.get("input")
            if raw_input is None:
                args_str = "{}"
            elif isinstance(raw_input, str):
                args_str = raw_input
            else:
                args_str = ujson.dumps(raw_input, ensure_ascii=False)
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                }
            )
        primary["tool_calls"] = tool_calls

    out: List[Dict[str, Any]] = [primary]

    for tr in tool_results:
        tool_use_id = tr.get("tool_use_id") or tr.get("tool_call_id") or tr.get("id")
        tr_content = tr.get("content")
        tool_content = _extract_text_from_content(tr_content)
        tool_msg: Dict[str, Any] = {"role": "tool", "name": tr.get("name") or "tool_result", "content": tool_content}
        if tool_use_id is not None:
            tool_msg["tool_call_id"] = str(tool_use_id)
        out.append(tool_msg)

    return out

def _claude_message_to_openai_messages_for_upstream(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    role = msg.get("role") or "user"
    content = msg.get("content")

    if isinstance(content, str):
        return [{"role": role, "content": content}]

    if not isinstance(content, list):
        return [{"role": role, "content": _extract_text_from_content(content)}]

    text_parts: List[str] = []
    tool_results: List[Dict[str, Any]] = []
    tool_uses: List[Dict[str, Any]] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            txt = block.get("text")
            if isinstance(txt, str) and txt.strip():
                text_parts.append(txt)
        elif btype == "tool_result":
            tool_results.append(block)
        elif btype == "tool_use":
            tool_uses.append(block)

    primary_text = " ".join(text_parts).strip() if text_parts else ""

    tool_calls: List[Dict[str, Any]] = []
    if role == "assistant" and tool_uses:
        for tu in tool_uses:
            name = tu.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            call_id = tu.get("id") or f"call_{fast_uuid()}"
            raw_input = tu.get("input")
            if raw_input is None:
                args_str = "{}"
            elif isinstance(raw_input, str):
                args_str = raw_input
            else:
                args_str = ujson.dumps(raw_input, ensure_ascii=False)
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name.strip(), "arguments": args_str},
                }
            )

    out: List[Dict[str, Any]] = []

    if role == "assistant":
        primary_content: Any = primary_text
        if tool_calls and not primary_text:
            primary_content = None
        if tool_calls or primary_text or not tool_results:
            primary: Dict[str, Any] = {"role": role, "content": primary_content}
            if tool_calls:
                primary["tool_calls"] = tool_calls
            out.append(primary)
    elif role == "user":
        if primary_text or not tool_results:
            out.append({"role": role, "content": primary_text})
    else:
        out.append({"role": role, "content": primary_text})

    for tr in tool_results:
        tool_use_id = tr.get("tool_use_id") or tr.get("tool_call_id") or tr.get("id")
        tool_content = _extract_text_from_content(tr.get("content"))
        if tool_use_id is None or (isinstance(tool_use_id, str) and not tool_use_id.strip()):
            out.append({"role": "user", "content": tool_content})
            continue
        out.append({"role": "tool", "tool_call_id": str(tool_use_id), "content": tool_content})

    if out:
        return out
    return [{"role": role, "content": primary_text}]

def _summarize_openai_chat_completion_request(req: Dict[str, Any]) -> Dict[str, Any]:
    try:
        messages = req.get("messages") or []
        roles: List[str] = []
        tool_msgs_missing_id = 0

        if isinstance(messages, list):
            for m in messages:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                if isinstance(role, str):
                    roles.append(role)
                    if role == "tool":
                        tcid = m.get("tool_call_id")
                        if not (isinstance(tcid, str) and tcid.strip()):
                            tool_msgs_missing_id += 1

        tools = req.get("tools")
        tool_count = len(tools) if isinstance(tools, list) else 0

        return {
            "model": req.get("model"),
            "stream": bool(req.get("stream", False)),
            "max_tokens": req.get("max_tokens"),
            "message_count": len(messages) if isinstance(messages, list) else None,
            "roles": roles[:32],
            "tools_count": tool_count,
            "tool_choice": req.get("tool_choice"),
            "tool_msgs_missing_tool_call_id": tool_msgs_missing_id,
        }
    except Exception:
        return {"summary_error": True}

def _claude_tool_choice_to_openai(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, dict):
        return None
    choice_type = tool_choice.get("type")
    if choice_type in (None, "auto"):
        return "auto"
    if choice_type in ("any", "required"):
        return "required"
    if choice_type in ("tool", "function"):
        name = tool_choice.get("name")
        if not isinstance(name, str) or not name.strip():
            return None
        return {"type": "function", "function": {"name": name}}
    return None

async def transform_claude_messages_to_dify(
    claude_request: Dict[str, Any],
    conversation_id: Optional[str],
    dify_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    model = claude_request.get("model")
    if not isinstance(model, str) or not model.strip():
        return None

    claude_messages = claude_request.get("messages")
    if not isinstance(claude_messages, list) or not claude_messages:
        return None

    messages: List[Dict[str, Any]] = []

    system = claude_request.get("system")
    system_text = _extract_text_from_content(system)
    if system_text.strip():
        messages.append({"role": "system", "content": system_text})

    for msg in claude_messages:
        if not isinstance(msg, dict):
            continue
        messages.extend(_claude_message_to_openai_messages(msg))

    openai_like: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": bool(claude_request.get("stream", False)),
    }

    openai_tools = _claude_tools_to_openai_tools(claude_request.get("tools"))
    if openai_tools:
        openai_like["tools"] = openai_tools

    mapped_tool_choice = _claude_tool_choice_to_openai(claude_request.get("tool_choice"))
    if mapped_tool_choice is not None:
        openai_like["tool_choice"] = mapped_tool_choice

    metadata = claude_request.get("metadata")
    if isinstance(metadata, dict) and metadata.get("user_id") is not None:
        openai_like["user"] = str(metadata.get("user_id"))

    return await transform_openai_to_dify(openai_like, conversation_id, dify_key=dify_key)

def _openai_tool_call_to_anthropic_tool_use(tc: Dict[str, Any]) -> Dict[str, Any]:
    call_id = tc.get("id") or f"call_{fast_uuid()}"
    func = tc.get("function") or {}
    name = (func or {}).get("name") or ""
    args = (func or {}).get("arguments") or ""
    input_obj: Any = {}
    if isinstance(args, str) and args.strip():
        try:
            input_obj = ujson.loads(args)
        except Exception:
            input_obj = {"_raw": args}
    return {"type": "tool_use", "id": call_id, "name": name, "input": input_obj}

def _openai_finish_reason_to_anthropic_stop_reason(finish_reason: Any, *, tool_calls_present: bool) -> str:
    if tool_calls_present or finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "content_filter":
        return "refusal"
    return "end_turn"

def transform_claude_messages_to_openai_chat_completion_request(
    claude_request: Dict[str, Any],
    *,
    upstream_model: str,
    default_max_tokens: int,
) -> Dict[str, Any]:
    claude_messages = claude_request.get("messages")
    if not isinstance(claude_messages, list) or not claude_messages:
        raise OpenAIHTTPError(400, "Missing required parameter: 'messages'", param="messages", code="missing_required_parameter")

    messages: List[Dict[str, Any]] = []

    system = claude_request.get("system")
    system_text = _extract_text_from_content(system)
    if system_text.strip():
        messages.append({"role": "system", "content": system_text})

    for msg in claude_messages:
        if not isinstance(msg, dict):
            continue
        messages.extend(_claude_message_to_openai_messages_for_upstream(msg))

    max_tokens = claude_request.get("max_tokens")
    if max_tokens is None:
        max_tokens_int = default_max_tokens
    else:
        try:
            max_tokens_int = int(max_tokens)
        except Exception:
            max_tokens_int = default_max_tokens
    if max_tokens_int <= 0:
        max_tokens_int = default_max_tokens

    openai_like: Dict[str, Any] = {
        "model": upstream_model,
        "messages": messages,
        "stream": bool(claude_request.get("stream", False)),
        "max_tokens": int(max_tokens_int),
    }

    temperature = claude_request.get("temperature")
    if temperature is not None:
        try:
            openai_like["temperature"] = float(temperature)
        except Exception:
            pass

    top_p = claude_request.get("top_p")
    if top_p is not None:
        try:
            openai_like["top_p"] = float(top_p)
        except Exception:
            pass

    stop_sequences = claude_request.get("stop_sequences")
    if stop_sequences is not None:
        if isinstance(stop_sequences, str) and stop_sequences.strip():
            openai_like["stop"] = stop_sequences.strip()
        elif isinstance(stop_sequences, list):
            stops = [s for s in stop_sequences if isinstance(s, str) and s.strip()]
            if stops:
                openai_like["stop"] = stops

    openai_tools = _claude_tools_to_openai_tools(claude_request.get("tools"))
    if openai_tools:
        openai_like["tools"] = openai_tools

    mapped_tool_choice = _claude_tool_choice_to_openai(claude_request.get("tool_choice"))
    if mapped_tool_choice is not None:
        openai_like["tool_choice"] = mapped_tool_choice

    metadata = claude_request.get("metadata")
    if isinstance(metadata, dict) and metadata.get("user_id") is not None:
        openai_like["user"] = str(metadata.get("user_id"))

    return openai_like

def transform_openai_chat_completion_to_claude_message(openai_resp: Dict[str, Any], *, request_model: str) -> Dict[str, Any]:
    choices = openai_resp.get("choices") or []
    if not isinstance(choices, list) or not choices:
        raise OpenAIHTTPError(502, "Upstream returned no choices", type="api_error", code="bad_upstream", param=None)

    choice0 = choices[0] if isinstance(choices[0], dict) else {}
    message = choice0.get("message") or {}
    if not isinstance(message, dict):
        message = {}

    content_text = message.get("content")
    if not isinstance(content_text, str):
        content_text = "" if content_text is None else str(content_text)

    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        tool_calls = []

    content_blocks: List[Dict[str, Any]] = []
    if content_text.strip():
        content_blocks.append({"type": "text", "text": content_text})

    for tc in tool_calls:
        if isinstance(tc, dict):
            content_blocks.append(_openai_tool_call_to_anthropic_tool_use(tc))

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    usage = openai_resp.get("usage") if isinstance(openai_resp.get("usage"), dict) else {}
    finish_reason = choice0.get("finish_reason")
    stop_reason = _openai_finish_reason_to_anthropic_stop_reason(finish_reason, tool_calls_present=bool(tool_calls))

    return {
        "id": _ensure_anthropic_msg_id(openai_resp.get("id")),
        "type": "message",
        "role": "assistant",
        "model": request_model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)},
    }

def transform_dify_to_claude_message(dify_response: Dict[str, Any], model: str) -> Tuple[Dict[str, Any], Optional[str]]:
    answer = dify_response.get("answer", "")
    tool_calls: List[Dict[str, Any]] = []

    if answer and TOOL_SUPPORT:
        extracted_tools = extract_tool_invocations(answer)
        if extracted_tools:
            tool_calls = _normalize_tool_calls(extracted_tools)
            answer = remove_tool_json_content(answer)

    if "tool_calls" in dify_response and not tool_calls:
        tool_calls = _normalize_tool_calls(list(dify_response.get("tool_calls") or []))

    content: List[Dict[str, Any]] = []
    answer_text = answer.strip() if isinstance(answer, str) else str(answer)
    if answer_text:
        content.append({"type": "text", "text": answer_text})

    if tool_calls:
        for tc in tool_calls:
            content.append(_openai_tool_call_to_anthropic_tool_use(tc))

    if not content:
        content.append({"type": "text", "text": ""})

    meta_usage = (dify_response.get("metadata") or {}).get("usage") if isinstance(dify_response.get("metadata"), dict) else None
    if not isinstance(meta_usage, dict):
        meta_usage = {}

    stop_reason = "tool_use" if tool_calls else "end_turn"

    resp = {
        "id": _ensure_anthropic_msg_id(dify_response.get("message_id")),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": meta_usage.get("prompt_tokens", 0),
            "output_tokens": meta_usage.get("completion_tokens", 0),
        },
    }

    conversation_id = dify_response.get("conversation_id")
    return resp, conversation_id

def _anthropic_stop_reason_to_openai_finish_reason(stop_reason: Any, *, tool_calls_sent: bool) -> str:
    if tool_calls_sent or stop_reason == "tool_use":
        return "tool_calls"
    if stop_reason == "max_tokens":
        return "length"
    if stop_reason == "refusal":
        return "content_filter"
    return "stop"

def _openai_tools_to_anthropic_tools(openai_tools: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(openai_tools, list):
        return out

    for tool in openai_tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue

        func = tool.get("function") or {}
        if not isinstance(func, dict):
            func = {}

        name = (func.get("name") or "").strip()
        if not name:
            continue

        description = func.get("description") or ""
        parameters = func.get("parameters")
        if not isinstance(parameters, dict) or not parameters:
            parameters = {"type": "object", "properties": {}}
        if parameters.get("type") is None:
            parameters = dict(parameters)
            parameters["type"] = "object"

        out.append({"name": name, "description": description, "input_schema": parameters})
    return out

def _openai_tool_choice_to_anthropic_tool_choice(tool_choice: Any) -> Optional[Dict[str, Any]]:
    if tool_choice is None or tool_choice == "auto":
        return None

    if tool_choice == "none":
        return {"type": "none"}
    if tool_choice == "required":
        return {"type": "any"}

    if not isinstance(tool_choice, dict):
        return None

    tc_type = tool_choice.get("type")
    if tc_type == "function":
        function_obj = tool_choice.get("function") or {}
        if not isinstance(function_obj, dict):
            function_obj = {}
        fname = function_obj.get("name")
        if isinstance(fname, str) and fname.strip():
            return {"type": "tool", "name": fname.strip()}
        return None

    if tc_type == "tool":
        name = tool_choice.get("name")
        if isinstance(name, str) and name.strip():
            return {"type": "tool", "name": name.strip()}
        return None

    if tc_type in ("any", "none"):
        return {"type": tc_type}

    return None

def transform_openai_chat_to_anthropic_request(
    openai_request: Dict[str, Any],
    *,
    upstream_model: str,
    default_max_tokens: int,
) -> Dict[str, Any]:
    messages = openai_request.get("messages") or []

    system_parts: List[str] = []
    anthropic_messages: List[Dict[str, Any]] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")

        if role in ("system", "developer"):
            system_text = _extract_text_from_content(msg.get("content"))
            if system_text.strip():
                system_parts.append(system_text.strip())
            continue

        if role == "user":
            user_text = _extract_text_from_content(msg.get("content"))
            anthropic_messages.append({"role": "user", "content": user_text})
            continue

        if role == "assistant":
            blocks: List[Dict[str, Any]] = []
            assistant_text = _extract_text_from_content(msg.get("content"))
            if assistant_text.strip():
                blocks.append({"type": "text", "text": assistant_text})

            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    func = tc.get("function") or {}
                    if not isinstance(func, dict):
                        func = {}
                    name = (func.get("name") or "").strip()
                    if not name:
                        continue

                    call_id = tc.get("id") or f"call_{fast_uuid()}"
                    args = func.get("arguments")
                    input_obj: Dict[str, Any] = {}
                    if isinstance(args, str):
                        args_str = args.strip()
                        if args_str:
                            try:
                                parsed = ujson.loads(args_str)
                                if isinstance(parsed, dict):
                                    input_obj = parsed
                            except Exception:
                                input_obj = {}
                    elif isinstance(args, dict):
                        input_obj = args
                    blocks.append({"type": "tool_use", "id": call_id, "name": name, "input": input_obj})

            if not blocks:
                blocks.append({"type": "text", "text": ""})
            anthropic_messages.append({"role": "assistant", "content": blocks})
            continue

        if role == "tool":
            tool_text = _extract_text_from_content(msg.get("content"))
            tool_call_id = msg.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id.strip():
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": tool_call_id.strip(), "content": tool_text}],
                    }
                )
            else:
                anthropic_messages.append({"role": "user", "content": tool_text})
            continue

        fallback_text = _extract_text_from_content(msg.get("content"))
        anthropic_messages.append({"role": "user", "content": fallback_text})

    if not anthropic_messages:
        anthropic_messages.append({"role": "user", "content": ""})

    max_tokens = openai_request.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = openai_request.get("max_tokens")
    if max_tokens is None:
        max_tokens = default_max_tokens
    try:
        max_tokens_int = int(max_tokens)
    except Exception:
        max_tokens_int = default_max_tokens
    if max_tokens_int <= 0:
        max_tokens_int = default_max_tokens

    req: Dict[str, Any] = {
        "model": upstream_model,
        "max_tokens": max_tokens_int,
        "messages": anthropic_messages,
        "stream": bool(openai_request.get("stream", False)),
    }

    if system_parts:
        req["system"] = "\n".join(system_parts)

    temperature = openai_request.get("temperature")
    if temperature is not None:
        try:
            req["temperature"] = float(temperature)
        except Exception:
            pass

    top_p = openai_request.get("top_p")
    if top_p is not None:
        try:
            req["top_p"] = float(top_p)
        except Exception:
            pass

    stop = openai_request.get("stop")
    if stop is not None:
        if isinstance(stop, str) and stop.strip():
            req["stop_sequences"] = [stop]
        elif isinstance(stop, list):
            stops = [s for s in stop if isinstance(s, str) and s.strip()]
            if stops:
                req["stop_sequences"] = stops

    openai_tools = openai_request.get("tools")
    if not openai_tools and isinstance(openai_request.get("functions"), list):
        openai_tools = [{"type": "function", "function": f} for f in openai_request.get("functions") if isinstance(f, dict)]

    anthropic_tools = _openai_tools_to_anthropic_tools(openai_tools)
    if anthropic_tools:
        req["tools"] = anthropic_tools

        tool_choice = _openai_tool_choice_to_anthropic_tool_choice(openai_request.get("tool_choice"))
        if tool_choice is not None:
            req["tool_choice"] = tool_choice

    return req

def transform_anthropic_to_openai_chat_completion(
    anthropic_message: Dict[str, Any],
    *,
    request_model: str,
) -> Dict[str, Any]:
    content_blocks = anthropic_message.get("content") or []
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    if isinstance(content_blocks, list):
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    text_parts.append(text)
            elif btype == "tool_use":
                call_id = block.get("id") or f"call_{fast_uuid()}"
                name = (block.get("name") or "").strip()
                input_obj = block.get("input")
                if not isinstance(input_obj, dict):
                    input_obj = {}
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": ujson.dumps(input_obj, ensure_ascii=False)},
                    }
                )

    assistant_text = "".join(text_parts)
    message_content: Dict[str, Any] = {
        "role": "assistant",
        "content": assistant_text if assistant_text.strip() else None,
    }
    if tool_calls:
        message_content["tool_calls"] = tool_calls

    usage = anthropic_message.get("usage") if isinstance(anthropic_message.get("usage"), dict) else {}
    prompt_tokens = usage.get("input_tokens", 0) if isinstance(usage, dict) else 0
    completion_tokens = usage.get("output_tokens", 0) if isinstance(usage, dict) else 0
    try:
        prompt_tokens_int = int(prompt_tokens)
    except Exception:
        prompt_tokens_int = 0
    try:
        completion_tokens_int = int(completion_tokens)
    except Exception:
        completion_tokens_int = 0

    stop_reason = anthropic_message.get("stop_reason")
    finish_reason = _anthropic_stop_reason_to_openai_finish_reason(stop_reason, tool_calls_sent=bool(tool_calls))

    return {
        "id": _ensure_chatcmpl_id(anthropic_message.get("id")),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request_model,
        "system_fingerprint": "fp_anthropic",
        "service_tier": None,
        "choices": [
            {
                "index": 0,
                "message": message_content,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens_int,
            "completion_tokens": completion_tokens_int,
            "total_tokens": prompt_tokens_int + completion_tokens_int,
        },
    }

async def _iter_anthropic_sse_events(rsp: httpx.Response) -> AsyncGenerator[Tuple[Optional[str], Dict[str, Any]], None]:
    event_name: Optional[str] = None
    data_lines: List[str] = []

    async for line in rsp.aiter_lines():
        if line == "":
            if data_lines:
                raw = "\n".join(data_lines).strip()
                data_lines = []
                try:
                    data = ujson.loads(raw)
                except Exception:
                    data = None
                if isinstance(data, dict):
                    yield event_name, data
            event_name = None
            continue

        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
            continue

        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
            continue

    if data_lines:
        raw = "\n".join(data_lines).strip()
        try:
            data = ujson.loads(raw)
        except Exception:
            data = None
        if isinstance(data, dict):
            yield event_name, data

async def stream_anthropic_chat_completion(
    rsp: httpx.Response,
    *,
    model: str,
    message_id: str,
    stream_include_usage: bool,
) -> AsyncGenerator[str, None]:
    tool_calls_sent = False
    stop_reason: Optional[str] = None
    usage_obj: Dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    tool_by_content_index: Dict[int, Dict[str, Any]] = {}
    next_tool_index = 0

    def make_sse_chunk(
        delta_content: Optional[str] = None,
        delta_role: Optional[str] = None,
        delta_tool_calls: Optional[List[Dict[str, Any]]] = None,
        finish_reason: Optional[str] = None,
    ) -> str:
        chunk: Dict[str, Any] = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "system_fingerprint": "fp_anthropic",
            "service_tier": None,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
        }
        if delta_role:
            chunk["choices"][0]["delta"]["role"] = delta_role
        if delta_content is not None:
            chunk["choices"][0]["delta"]["content"] = delta_content
        if delta_tool_calls:
            chunk["choices"][0]["delta"]["tool_calls"] = delta_tool_calls
        return f"data: {ujson.dumps(chunk, ensure_ascii=False)}\n\n"

    try:
        yield make_sse_chunk(delta_role="assistant")

        async for event_name, data in _iter_anthropic_sse_events(rsp):
            ev_type = data.get("type") or event_name

            if ev_type == "content_block_start":
                content_block = data.get("content_block") or {}
                if not isinstance(content_block, dict):
                    continue

                if content_block.get("type") == "tool_use":
                    idx = data.get("index")
                    if not isinstance(idx, int):
                        continue

                    call_id = content_block.get("id") or f"call_{fast_uuid()}"
                    name = (content_block.get("name") or "").strip()
                    tool_by_content_index[idx] = {"tool_index": next_tool_index, "id": call_id, "name": name}
                    next_tool_index += 1
                    tool_calls_sent = True

                    yield make_sse_chunk(
                        delta_tool_calls=[
                            {
                                "index": tool_by_content_index[idx]["tool_index"],
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""},
                            }
                        ]
                    )

            elif ev_type == "content_block_delta":
                delta = data.get("delta") or {}
                if not isinstance(delta, dict):
                    continue

                if delta.get("type") == "text_delta":
                    text = delta.get("text")
                    if isinstance(text, str) and text:
                        yield make_sse_chunk(delta_content=text)

                elif delta.get("type") == "input_json_delta":
                    idx = data.get("index")
                    if not isinstance(idx, int):
                        continue
                    info = tool_by_content_index.get(idx)
                    if not info:
                        continue
                    partial_json = delta.get("partial_json")
                    if isinstance(partial_json, str) and partial_json:
                        tool_calls_sent = True
                        yield make_sse_chunk(
                            delta_tool_calls=[
                                {
                                    "index": info["tool_index"],
                                    "id": info["id"],
                                    "type": "function",
                                    "function": {"name": info["name"], "arguments": partial_json},
                                }
                            ]
                        )

            elif ev_type == "message_delta":
                delta_obj = data.get("delta") or {}
                if isinstance(delta_obj, dict) and isinstance(delta_obj.get("stop_reason"), str):
                    stop_reason = delta_obj.get("stop_reason")

                usage = data.get("usage") or {}
                if isinstance(usage, dict):
                    try:
                        prompt_tokens_int = int(usage.get("input_tokens") or 0)
                    except Exception:
                        prompt_tokens_int = 0
                    try:
                        completion_tokens_int = int(usage.get("output_tokens") or 0)
                    except Exception:
                        completion_tokens_int = 0
                    usage_obj = {
                        "prompt_tokens": prompt_tokens_int,
                        "completion_tokens": completion_tokens_int,
                        "total_tokens": prompt_tokens_int + completion_tokens_int,
                    }

            elif ev_type == "message_stop":
                finish_reason = _anthropic_stop_reason_to_openai_stop_reason_compat(stop_reason, tool_calls_sent=tool_calls_sent)
                yield make_sse_chunk(finish_reason=finish_reason)

                if stream_include_usage:
                    yield (
                        "data: "
                        + ujson.dumps(
                            {
                                "id": message_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "system_fingerprint": "fp_anthropic",
                                "service_tier": None,
                                "choices": [],
                                "usage": usage_obj,
                            },
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    )

                yield "data: [DONE]\n\n"
                return

        yield make_sse_chunk(finish_reason="stop")
        if stream_include_usage:
            yield (
                "data: "
                + ujson.dumps(
                    {
                        "id": message_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "system_fingerprint": "fp_anthropic",
                        "service_tier": None,
                        "choices": [],
                        "usage": usage_obj,
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Anthropic stream error: {e}")
        yield make_sse_chunk(finish_reason="error")
        yield "data: [DONE]\n\n"

def _anthropic_stop_reason_to_openai_stop_reason_compat(stop_reason: Any, *, tool_calls_sent: bool) -> str:
    return _anthropic_stop_reason_to_openai_finish_reason(stop_reason, tool_calls_sent=tool_calls_sent)

async def _iter_dify_sse_json(rsp: httpx.Response) -> AsyncGenerator[Dict[str, Any], None]:
    buffer = bytearray()
    async for chunk in rsp.aiter_bytes(8192):
        buffer.extend(chunk)
        while b"\n" in buffer:
            idx = buffer.find(b"\n")
            line = bytes(buffer[:idx]).strip()
            buffer = buffer[idx + 1 :]
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].lstrip()
            if not payload:
                continue
            try:
                data = ujson.loads(payload)
            except Exception:
                continue
            if isinstance(data, dict):
                yield data

    if buffer:
        line = bytes(buffer).strip()
        if line.startswith(b"data:"):
            payload = line[5:].lstrip()
            if payload:
                try:
                    data = ujson.loads(payload)
                    if isinstance(data, dict):
                        yield data
                except Exception:
                    pass

async def _call_dify_non_streaming(dify_key: str, dify_req: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行 Dify 非流式请求。
    当 DIFY_FORCE_STREAMING=true（默认）时，内部走 streaming 聚合成标准响应字典，
    从而原生兼容 Dify Agent 应用（Agent 不支持 blocking 模式）并防止反向代理空闲超时。
    """
    headers = {"Authorization": f"Bearer {dify_key}", "Content-Type": "application/json"}
    endpoint = f"{DIFY_API_BASE}/chat-messages"

    async def _run_via_stream() -> Dict[str, Any]:
        stream_req = dict(dify_req)
        stream_req["response_mode"] = "streaming"

        async with model_manager.client.stream(
            "POST",
            endpoint,
            content=ujson.dumps(stream_req, ensure_ascii=False),
            headers=headers,
            timeout=TIMEOUT,
        ) as rsp:
            if rsp.status_code != 200:
                body = await rsp.aread()
                raise _build_dify_http_error(rsp.status_code, body)

            answer_parts: List[str] = []
            native_tool_calls: List[Dict[str, Any]] = []
            message_id: Optional[str] = None
            conversation_id: Optional[str] = None
            created_at: Optional[Any] = None
            metadata: Dict[str, Any] = {}

            async for data in _iter_dify_sse_json(rsp):
                if data.get("message_id") and not message_id:
                    message_id = data.get("message_id")
                if data.get("conversation_id") and not conversation_id:
                    conversation_id = data.get("conversation_id")
                if data.get("created_at") and created_at is None:
                    created_at = data.get("created_at")

                ev = data.get("event")
                if ev == "error":
                    status = int(data.get("status") or 502)
                    err_msg = data.get("message") or "Dify stream error"
                    err_code = data.get("code") or "dify_error"
                    err_type = "rate_limit_error" if status == 429 else ("server_error" if status >= 500 else "invalid_request_error")
                    raise OpenAIHTTPError(status, err_msg, type=err_type, code=str(err_code))
                elif ev in ("message", "agent_message"):
                    delta = data.get("answer")
                    if isinstance(delta, str) and delta:
                        answer_parts.append(delta)
                elif ev == "tool_calls":
                    tcs = data.get("tool_calls")
                    if isinstance(tcs, list):
                        native_tool_calls.extend(tcs)
                elif ev == "message_end":
                    if isinstance(data.get("metadata"), dict):
                        metadata = data["metadata"]

            result: Dict[str, Any] = {
                "message_id": message_id or fast_uuid(),
                "conversation_id": conversation_id,
                "created_at": created_at or int(time.time()),
                "answer": "".join(answer_parts),
                "metadata": metadata,
            }
            if native_tool_calls:
                result["tool_calls"] = native_tool_calls
            return result

    if DIFY_FORCE_STREAMING:
        return await _run_via_stream()

    blocking_req = dict(dify_req)
    blocking_req["response_mode"] = "blocking"
    resp = await model_manager.client.post(
        endpoint,
        content=ujson.dumps(blocking_req, ensure_ascii=False),
        headers=headers,
        timeout=TIMEOUT,
    )
    if resp.status_code != 200:
        if resp.status_code == 400 and "blocking" in (resp.text or "").lower():
            return await _run_via_stream()
        raise _build_dify_http_error(resp.status_code, resp.content)

    return ujson.loads(resp.content)

def _build_dify_http_error(status_code: int, body: bytes) -> OpenAIHTTPError:
    try:
        dify_error = ujson.loads(body or b"{}")
        error_message = dify_error.get("message") or (body.decode("utf-8", errors="ignore") if body else "Upstream error")
        error_code = dify_error.get("code", "unknown_error")
    except Exception:
        error_message = body.decode("utf-8", errors="ignore") if body else "Upstream error"
        error_code = "unknown_error"

    if status_code == 503:
        err_type = "server_error"
        error_message = f"The model is currently overloaded. Please try again later. (Original: {error_message})"
    elif status_code == 429:
        err_type = "rate_limit_error"
    elif status_code >= 500:
        err_type = "server_error"
    elif status_code >= 400:
        err_type = "invalid_request_error"
    else:
        err_type = "api_error"
    return OpenAIHTTPError(status_code, error_message, type=err_type, code=str(error_code))

async def _iter_openai_sse_json(rsp: httpx.Response) -> AsyncGenerator[Dict[str, Any], None]:
    async for line in rsp.aiter_lines():
        if not line or not isinstance(line, str):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].lstrip()
        if not payload:
            continue
        if payload == "[DONE]":
            return
        try:
            data = ujson.loads(payload)
        except Exception:
            continue
        if isinstance(data, dict):
            yield data

async def stream_openai_chat_completion_as_claude_message(
    rsp: httpx.Response,
    *,
    model: str,
    message_id: str,
) -> AsyncGenerator[str, None]:
    def sse(event_name: str, payload: Dict[str, Any]) -> str:
        return f"event: {event_name}\ndata: {ujson.dumps(payload, ensure_ascii=False)}\n\n"

    accumulated_text = ""
    usage_obj: Dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}

    next_content_index = 0
    text_block_open = False
    current_text_index: Optional[int] = None

    tool_by_openai_index: Dict[int, Dict[str, Any]] = {}

    def _start_text_block() -> int:
        nonlocal next_content_index, text_block_open, current_text_index
        idx = next_content_index
        next_content_index += 1
        current_text_index = idx
        text_block_open = True
        return idx

    yield sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    text_idx = _start_text_block()
    yield sse("content_block_start", {"type": "content_block_start", "index": text_idx, "content_block": {"type": "text", "text": ""}})

    stop_reason: Optional[str] = None
    tool_calls_present = False

    try:
        async for chunk in _iter_openai_sse_json(rsp):
            if isinstance(chunk.get("usage"), dict):
                u = chunk.get("usage") or {}
                if isinstance(u.get("prompt_tokens"), int):
                    usage_obj["input_tokens"] = u.get("prompt_tokens", 0)
                if isinstance(u.get("completion_tokens"), int):
                    usage_obj["output_tokens"] = u.get("completion_tokens", 0)

            choices = chunk.get("choices") or []
            if not isinstance(choices, list) or not choices:
                continue
            choice0 = choices[0] if isinstance(choices[0], dict) else {}
            delta = choice0.get("delta") or {}
            if not isinstance(delta, dict):
                delta = {}

            delta_text = delta.get("content")
            if delta_text is not None:
                if not isinstance(delta_text, str):
                    delta_text = str(delta_text)
                if delta_text:
                    if not text_block_open:
                        text_idx = _start_text_block()
                        yield sse(
                            "content_block_start",
                            {"type": "content_block_start", "index": text_idx, "content_block": {"type": "text", "text": ""}},
                        )
                    accumulated_text += delta_text
                    yield sse(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": text_idx, "delta": {"type": "text_delta", "text": delta_text}},
                    )

            delta_tool_calls = delta.get("tool_calls")
            if isinstance(delta_tool_calls, list) and delta_tool_calls:
                tool_calls_present = True
                if text_block_open and current_text_index is not None:
                    yield sse("content_block_stop", {"type": "content_block_stop", "index": current_text_index})
                    text_block_open = False
                    current_text_index = None

                for fallback_idx, tc in enumerate(delta_tool_calls):
                    if not isinstance(tc, dict):
                        continue
                    openai_idx = tc.get("index")
                    if not isinstance(openai_idx, int):
                        openai_idx = fallback_idx

                    info = tool_by_openai_index.get(openai_idx)
                    if not info:
                        call_id = tc.get("id") or f"call_{fast_uuid()}"
                        func = tc.get("function") or {}
                        name = (func or {}).get("name") or ""
                        idx = next_content_index
                        next_content_index += 1
                        info = {"content_index": idx, "id": call_id, "name": name, "open": True}
                        tool_by_openai_index[openai_idx] = info
                        yield sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": idx,
                                "content_block": {"type": "tool_use", "id": info["id"], "name": info["name"], "input": {}},
                            },
                        )

                    func = tc.get("function") or {}
                    if isinstance(func, dict):
                        if isinstance(func.get("name"), str) and func.get("name") and not info.get("name"):
                            info["name"] = func.get("name")
                        args = func.get("arguments")
                        if args is not None and not isinstance(args, str):
                            args = ujson.dumps(args, ensure_ascii=False)
                        if isinstance(args, str) and args:
                            yield sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": info["content_index"],
                                    "delta": {"type": "input_json_delta", "partial_json": args},
                                },
                            )

            finish_reason = choice0.get("finish_reason")
            if finish_reason is not None:
                stop_reason = _openai_finish_reason_to_anthropic_stop_reason(finish_reason, tool_calls_present=tool_calls_present)
                break

    except Exception as e:
        logger.error(f"OpenAI->Claude stream error: {e}")
        stop_reason = "end_turn"

    if text_block_open and current_text_index is not None:
        yield sse("content_block_stop", {"type": "content_block_stop", "index": current_text_index})
        text_block_open = False
        current_text_index = None

    for info in tool_by_openai_index.values():
        if info.get("open"):
            yield sse("content_block_stop", {"type": "content_block_stop", "index": info.get("content_index")})
            info["open"] = False

    yield sse(
        "message_delta",
        {"type": "message_delta", "delta": {"stop_reason": stop_reason or "end_turn", "stop_sequence": None}, "usage": usage_obj},
    )
    yield sse("message_stop", {"type": "message_stop"})

async def stream_openai_response(
    rsp: httpx.Response,
    *,
    model: str,
    message_id: str,
    stream_include_usage: bool,
) -> AsyncGenerator[str, None]:
    """流式返回OpenAI标准SSE响应（带防原文泄露状态机与error事件转发）"""
    created_ts = int(time.time())
    tool_calls_sent = False
    next_tool_index = 0
    fallback_text_chunks: List[str] = []
    fallback_text_len = 0
    usage_obj: Optional[Dict[str, Any]] = None
    parser = _OpenDifyStreamParser()

    def make_sse_chunk(
        delta_content: Optional[str] = None,
        delta_role: Optional[str] = None,
        delta_tool_calls: Optional[List[Dict[str, Any]]] = None,
        finish_reason: Optional[str] = None,
    ) -> str:
        chunk: Dict[str, Any] = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": model,
            "system_fingerprint": "fp_dify",
            "service_tier": None,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
        }
        if delta_role:
            chunk["choices"][0]["delta"]["role"] = delta_role
        if delta_content is not None:
            chunk["choices"][0]["delta"]["content"] = delta_content
        if delta_tool_calls:
            chunk["choices"][0]["delta"]["tool_calls"] = delta_tool_calls
        return f"data: {ujson.dumps(chunk, ensure_ascii=False)}\n\n"

    def emit_tool_calls(calls: List[Dict[str, Any]]) -> List[str]:
        nonlocal tool_calls_sent, next_tool_index
        out_chunks: List[str] = []
        for tc in calls:
            idx = next_tool_index
            next_tool_index += 1
            tool_calls_sent = True

            func = tc.get("function") or {}
            name = func.get("name") or ""
            args_str = func.get("arguments")
            if args_str is None:
                args_str = "{}"
            elif not isinstance(args_str, str):
                args_str = ujson.dumps(args_str, ensure_ascii=False)

            out_chunks.append(
                make_sse_chunk(
                    delta_tool_calls=[
                        {
                            "index": idx,
                            "id": tc.get("id") or f"call_{fast_uuid()}",
                            "type": tc.get("type") or "function",
                            "function": {"name": name, "arguments": ""},
                        }
                    ]
                )
            )
            if args_str:
                out_chunks.append(
                    make_sse_chunk(
                        delta_tool_calls=[
                            {
                                "index": idx,
                                "function": {"arguments": args_str},
                            }
                        ]
                    )
                )
        return out_chunks

    try:
        yield make_sse_chunk(delta_role="assistant")

        async for data in _iter_dify_sse_json(rsp):
            event_type = data.get("event")

            if event_type == "error":
                err_msg = data.get("message") or "Dify upstream stream error"
                err_code = str(data.get("code") or "dify_error")
                status = int(data.get("status") or 502)
                err_type = "rate_limit_error" if status == 429 else ("server_error" if status >= 500 else "invalid_request_error")
                logger.error(f"Dify SSE error event: status={status} code={err_code} message={err_msg}")
                yield f"data: {ujson.dumps(_openai_error_payload(err_msg, type=err_type, code=err_code), ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            if event_type in ("message", "agent_message"):
                answer_delta = data.get("answer", "")
                if answer_delta:
                    if not isinstance(answer_delta, str):
                        answer_delta = str(answer_delta)

                    if TOOL_SUPPORT:
                        for ev_kind, ev_val in parser.feed(answer_delta):
                            if ev_kind == "text":
                                if ev_val:
                                    if not tool_calls_sent and (TOOL_BUFFER_LIMIT <= 0 or fallback_text_len < TOOL_BUFFER_LIMIT):
                                        fallback_text_chunks.append(ev_val)
                                        fallback_text_len += len(ev_val)
                                    yield make_sse_chunk(delta_content=ev_val)
                            elif ev_kind == "tool_calls":
                                for sse_chunk in emit_tool_calls(ev_val):
                                    yield sse_chunk
                    else:
                        yield make_sse_chunk(delta_content=answer_delta)

            elif event_type == "tool_calls":
                dify_tool_calls = data.get("tool_calls", [])
                if isinstance(dify_tool_calls, list) and dify_tool_calls:
                    normalized = _normalize_tool_calls(dify_tool_calls)
                    for sse_chunk in emit_tool_calls(normalized):
                        yield sse_chunk

            elif event_type == "message_end":
                meta_usage = (data.get("metadata") or {}).get("usage") if isinstance(data.get("metadata"), dict) else None
                if isinstance(meta_usage, dict):
                    usage_obj = {
                        "prompt_tokens": meta_usage.get("prompt_tokens", 0),
                        "completion_tokens": meta_usage.get("completion_tokens", 0),
                        "total_tokens": meta_usage.get("total_tokens", 0),
                    }

                if TOOL_SUPPORT:
                    for ev_kind, ev_val in parser.finalize():
                        if ev_kind == "text":
                            if ev_val:
                                if not tool_calls_sent and (TOOL_BUFFER_LIMIT <= 0 or fallback_text_len < TOOL_BUFFER_LIMIT):
                                    fallback_text_chunks.append(ev_val)
                                    fallback_text_len += len(ev_val)
                                yield make_sse_chunk(delta_content=ev_val)
                        elif ev_kind == "tool_calls":
                            for sse_chunk in emit_tool_calls(ev_val):
                                yield sse_chunk

                    if not tool_calls_sent and fallback_text_chunks:
                        legacy_spans = _find_legacy_fenced_tool_spans("".join(fallback_text_chunks))
                        if legacy_spans:
                            legacy_calls: List[Dict[str, Any]] = []
                            for _, _, calls in legacy_spans:
                                legacy_calls.extend(calls)
                            if legacy_calls:
                                for sse_chunk in emit_tool_calls(legacy_calls):
                                    yield sse_chunk

                finish_reason = "tool_calls" if tool_calls_sent else "stop"
                yield make_sse_chunk(finish_reason=finish_reason)

                if stream_include_usage:
                    yield (
                        "data: "
                        + ujson.dumps(
                            {
                                "id": message_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": model,
                                "system_fingerprint": "fp_dify",
                                "service_tier": None,
                                "choices": [],
                                "usage": usage_obj or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                            },
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    )
                yield "data: [DONE]\n\n"
                return

        if TOOL_SUPPORT:
            for ev_kind, ev_val in parser.finalize():
                if ev_kind == "text" and ev_val:
                    yield make_sse_chunk(delta_content=ev_val)
                elif ev_kind == "tool_calls":
                    for sse_chunk in emit_tool_calls(ev_val):
                        yield sse_chunk

        yield make_sse_chunk(finish_reason="tool_calls" if tool_calls_sent else "stop")
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {ujson.dumps(_openai_error_payload(str(e), type='server_error', code='stream_error'), ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

async def stream_openai_responses(
    rsp: httpx.Response,
    *,
    model: str,
    response_id: str,
) -> AsyncGenerator[str, None]:
    created_at = int(time.time())
    sequence_number = 0

    output_index = 0
    content_index = 0

    message_item_id: Optional[str] = None
    message_open = False
    content_part_open = False
    accumulated_text = ""

    tool_items_in_order: List[Dict[str, Any]] = []
    tool_items_by_id: Dict[str, Dict[str, Any]] = {}

    final_output_by_index: Dict[int, Dict[str, Any]] = {}
    usage_obj: Dict[str, Any] = {
        "input_tokens": 0,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 0,
    }
    parser = _OpenDifyStreamParser()

    def sse(event: Dict[str, Any]) -> str:
        nonlocal sequence_number
        payload = dict(event)
        payload["sequence_number"] = sequence_number
        sequence_number += 1
        return f"data: {ujson.dumps(payload, ensure_ascii=False)}\n\n"

    def _start_message_item() -> List[Dict[str, Any]]:
        nonlocal message_item_id, message_open, content_part_open, content_index
        if message_open:
            return []
        message_item_id = f"msg_{fast_uuid()}"
        message_open = True
        content_part_open = True
        content_index = 0
        return [
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {
                    "id": message_item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            },
            {
                "type": "response.content_part.added",
                "item_id": message_item_id,
                "output_index": output_index,
                "content_index": content_index,
                "part": {"type": "output_text", "text": ""},
            },
        ]

    def _close_message_item(final_text: str) -> List[Dict[str, Any]]:
        nonlocal message_item_id, message_open, content_part_open, output_index, accumulated_text, content_index
        if not message_open or not message_item_id:
            return []

        events: List[Dict[str, Any]] = []
        if content_part_open:
            events.append(
                {
                    "type": "response.output_text.done",
                    "item_id": message_item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "text": final_text,
                }
            )
            events.append(
                {
                    "type": "response.content_part.done",
                    "item_id": message_item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": {"type": "output_text", "text": final_text},
                }
            )
            content_part_open = False

        item = {
            "id": message_item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": final_text, "annotations": []}],
        }
        events.append({"type": "response.output_item.done", "output_index": output_index, "item": item})
        final_output_by_index[output_index] = item

        message_item_id = None
        message_open = False
        accumulated_text = ""
        content_index = 0

        output_index += 1
        return events

    def _start_tool_items(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        nonlocal output_index
        events: List[Dict[str, Any]] = []
        for tc in tool_calls or []:
            if not isinstance(tc, dict):
                continue
            call_id = tc.get("id") or f"call_{fast_uuid()}"
            if call_id in tool_items_by_id:
                continue
            func = tc.get("function") or {}
            name = (func or {}).get("name") or ""
            args = (func or {}).get("arguments") or ""
            tool_item = {"id": call_id, "output_index": output_index, "call_id": call_id, "name": name, "arguments": ""}
            tool_items_by_id[call_id] = tool_item
            tool_items_in_order.append(tool_item)

            events.append(
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {"id": call_id, "type": "function_call", "status": "in_progress", "call_id": call_id, "name": name},
                }
            )
            if isinstance(args, str) and args:
                tool_item["arguments"] = args
                events.append(
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": call_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "delta": args,
                    }
                )

            output_index += 1
        return events

    def _finish_tool_items() -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for tool_item in tool_items_in_order:
            call_id = tool_item["id"]
            out_idx = tool_item["output_index"]
            args = tool_item.get("arguments") or ""
            name = tool_item.get("name") or ""

            events.append(
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": call_id,
                    "output_index": out_idx,
                    "arguments": args,
                }
            )
            item = {
                "id": call_id,
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": name,
                "arguments": args,
            }
            events.append({"type": "response.output_item.done", "output_index": out_idx, "item": item})
            final_output_by_index[out_idx] = item

        tool_items_in_order.clear()
        tool_items_by_id.clear()
        return events

    base_response = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "model": model,
        "status": "in_progress",
        "output": [],
        "error": None,
    }

    yield sse({"type": "response.created", "response": base_response})
    yield sse({"type": "response.in_progress", "response": base_response})

    try:
        async for data in _iter_dify_sse_json(rsp):
            event_type = data.get("event")

            if event_type == "error":
                failed_response: Dict[str, Any] = {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "model": model,
                    "status": "failed",
                    "output": [final_output_by_index[idx] for idx in sorted(final_output_by_index.keys())],
                    "usage": usage_obj,
                    "error": {"code": data.get("code") or 500, "message": data.get("message") or "Upstream error"},
                }
                yield sse({"type": "response.completed", "response": failed_response})
                yield "data: [DONE]\n\n"
                return

            if event_type in ("message", "agent_message"):
                answer_delta = data.get("answer", "")
                if answer_delta:
                    if not isinstance(answer_delta, str):
                        answer_delta = str(answer_delta)
                    events_to_handle = parser.feed(answer_delta) if TOOL_SUPPORT else [("text", answer_delta)]
                    for ev_kind, ev_val in events_to_handle:
                        if ev_kind == "text" and ev_val:
                            if not message_open:
                                for ev in _start_message_item():
                                    yield sse(ev)
                            accumulated_text += ev_val
                            yield sse(
                                {
                                    "type": "response.output_text.delta",
                                    "item_id": message_item_id,
                                    "output_index": output_index,
                                    "content_index": content_index,
                                    "delta": ev_val,
                                }
                            )
                        elif ev_kind == "tool_calls":
                            if message_open:
                                for ev in _close_message_item(accumulated_text):
                                    yield sse(ev)
                            for ev in _start_tool_items(ev_val):
                                yield sse(ev)

            elif event_type == "tool_calls":
                dify_tool_calls = data.get("tool_calls", [])
                if isinstance(dify_tool_calls, list) and dify_tool_calls:
                    normalized = _normalize_tool_calls(dify_tool_calls)
                    if message_open:
                        for ev in _close_message_item(accumulated_text):
                            yield sse(ev)
                    for ev in _start_tool_items(normalized):
                        yield sse(ev)

            elif event_type == "message_end":
                meta_usage = (data.get("metadata") or {}).get("usage") if isinstance(data.get("metadata"), dict) else None
                if isinstance(meta_usage, dict):
                    usage_obj = {
                        "input_tokens": meta_usage.get("prompt_tokens", 0),
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": meta_usage.get("completion_tokens", 0),
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": meta_usage.get("total_tokens", 0),
                    }

                if TOOL_SUPPORT:
                    for ev_kind, ev_val in parser.finalize():
                        if ev_kind == "text" and ev_val:
                            if not message_open:
                                for ev in _start_message_item():
                                    yield sse(ev)
                            accumulated_text += ev_val
                            yield sse(
                                {
                                    "type": "response.output_text.delta",
                                    "item_id": message_item_id,
                                    "output_index": output_index,
                                    "content_index": content_index,
                                    "delta": ev_val,
                                }
                            )
                        elif ev_kind == "tool_calls":
                            if message_open:
                                for ev in _close_message_item(accumulated_text):
                                    yield sse(ev)
                            for ev in _start_tool_items(ev_val):
                                yield sse(ev)

                if message_open:
                    for ev in _close_message_item(accumulated_text):
                        yield sse(ev)

                for ev in _finish_tool_items():
                    yield sse(ev)

                if not final_output_by_index:
                    for ev in _start_message_item():
                        yield sse(ev)
                    for ev in _close_message_item(""):
                        yield sse(ev)

                completed_response: Dict[str, Any] = {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "model": model,
                    "status": "completed",
                    "output": [final_output_by_index[idx] for idx in sorted(final_output_by_index.keys())],
                    "usage": usage_obj,
                    "error": None,
                }
                yield sse({"type": "response.completed", "response": completed_response})
                yield "data: [DONE]\n\n"
                return

    except Exception as e:
        logger.error(f"Responses stream error: {e}")
        failed_response = {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "model": model,
            "status": "failed",
            "output": [],
            "usage": usage_obj,
            "error": {"code": 500, "message": str(e)},
        }
        yield sse({"type": "response.completed", "response": failed_response})
        yield "data: [DONE]\n\n"

async def stream_claude_message(
    rsp: httpx.Response,
    *,
    model: str,
    message_id: str,
) -> AsyncGenerator[str, None]:
    tool_calls_sent = False
    content_index = 0
    text_block_open = False
    current_text_index = 0
    parser = _OpenDifyStreamParser()

    def sse(event_name: str, payload: Dict[str, Any]) -> str:
        return f"event: {event_name}\ndata: {ujson.dumps(payload, ensure_ascii=False)}\n\n"

    yield sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    usage_obj: Dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}

    def _emit_tool_calls(calls: List[Dict[str, Any]]) -> List[str]:
        nonlocal tool_calls_sent, text_block_open, content_index
        out_events: List[str] = []
        if text_block_open:
            out_events.append(sse("content_block_stop", {"type": "content_block_stop", "index": current_text_index}))
            text_block_open = False
            content_index = current_text_index + 1

        for tc in calls:
            tool_calls_sent = True
            call_id = tc.get("id") or f"call_{fast_uuid()}"
            func = tc.get("function") or {}
            name = (func or {}).get("name") or ""
            args = (func or {}).get("arguments") or ""

            tool_idx = content_index
            out_events.append(
                sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": tool_idx,
                        "content_block": {"type": "tool_use", "id": call_id, "name": name, "input": {}},
                    },
                )
            )
            if isinstance(args, str) and args:
                out_events.append(
                    sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": tool_idx,
                            "delta": {"type": "input_json_delta", "partial_json": args},
                        },
                    )
                )
            out_events.append(sse("content_block_stop", {"type": "content_block_stop", "index": tool_idx}))
            content_index += 1
        return out_events

    try:
        async for data in _iter_dify_sse_json(rsp):
            event_type = data.get("event")

            if event_type == "error":
                yield sse(
                    "error",
                    {
                        "type": "error",
                        "error": {"type": "api_error", "message": data.get("message") or "Upstream error"},
                    },
                )
                return

            if event_type in ("message", "agent_message"):
                answer_delta = data.get("answer", "")
                if answer_delta:
                    if not isinstance(answer_delta, str):
                        answer_delta = str(answer_delta)

                    events_to_handle = parser.feed(answer_delta) if TOOL_SUPPORT else [("text", answer_delta)]
                    for ev_kind, ev_val in events_to_handle:
                        if ev_kind == "text" and ev_val:
                            if not text_block_open:
                                current_text_index = content_index
                                content_index += 1
                                text_block_open = True
                                yield sse(
                                    "content_block_start",
                                    {
                                        "type": "content_block_start",
                                        "index": current_text_index,
                                        "content_block": {"type": "text", "text": ""},
                                    },
                                )
                            yield sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": current_text_index,
                                    "delta": {"type": "text_delta", "text": ev_val},
                                },
                            )
                        elif ev_kind == "tool_calls":
                            for ev_str in _emit_tool_calls(ev_val):
                                yield ev_str

            elif event_type == "tool_calls":
                dify_tool_calls = data.get("tool_calls", [])
                if isinstance(dify_tool_calls, list) and dify_tool_calls:
                    normalized = _normalize_tool_calls(dify_tool_calls)
                    for ev_str in _emit_tool_calls(normalized):
                        yield ev_str

            elif event_type == "message_end":
                meta_usage = (data.get("metadata") or {}).get("usage") if isinstance(data.get("metadata"), dict) else None
                if isinstance(meta_usage, dict):
                    usage_obj = {
                        "input_tokens": meta_usage.get("prompt_tokens", 0),
                        "output_tokens": meta_usage.get("completion_tokens", 0),
                    }

                if TOOL_SUPPORT:
                    for ev_kind, ev_val in parser.finalize():
                        if ev_kind == "text" and ev_val:
                            if not text_block_open:
                                current_text_index = content_index
                                content_index += 1
                                text_block_open = True
                                yield sse(
                                    "content_block_start",
                                    {
                                        "type": "content_block_start",
                                        "index": current_text_index,
                                        "content_block": {"type": "text", "text": ""},
                                    },
                                )
                            yield sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": current_text_index,
                                    "delta": {"type": "text_delta", "text": ev_val},
                                },
                            )
                        elif ev_kind == "tool_calls":
                            for ev_str in _emit_tool_calls(ev_val):
                                yield ev_str

                if text_block_open:
                    yield sse("content_block_stop", {"type": "content_block_stop", "index": current_text_index})
                    text_block_open = False

                stop_reason = "tool_use" if tool_calls_sent else "end_turn"
                yield sse(
                    "message_delta",
                    {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": usage_obj},
                )
                yield sse("message_stop", {"type": "message_stop"})
                return

    except Exception as e:
        logger.error(f"Claude stream error: {e}")
        yield sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": usage_obj})
        yield sse("message_stop", {"type": "message_stop"})

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        request_body = await request.body()
        openai_request = ujson.loads(request_body)
        model = openai_request.get("model")
        if not isinstance(model, str) or not model.strip():
            raise OpenAIHTTPError(400, "Missing required parameter: 'model'", param="model", code="missing_required_parameter")

        messages = openai_request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise OpenAIHTTPError(400, "Missing required parameter: 'messages'", param="messages", code="missing_required_parameter")

        n = openai_request.get("n")
        if n is not None and n != 1:
            raise OpenAIHTTPError(400, "Only 'n=1' is supported", param="n", code="invalid_request_error")

        await model_manager.ensure_loaded()

        conversation_id = request.headers.get("X-Dify-Conversation-Id") or openai_request.get("conversation_id")
        if conversation_id is not None and not isinstance(conversation_id, str):
            conversation_id = None

        dify_key = model_manager.get_api_key(model)
        if not dify_key:
            raise OpenAIHTTPError(
                404,
                f"The model '{model}' does not exist",
                type="invalid_request_error",
                code="model_not_found",
                param="model",
            )

        dify_req = await transform_openai_to_dify(openai_request, conversation_id, dify_key=dify_key)
        if not dify_req:
            raise OpenAIHTTPError(400, "Invalid format", code="invalid_request_error", param=None)

        stream = bool(openai_request.get("stream", False))
        if stream:
            stream_options = openai_request.get("stream_options") or {}
            stream_include_usage = bool((stream_options or {}).get("include_usage")) if isinstance(stream_options, dict) else False

            headers = {"Authorization": f"Bearer {dify_key}", "Content-Type": "application/json"}
            endpoint = f"{DIFY_API_BASE}/chat-messages"
            message_id = f"chatcmpl-{fast_uuid()}"

            upstream_cm = model_manager.client.stream(
                "POST",
                endpoint,
                content=ujson.dumps(dify_req, ensure_ascii=False),
                headers=headers,
                timeout=TIMEOUT,
            )
            upstream_rsp = await upstream_cm.__aenter__()
            if upstream_rsp.status_code != 200:
                body = await upstream_rsp.aread()
                await upstream_cm.__aexit__(None, None, None)
                raise _build_dify_http_error(upstream_rsp.status_code, body)

            return StreamingResponse(
                stream_openai_response(
                    upstream_rsp,
                    model=model,
                    message_id=message_id,
                    stream_include_usage=stream_include_usage,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                },
                background=BackgroundTask(upstream_cm.__aexit__, None, None, None),
            )

        dify_resp = await _call_dify_non_streaming(dify_key, dify_req)
        openai_resp, resp_conversation_id = transform_dify_to_openai_response(dify_resp, model)

        response_headers = {"access-control-allow-origin": "*"}
        if resp_conversation_id:
            response_headers["X-Dify-Conversation-Id"] = str(resp_conversation_id)
        return UJSONResponse(content=openai_resp, headers=response_headers)

    except OpenAIHTTPError:
        raise
    except HTTPException:
        raise
    except ujson.JSONDecodeError:
        raise OpenAIHTTPError(400, "Invalid JSON format in request", type="invalid_request_error", code="invalid_json", param=None)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise OpenAIHTTPError(500, str(e), type="server_error", code="internal_error", param=None)

@app.post("/anthropic/v1/chat/completions")
async def anthropic_chat_completions(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        request_body = await request.body()
        openai_request = ujson.loads(request_body)

        messages = openai_request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise OpenAIHTTPError(400, "Missing required parameter: 'messages'", param="messages", code="missing_required_parameter")

        n = openai_request.get("n")
        if n is not None and n != 1:
            raise OpenAIHTTPError(400, "Only 'n=1' is supported", param="n", code="invalid_request_error")

        request_model = openai_request.get("model")
        if isinstance(request_model, str):
            request_model = request_model.strip() or None
        else:
            request_model = None

        anthropic_api_base = (_env_first("ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL", default="https://api.anthropic.com") or "").rstrip("/")
        if anthropic_api_base.endswith("/v1"):
            anthropic_api_base = anthropic_api_base[:-3]
        anthropic_api_key = _env_first("ANTHROPIC_API_KEY", "ANTHROPIC_KEY", default="") or ""
        anthropic_model = _env_first("ANTHROPIC_MODEL", default="") or ""
        anthropic_version = _env_first("ANTHROPIC_VERSION", default="2023-06-01") or "2023-06-01"

        raw_default_max_tokens = _env_first("ANTHROPIC_MAX_TOKENS", default="1024") or "1024"
        try:
            default_max_tokens = int(raw_default_max_tokens)
        except Exception:
            default_max_tokens = 1024
        if default_max_tokens <= 0:
            default_max_tokens = 1024

        upstream_model = anthropic_model or request_model
        if not upstream_model:
            raise OpenAIHTTPError(500, "Anthropic model is not configured", type="server_error", code="config_error", param=None)
        if not anthropic_api_key:
            raise OpenAIHTTPError(500, "Anthropic api_key is not configured", type="server_error", code="config_error", param=None)

        response_model = request_model or upstream_model

        anthropic_req = transform_openai_chat_to_anthropic_request(
            openai_request,
            upstream_model=upstream_model,
            default_max_tokens=default_max_tokens,
        )

        headers = {
            "x-api-key": anthropic_api_key,
            "anthropic-version": anthropic_version,
            "content-type": "application/json",
        }
        endpoint = f"{anthropic_api_base}/v1/messages"

        stream = bool(openai_request.get("stream", False))
        if stream:
            stream_options = openai_request.get("stream_options") or {}
            stream_include_usage = bool((stream_options or {}).get("include_usage")) if isinstance(stream_options, dict) else False

            message_id = f"chatcmpl-{fast_uuid()}"
            upstream_cm = anthropic_client.stream(
                "POST",
                endpoint,
                content=ujson.dumps(anthropic_req, ensure_ascii=False),
                headers=headers,
                timeout=TIMEOUT,
            )
            upstream_rsp = await upstream_cm.__aenter__()
            if upstream_rsp.status_code != 200:
                body = await upstream_rsp.aread()
                await upstream_cm.__aexit__(None, None, None)
                try:
                    err_payload = ujson.loads(body or b"{}")
                    err_obj = err_payload.get("error") if isinstance(err_payload, dict) else None
                    if not isinstance(err_obj, dict):
                        err_obj = {}
                    error_message = err_obj.get("message") or err_payload.get("message") or (body.decode("utf-8", errors="ignore") if body else "Upstream error")
                    error_code = err_obj.get("type") or err_obj.get("code") or err_payload.get("type") or "unknown_error"
                except Exception:
                    error_message = body.decode("utf-8", errors="ignore") if body else "Upstream error"
                    error_code = "unknown_error"

                if upstream_rsp.status_code == 429:
                    err_type = "rate_limit_error"
                elif upstream_rsp.status_code >= 500:
                    err_type = "server_error"
                else:
                    err_type = "invalid_request_error"
                raise OpenAIHTTPError(upstream_rsp.status_code, error_message, type=err_type, code=error_code, param=None)

            return StreamingResponse(
                stream_anthropic_chat_completion(
                    upstream_rsp,
                    model=response_model,
                    message_id=message_id,
                    stream_include_usage=stream_include_usage,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                },
                background=BackgroundTask(upstream_cm.__aexit__, None, None, None),
            )

        resp = await anthropic_client.post(
            endpoint,
            content=ujson.dumps(anthropic_req, ensure_ascii=False),
            headers=headers,
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            try:
                err_payload = ujson.loads(resp.content)
                err_obj = err_payload.get("error") if isinstance(err_payload, dict) else None
                if not isinstance(err_obj, dict):
                    err_obj = {}
                error_message = err_obj.get("message") or err_payload.get("message") or resp.text
                error_code = err_obj.get("type") or err_obj.get("code") or err_payload.get("type") or "unknown_error"
            except Exception:
                error_message = resp.text
                error_code = "unknown_error"

            if resp.status_code == 429:
                err_type = "rate_limit_error"
            elif resp.status_code >= 500:
                err_type = "server_error"
            else:
                err_type = "invalid_request_error"

            raise OpenAIHTTPError(resp.status_code, error_message, type=err_type, code=error_code, param=None)

        anthropic_message = ujson.loads(resp.content)
        openai_resp = transform_anthropic_to_openai_chat_completion(anthropic_message, request_model=response_model)

        return UJSONResponse(content=openai_resp, headers={"access-control-allow-origin": "*"})

    except OpenAIHTTPError:
        raise
    except HTTPException:
        raise
    except ujson.JSONDecodeError:
        raise OpenAIHTTPError(400, "Invalid JSON format in request", type="invalid_request_error", code="invalid_json", param=None)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise OpenAIHTTPError(500, str(e), type="server_error", code="internal_error", param=None)

@app.post("/v1/responses")
async def responses_api(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        request_body = await request.body()
        responses_request = ujson.loads(request_body)

        model = responses_request.get("model")
        if not isinstance(model, str) or not model.strip():
            raise OpenAIHTTPError(400, "Missing required parameter: 'model'", param="model", code="missing_required_parameter")

        if "input" not in responses_request:
            raise OpenAIHTTPError(400, "Missing required parameter: 'input'", param="input", code="missing_required_parameter")

        await model_manager.ensure_loaded()

        conversation_id = request.headers.get("X-Dify-Conversation-Id") or responses_request.get("conversation_id")
        if conversation_id is not None and not isinstance(conversation_id, str):
            conversation_id = None

        dify_key = model_manager.get_api_key(model)
        if not dify_key:
            raise OpenAIHTTPError(
                404,
                f"The model '{model}' does not exist",
                type="invalid_request_error",
                code="model_not_found",
                param="model",
            )

        dify_req = await transform_openai_responses_to_dify(responses_request, conversation_id, dify_key=dify_key)
        if not dify_req:
            raise OpenAIHTTPError(400, "Invalid format", code="invalid_request_error", param=None)

        stream = bool(responses_request.get("stream", False))
        if stream:
            headers = {"Authorization": f"Bearer {dify_key}", "Content-Type": "application/json"}
            endpoint = f"{DIFY_API_BASE}/chat-messages"
            response_id = f"resp_{fast_uuid()}"

            upstream_cm = model_manager.client.stream(
                "POST",
                endpoint,
                content=ujson.dumps(dify_req, ensure_ascii=False),
                headers=headers,
                timeout=TIMEOUT,
            )
            upstream_rsp = await upstream_cm.__aenter__()
            if upstream_rsp.status_code != 200:
                body = await upstream_rsp.aread()
                await upstream_cm.__aexit__(None, None, None)
                raise _build_dify_http_error(upstream_rsp.status_code, body)

            return StreamingResponse(
                stream_openai_responses(upstream_rsp, model=model, response_id=response_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                },
                background=BackgroundTask(upstream_cm.__aexit__, None, None, None),
            )

        dify_resp = await _call_dify_non_streaming(dify_key, dify_req)
        out_resp, resp_conversation_id = transform_dify_to_openai_responses(dify_resp, model)

        response_headers = {"access-control-allow-origin": "*"}
        if resp_conversation_id:
            response_headers["X-Dify-Conversation-Id"] = str(resp_conversation_id)
        return UJSONResponse(content=out_resp, headers=response_headers)

    except OpenAIHTTPError:
        raise
    except HTTPException:
        raise
    except ujson.JSONDecodeError:
        raise OpenAIHTTPError(400, "Invalid JSON format in request", type="invalid_request_error", code="invalid_json", param=None)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise OpenAIHTTPError(500, str(e), type="server_error", code="internal_error", param=None)

@app.post("/v1/messages/count_tokens")
async def claude_messages_count_tokens_api(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        request_body = await request.body()
        claude_request = ujson.loads(request_body)

        model = claude_request.get("model")
        if not isinstance(model, str) or not model.strip():
            raise OpenAIHTTPError(400, "Missing required parameter: 'model'", param="model", code="missing_required_parameter")
        model = model.strip()
        claude_request["model"] = model

        messages = claude_request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise OpenAIHTTPError(400, "Missing required parameter: 'messages'", param="messages", code="missing_required_parameter")

        await _resolve_claude_model_to_dify(model)

        text = _claude_request_to_text_for_token_count(claude_request)
        token_count = _estimate_token_count_from_text(text)

        return UJSONResponse(content={"token_count": int(token_count)}, headers={"access-control-allow-origin": "*"})

    except OpenAIHTTPError:
        raise
    except HTTPException:
        raise
    except ujson.JSONDecodeError:
        raise OpenAIHTTPError(400, "Invalid JSON format in request", type="invalid_request_error", code="invalid_json", param=None)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise OpenAIHTTPError(500, str(e), type="server_error", code="internal_error", param=None)

@app.post("/anthropic/v1/messages/count_tokens")
async def anthropic_messages_count_tokens_api(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        request_body = await request.body()
        claude_request = ujson.loads(request_body)

        model = claude_request.get("model")
        if not isinstance(model, str) or not model.strip():
            raise OpenAIHTTPError(400, "Missing required parameter: 'model'", param="model", code="missing_required_parameter")

        messages = claude_request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise OpenAIHTTPError(400, "Missing required parameter: 'messages'", param="messages", code="missing_required_parameter")

        text = _claude_request_to_text_for_token_count(claude_request)
        token_count = _estimate_token_count_from_text(text)
        return UJSONResponse(content={"token_count": int(token_count)}, headers={"access-control-allow-origin": "*"})

    except OpenAIHTTPError:
        raise
    except HTTPException:
        raise
    except ujson.JSONDecodeError:
        raise OpenAIHTTPError(400, "Invalid JSON format in request", type="invalid_request_error", code="invalid_json", param=None)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise OpenAIHTTPError(500, str(e), type="server_error", code="internal_error", param=None)

@app.post("/v1/messages")
async def claude_messages_api(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        request_body = await request.body()
        claude_request = ujson.loads(request_body)

        model = claude_request.get("model")
        if not isinstance(model, str) or not model.strip():
            raise OpenAIHTTPError(400, "Missing required parameter: 'model'", param="model", code="missing_required_parameter")

        messages = claude_request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise OpenAIHTTPError(400, "Missing required parameter: 'messages'", param="messages", code="missing_required_parameter")

        conversation_id = request.headers.get("X-Dify-Conversation-Id") or claude_request.get("conversation_id")
        if conversation_id is not None and not isinstance(conversation_id, str):
            conversation_id = None

        _, dify_key = await _resolve_claude_model_to_dify(model)

        dify_req = await transform_claude_messages_to_dify(claude_request, conversation_id, dify_key=dify_key)
        if not dify_req:
            raise OpenAIHTTPError(400, "Invalid format", code="invalid_request_error", param=None)

        stream = bool(claude_request.get("stream", False))
        if stream:
            headers = {"Authorization": f"Bearer {dify_key}", "Content-Type": "application/json"}
            endpoint = f"{DIFY_API_BASE}/chat-messages"
            message_id = f"msg_{fast_uuid()}"

            upstream_cm = model_manager.client.stream(
                "POST",
                endpoint,
                content=ujson.dumps(dify_req, ensure_ascii=False),
                headers=headers,
                timeout=TIMEOUT,
            )
            upstream_rsp = await upstream_cm.__aenter__()
            if upstream_rsp.status_code != 200:
                body = await upstream_rsp.aread()
                await upstream_cm.__aexit__(None, None, None)
                raise _build_dify_http_error(upstream_rsp.status_code, body)

            return StreamingResponse(
                stream_claude_message(upstream_rsp, model=model, message_id=message_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                },
                background=BackgroundTask(upstream_cm.__aexit__, None, None, None),
            )

        dify_resp = await _call_dify_non_streaming(dify_key, dify_req)
        out_resp, resp_conversation_id = transform_dify_to_claude_message(dify_resp, model)

        response_headers = {"access-control-allow-origin": "*"}
        if resp_conversation_id:
            response_headers["X-Dify-Conversation-Id"] = str(resp_conversation_id)
        return UJSONResponse(content=out_resp, headers=response_headers)

    except OpenAIHTTPError:
        raise
    except HTTPException:
        raise
    except ujson.JSONDecodeError:
        raise OpenAIHTTPError(400, "Invalid JSON format in request", type="invalid_request_error", code="invalid_json", param=None)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise OpenAIHTTPError(500, str(e), type="server_error", code="internal_error", param=None)

@app.post("/anthropic/v1/messages")
async def anthropic_messages_api(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        request_body = await request.body()
        claude_request = ujson.loads(request_body)

        request_model = claude_request.get("model")
        if not isinstance(request_model, str) or not request_model.strip():
            raise OpenAIHTTPError(400, "Missing required parameter: 'model'", param="model", code="missing_required_parameter")
        request_model = request_model.strip()

        messages = claude_request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise OpenAIHTTPError(400, "Missing required parameter: 'messages'", param="messages", code="missing_required_parameter")

        upstream_url_raw = _env_first("UPSTREAM_OPENAI_BASE_URL", "UPSTREAM_BASE_URL", "base_url", default="") or ""
        upstream_url = _normalize_openai_chat_completions_url(upstream_url_raw)

        upstream_key = _env_first("UPSTREAM_OPENAI_API_KEY", "UPSTREAM_API_KEY", "api_key", "key", default="") or ""

        if not upstream_url or not upstream_key:
            return await claude_messages_api(request, api_key)

        upstream_model = _env_first("UPSTREAM_OPENAI_MODEL", "UPSTREAM_MODEL", "model", default="") or ""
        upstream_model_map = _parse_model_map(_env_first("UPSTREAM_OPENAI_MODEL_MAP", default=""))
        mapped_model = upstream_model_map.get(request_model) if upstream_model_map else None
        if isinstance(mapped_model, str) and mapped_model.strip():
            upstream_model_final = mapped_model.strip()
        else:
            upstream_model_final = upstream_model.strip() if upstream_model.strip() else request_model

        raw_default_max_tokens = _env_first("UPSTREAM_OPENAI_MAX_TOKENS", "ANTHROPIC_MAX_TOKENS", default="1024") or "1024"
        try:
            default_max_tokens = int(raw_default_max_tokens)
        except Exception:
            default_max_tokens = 1024
        if default_max_tokens <= 0:
            default_max_tokens = 1024

        openai_req = transform_claude_messages_to_openai_chat_completion_request(
            claude_request,
            upstream_model=upstream_model_final,
            default_max_tokens=default_max_tokens,
        )

        headers = {"Authorization": f"Bearer {upstream_key}", "Content-Type": "application/json"}
        stream = bool(claude_request.get("stream", False))
        if stream:
            headers["Accept"] = "text/event-stream"

        if stream:
            message_id = f"msg_{fast_uuid()}"
            upstream_cm = openai_upstream_client.stream(
                "POST",
                upstream_url,
                content=ujson.dumps(openai_req, ensure_ascii=False),
                headers=headers,
                timeout=TIMEOUT,
            )
            upstream_rsp = await upstream_cm.__aenter__()
            if upstream_rsp.status_code != 200:
                body = await upstream_rsp.aread()
                await upstream_cm.__aexit__(None, None, None)
                try:
                    err_payload = ujson.loads(body or b"{}")
                    err_obj = err_payload.get("error") if isinstance(err_payload, dict) else None
                    if not isinstance(err_obj, dict):
                        err_obj = {}
                    error_message = err_obj.get("message") or err_payload.get("message") or (body.decode("utf-8", errors="ignore") if body else "Upstream error")
                    error_code = err_obj.get("code") or err_obj.get("type") or err_payload.get("type") or "unknown_error"
                except Exception:
                    error_message = body.decode("utf-8", errors="ignore") if body else "Upstream error"
                    error_code = "unknown_error"

                err_type = "server_error" if upstream_rsp.status_code >= 500 else "invalid_request_error"
                logger.error(
                    "2anthropic upstream error (stream) status=%s url=%s code=%s message=%s req=%s",
                    upstream_rsp.status_code,
                    upstream_url,
                    error_code,
                    (str(error_message)[:500] if error_message is not None else ""),
                    _summarize_openai_chat_completion_request(openai_req),
                )
                raise OpenAIHTTPError(upstream_rsp.status_code, error_message, type=err_type, code=error_code, param=None)

            return StreamingResponse(
                stream_openai_chat_completion_as_claude_message(
                    upstream_rsp,
                    model=request_model,
                    message_id=message_id,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                },
                background=BackgroundTask(upstream_cm.__aexit__, None, None, None),
            )

        resp = await openai_upstream_client.post(
            upstream_url,
            content=ujson.dumps(openai_req, ensure_ascii=False),
            headers=headers,
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            try:
                err_payload = ujson.loads(resp.content)
                err_obj = err_payload.get("error") if isinstance(err_payload, dict) else None
                if not isinstance(err_obj, dict):
                    err_obj = {}
                error_message = err_obj.get("message") or err_payload.get("message") or resp.text
                error_code = err_obj.get("code") or err_obj.get("type") or err_payload.get("type") or "unknown_error"
            except Exception:
                error_message = resp.text
                error_code = "unknown_error"

            err_type = "server_error" if resp.status_code >= 500 else "invalid_request_error"
            logger.error(
                "2anthropic upstream error status=%s url=%s code=%s message=%s req=%s",
                resp.status_code,
                upstream_url,
                error_code,
                (str(error_message)[:500] if error_message is not None else ""),
                _summarize_openai_chat_completion_request(openai_req),
            )
            raise OpenAIHTTPError(resp.status_code, error_message, type=err_type, code=error_code, param=None)

        openai_resp = ujson.loads(resp.content)
        claude_resp = transform_openai_chat_completion_to_claude_message(openai_resp, request_model=request_model)
        return UJSONResponse(content=claude_resp, headers={"access-control-allow-origin": "*"})

    except OpenAIHTTPError:
        raise
    except HTTPException:
        raise
    except ujson.JSONDecodeError:
        raise OpenAIHTTPError(400, "Invalid JSON format in request", type="invalid_request_error", code="invalid_json", param=None)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise OpenAIHTTPError(500, str(e), type="server_error", code="internal_error", param=None)

@app.get("/anthropic/v1/models")
async def list_anthropic_models(request: Request, api_key: str = Depends(verify_api_key)):
    if request.headers.get("anthropic-version") or request.headers.get("anthropic-beta"):
        payload = await _anthropic_models_page(request)
        return UJSONResponse(content=payload, headers={"access-control-allow-origin": "*"})

    anthropic_model = _env_first("ANTHROPIC_MODEL", "model", default="") or ""
    timestamp = int(time.time())
    models = [{"id": anthropic_model, "object": "model", "created": timestamp, "owned_by": "anthropic"}] if anthropic_model else []
    return UJSONResponse(content={"object": "list", "data": models}, headers={"access-control-allow-origin": "*"})

@app.get("/anthropic/v1/models/{model_id}")
async def get_anthropic_model(request: Request, model_id: str, api_key: str = Depends(verify_api_key)):
    if request.headers.get("anthropic-version") or request.headers.get("anthropic-beta"):
        payload = await _anthropic_model_info(model_id)
        return UJSONResponse(content=payload, headers={"access-control-allow-origin": "*"})

    anthropic_model = _env_first("ANTHROPIC_MODEL", "model", default="") or ""
    if not anthropic_model or model_id != anthropic_model:
        raise OpenAIHTTPError(
            404,
            f"The model '{model_id}' does not exist",
            type="invalid_request_error",
            code="model_not_found",
            param="model",
        )

    timestamp = int(time.time())
    return UJSONResponse(
        content={"id": anthropic_model, "object": "model", "created": timestamp, "owned_by": "anthropic"},
        headers={"access-control-allow-origin": "*"},
    )

@app.get("/v1/models")
async def list_models(request: Request, api_key: str = Depends(verify_api_key)):
    if _is_anthropic_compat_request(request):
        payload = await _anthropic_models_page(request)
        return UJSONResponse(content=payload, headers={"access-control-allow-origin": "*"})

    await model_manager.refresh_model_info()
    models = model_manager.get_available_models()
    return UJSONResponse(content={"object": "list", "data": models}, headers={"access-control-allow-origin": "*"})

@app.get("/v1/models/{model_id}")
async def get_model(request: Request, model_id: str, api_key: str = Depends(verify_api_key)):
    if _is_anthropic_compat_request(request):
        payload = await _anthropic_model_info(model_id)
        return UJSONResponse(content=payload, headers={"access-control-allow-origin": "*"})

    await model_manager.ensure_loaded()

    dify_key = model_manager.get_api_key(model_id)
    if not dify_key:
        raise OpenAIHTTPError(
            404,
            f"The model '{model_id}' does not exist",
            type="invalid_request_error",
            code="model_not_found",
            param="model",
        )

    timestamp = int(time.time())
    return UJSONResponse(
        content={"id": model_id, "object": "model", "created": timestamp, "owned_by": "dify"},
        headers={"access-control-allow-origin": "*"},
    )

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("SERVER_PORT", 5000))
    workers = int(os.getenv("WORKERS", 1))
    target: Any = f"{os.path.splitext(os.path.basename(__file__))[0]}:app" if workers > 1 else app

    uvicorn.run(
        target,
        host=host,
        port=port,
        workers=workers,
        access_log=False,
        server_header=False,
        date_header=False,
        loop="uvloop" if os.name != "nt" else "asyncio",
    )