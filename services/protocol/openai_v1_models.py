from __future__ import annotations

from copy import deepcopy
from threading import Lock
from time import monotonic
from typing import Any

from services.account_service import account_service
from services.openai_backend_api import OpenAIBackendAPI
from utils.helper import CODEX_IMAGE_MODEL


MODEL_CACHE_TTL_SECONDS = 60.0
_model_cache_lock = Lock()
_model_cache: tuple[float, dict[str, Any]] | None = None


def clear_model_cache() -> None:
    global _model_cache
    with _model_cache_lock:
        _model_cache = None


def _backend_models() -> dict[str, Any]:
    global _model_cache
    now = monotonic()
    with _model_cache_lock:
        if _model_cache is not None:
            cached_at, cached_result = _model_cache
            if now - cached_at < MODEL_CACHE_TTL_SECONDS:
                return deepcopy(cached_result)

    backend = OpenAIBackendAPI()
    try:
        result = backend.list_models()
    finally:
        backend.close()
    with _model_cache_lock:
        _model_cache = (monotonic(), deepcopy(result))
    return result


def list_models() -> dict[str, Any]:
    try:
        result = _backend_models()
    except Exception:
        result = {"object": "list", "data": []}
    data = result.get("data")
    if not isinstance(data, list):
        return result
    seen = {str(item.get("id") or "").strip() for item in data if isinstance(item, dict)}
    dynamic_models: set[str] = set()
    accounts = account_service.list_accounts()
    web_image_accounts = [
        account
        for account in accounts
        if isinstance(account, dict)
    ]
    codex_types = {
        normalized
        for account in accounts
        if isinstance(account, dict)
           and account_service._normalize_source_type(account.get("source_type")) == "codex"
           and (normalized := account_service._normalize_account_type(account.get("type")))
    }

    if web_image_accounts:
        dynamic_models.add("gpt-image-2")
    if codex_types & {"Plus", "Team", "Pro"}:
        dynamic_models.add(CODEX_IMAGE_MODEL)
    if "Plus" in codex_types:
        dynamic_models.add(f"plus-{CODEX_IMAGE_MODEL}")
    if "Team" in codex_types:
        dynamic_models.add(f"team-{CODEX_IMAGE_MODEL}")
    if "Pro" in codex_types:
        dynamic_models.add(f"pro-{CODEX_IMAGE_MODEL}")

    for model in sorted(dynamic_models):
        if model not in seen:
            data.append({
                "id": model,
                "object": "model",
                "created": 0,
                "owned_by": "chatgpt2api",
                "permission": [],
                "root": model,
                "parent": None,
            })
    return result
