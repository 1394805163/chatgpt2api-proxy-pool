from __future__ import annotations

import time

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask, BackgroundTasks

from api.image_inputs import cleanup_spooled_image_sources, parse_image_edit_request, read_image_sources
from api.support import acquire_image_concurrency, enforce_image_request_limit, require_identity, resolve_image_base_url
from services.content_filter import check_request, request_shape, request_text
from services.auth_service import DailyRequestQuotaExceeded, auth_service
from services.config import config
from services.editable_file_task_service import editable_file_task_service
from services.image_idempotency import (
    CachedImageResponse,
    ImageIdempotencyCapacityExceeded,
    ImageIdempotencyConflict,
    ImageIdempotencyRegistry,
    image_request_fingerprint,
)
from services.log_service import LoggedCall
from services.protocol import (
    anthropic_v1_messages,
    openai_v1_chat_complete,
    openai_v1_image_edit,
    openai_v1_image_generations,
    openai_v1_models,
    openai_v1_response,
    openai_search,
)
from utils.helper import is_image_chat_request, parse_image_count
from utils.log import logger


IMAGE_TIMEOUT_MIN_SECS = 30.0
IMAGE_TIMEOUT_MAX_SECS = 600.0


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str = "gpt-image-2"
    n: int = Field(default=1, ge=1, le=100)
    size: str | None = None
    quality: str = "auto"
    response_format: str = "b64_json"
    history_disabled: bool = True
    stream: bool | None = None
    timeout_secs: float | None = Field(default=None, ge=IMAGE_TIMEOUT_MIN_SECS, le=IMAGE_TIMEOUT_MAX_SECS)
    client_task_id: str | None = Field(default=None, max_length=128)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    prompt: str | None = None
    n: int | None = None
    stream: bool | None = None
    modalities: list[str] | None = None
    messages: list[dict[str, object]] | None = None


class ResponseCreateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    input: object | None = None
    tools: list[dict[str, object]] | None = None
    tool_choice: object | None = None
    stream: bool | None = None


class AnthropicMessageRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    messages: list[dict[str, object]] | None = None
    system: object | None = None
    stream: bool | None = None


class SearchRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class EditableFileTaskRequest(BaseModel):
    prompt: str = ""
    base64_images: list[str] = Field(default_factory=list)
    client_task_id: str | None = None


def apply_image_timeout(identity: dict[str, object], payload: dict[str, object]) -> float:
    configured_timeout = config.user_image_task_timeout_secs if identity.get("role") == "user" else config.image_task_timeout_secs
    requested_timeout = payload.pop("timeout_secs", None)
    timeout_secs = configured_timeout
    if requested_timeout is not None:
        try:
            requested_timeout = float(requested_timeout)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail={"error": "timeout_secs must be a number"}) from exc
        if requested_timeout < IMAGE_TIMEOUT_MIN_SECS or requested_timeout > IMAGE_TIMEOUT_MAX_SECS:
            raise HTTPException(
                status_code=400,
                detail={"error": f"timeout_secs must be between {IMAGE_TIMEOUT_MIN_SECS:g} and {IMAGE_TIMEOUT_MAX_SECS:g}"},
            )
        timeout_secs = min(requested_timeout, configured_timeout) if identity.get("role") == "user" else requested_timeout
    payload["task_timeout_secs"] = timeout_secs
    payload["task_deadline_ts"] = time.time() + timeout_secs
    payload["image_retention_seconds"] = auth_service.image_retention_seconds(identity)
    payload["image_owner_id"] = str(identity.get("id") or "")
    return timeout_secs


def log_image_request(request: Request, identity: dict[str, object], endpoint: str, payload: dict[str, object], timeout_secs: float) -> None:
    try:
        content_length = max(0, int(request.headers.get("content-length") or 0))
    except (TypeError, ValueError):
        content_length = 0
    logger.info({
        "event": "image_api_request",
        "endpoint": endpoint,
        "client_task_id": str(payload.get("client_task_id") or "").strip(),
        "role": identity.get("role"),
        "timeout_secs": timeout_secs,
        "content_length": content_length,
    })


def image_idempotency_key(request: Request, client_task_id: object = "") -> str:
    header_key = str(request.headers.get("idempotency-key") or "").strip()
    body_key = str(client_task_id or "").strip()
    if header_key and body_key and header_key != body_key:
        raise HTTPException(status_code=400, detail={"error": "Idempotency-Key and client_task_id must match"})
    key = header_key or body_key
    if len(key) > 128:
        raise HTTPException(status_code=400, detail={"error": "idempotency key must be at most 128 characters"})
    return key


def image_idempotency_scope(identity: dict[str, object], endpoint: str) -> str:
    return f"{str(identity.get('role') or 'user').strip()}:{str(identity.get('id') or '').strip()}:{endpoint}"


def raise_image_idempotency_error(exc: Exception) -> None:
    if isinstance(exc, ImageIdempotencyConflict):
        raise HTTPException(status_code=409, detail={"error": str(exc)}) from exc
    if isinstance(exc, ImageIdempotencyCapacityExceeded):
        raise HTTPException(status_code=429, detail={"error": str(exc), "retryable": True}) from exc
    raise exc


async def filter_or_log(call: LoggedCall, text: str) -> None:
    try:
        await run_in_threadpool(check_request, text)
    except HTTPException as exc:
        call.log("调用失败", status="failed", error=str(exc.detail))
        raise


def release_image_concurrency_after_response(response, lease):
    if not isinstance(response, StreamingResponse):
        lease.release()
        return response
    if response.background is None:
        response.background = BackgroundTask(lease.release)
    else:
        cleanup_tasks = BackgroundTasks()
        cleanup_tasks.tasks.append(response.background)
        cleanup_tasks.add_task(lease.release)
        response.background = cleanup_tasks
    return response


def create_router() -> APIRouter:
    router = APIRouter()
    image_idempotency = ImageIdempotencyRegistry()

    @router.get("/v1/models")
    async def list_models(authorization: str | None = Header(default=None)):
        require_identity(authorization)
        try:
            return await run_in_threadpool(openai_v1_models.list_models)
        except Exception as exc:
            raise HTTPException(status_code=502, detail={"error": str(exc)}) from exc

    @router.post("/v1/images/generations")
    async def generate_images(
            body: ImageGenerationRequest,
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        enforce_image_request_limit(identity, body.n)
        payload = body.model_dump(mode="python")
        idempotency_key = image_idempotency_key(request, payload.get("client_task_id"))
        if idempotency_key:
            if payload.get("stream"):
                raise HTTPException(status_code=400, detail={"error": "idempotency is not supported for streaming image requests"})
            payload["client_task_id"] = idempotency_key
        fingerprint = image_request_fingerprint("/v1/images/generations", payload) if idempotency_key else ""
        timeout_secs = apply_image_timeout(identity, payload)
        payload["base_url"] = resolve_image_base_url(request)
        log_image_request(request, identity, "/v1/images/generations", payload, timeout_secs)

        async def execute_response():
            call = LoggedCall(
                identity,
                "/v1/images/generations",
                body.model,
                "文生图",
                request_text=body.prompt,
                client_task_id=idempotency_key,
                request_timeout_secs=timeout_secs,
            )
            lease = acquire_image_concurrency(identity, units=body.n)
            try:
                await filter_or_log(call, body.prompt)
                result = await call.run(openai_v1_image_generations.handle, payload)
            except BaseException:
                lease.release()
                raise
            return release_image_concurrency_after_response(result, lease)

        if not idempotency_key:
            return await execute_response()

        async def execute() -> CachedImageResponse:
            return CachedImageResponse.capture(await execute_response())

        try:
            cached, replayed = await image_idempotency.run(
                image_idempotency_scope(identity, "/v1/images/generations"),
                idempotency_key,
                fingerprint,
                execute,
            )
        except (ImageIdempotencyConflict, ImageIdempotencyCapacityExceeded) as exc:
            raise_image_idempotency_error(exc)
        if replayed:
            logger.info({"event": "image_idempotency_replay", "endpoint": "/v1/images/generations", "client_task_id": idempotency_key})
        return cached.materialize()

    @router.post("/v1/images/edits")
    async def edit_images(
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        request_started = time.time()
        identity = require_identity(authorization)
        payload, image_sources, mask_sources = await parse_image_edit_request(request)
        enforce_image_request_limit(identity, int(payload.get("n") or 1))
        prompt = str(payload["prompt"])
        model = str(payload["model"])
        idempotency_key = image_idempotency_key(request, payload.get("client_task_id"))
        if idempotency_key:
            if payload.get("stream"):
                raise HTTPException(status_code=400, detail={"error": "idempotency is not supported for streaming image requests"})
            payload["client_task_id"] = idempotency_key
        timeout_secs = apply_image_timeout(identity, payload)
        log_image_request(request, identity, "/v1/images/edits", payload, timeout_secs)
        images = []
        masks = []
        try:
            images = await read_image_sources(image_sources, spool_to_disk=True)
            masks = await read_image_sources(mask_sources, spool_to_disk=True) if mask_sources else []
            payload["images"] = images
            if masks:
                payload["mask"] = masks
            payload["base_url"] = resolve_image_base_url(request)
        except BaseException:
            cleanup_spooled_image_sources(images)
            cleanup_spooled_image_sources(masks)
            raise

        def cleanup_inputs() -> None:
            cleanup_spooled_image_sources(images)
            cleanup_spooled_image_sources(masks)

        fingerprint = image_request_fingerprint("/v1/images/edits", payload) if idempotency_key else ""

        async def execute_response():
            call = LoggedCall(
                identity,
                "/v1/images/edits",
                model,
                "图生图",
                started=request_started,
                request_text=prompt,
                client_task_id=idempotency_key,
                request_timeout_secs=timeout_secs,
            )
            lease = acquire_image_concurrency(identity, units=int(payload.get("n") or 1))
            try:
                await filter_or_log(call, prompt)
                result = await call.run(openai_v1_image_edit.handle, payload)
                if isinstance(result, StreamingResponse):
                    if result.background is None:
                        result.background = BackgroundTask(cleanup_inputs)
                    else:
                        cleanup_tasks = BackgroundTasks()
                        cleanup_tasks.tasks.append(result.background)
                        cleanup_tasks.add_task(cleanup_inputs)
                        result.background = cleanup_tasks
                else:
                    cleanup_inputs()
                return release_image_concurrency_after_response(result, lease)
            except BaseException:
                cleanup_inputs()
                lease.release()
                raise

        if not idempotency_key:
            return await execute_response()

        async def execute() -> CachedImageResponse:
            return CachedImageResponse.capture(await execute_response())

        try:
            cached, replayed = await image_idempotency.run(
                image_idempotency_scope(identity, "/v1/images/edits"),
                idempotency_key,
                fingerprint,
                execute,
                on_replay=cleanup_inputs,
            )
        except (ImageIdempotencyConflict, ImageIdempotencyCapacityExceeded) as exc:
            cleanup_inputs()
            raise_image_idempotency_error(exc)
        if replayed:
            logger.info({"event": "image_idempotency_replay", "endpoint": "/v1/images/edits", "client_task_id": idempotency_key})
        return cached.materialize()

    @router.post("/v1/chat/completions")
    async def create_chat_completion(
        body: ChatCompletionRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        payload = body.model_dump(mode="python")
        is_image_request = is_image_chat_request(payload)
        image_units = parse_image_count(payload.get("n")) if is_image_request else 1
        idempotency_key = ""
        fingerprint = ""
        if is_image_request:
            enforce_image_request_limit(identity, image_units)
            idempotency_key = image_idempotency_key(request, payload.get("client_task_id"))
            if idempotency_key:
                if payload.get("stream"):
                    raise HTTPException(status_code=400, detail={"error": "idempotency is not supported for streaming image requests"})
                payload["client_task_id"] = idempotency_key
            fingerprint = image_request_fingerprint("/v1/chat/completions", payload) if idempotency_key else ""
            apply_image_timeout(identity, payload)
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("prompt"), payload.get("messages"))

        async def execute_response():
            call = LoggedCall(
                identity,
                "/v1/chat/completions",
                model,
                "图片生成" if is_image_request else "文本生成",
                request_text=request_preview,
                request_shape=request_shape(payload.get("messages")),
                client_task_id=idempotency_key,
            )
            lease = acquire_image_concurrency(identity, units=image_units) if is_image_request else None
            try:
                await filter_or_log(call, request_preview)
                result = await call.run(openai_v1_chat_complete.handle, payload)
            except BaseException:
                if lease is not None:
                    lease.release()
                raise
            return release_image_concurrency_after_response(result, lease) if lease is not None else result

        if not idempotency_key:
            return await execute_response()

        async def execute() -> CachedImageResponse:
            return CachedImageResponse.capture(await execute_response())

        try:
            cached, replayed = await image_idempotency.run(
                image_idempotency_scope(identity, "/v1/chat/completions"),
                idempotency_key,
                fingerprint,
                execute,
            )
        except (ImageIdempotencyConflict, ImageIdempotencyCapacityExceeded) as exc:
            raise_image_idempotency_error(exc)
        if replayed:
            logger.info({"event": "image_idempotency_replay", "endpoint": "/v1/chat/completions", "client_task_id": idempotency_key})
        return cached.materialize()

    @router.post("/v1/responses")
    async def create_response(
        body: ResponseCreateRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        payload = body.model_dump(mode="python")
        is_image_request = not openai_v1_response.is_text_response_request(payload)
        idempotency_key = ""
        fingerprint = ""
        if is_image_request:
            idempotency_key = image_idempotency_key(request, payload.get("client_task_id"))
            if idempotency_key:
                if payload.get("stream"):
                    raise HTTPException(status_code=400, detail={"error": "idempotency is not supported for streaming image requests"})
                payload["client_task_id"] = idempotency_key
            fingerprint = image_request_fingerprint("/v1/responses", payload) if idempotency_key else ""
            apply_image_timeout(identity, payload)
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("input"), payload.get("instructions"))

        async def execute_response():
            call = LoggedCall(
                identity,
                "/v1/responses",
                model,
                "Responses",
                request_text=request_preview,
                request_shape=request_shape(payload.get("input")),
                client_task_id=idempotency_key,
            )
            lease = acquire_image_concurrency(identity) if is_image_request else None
            try:
                await filter_or_log(call, request_preview)
                result = await call.run(openai_v1_response.handle, payload)
            except BaseException:
                if lease is not None:
                    lease.release()
                raise
            return release_image_concurrency_after_response(result, lease) if lease is not None else result

        if not idempotency_key:
            return await execute_response()

        async def execute() -> CachedImageResponse:
            return CachedImageResponse.capture(await execute_response())

        try:
            cached, replayed = await image_idempotency.run(
                image_idempotency_scope(identity, "/v1/responses"),
                idempotency_key,
                fingerprint,
                execute,
            )
        except (ImageIdempotencyConflict, ImageIdempotencyCapacityExceeded) as exc:
            raise_image_idempotency_error(exc)
        if replayed:
            logger.info({"event": "image_idempotency_replay", "endpoint": "/v1/responses", "client_task_id": idempotency_key})
        return cached.materialize()

    @router.post("/v1/messages")
    async def create_message(
            body: AnthropicMessageRequest,
            authorization: str | None = Header(default=None),
            x_api_key: str | None = Header(default=None, alias="x-api-key"),
            anthropic_version: str | None = Header(default=None, alias="anthropic-version"),
    ):
        identity = require_identity(authorization or (f"Bearer {x_api_key}" if x_api_key else None))
        payload = body.model_dump(mode="python")
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("system"), payload.get("messages"), payload.get("tools"))
        call = LoggedCall(identity, "/v1/messages", model, "Messages", request_text=request_preview)
        await filter_or_log(call, request_preview)
        return await call.run(anthropic_v1_messages.handle, payload, sse="anthropic")

    @router.post("/v1/search")
    async def search(body: SearchRequest, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        call = LoggedCall(identity, "/v1/search", openai_search.MODEL, "搜索", request_text=body.prompt)
        await filter_or_log(call, body.prompt)
        return await call.run(openai_search.handle, body.model_dump(mode="python"))

    @router.get("/v1/editable-file-tasks")
    async def list_editable_file_tasks(ids: str = "", authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        task_ids = [item.strip() for item in ids.split(",") if item.strip()]
        return await run_in_threadpool(editable_file_task_service.list_tasks, identity, task_ids)

    @router.get("/files/{file_path:path}")
    async def download_editable_file(file_path: str):
        try:
            path = await run_in_threadpool(editable_file_task_service.public_file_path, file_path)
        except Exception as exc:
            raise HTTPException(status_code=404, detail={"error": "file not found"}) from exc
        return FileResponse(path, filename=path.name)

    @router.post("/v1/ppt/generations")
    async def create_ppt_task(body: EditableFileTaskRequest, request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        await filter_or_log(LoggedCall(identity, "/v1/ppt/generations", "gpt-5-5-thinking", "PPT生成任务", request_text=body.prompt), body.prompt)
        try:
            return await run_in_threadpool(
                editable_file_task_service.submit_ppt,
                identity,
                client_task_id=body.client_task_id or "",
                prompt=body.prompt,
                base64_images=body.base64_images,
                base_url=resolve_image_base_url(request),
            )
        except DailyRequestQuotaExceeded as exc:
            raise HTTPException(status_code=429, detail={"error": "daily request quota exhausted"}) from exc

    @router.post("/v1/psd/generations")
    async def create_psd_task(body: EditableFileTaskRequest, request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        await filter_or_log(LoggedCall(identity, "/v1/psd/generations", "gpt-5-5-thinking", "PSD生成任务", request_text=body.prompt), body.prompt)
        try:
            return await run_in_threadpool(
                editable_file_task_service.submit_psd,
                identity,
                client_task_id=body.client_task_id or "",
                prompt=body.prompt,
                base64_images=body.base64_images,
                base_url=resolve_image_base_url(request),
            )
        except DailyRequestQuotaExceeded as exc:
            raise HTTPException(status_code=429, detail={"error": "daily request quota exhausted"}) from exc

    return router
