from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import hashlib
import json
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from starlette.responses import Response, StreamingResponse


class ImageIdempotencyConflict(ValueError):
    pass


class ImageIdempotencyCapacityExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class CachedImageResponse:
    payload: object | None = None
    status_code: int | None = None
    body: bytes = b""
    headers: tuple[tuple[str, str], ...] = ()
    size_bytes: int = 0

    @classmethod
    def capture(cls, value: object) -> "CachedImageResponse":
        if isinstance(value, StreamingResponse):
            raise ValueError("idempotency is not supported for streaming image responses")
        if isinstance(value, Response):
            body = bytes(value.body)
            return cls(
                status_code=int(value.status_code),
                body=body,
                headers=tuple((str(key), str(item)) for key, item in value.headers.items()),
                size_bytes=len(body),
            )
        payload = copy.deepcopy(value)
        return cls(payload=payload, size_bytes=_payload_size(payload))

    def materialize(self) -> object:
        if self.status_code is not None:
            return Response(
                content=self.body,
                status_code=self.status_code,
                headers=dict(self.headers),
            )
        return copy.deepcopy(self.payload)


@dataclass
class _Entry:
    fingerprint: str
    future: concurrent.futures.Future[CachedImageResponse]
    created_at: float
    completed_at: float | None = None
    response_size: int = 0


def _payload_size(value: object) -> int:
    if value is None:
        return 4
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8", errors="replace"))
    if isinstance(value, dict):
        return sum(_payload_size(key) + _payload_size(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return sum(_payload_size(item) for item in value)
    return len(str(value).encode("utf-8", errors="replace"))


class ImageIdempotencyRegistry:
    def __init__(
        self,
        *,
        ttl_seconds: float = 600.0,
        max_entries: int = 500,
        max_response_bytes: int = 16 * 1024 * 1024,
        max_cached_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self.max_response_bytes = max(1, int(max_response_bytes))
        self.max_cached_bytes = max(self.max_response_bytes, int(max_cached_bytes))
        self._cached_bytes = 0
        self._lock = threading.RLock()
        self._entries: dict[str, _Entry] = {}

    def _remove_locked(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._cached_bytes = max(0, self._cached_bytes - entry.response_size)

    def _prune_locked(self, now: float) -> None:
        expired = [
            key
            for key, entry in self._entries.items()
            if entry.completed_at is not None and now - entry.completed_at >= self.ttl_seconds
        ]
        for key in expired:
            self._remove_locked(key)
        if len(self._entries) < self.max_entries:
            return
        completed = sorted(
            (
                (entry.completed_at, key)
                for key, entry in self._entries.items()
                if entry.completed_at is not None
            ),
            key=lambda item: item[0] or 0.0,
        )
        while completed and len(self._entries) >= self.max_entries:
            _, key = completed.pop(0)
            self._remove_locked(key)

    def _enforce_memory_limit_locked(self, protected_key: str) -> None:
        completed = sorted(
            (
                (entry.completed_at, key)
                for key, entry in self._entries.items()
                if key != protected_key and entry.completed_at is not None and entry.response_size > 0
            ),
            key=lambda item: item[0] or 0.0,
        )
        while completed and self._cached_bytes > self.max_cached_bytes:
            _, key = completed.pop(0)
            self._remove_locked(key)

    async def run(
        self,
        scope: str,
        key: str,
        fingerprint: str,
        factory: Callable[[], Awaitable[CachedImageResponse]],
        *,
        on_replay: Callable[[], None] | None = None,
    ) -> tuple[CachedImageResponse, bool]:
        if not key:
            return await factory(), False

        registry_key = f"{scope}:{key}"
        replayed = False
        leader = False
        with self._lock:
            now = time.monotonic()
            self._prune_locked(now)
            entry = self._entries.get(registry_key)
            if entry is not None:
                if entry.fingerprint != fingerprint:
                    raise ImageIdempotencyConflict("idempotency key was already used with a different image request")
                replayed = True
                future = entry.future
            else:
                if len(self._entries) >= self.max_entries:
                    raise ImageIdempotencyCapacityExceeded("too many in-flight idempotent image requests")
                future = concurrent.futures.Future()
                entry = _Entry(fingerprint=fingerprint, future=future, created_at=now)
                self._entries[registry_key] = entry
                leader = True

        if leader:
            task = asyncio.create_task(factory())

            def settle(done_task: asyncio.Task[CachedImageResponse]) -> None:
                if done_task.cancelled():
                    future.cancel()
                    result = None
                    error: BaseException | None = None
                else:
                    error = done_task.exception()
                    result = None if error is not None else done_task.result()
                    if error is not None:
                        future.set_exception(error)
                    else:
                        future.set_result(result)
                with self._lock:
                    current = self._entries.get(registry_key)
                    if current is not entry:
                        return
                    entry.completed_at = time.monotonic()
                    if done_task.cancelled():
                        self._remove_locked(registry_key)
                        return
                    if error is not None:
                        self._remove_locked(registry_key)
                        return
                    if result is not None and result.size_bytes > self.max_response_bytes:
                        # Existing waiters hold the task reference, while later
                        # requests may retry without retaining a large b64 body.
                        self._remove_locked(registry_key)
                        return
                    if result is not None:
                        entry.response_size = result.size_bytes
                        self._cached_bytes += result.size_bytes
                        self._enforce_memory_limit_locked(registry_key)

            task.add_done_callback(settle)

        if replayed and on_replay is not None:
            on_replay()
        return await asyncio.shield(asyncio.wrap_future(future)), replayed


def image_request_fingerprint(endpoint: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(endpoint.encode("utf-8"))
    ignored = {
        "base_url",
        "cancel_event",
        "progress_callback",
        "task_deadline_ts",
        "image_owner_id",
        "image_retention_seconds",
    }
    scalar_payload = {
        key: value
        for key, value in payload.items()
        if key not in ignored and key not in {"images", "mask"}
    }
    digest.update(json.dumps(scalar_payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    for field in ("images", "mask"):
        digest.update(field.encode("ascii"))
        for item in payload.get(field) or []:
            if isinstance(item, tuple) and len(item) >= 3:
                data, filename, mime_type = item[:3]
            else:
                data, filename, mime_type = item, "", ""
            digest.update(str(filename).encode("utf-8", errors="replace"))
            digest.update(str(mime_type).encode("utf-8", errors="replace"))
            if isinstance(data, (str, Path)):
                try:
                    is_file = Path(data).is_file()
                except OSError:
                    is_file = False
                if is_file:
                    with Path(data).open("rb") as source:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(chunk)
                else:
                    digest.update(str(data).encode("utf-8", errors="replace"))
            elif isinstance(data, bytes):
                digest.update(data)
            else:
                digest.update(str(data).encode("utf-8", errors="replace"))
    return digest.hexdigest()
