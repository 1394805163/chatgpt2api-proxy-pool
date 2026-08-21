from __future__ import annotations

import gc
import json
import shutil
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from services.config import DATA_DIR, config
from services.content_filter import request_text
from services.auth_service import auth_service
from services.log_service import LOG_TYPE_CALL, log_service
from services.image_concurrency import (
    ImageConcurrencyGate,
    ImageConcurrencyLease,
    image_concurrency_gate,
    image_owner_limit,
)
from services.protocol import openai_v1_image_edit, openai_v1_image_generations
from services.time_utils import utc_now_iso, utc_timestamp_iso
from utils.log import logger

TASK_STATUS_QUEUED = "queued"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_SUCCESS = "success"
TASK_STATUS_ERROR = "error"
TERMINAL_STATUSES = {TASK_STATUS_SUCCESS, TASK_STATUS_ERROR}
UNFINISHED_STATUSES = {TASK_STATUS_QUEUED, TASK_STATUS_RUNNING}


class ImageQueueLimitExceeded(ValueError):
    """Raised when the bounded asynchronous image queue is full."""

    def __init__(self, scope: str, limit: int):
        self.scope = scope
        self.limit = limit
        super().__init__(f"{scope} image task queue limit of {limit} reached")


def _now_iso() -> str:
    return utc_now_iso()


def _timestamp(value: object) -> float:
    if not isinstance(value, str) or not value.strip():
        return 0.0
    raw = value.strip()
    if raw.endswith("Z") or "+" in raw[10:] or "-" in raw[10:]:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except Exception:
            pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw[:26], fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except Exception:
        return 0.0


def _clean(value: object, default: str = "") -> str:
    return str(value or default).strip()


def _owner_id(identity: dict[str, object]) -> str:
    return _clean(identity.get("id")) or "anonymous"


def _task_key(owner_id: str, task_id: str) -> str:
    return f"{owner_id}:{task_id}"


def _collect_image_urls(data: list[Any]) -> list[str]:
    urls: list[str] = []
    for item in data:
        if isinstance(item, dict):
            url = item.get("url")
            if isinstance(url, str) and url:
                urls.append(url)
    return urls


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    item = {
        "id": task.get("id"),
        "status": task.get("status"),
        "mode": task.get("mode"),
        "model": task.get("model"),
        "size": task.get("size"),
        "quality": task.get("quality"),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
    }
    if task.get("conversation_id"):
        item["conversation_id"] = task.get("conversation_id")
    if task.get("data") is not None:
        item["data"] = task.get("data")
    if task.get("usage") is not None:
        item["usage"] = task.get("usage")
    if task.get("error"):
        item["error"] = task.get("error")
    if task.get("progress"):
        item["progress"] = task.get("progress")
    if task.get("duration_ms") is not None:
        item["duration_ms"] = task.get("duration_ms")
    if task.get("queue_duration_ms") is not None:
        item["queue_duration_ms"] = task.get("queue_duration_ms")
    if task.get("status") in (TASK_STATUS_RUNNING, TASK_STATUS_QUEUED):
        if task.get("status") == TASK_STATUS_RUNNING:
            # RUNNING 状态仅在 started_ts 被设置后（image_stream_resolve_start）才计时
            base_ts = task.get("started_ts")
        else:
            # QUEUED 状态从 created_ts 开始计时（排队等待中）
            base_ts = task.get("queued_ts") or task.get("created_ts") or task.get("updated_ts")
        if base_ts:
            item["elapsed_secs"] = round(time.time() - base_ts, 1)
    return item


class ImageTaskService:
    def __init__(
        self,
        path: Path,
        *,
        generation_handler: Callable[[dict[str, Any]], dict[str, Any]] = openai_v1_image_generations.handle,
        edit_handler: Callable[[dict[str, Any]], dict[str, Any]] = openai_v1_image_edit.handle,
        retention_days_getter: Callable[[], int] | None = None,
        stale_task_timeout_getter: Callable[[], float] | None = None,
        max_task_duration_getter: Callable[[], float] | None = None,
        global_concurrency_getter: Callable[[], int] | None = None,
        per_owner_concurrency_getter: Callable[[], int] | None = None,
        queue_timeout_getter: Callable[[], float] | None = None,
        queue_limit_getter: Callable[[], int] | None = None,
        queue_owner_limit_getter: Callable[[], int] | None = None,
        concurrency_gate: ImageConcurrencyGate | None = None,
        reject_when_busy: bool = False,
        queue_when_busy: bool = False,
    ):
        self.path = path
        self.generation_handler = generation_handler
        self.edit_handler = edit_handler
        self.retention_days_getter = retention_days_getter or (lambda: config.image_retention_days)
        self.stale_task_timeout_getter = stale_task_timeout_getter or self._default_stale_task_timeout
        self.max_task_duration_getter = max_task_duration_getter or (lambda: config.image_task_timeout_secs)
        self.global_concurrency_getter = global_concurrency_getter or (lambda: config.image_global_concurrency)
        self.per_owner_concurrency_getter = per_owner_concurrency_getter or (lambda: config.image_user_concurrency)
        self.queue_timeout_getter = queue_timeout_getter or (lambda: config.image_queue_timeout_secs)
        self.queue_limit_getter = queue_limit_getter or (lambda: config.image_async_queue_limit)
        self.queue_owner_limit_getter = queue_owner_limit_getter or (lambda: config.image_async_queue_owner_limit)
        self.concurrency_gate = concurrency_gate or ImageConcurrencyGate()
        self.reject_when_busy = reject_when_busy
        self.queue_when_busy = queue_when_busy
        self._lock = threading.RLock()
        self._dispatch_lock = threading.Lock()
        self._tasks: dict[str, dict[str, Any]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._pending: dict[
            str,
            tuple[Callable[..., None], tuple[Any, ...], dict[str, object], str],
        ] = {}
        self._pending_by_owner: dict[str, deque[str]] = {}
        self._ready_owners: deque[str] = deque()
        self._queue_timer: threading.Timer | None = None
        self._queue_timer_deadline: float | None = None
        self._input_spool_dir = self.path.parent / "image_task_inputs"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._input_spool_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_orphaned_input_spools()
        with self._lock:
            self._tasks = self._load_locked()
            changed = self._recover_unfinished_locked()
            changed = self._cleanup_locked() or changed
            if changed:
                self._save_locked()

    def submit_generation(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        prompt: str,
        model: str,
        size: str | None,
        quality: str = "auto",
        n: int = 1,
        base_url: str = "",
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "model": model,
            "n": max(1, min(100, int(n))),
            "size": size,
            "quality": quality,
            "response_format": "url",
            "base_url": base_url,
        }
        return self._submit(identity, client_task_id=client_task_id, mode="generate", payload=payload)

    def submit_edit(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        prompt: str,
        model: str,
        size: str | None,
        quality: str = "auto",
        n: int = 1,
        base_url: str = "",
        images: list[tuple[bytes | str, str, str]] | None = None,
        masks: list[tuple[bytes | str, str, str]] | None = None,
    ) -> dict[str, Any]:
        task_id = _clean(client_task_id)
        if not task_id:
            raise ValueError("client_task_id is required")
        payload = {
            "prompt": prompt,
            "images": images or [],
            "mask": masks or [],
            "model": model,
            "n": max(1, min(100, int(n))),
            "size": size,
            "quality": quality,
            "response_format": "url",
            "base_url": base_url,
        }
        spooled_payload = self._spool_edit_payload(payload)
        try:
            result = self._submit(identity, client_task_id=task_id, mode="edit", payload=spooled_payload)
        except BaseException:
            self._cleanup_payload_spool(spooled_payload)
            raise
        if not spooled_payload.get("_spool_enqueued"):
            self._cleanup_payload_spool(spooled_payload)
        return result

    def _spool_edit_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        spool_path = self._input_spool_dir / uuid4().hex
        spool_path.mkdir(parents=True, exist_ok=False)
        references: dict[str, list[dict[str, str]]] = {"images": [], "mask": []}
        try:
            for field in ("images", "mask"):
                for index, item in enumerate(payload.get(field) or []):
                    data, filename, mime_type = item
                    suffix = Path(str(filename or "")).suffix.lower()
                    if not (1 < len(suffix) <= 10 and suffix[1:].isalnum()):
                        suffix = ".bin"
                    file_path = spool_path / f"{field}-{index}{suffix}"
                    if isinstance(data, (str, Path)):
                        shutil.copyfile(Path(data), file_path)
                    else:
                        file_path.write_bytes(bytes(data))
                    references[field].append({
                        "path": str(file_path),
                        "filename": str(filename or "image.png"),
                        "mime_type": str(mime_type or "image/png"),
                    })
        except BaseException:
            shutil.rmtree(spool_path, ignore_errors=True)
            raise
        return {
            **payload,
            "images": [],
            "mask": [],
            "_input_spool_dir": str(spool_path),
            "_spooled_inputs": references,
        }

    @staticmethod
    def _materialize_payload(payload: dict[str, Any]) -> dict[str, Any]:
        references = payload.get("_spooled_inputs")
        if not isinstance(references, dict):
            return payload
        materialized = {
            key: value
            for key, value in payload.items()
            if key not in {"_input_spool_dir", "_spooled_inputs", "_spool_enqueued"}
        }
        for field in ("images", "mask"):
            items: list[tuple[str, str, str]] = []
            for reference in references.get(field) or []:
                if not isinstance(reference, dict):
                    continue
                file_path = Path(str(reference.get("path") or ""))
                items.append((
                    str(file_path),
                    str(reference.get("filename") or "image.png"),
                    str(reference.get("mime_type") or "image/png"),
                ))
            materialized[field] = items
        return materialized

    @staticmethod
    def _cleanup_payload_spool(payload: object) -> None:
        if not isinstance(payload, dict):
            return
        spool_dir = _clean(payload.get("_input_spool_dir"))
        if spool_dir:
            shutil.rmtree(spool_dir, ignore_errors=True)

    @staticmethod
    def _pending_payload(pending: object) -> dict[str, Any] | None:
        if not isinstance(pending, tuple) or len(pending) < 2:
            return None
        args = pending[1]
        if not isinstance(args, tuple) or len(args) < 3:
            return None
        payload = args[2]
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _pending_lease(pending: object) -> ImageConcurrencyLease | None:
        if not isinstance(pending, tuple) or len(pending) < 2:
            return None
        args = pending[1]
        if not isinstance(args, tuple):
            return None
        return next((item for item in args if isinstance(item, ImageConcurrencyLease)), None)

    def _cleanup_orphaned_input_spools(self) -> None:
        for path in self._input_spool_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    @staticmethod
    def _release_process_memory() -> None:
        gc.collect()
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    def list_tasks(self, identity: dict[str, object], task_ids: list[str]) -> dict[str, Any]:
        self._expire_queued_tasks()
        self._dispatch_available()
        owner = _owner_id(identity)
        requested_ids = [_clean(task_id) for task_id in task_ids if _clean(task_id)]
        with self._lock:
            changed = self._mark_stale_unfinished_locked()
            changed = self._cleanup_locked() or changed
            if changed:
                self._save_locked()
            items = []
            missing_ids = []
            for task_id in requested_ids:
                task = self._tasks.get(_task_key(owner, task_id))
                if task is None:
                    missing_ids.append(task_id)
                else:
                    items.append(_public_task(task))
            if not requested_ids:
                items = [
                    _public_task(task)
                    for task in self._tasks.values()
                    if task.get("owner_id") == owner
                ]
                items.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
                missing_ids = []
            return {"items": items, "missing_ids": missing_ids}

    def _submit(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        mode: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        task_id = _clean(client_task_id)
        if not task_id:
            raise ValueError("client_task_id is required")
        payload["client_task_id"] = task_id
        owner = _owner_id(identity)
        try:
            retention_seconds = int(auth_service.image_retention_seconds(identity))
            if retention_seconds <= 0:
                raise ValueError("image retention must be positive")
        except (AttributeError, TypeError, ValueError):
            try:
                retention_seconds = max(1, int(self.retention_days_getter())) * 86400
            except (TypeError, ValueError):
                retention_seconds = 30 * 86400
        payload.setdefault("_image_retention_seconds", retention_seconds)
        payload.setdefault("_image_owner_id", owner)
        key = _task_key(owner, task_id)
        quota_reservation_id = f"image-task:{key}"
        now = _now_iso()
        now_ts = time.time()
        should_start = False
        quota_reserved = False
        with self._lock:
            cleaned = self._mark_stale_unfinished_locked()
            cleaned = self._cleanup_locked() or cleaned
            task = self._tasks.get(key)
            if task is not None:
                self._cleanup_payload_spool(payload)
                if cleaned:
                    self._save_locked()
                return _public_task(task)
            quota_units = self._validate_requested_units(
                identity,
                max(1, int(payload.get("n") or 1)),
            )
            if self.queue_when_busy:
                self._ensure_queue_capacity_locked(identity)
                lease = None
            else:
                lease = self._acquire_concurrency(identity, units=quota_units)
            try:
                quota_reserved = auth_service.reserve_daily_request(
                    identity,
                    quota_reservation_id,
                    units=quota_units,
                )
            except BaseException:
                if lease is not None:
                    lease.release()
                raise
            task = {
                "id": task_id,
                "owner_id": owner,
                "status": TASK_STATUS_QUEUED,
                "mode": mode,
                "model": _clean(payload.get("model"), "gpt-image-2"),
                "size": _clean(payload.get("size")),
                "quality": _clean(payload.get("quality"), "auto"),
                "quota_units": quota_units,
                "base_url": _clean(payload.get("base_url")),
                "image_retention_seconds": payload.get("_image_retention_seconds"),
                "created_at": now,
                "updated_at": now,
                "created_ts": now_ts,
                "queued_ts": now_ts,
                "quota_reservation_id": quota_reservation_id if quota_reserved else "",
            }
            self._tasks[key] = task
            try:
                self._save_locked()
            except Exception:
                self._tasks.pop(key, None)
                if lease is not None:
                    lease.release()
                if quota_reserved:
                    auth_service.finish_daily_request(identity, quota_reservation_id, success=False)
                raise
            self._enqueue_pending_locked(
                key,
                self._run_task,
                (key, mode, payload, dict(identity), _clean(payload.get("model"), "gpt-image-2"), lease),
                dict(identity),
                f"image-task-{task_id[:16]}",
            )
            payload["_spool_enqueued"] = True
            should_start = True

        if should_start:
            self._dispatch_available()
        with self._lock:
            return _public_task(self._tasks.get(key, task))

    def _enqueue_pending_locked(
        self,
        key: str,
        target: Callable[..., None],
        args: tuple[Any, ...],
        identity: dict[str, object],
        name: str,
    ) -> None:
        owner = _owner_id(identity)
        owner_queue = self._pending_by_owner.get(owner)
        if owner_queue is None:
            owner_queue = deque()
            self._pending_by_owner[owner] = owner_queue
        if not owner_queue:
            self._ready_owners.append(owner)
        owner_queue.append(key)
        self._pending[key] = (target, args, identity, name)
        self._schedule_queue_expiry_locked()

    def _running_count_locked(self, owner: str | None = None) -> int:
        count = 0
        stale_keys: list[str] = []
        for key, thread in self._threads.items():
            if not thread.is_alive():
                stale_keys.append(key)
                continue
            task = self._tasks.get(key)
            if (
                task
                and task.get("status") in UNFINISHED_STATUSES
                and (owner is None or task.get("owner_id") == owner)
            ):
                count += 1
        for key in stale_keys:
            self._threads.pop(key, None)
        return count

    def _physical_thread_count_locked(self) -> int:
        return sum(1 for thread in self._threads.values() if thread.is_alive())

    def _global_concurrency(self) -> int:
        try:
            return max(1, int(self.global_concurrency_getter()))
        except Exception:
            return 10

    def _per_owner_concurrency(self) -> int:
        try:
            return max(1, int(self.per_owner_concurrency_getter()))
        except Exception:
            return 2

    def _owner_concurrency(self, identity: dict[str, object]) -> int:
        global_limit = self._global_concurrency()
        if _clean(identity.get("role")).lower() == "admin":
            return image_owner_limit(identity, global_limit)
        if identity.get("image_concurrency_limit") is None:
            return min(self._per_owner_concurrency(), global_limit)
        return image_owner_limit(identity, global_limit)

    def _acquire_concurrency(
        self,
        identity: dict[str, object],
        *,
        units: int = 1,
    ) -> ImageConcurrencyLease | None:
        if not self.reject_when_busy:
            return None
        return self.concurrency_gate.try_acquire(
            identity,
            global_limit=self._global_concurrency(),
            owner_limit=self._owner_concurrency(identity),
            units=max(1, int(units)),
        )

    def _validate_requested_units(self, identity: dict[str, object], units: int) -> int:
        """Reject work that can never fit instead of leaving it queued forever."""
        normalized_units = max(1, int(units))
        global_limit = self._global_concurrency()
        if normalized_units > global_limit:
            raise ImageConcurrencyLimitExceeded("request", global_limit)
        owner_limit = self._owner_concurrency(identity)
        if normalized_units > owner_limit:
            raise ImageConcurrencyLimitExceeded("owner", owner_limit)
        return normalized_units

    @classmethod
    def _pending_units(cls, pending: object) -> int:
        payload = cls._pending_payload(pending)
        try:
            return max(1, int(payload.get("n") or 1)) if payload is not None else 1
        except (TypeError, ValueError):
            return 1

    def _queue_timeout(self) -> float:
        try:
            return max(0.01, float(self.queue_timeout_getter()))
        except Exception:
            return 600.0

    def _queue_limit(self) -> int:
        try:
            return max(0, int(self.queue_limit_getter()))
        except Exception:
            return 100

    def _queue_owner_limit(self) -> int:
        try:
            return max(0, int(self.queue_owner_limit_getter()))
        except Exception:
            return 50

    def _ensure_queue_capacity_locked(self, identity: dict[str, object]) -> None:
        owner = _owner_id(identity)
        owner_queue = self._pending_by_owner.get(owner)
        if len(self._pending) >= self._queue_limit():
            raise ImageQueueLimitExceeded("global_queue", self._queue_limit())
        if owner_queue is not None and len(owner_queue) >= self._queue_owner_limit():
            raise ImageQueueLimitExceeded("owner_queue", self._queue_owner_limit())

    def _restore_pending_locked(
        self,
        key: str,
        pending: tuple[Callable[..., None], tuple[Any, ...], dict[str, object], str],
    ) -> None:
        """Put a task back at the head when a shared concurrency slot is busy."""
        self._pending[key] = pending
        owner = _owner_id(pending[2])
        owner_queue = self._pending_by_owner.get(owner)
        if owner_queue is None:
            owner_queue = deque()
            self._pending_by_owner[owner] = owner_queue
        owner_queue.appendleft(key)
        if owner not in self._ready_owners:
            self._ready_owners.appendleft(owner)
        self._schedule_queue_expiry_locked()

    def _take_next_pending_locked(
        self,
    ) -> tuple[str, tuple[Callable[..., None], tuple[Any, ...], dict[str, object], str]] | None:
        owner_attempts = len(self._ready_owners)
        per_owner_limit = self._per_owner_concurrency()
        for _ in range(owner_attempts):
            owner = self._ready_owners.popleft()
            owner_queue = self._pending_by_owner.get(owner)
            if not owner_queue:
                self._pending_by_owner.pop(owner, None)
                continue
            pending = self._pending.get(owner_queue[0])
            if pending is None:
                owner_queue.popleft()
                if owner_queue:
                    self._ready_owners.append(owner)
                else:
                    self._pending_by_owner.pop(owner, None)
                continue
            identity = pending[2]
            owner_limit = self._owner_concurrency(identity)
            if self._running_count_locked(owner) >= owner_limit:
                self._ready_owners.append(owner)
                continue
            key = owner_queue.popleft()
            if owner_queue:
                self._ready_owners.append(owner)
            else:
                self._pending_by_owner.pop(owner, None)
            pending = self._pending.pop(key, None)
            if pending is None:
                continue
            task = self._tasks.get(key)
            if not task or task.get("status") != TASK_STATUS_QUEUED:
                lease = self._pending_lease(pending)
                if lease is not None:
                    lease.release()
                continue
            self._schedule_queue_expiry_locked()
            return key, pending
        self._schedule_queue_expiry_locked()
        return None

    def _dispatch_available(self) -> None:
        with self._dispatch_lock:
            self._expire_queued_tasks()
            while True:
                with self._lock:
                    global_limit = self._global_concurrency()
                    # A timed-out task still owns sockets and memory until its worker
                    # exits, so do not create bypass threads above the configured limit.
                    if self._running_count_locked() >= global_limit:
                        return
                    selected = self._take_next_pending_locked()
                if selected is None:
                    return
                key, (target, args, identity, name) = selected
                running_args = args
                try:
                    if self.queue_when_busy:
                        lease = self._acquire_concurrency(
                            identity,
                            units=self._pending_units((target, args, identity, name)),
                        )
                        if lease is None:
                            raise RuntimeError("image concurrency gate is not configured")
                        running_args = (*args[:-1], lease)
                    self._start_tracked_thread(
                        key,
                        target,
                        args=running_args,
                        name=name,
                    )
                except ImageConcurrencyLimitExceeded:
                    with self._lock:
                        self._restore_pending_locked(key, (target, args, identity, name))
                    return
                except BaseException as exc:
                    failed_pending = (target, running_args, identity, name)
                    self._cleanup_payload_spool(self._pending_payload(failed_pending))
                    lease = self._pending_lease(failed_pending)
                    if lease is not None:
                        lease.release()
                    self._settle_quota(key, identity, success=False)
                    self._update_task(
                        key,
                        status=TASK_STATUS_ERROR,
                        error=str(exc) or "image task failed to start",
                        data=[],
                    )

    def _remove_pending_locked(self, key: str) -> None:
        pending = self._pending.pop(key, None)
        if pending is None:
            return
        task = self._tasks.get(key)
        owner = _clean(task.get("owner_id")) if task else ""
        owner_queue = self._pending_by_owner.get(owner)
        if owner_queue is None:
            return
        try:
            owner_queue.remove(key)
        except ValueError:
            pass
        if owner_queue:
            return
        self._pending_by_owner.pop(owner, None)
        self._ready_owners = deque(item for item in self._ready_owners if item != owner)

    def _schedule_queue_expiry_locked(self) -> None:
        if not self._pending:
            if self._queue_timer is not None:
                self._queue_timer.cancel()
            self._queue_timer = None
            self._queue_timer_deadline = None
            return
        now = time.time()
        timeout = self._queue_timeout()
        deadlines: list[float] = []
        for key in self._pending:
            task = self._tasks.get(key)
            try:
                created_ts = float(task.get("queued_ts") or task.get("created_ts") or now) if task else now
            except (TypeError, ValueError):
                created_ts = now
            deadlines.append(created_ts + timeout)
        next_deadline = min(deadlines)
        if (
            self._queue_timer is not None
            and self._queue_timer.is_alive()
            and self._queue_timer_deadline is not None
            and self._queue_timer_deadline <= next_deadline
        ):
            return
        if self._queue_timer is not None:
            self._queue_timer.cancel()
        timer = threading.Timer(max(0.01, next_deadline - now), self._on_queue_timer)
        timer.daemon = True
        self._queue_timer = timer
        self._queue_timer_deadline = next_deadline
        timer.start()

    def _on_queue_timer(self) -> None:
        current = threading.current_thread()
        with self._lock:
            if self._queue_timer is not current:
                return
            self._queue_timer = None
            self._queue_timer_deadline = None
        self._expire_queued_tasks()
        self._dispatch_available()

    def _expire_queued_tasks(self) -> None:
        now = time.time()
        timeout = self._queue_timeout()
        releases: list[tuple[dict[str, object], str]] = []
        changed = False
        with self._lock:
            for key, pending in list(self._pending.items()):
                task = self._tasks.get(key)
                if not task or task.get("status") != TASK_STATUS_QUEUED:
                    self._remove_pending_locked(key)
                    continue
                try:
                    created_ts = float(task.get("queued_ts") or task.get("created_ts") or now)
                except (TypeError, ValueError):
                    created_ts = now
                if now - created_ts < timeout:
                    continue
                identity = pending[2]
                payload = self._pending_payload(pending)
                lease = self._pending_lease(pending)
                reservation_id = _clean(task.get("quota_reservation_id"))
                task["status"] = TASK_STATUS_ERROR
                task["error"] = f"图片任务排队超过 {timeout:g} 秒，已取消；请稍后重新提交"
                task["data"] = []
                task["queue_duration_ms"] = int(max(0.0, now - created_ts) * 1000)
                task["duration_ms"] = task["queue_duration_ms"]
                task["updated_at"] = _now_iso()
                task["updated_ts"] = now
                task["quota_reservation_id"] = ""
                self._remove_pending_locked(key)
                self._cleanup_payload_spool(payload)
                if lease is not None:
                    lease.release()
                if reservation_id:
                    releases.append((identity, reservation_id))
                changed = True
            if changed:
                self._save_locked()
            self._schedule_queue_expiry_locked()
        for identity, reservation_id in releases:
            try:
                auth_service.finish_daily_request(identity, reservation_id, success=False)
            except Exception as exc:
                logger.error(f"Failed to release queued image task quota: {exc}")

    def _start_tracked_thread(
        self,
        key: str,
        target: Callable[..., None],
        *,
        args: tuple[Any, ...],
        name: str,
    ) -> None:
        def run() -> None:
            try:
                target(*args)
            finally:
                pending_payload = args[2] if len(args) >= 3 and isinstance(args[2], dict) else None
                self._cleanup_payload_spool(pending_payload)
                lease = next((item for item in args if isinstance(item, ImageConcurrencyLease)), None)
                if lease is not None:
                    lease.release()
                current = threading.current_thread()
                with self._lock:
                    if self._threads.get(key) is current:
                        self._threads.pop(key, None)
                self._release_process_memory()
                self._dispatch_available()

        thread = threading.Thread(target=run, name=name, daemon=True)
        with self._lock:
            self._threads[key] = thread
            try:
                thread.start()
            except BaseException:
                if self._threads.get(key) is thread:
                    self._threads.pop(key, None)
                raise

    def _run_task(
        self,
        key: str,
        mode: str,
        payload: dict[str, Any],
        identity: dict[str, object],
        model: str,
        _lease: ImageConcurrencyLease,
    ) -> None:
        started = time.time()
        max_duration = self._max_task_duration(identity)
        cancel_event = threading.Event()
        deadline_timer = threading.Timer(
            max_duration,
            self._expire_task,
            args=(key, identity, mode, model, started, payload, cancel_event, max_duration),
        )
        deadline_timer.daemon = True
        with self._lock:
            task = self._tasks.get(key)
            try:
                created_ts = float(task.get("queued_ts") or task.get("created_ts") or started) if task else started
            except (TypeError, ValueError):
                created_ts = started
        self._update_task(
            key,
            status=TASK_STATUS_RUNNING,
            error="",
            queue_duration_ms=int(max(0.0, started - created_ts) * 1000),
        )
        deadline_timer.start()
        # 创建进度回调，每个步骤完成后更新任务状态
        def progress_callback(step: str) -> None:
            if not self._task_can_finish(key):
                return
            if step.startswith("account_email:"):
                account_email = _clean(step.split(":", 1)[1])
                if account_email:
                    self._update_task(key, account_email=account_email)
                return
            if step == "image_stream_resolve_start":
                self._update_task(key, started_ts=time.time())
            self._update_task(key, progress=step)
        # 将进度回调添加到 payload 中（handler 会提取并传递给 ConversationRequest）
        payload_with_progress = {
            **payload,
            "progress_callback": progress_callback,
            "task_deadline_ts": started + max_duration,
            "task_timeout_secs": max_duration,
            "cancel_event": cancel_event,
        }
        try:
            handler = self.edit_handler if mode == "edit" else self.generation_handler
            materialized_payload = self._materialize_payload(payload_with_progress)
            result = handler(materialized_payload)
            if not isinstance(result, dict):
                raise RuntimeError("image task returned streaming result unexpectedly")
            data = result.get("data")
            account_email = _clean(result.get("_account_email") or result.get("account_email"))
            if not isinstance(data, list) or not data:
                upstream = _clean(result.get("message"))
                if upstream:
                    message = upstream
                else:
                    message = "号池中没有可用账号或所有账号均被限流，请检查号池状态（账号额度、是否被封禁、是否到达生图上限）"
                error = RuntimeError(message)
                if account_email:
                    setattr(error, "account_email", account_email)
                raise error
            usage = result.get("usage")
            duration_ms = int((time.time() - started) * 1000)
            if not self._update_task(
                key,
                _expected_status=TASK_STATUS_RUNNING,
                status=TASK_STATUS_SUCCESS,
                data=data,
                usage=usage,
                error="",
                **({"account_email": account_email} if account_email else {}),
                duration_ms=duration_ms,
            ):
                return
            self._settle_quota(key, identity, success=True, units=len(data))
            self._log_call(
                identity,
                mode,
                model,
                started,
                "调用完成",
                request_preview=request_text(payload.get("prompt")),
                urls=_collect_image_urls(data),
                account_email=account_email,
            )
        except Exception as exc:
            error_message = str(exc) or "image task failed"
            account_email = _clean(getattr(exc, "account_email", ""))
            conversation_id = _clean(getattr(exc, "conversation_id", ""))
            duration_ms = int((time.time() - started) * 1000)
            if not self._update_task(
                key,
                _expected_status=TASK_STATUS_RUNNING,
                status=TASK_STATUS_ERROR,
                error=error_message,
                data=[],
                **({"account_email": account_email} if account_email else {}),
                duration_ms=duration_ms,
                **({"conversation_id": conversation_id} if conversation_id else {}),
            ):
                return
            self._settle_quota(key, identity, success=False, units=0)
            self._log_call(
                identity,
                mode,
                model,
                started,
                "调用失败",
                request_preview=request_text(payload.get("prompt")),
                status="failed",
                error=error_message,
                account_email=account_email,
            )
        finally:
            deadline_timer.cancel()
            cancel_event.set()

    def _task_can_finish(self, key: str) -> bool:
        with self._lock:
            task = self._tasks.get(key)
            return bool(task and task.get("status") == TASK_STATUS_RUNNING)

    def _settle_quota(
        self,
        key: str,
        identity: dict[str, object],
        *,
        success: bool,
        units: int | None = None,
    ) -> None:
        with self._lock:
            task = self._tasks.get(key)
            reservation_id = _clean(task.get("quota_reservation_id")) if task else ""
            if not reservation_id:
                return
        try:
            auth_service.finish_daily_request(
                identity,
                reservation_id,
                success=success,
                units=units,
                is_image=True,
            )
        except Exception as exc:
            logger.error(f"Failed to settle image task daily usage: {exc}")
            return
        with self._lock:
            task = self._tasks.get(key)
            if task is None or _clean(task.get("quota_reservation_id")) != reservation_id:
                return
            task["quota_reservation_id"] = ""
            self._save_locked()

    def _max_task_duration(self, identity: dict[str, object] | None = None) -> float:
        if identity and identity.get("role") == "user":
            return config.user_image_task_timeout_secs
        try:
            return max(0.01, float(self.max_task_duration_getter()))
        except Exception:
            return 150.0

    def _expire_task(
        self,
        key: str,
        identity: dict[str, object],
        mode: str,
        model: str,
        started: float,
        payload: dict[str, Any],
        cancel_event: threading.Event,
        max_duration: float,
    ) -> None:
        cancel_event.set()
        now = time.time()
        timeout_label = f"{max_duration:g}"
        error_message = f"图片任务已达到 {timeout_label} 秒总时限；已停止等待，请重新提交"
        with self._lock:
            task = self._tasks.get(key)
            if not task or task.get("status") not in UNFINISHED_STATUSES:
                return
            task["status"] = TASK_STATUS_ERROR
            task["error"] = error_message
            task["data"] = []
            task.pop("progress", None)
            task["duration_ms"] = int(max(0.0, now - started) * 1000)
            task["updated_at"] = _now_iso()
            task["updated_ts"] = now
            reservation_id = _clean(task.get("quota_reservation_id"))
            task["quota_reservation_id"] = ""
            self._save_locked()
        if reservation_id:
            try:
                auth_service.finish_daily_request(identity, reservation_id, success=False)
            except Exception as exc:
                logger.error(f"Failed to release expired image task quota: {exc}")
        self._log_call(
            identity,
            mode,
            model,
            started,
            "调用失败",
            request_preview=request_text(payload.get("prompt")),
            status="failed",
            error=error_message,
        )
        self._dispatch_available()

    def _log_call(
        self,
        identity: dict[str, object],
        mode: str,
        model: str,
        started: float,
        suffix: str,
        *,
        request_preview: str = "",
        status: str = "success",
        error: str = "",
        urls: list[str] | None = None,
        account_email: str = "",
    ) -> None:
        endpoint = "/v1/images/edits" if mode == "edit" else "/v1/images/generations"
        summary_prefix = "图生图" if mode == "edit" else "文生图"
        detail = {
            "key_id": identity.get("id"),
            "key_name": identity.get("name"),
            "role": identity.get("role"),
            "endpoint": endpoint,
            "model": model,
            "started_at": utc_timestamp_iso(started),
            "ended_at": _now_iso(),
            "duration_ms": int((time.time() - started) * 1000),
            "status": status,
        }
        if request_preview:
            detail["request_text"] = request_preview
        if error:
            detail["error"] = error
        if account_email:
            detail["account_email"] = account_email
        if urls:
            detail["urls"] = list(dict.fromkeys(urls))
        try:
            log_service.add(LOG_TYPE_CALL, f"{summary_prefix}{suffix}", detail)
        except Exception:
            pass

    def _update_task(self, key: str, **updates: Any) -> bool:
        with self._lock:
            task = self._tasks.get(key)
            if task is None:
                return False
            expected_status = updates.pop("_expected_status", None)
            if expected_status is not None and task.get("status") != expected_status:
                return False
            task.update(updates)
            task["updated_at"] = _now_iso()
            task["updated_ts"] = time.time()
            self._save_locked()
            return True

    def _load_locked(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        raw_items = raw.get("tasks") if isinstance(raw, dict) else raw
        if not isinstance(raw_items, list):
            return {}
        tasks: dict[str, dict[str, Any]] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            task_id = _clean(item.get("id"))
            owner = _clean(item.get("owner_id"))
            if not task_id or not owner:
                continue
            status = _clean(item.get("status"))
            if status not in {TASK_STATUS_QUEUED, TASK_STATUS_RUNNING, TASK_STATUS_SUCCESS, TASK_STATUS_ERROR}:
                status = TASK_STATUS_ERROR
            try:
                quota_units = max(1, int(item.get("quota_units") or 1))
            except (TypeError, ValueError):
                quota_units = 1
            task = {
                "id": task_id,
                "owner_id": owner,
                "status": status,
                "mode": "edit" if item.get("mode") == "edit" else "generate",
                "model": _clean(item.get("model"), "gpt-image-2"),
                "size": _clean(item.get("size")),
                "quality": _clean(item.get("quality"), "auto"),
                "quota_units": quota_units,
                "base_url": _clean(item.get("base_url")),
                "created_at": _clean(item.get("created_at"), _now_iso()),
                "updated_at": _clean(item.get("updated_at"), _clean(item.get("created_at"), _now_iso())),
                "created_ts": item.get("created_ts"),
                "queued_ts": item.get("queued_ts"),
                "updated_ts": item.get("updated_ts"),
                "started_ts": item.get("started_ts"),
                "duration_ms": item.get("duration_ms"),
                "queue_duration_ms": item.get("queue_duration_ms"),
            }
            conversation_id = _clean(item.get("conversation_id"))
            if conversation_id:
                task["conversation_id"] = conversation_id
            account_email = _clean(item.get("account_email"))
            if account_email:
                task["account_email"] = account_email
            quota_reservation_id = _clean(item.get("quota_reservation_id"))
            if quota_reservation_id:
                task["quota_reservation_id"] = quota_reservation_id
            data = item.get("data")
            if isinstance(data, list):
                task["data"] = data
            usage = item.get("usage")
            if isinstance(usage, dict):
                task["usage"] = usage
            error = _clean(item.get("error"))
            if error:
                task["error"] = error
            tasks[_task_key(owner, task_id)] = task
        return tasks

    def _save_locked(self) -> None:
        items = sorted(self._tasks.values(), key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps({"tasks": items}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(self.path)

    def _default_stale_task_timeout(self) -> float:
        try:
            return max(
                30.0,
                max(float(config.image_task_timeout_secs), float(config.user_image_task_timeout_secs))
                + float(config.image_poll_interval_secs)
                + float(config.image_timeout_retry_secs),
            )
        except Exception:
            return 180.0

    def _stale_task_timeout(self) -> float:
        try:
            return max(0.0, float(self.stale_task_timeout_getter()))
        except Exception:
            return 600.0

    def _mark_stale_unfinished_locked(self) -> bool:
        timeout = self._stale_task_timeout()
        if timeout <= 0:
            return False
        now = time.time()
        changed = False
        for key, task in self._tasks.items():
            if task.get("status") not in UNFINISHED_STATUSES:
                continue
            if task.get("status") == TASK_STATUS_QUEUED and key in self._pending:
                continue
            thread = self._threads.get(key)
            if thread is not None and thread.is_alive():
                continue
            base_ts = task.get("updated_ts") or task.get("started_ts") or task.get("created_ts")
            try:
                base = float(base_ts)
            except (TypeError, ValueError):
                base = 0.0
            if base and now - base <= timeout:
                continue
            task["status"] = TASK_STATUS_ERROR
            task["error"] = "图片任务超时，后台生成线程可能已卡住；请重新提交或换账号重试"
            task["duration_ms"] = int(max(0.0, now - float(task.get("created_ts") or base or now)) * 1000)
            task["updated_at"] = _now_iso()
            task["updated_ts"] = now
            reservation_id = _clean(task.get("quota_reservation_id"))
            task["quota_reservation_id"] = ""
            if reservation_id:
                try:
                    auth_service.finish_daily_request(
                        {"id": task.get("owner_id"), "role": "user"},
                        reservation_id,
                        success=False,
                    )
                except Exception as exc:
                    task["quota_reservation_id"] = reservation_id
                    logger.error(f"Failed to release stale image task quota: {exc}")
            changed = True
        return changed

    def _recover_unfinished_locked(self) -> bool:
        changed = False
        for task in self._tasks.values():
            if task.get("status") in UNFINISHED_STATUSES:
                task["status"] = TASK_STATUS_ERROR
                task["error"] = "服务已重启，未完成的图片任务已中断"
                task["updated_at"] = _now_iso()
                task["quota_reservation_id"] = ""
                changed = True
        return changed

    def _cleanup_locked(self) -> bool:
        try:
            retention_days = max(1, int(self.retention_days_getter()))
        except Exception:
            retention_days = 30
        cutoff = time.time() - retention_days * 86400
        removed_keys = [
            key
            for key, task in self._tasks.items()
            if task.get("status") in TERMINAL_STATUSES and _timestamp(task.get("updated_at")) < cutoff
        ]
        for key in removed_keys:
            self._tasks.pop(key, None)
        return bool(removed_keys)

    def resume_poll(
        self,
        identity: dict[str, object],
        task_id: str,
        extra_timeout_secs: float = 30.0,
    ) -> dict[str, Any]:
        """Resume polling for a timed-out image task as a new billable attempt."""
        owner = _owner_id(identity)
        key = _task_key(owner, _clean(task_id))
        reservation_id = f"image-resume:{key}:{uuid4().hex}"
        quota_reserved = False
        with self._lock:
            task = self._tasks.get(key)
            if task is None:
                raise ValueError("task not found")
            if task.get("status") != TASK_STATUS_ERROR:
                raise ValueError("task is not in error state")
            error_msg = _clean(task.get("error"))
            if "超时" not in error_msg:
                raise ValueError("task error is not a timeout error")
            conversation_id = _clean(task.get("conversation_id"))
            if not conversation_id:
                raise ValueError("task has no conversation_id")
            existing_thread = self._threads.get(key)
            if existing_thread is not None and existing_thread.is_alive():
                raise ValueError("original image task is still shutting down")
            mode = task.get("mode", "generate")
            model = task.get("model", "gpt-image-2")
            requested_units = max(1, int(task.get("quota_units") or 1))
            lease = self._acquire_concurrency(identity, units=requested_units)
            try:
                quota_reserved = auth_service.reserve_daily_request(
                    identity,
                    reservation_id,
                    units=requested_units,
                )
            except BaseException:
                if lease is not None:
                    lease.release()
                raise
            previous_task = dict(task)
            queued_ts = time.time()
            task["status"] = TASK_STATUS_QUEUED
            task["error"] = ""
            task["quota_reservation_id"] = reservation_id if quota_reserved else ""
            task["updated_at"] = _now_iso()
            task["updated_ts"] = queued_ts
            task["queued_ts"] = queued_ts
            try:
                self._save_locked()
            except Exception:
                self._tasks[key] = previous_task
                if lease is not None:
                    lease.release()
                if quota_reserved:
                    auth_service.finish_daily_request(identity, reservation_id, success=False)
                raise
            self._enqueue_pending_locked(
                key,
                self._run_resume_poll,
                (key, conversation_id, extra_timeout_secs, dict(identity), mode, model, lease),
                dict(identity),
                f"image-resume-{_clean(task_id)[:16]}",
            )
        self._dispatch_available()
        with self._lock:
            return _public_task(self._tasks.get(key, task))

    def _run_resume_poll(
        self,
        key: str,
        conversation_id: str,
        extra_timeout_secs: float,
        identity: dict[str, object],
        mode: str,
        model: str,
        _lease: ImageConcurrencyLease,
    ) -> None:
        """后台线程：继续轮询已有 conversation_id 的图片结果。"""
        started = time.time()
        backend = None
        with self._lock:
            task = self._tasks.get(key)
            try:
                created_ts = float(task.get("queued_ts") or task.get("updated_ts") or started) if task else started
            except (TypeError, ValueError):
                created_ts = started
        if not self._update_task(
            key,
            _expected_status=TASK_STATUS_QUEUED,
            status=TASK_STATUS_RUNNING,
            error="",
            queue_duration_ms=int(max(0.0, started - created_ts) * 1000),
        ):
            return
        try:
            from services.openai_backend_api import OpenAIBackendAPI
            from services.protocol.conversation import format_image_result

            with self._lock:
                task = self._tasks.get(key)
                account_email = _clean(task.get("account_email")) if task else ""
                base_url = _clean(task.get("base_url")) if task else ""
                retention_seconds = task.get("image_retention_seconds") if task else None
                owner_id = _clean(task.get("owner_id")) if task else _owner_id(identity)
            access_token = ""
            if account_email:
                from services.account_service import account_service

                access_token = next(
                    (
                        _clean(account.get("access_token"))
                        for account in account_service.list_accounts()
                        if _clean(account.get("email")).lower() == account_email.lower()
                        and _clean(account.get("access_token"))
                    ),
                    "",
                )
                if not access_token:
                    raise RuntimeError("original image account is no longer available")
            backend = OpenAIBackendAPI(access_token=access_token)
            file_ids, sediment_ids = backend._poll_image_results(
                conversation_id,
                extra_timeout_secs,
            )
            if not file_ids and not sediment_ids:
                raise RuntimeError(
                    f"继续等待 {extra_timeout_secs} 秒后仍未找到图片结果。"
                )

            image_urls = backend.resolve_conversation_image_urls(
                conversation_id, file_ids, sediment_ids, poll=False,
            )
            if not image_urls:
                raise RuntimeError("图片 URL 解析失败")

            image_items = [
                {"image_bytes": image_data}
                for image_data in backend.download_image_bytes(image_urls)
            ]
            data = format_image_result(
                image_items,
                "",  # prompt 已不重要，结果已经拿到了
                "url",
                base_url,
                int(time.time()),
                retention_seconds=int(retention_seconds) if retention_seconds else None,
                owner_id=owner_id,
            )["data"]
            if not self._update_task(
                key,
                _expected_status=TASK_STATUS_RUNNING,
                status=TASK_STATUS_SUCCESS,
                data=data,
                error="",
                duration_ms=int((time.time() - started) * 1000),
            ):
                return
            self._settle_quota(key, identity, success=True, units=len(data))
            self._log_call(
                identity,
                mode,
                model,
                started,
                "调用完成（续轮询）",
                status="success",
                urls=_collect_image_urls(data),
            )
        except Exception as exc:
            error_message = str(exc) or "resume poll failed"
            duration_ms = int((time.time() - started) * 1000)
            if not self._update_task(
                key,
                _expected_status=TASK_STATUS_RUNNING,
                status=TASK_STATUS_ERROR,
                error=error_message,
                data=[],
                duration_ms=duration_ms,
            ):
                return
            self._settle_quota(key, identity, success=False)
            self._log_call(
                identity,
                mode,
                model,
                started,
                "调用失败（续轮询）",
                status="failed",
                error=error_message,
            )
        finally:
            if backend is not None:
                backend.close()


image_task_service = ImageTaskService(
    DATA_DIR / "image_tasks.json",
    concurrency_gate=image_concurrency_gate,
    reject_when_busy=True,
    queue_when_busy=True,
)
