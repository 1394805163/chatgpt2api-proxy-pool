from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from services.config import DATA_DIR, config
from services.content_filter import request_text
from services.auth_service import ImageRequestLimitExceeded, auth_service
from services.log_service import LOG_TYPE_CALL, log_service
from services.protocol import openai_v1_image_edit, openai_v1_image_generations
from services.time_utils import utc_now_iso, utc_timestamp_iso
from utils.log import logger

TASK_STATUS_QUEUED = "queued"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_SUCCESS = "success"
TASK_STATUS_ERROR = "error"
TERMINAL_STATUSES = {TASK_STATUS_SUCCESS, TASK_STATUS_ERROR}
UNFINISHED_STATUSES = {TASK_STATUS_QUEUED, TASK_STATUS_RUNNING}


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
    if task.get("status") in (TASK_STATUS_RUNNING, TASK_STATUS_QUEUED):
        if task.get("status") == TASK_STATUS_RUNNING:
            # RUNNING 状态仅在 started_ts 被设置后（image_stream_resolve_start）才计时
            base_ts = task.get("started_ts")
        else:
            # QUEUED 状态从 created_ts 开始计时（排队等待中）
            base_ts = task.get("created_ts") or task.get("updated_ts")
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
    ):
        self.path = path
        self.generation_handler = generation_handler
        self.edit_handler = edit_handler
        self.retention_days_getter = retention_days_getter or (lambda: config.image_retention_days)
        self.stale_task_timeout_getter = stale_task_timeout_getter or self._default_stale_task_timeout
        self.max_task_duration_getter = max_task_duration_getter or (lambda: config.image_task_timeout_secs)
        self._lock = threading.RLock()
        self._tasks: dict[str, dict[str, Any]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
        base_url: str = "",
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "model": model,
            "n": 1,
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
        base_url: str = "",
        images: list[tuple[bytes, str, str]] | None = None,
        masks: list[tuple[bytes, str, str]] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "images": images or [],
            "mask": masks or [],
            "model": model,
            "n": 1,
            "size": size,
            "quality": quality,
            "response_format": "url",
            "base_url": base_url,
        }
        return self._submit(identity, client_task_id=client_task_id, mode="edit", payload=payload)

    def list_tasks(self, identity: dict[str, object], task_ids: list[str]) -> dict[str, Any]:
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
        payload = {**payload, "client_task_id": task_id}
        owner = _owner_id(identity)
        key = _task_key(owner, task_id)
        quota_reservation_id = f"image-task:{key}"
        now = _now_iso()
        should_start = False
        quota_reserved = False
        with self._lock:
            cleaned = self._mark_stale_unfinished_locked()
            cleaned = self._cleanup_locked() or cleaned
            task = self._tasks.get(key)
            if task is not None:
                if cleaned:
                    self._save_locked()
                return _public_task(task)
            if identity.get("role") == "user":
                try:
                    active_limit = max(1, int(identity.get("image_request_limit") or 5))
                except (TypeError, ValueError):
                    active_limit = 5
                active_count = sum(
                    1
                    for existing in self._tasks.values()
                    if existing.get("owner_id") == owner and existing.get("status") in UNFINISHED_STATUSES
                )
                if active_count >= active_limit:
                    raise ImageRequestLimitExceeded(active_limit)
            quota_reserved = auth_service.reserve_daily_request(identity, quota_reservation_id)
            task = {
                "id": task_id,
                "owner_id": owner,
                "status": TASK_STATUS_QUEUED,
                "mode": mode,
                "model": _clean(payload.get("model"), "gpt-image-2"),
                "size": _clean(payload.get("size")),
                "quality": _clean(payload.get("quality"), "auto"),
                "base_url": _clean(payload.get("base_url")),
                "created_at": now,
                "updated_at": now,
                "created_ts": time.time(),
                "quota_reservation_id": quota_reservation_id if quota_reserved else "",
            }
            self._tasks[key] = task
            try:
                self._save_locked()
            except Exception:
                self._tasks.pop(key, None)
                if quota_reserved:
                    auth_service.finish_daily_request(identity, quota_reservation_id, success=False)
                raise
            should_start = True

        if should_start:
            try:
                self._start_tracked_thread(
                    key,
                    self._run_task,
                    args=(key, mode, payload, dict(identity), _clean(payload.get("model"), "gpt-image-2")),
                    name=f"image-task-{task_id[:16]}",
                )
            except BaseException:
                if quota_reserved:
                    self._settle_quota(key, identity, success=False)
                self._update_task(key, status=TASK_STATUS_ERROR, error="image task failed to start", data=[])
                raise
        return _public_task(task)

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
                current = threading.current_thread()
                with self._lock:
                    if self._threads.get(key) is current:
                        self._threads.pop(key, None)

        thread = threading.Thread(target=run, name=name, daemon=True)
        with self._lock:
            self._threads[key] = thread
        try:
            thread.start()
        except BaseException:
            with self._lock:
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
        self._update_task(key, status=TASK_STATUS_RUNNING, error="")
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
            result = handler(payload_with_progress)
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
            self._settle_quota(key, identity, success=True)
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
            self._settle_quota(key, identity, success=False)
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

    def _settle_quota(self, key: str, identity: dict[str, object], *, success: bool) -> None:
        with self._lock:
            task = self._tasks.get(key)
            reservation_id = _clean(task.get("quota_reservation_id")) if task else ""
            if not reservation_id:
                return
        try:
            auth_service.finish_daily_request(identity, reservation_id, success=success)
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
            task = {
                "id": task_id,
                "owner_id": owner,
                "status": status,
                "mode": "edit" if item.get("mode") == "edit" else "generate",
                "model": _clean(item.get("model"), "gpt-image-2"),
                "size": _clean(item.get("size")),
                "quality": _clean(item.get("quality"), "auto"),
                "base_url": _clean(item.get("base_url")),
                "created_at": _clean(item.get("created_at"), _now_iso()),
                "updated_at": _clean(item.get("updated_at"), _clean(item.get("created_at"), _now_iso())),
                "created_ts": item.get("created_ts"),
                "updated_ts": item.get("updated_ts"),
                "started_ts": item.get("started_ts"),
                "duration_ms": item.get("duration_ms"),
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
            mode = task.get("mode", "generate")
            model = task.get("model", "gpt-image-2")
            if identity.get("role") == "user":
                try:
                    active_limit = max(1, int(identity.get("image_request_limit") or 5))
                except (TypeError, ValueError):
                    active_limit = 5
                active_count = sum(
                    1
                    for existing_key, existing in self._tasks.items()
                    if existing_key != key
                    and existing.get("owner_id") == owner
                    and existing.get("status") in UNFINISHED_STATUSES
                )
                if active_count >= active_limit:
                    raise ImageRequestLimitExceeded(active_limit)

            quota_reserved = auth_service.reserve_daily_request(identity, reservation_id)
            previous_task = dict(task)
            task["status"] = TASK_STATUS_RUNNING
            task["error"] = ""
            task["quota_reservation_id"] = reservation_id if quota_reserved else ""
            task["updated_at"] = _now_iso()
            task["updated_ts"] = time.time()
            try:
                self._save_locked()
            except Exception:
                self._tasks[key] = previous_task
                if quota_reserved:
                    auth_service.finish_daily_request(identity, reservation_id, success=False)
                raise

        try:
            self._start_tracked_thread(
                key,
                self._run_resume_poll,
                args=(key, conversation_id, extra_timeout_secs, dict(identity), mode, model),
                name=f"image-resume-{_clean(task_id)[:16]}",
            )
        except BaseException:
            if quota_reserved:
                self._settle_quota(key, identity, success=False)
            self._update_task(
                key,
                _expected_status=TASK_STATUS_RUNNING,
                status=TASK_STATUS_ERROR,
                error="image resume task failed to start",
                data=[],
            )
            raise
        return _public_task(task)

    def _run_resume_poll(
        self,
        key: str,
        conversation_id: str,
        extra_timeout_secs: float,
        identity: dict[str, object],
        mode: str,
        model: str,
    ) -> None:
        """后台线程：继续轮询已有 conversation_id 的图片结果。"""
        started = time.time()
        try:
            from services.openai_backend_api import OpenAIBackendAPI
            from services.protocol.conversation import format_image_result

            with self._lock:
                task = self._tasks.get(key)
                account_email = _clean(task.get("account_email")) if task else ""
                base_url = _clean(task.get("base_url")) if task else ""
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
                {"b64_json": __import__("base64").b64encode(image_data).decode("ascii")}
                for image_data in backend.download_image_bytes(image_urls)
            ]
            data = format_image_result(
                image_items,
                "",  # prompt 已不重要，结果已经拿到了
                "url",
                base_url,
                int(time.time()),
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
            self._settle_quota(key, identity, success=True)
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


image_task_service = ImageTaskService(DATA_DIR / "image_tasks.json")
