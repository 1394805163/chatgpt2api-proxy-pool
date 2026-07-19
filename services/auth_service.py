from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timezone
from threading import RLock
from typing import Literal
from zoneinfo import ZoneInfo

from services.config import config
from services.storage.base import StorageBackend

AuthRole = Literal["admin", "user"]
DEFAULT_IMAGE_REQUEST_LIMIT = 5
MAX_IMAGE_REQUEST_LIMIT = 100


class DailyRequestQuotaExceeded(ValueError):
    pass


class ImageRequestLimitExceeded(ValueError):
    def __init__(self, limit: int):
        self.limit = limit
        super().__init__(f"single image request exceeds the configured limit of {limit}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class AuthService:
    def __init__(self, storage: StorageBackend):
        self.storage = storage
        self._lock = RLock()
        self._items = self._load()
        self._last_used_flush_at: dict[str, datetime] = {}
        self._daily_reservations: dict[str, set[str]] = {}

    @staticmethod
    def _clean(value: object) -> str:
        return str(value or "").strip()

    @staticmethod
    def _non_negative_int(value: object, default: int = 0) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _image_request_limit(value: object) -> int:
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            normalized = DEFAULT_IMAGE_REQUEST_LIMIT
        return min(MAX_IMAGE_REQUEST_LIMIT, max(1, normalized))

    @staticmethod
    def _today() -> str:
        return datetime.now(ZoneInfo(config.display_timezone)).date().isoformat()

    @staticmethod
    def _default_name(role: object) -> str:
        return "管理员密钥" if str(role or "").strip().lower() == "admin" else "普通用户"

    def _normalize_item(self, raw: object) -> dict[str, object] | None:
        if not isinstance(raw, dict):
            return None
        role = self._clean(raw.get("role")).lower()
        if role not in {"admin", "user"}:
            return None
        key_hash = self._clean(raw.get("key_hash"))
        if not key_hash:
            return None
        item_id = self._clean(raw.get("id")) or uuid.uuid4().hex[:12]
        name = self._clean(raw.get("name")) or self._default_name(role)
        created_at = self._clean(raw.get("created_at")) or _now_iso()
        last_used_at = self._clean(raw.get("last_used_at")) or None
        daily_request_limit = self._non_negative_int(raw.get("daily_request_limit"))
        daily_request_date = self._clean(raw.get("daily_request_date")) or self._today()
        daily_request_used = self._non_negative_int(raw.get("daily_request_used"))
        if daily_request_date != self._today():
            daily_request_date = self._today()
            daily_request_used = 0
        return {
            "id": item_id,
            "name": name,
            "role": role,
            "key_hash": key_hash,
            "enabled": bool(raw.get("enabled", True)),
            "created_at": created_at,
            "last_used_at": last_used_at,
            "daily_request_limit": daily_request_limit,
            "daily_request_used": daily_request_used,
            "daily_request_date": daily_request_date,
            "image_request_limit": self._image_request_limit(raw.get("image_request_limit")),
        }

    def _load(self) -> list[dict[str, object]]:
        try:
            items = self.storage.load_auth_keys()
        except Exception:
            return []
        if not isinstance(items, list):
            return []
        return [normalized for item in items if (normalized := self._normalize_item(item)) is not None]

    def _save(self) -> None:
        self.storage.save_auth_keys(self._items)

    def _save_item_locked(self, item: dict[str, object]) -> None:
        if not self.storage.save_auth_key(item):
            self._save()

    def _reload_locked(self) -> None:
        self._items = self._load()

    @staticmethod
    def _public_item(item: dict[str, object]) -> dict[str, object]:
        daily_request_limit = AuthService._non_negative_int(item.get("daily_request_limit"))
        daily_request_used = AuthService._non_negative_int(item.get("daily_request_used"))
        return {
            "id": item.get("id"),
            "name": item.get("name"),
            "role": item.get("role"),
            "enabled": bool(item.get("enabled", True)),
            "created_at": item.get("created_at"),
            "last_used_at": item.get("last_used_at"),
            "daily_request_limit": daily_request_limit,
            "daily_request_used": daily_request_used,
            "daily_request_remaining": (
                max(0, daily_request_limit - daily_request_used)
                if daily_request_limit > 0
                else None
            ),
            "daily_request_date": item.get("daily_request_date"),
            "image_request_limit": AuthService._image_request_limit(item.get("image_request_limit")),
        }

    def _find_item_locked(self, key_id: str) -> tuple[int, dict[str, object]] | None:
        for index, item in enumerate(self._items):
            if self._clean(item.get("id")) == key_id:
                return index, item
        return None

    def _reset_daily_if_needed_locked(self, index: int, item: dict[str, object]) -> bool:
        today = self._today()
        if self._clean(item.get("daily_request_date")) == today:
            return False
        next_item = dict(item)
        next_item["daily_request_date"] = today
        next_item["daily_request_used"] = 0
        self._items[index] = next_item
        return True

    def list_keys(self, role: AuthRole | None = None) -> list[dict[str, object]]:
        with self._lock:
            self._reload_locked()
            items = [item for item in self._items if role is None or item.get("role") == role]
            return [self._public_item(item) for item in items]

    def _has_key_hash_locked(self, key_hash: str, *, exclude_id: str = "") -> bool:
        for item in self._items:
            item_id = self._clean(item.get("id"))
            if exclude_id and item_id == exclude_id:
                continue
            stored_hash = self._clean(item.get("key_hash"))
            if stored_hash and hmac.compare_digest(stored_hash, key_hash):
                return True
        return False

    def _build_key_hash_locked(self, raw_key: str, *, exclude_id: str = "") -> str:
        candidate = self._clean(raw_key)
        if not candidate:
            raise ValueError("请输入新的专用密钥")
        admin_key = self._clean(config.auth_key)
        if admin_key and hmac.compare_digest(candidate, admin_key):
            raise ValueError("这个密钥和管理员密钥冲突了，请换一个新的密钥")
        key_hash = _hash_key(candidate)
        if self._has_key_hash_locked(key_hash, exclude_id=exclude_id):
            raise ValueError("这个专用密钥已经存在，请换一个新的密钥")
        return key_hash

    def _has_name_locked(self, name: str, *, role: AuthRole | None = None, exclude_id: str = "") -> bool:
        candidate = self._clean(name)
        if not candidate:
            return False
        for item in self._items:
            item_id = self._clean(item.get("id"))
            if exclude_id and item_id == exclude_id:
                continue
            if role is not None and item.get("role") != role:
                continue
            if self._clean(item.get("name")) == candidate:
                return True
        return False

    def _build_default_name_locked(self, role: AuthRole, *, exclude_id: str = "") -> str:
        base_name = self._default_name(role)
        if not self._has_name_locked(base_name, role=role, exclude_id=exclude_id):
            return base_name
        suffix = 2
        while True:
            candidate = f"{base_name} {suffix}"
            if not self._has_name_locked(candidate, role=role, exclude_id=exclude_id):
                return candidate
            suffix += 1

    def _build_name_locked(self, name: str, *, role: AuthRole, exclude_id: str = "") -> str:
        candidate = self._clean(name)
        if not candidate:
            return self._build_default_name_locked(role, exclude_id=exclude_id)
        if self._has_name_locked(candidate, role=role, exclude_id=exclude_id):
            raise ValueError("这个名称已经在使用中了，换一个更容易区分的名称吧")
        return candidate

    def create_key(
        self,
        *,
        role: AuthRole,
        name: str = "",
        daily_request_limit: int = 0,
        image_request_limit: int = DEFAULT_IMAGE_REQUEST_LIMIT,
    ) -> tuple[dict[str, object], str]:
        with self._lock:
            self._reload_locked()
            normalized_name = self._build_name_locked(name, role=role)
            while True:
                raw_key = f"sk-{secrets.token_urlsafe(24)}"
                try:
                    key_hash = self._build_key_hash_locked(raw_key)
                    break
                except ValueError:
                    continue
            item = {
                "id": uuid.uuid4().hex[:12],
                "name": normalized_name,
                "role": role,
                "key_hash": key_hash,
                "enabled": True,
                "created_at": _now_iso(),
                "last_used_at": None,
                "daily_request_limit": self._non_negative_int(daily_request_limit),
                "daily_request_used": 0,
                "daily_request_date": self._today(),
                "image_request_limit": self._image_request_limit(image_request_limit),
            }
            self._items.append(item)
            self._save()
            return self._public_item(item), raw_key

    def update_key(
        self,
        key_id: str,
        updates: dict[str, object],
        *,
        role: AuthRole | None = None,
    ) -> dict[str, object] | None:
        normalized_id = self._clean(key_id)
        if not normalized_id:
            return None
        with self._lock:
            self._reload_locked()
            for index, item in enumerate(self._items):
                if item.get("id") != normalized_id:
                    continue
                if role is not None and item.get("role") != role:
                    return None
                next_item = dict(item)
                next_role = "admin" if str(next_item.get("role") or "").strip().lower() == "admin" else "user"
                if "name" in updates and updates.get("name") is not None:
                    next_item["name"] = self._build_name_locked(
                        str(updates.get("name") or ""),
                        role=next_role,
                        exclude_id=normalized_id,
                    )
                if "enabled" in updates and updates.get("enabled") is not None:
                    next_item["enabled"] = bool(updates.get("enabled"))
                if "key" in updates and updates.get("key") is not None:
                    next_item["key_hash"] = self._build_key_hash_locked(str(updates.get("key") or ""), exclude_id=normalized_id)
                if "daily_request_limit" in updates and updates.get("daily_request_limit") is not None:
                    next_item["daily_request_limit"] = self._non_negative_int(updates.get("daily_request_limit"))
                if "image_request_limit" in updates and updates.get("image_request_limit") is not None:
                    next_item["image_request_limit"] = self._image_request_limit(updates.get("image_request_limit"))
                if bool(updates.get("reset_daily_usage")):
                    next_item["daily_request_used"] = 0
                    next_item["daily_request_date"] = self._today()
                self._items[index] = next_item
                self._save()
                return self._public_item(next_item)
        return None

    def delete_key(self, key_id: str, *, role: AuthRole | None = None) -> bool:
        normalized_id = self._clean(key_id)
        if not normalized_id:
            return False
        with self._lock:
            self._reload_locked()
            before = len(self._items)
            self._items = [
                item
                for item in self._items
                if not (item.get("id") == normalized_id and (role is None or item.get("role") == role))
            ]
            if len(self._items) == before:
                return False
            self._daily_reservations.pop(normalized_id, None)
            self._save()
            return True

    def reserve_daily_request(self, identity: dict[str, object], reservation_id: str) -> bool:
        if identity.get("role") != "user":
            return False
        key_id = self._clean(identity.get("id"))
        normalized_reservation_id = self._clean(reservation_id)
        if not key_id or not normalized_reservation_id:
            raise ValueError("user key and reservation id are required")
        with self._lock:
            found = self._find_item_locked(key_id)
            if found is None:
                raise DailyRequestQuotaExceeded("user key is no longer available")
            index, item = found
            previous_item = dict(item)
            reset = self._reset_daily_if_needed_locked(index, item)
            item = self._items[index]
            reservations = self._daily_reservations.setdefault(key_id, set())
            if normalized_reservation_id in reservations:
                return True
            limit = self._non_negative_int(item.get("daily_request_limit"))
            used = self._non_negative_int(item.get("daily_request_used"))
            if limit > 0 and used + len(reservations) >= limit:
                if reset:
                    try:
                        self._save_item_locked(item)
                    except Exception:
                        self._items[index] = previous_item
                        if not reservations:
                            self._daily_reservations.pop(key_id, None)
                        raise
                raise DailyRequestQuotaExceeded("daily request quota exhausted")
            if reset:
                try:
                    self._save_item_locked(item)
                except Exception:
                    self._items[index] = previous_item
                    if not reservations:
                        self._daily_reservations.pop(key_id, None)
                    raise
            reservations.add(normalized_reservation_id)
            return True

    def finish_daily_request(
        self,
        identity: dict[str, object],
        reservation_id: str,
        *,
        success: bool,
    ) -> bool:
        if identity.get("role") != "user":
            return False
        key_id = self._clean(identity.get("id"))
        normalized_reservation_id = self._clean(reservation_id)
        with self._lock:
            reservations = self._daily_reservations.get(key_id)
            if not reservations or normalized_reservation_id not in reservations:
                return False
            if not success:
                reservations.discard(normalized_reservation_id)
                if not reservations:
                    self._daily_reservations.pop(key_id, None)
                return False
            found = self._find_item_locked(key_id)
            if found is None:
                reservations.discard(normalized_reservation_id)
                if not reservations:
                    self._daily_reservations.pop(key_id, None)
                return False
            index, item = found
            previous_item = dict(item)
            self._reset_daily_if_needed_locked(index, item)
            next_item = dict(self._items[index])
            next_item["daily_request_used"] = self._non_negative_int(next_item.get("daily_request_used")) + 1
            self._items[index] = next_item
            try:
                self._save_item_locked(next_item)
            except Exception:
                self._items[index] = previous_item
                raise
            reservations.discard(normalized_reservation_id)
            if not reservations:
                self._daily_reservations.pop(key_id, None)
            return True

    def validate_image_request(self, identity: dict[str, object], count: int) -> None:
        if identity.get("role") != "user":
            return
        key_id = self._clean(identity.get("id"))
        with self._lock:
            found = self._find_item_locked(key_id)
            if found is None:
                raise ImageRequestLimitExceeded(DEFAULT_IMAGE_REQUEST_LIMIT)
            _, item = found
            limit = self._image_request_limit(item.get("image_request_limit"))
        if count > limit:
            raise ImageRequestLimitExceeded(limit)

    def authenticate(self, raw_key: str) -> dict[str, object] | None:
        candidate = self._clean(raw_key)
        if not candidate:
            return None
        candidate_hash = _hash_key(candidate)
        with self._lock:
            for index, item in enumerate(self._items):
                if not bool(item.get("enabled", True)):
                    continue
                stored_hash = self._clean(item.get("key_hash"))
                if not stored_hash or not hmac.compare_digest(stored_hash, candidate_hash):
                    continue
                next_item = dict(item)
                now = datetime.now(timezone.utc)
                reset = self._reset_daily_if_needed_locked(index, next_item)
                if reset:
                    next_item = dict(self._items[index])
                next_item["last_used_at"] = now.isoformat()
                self._items[index] = next_item
                item_id = self._clean(next_item.get("id"))
                last_flush_at = self._last_used_flush_at.get(item_id)
                if reset or last_flush_at is None or (now - last_flush_at).total_seconds() >= 60:
                    try:
                        self._save_item_locked(next_item)
                        self._last_used_flush_at[item_id] = now
                    except Exception:
                        pass
                return self._public_item(next_item)
        return None


auth_service = AuthService(config.get_storage_backend())
