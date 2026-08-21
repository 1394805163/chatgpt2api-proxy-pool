from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

from services.config import config


ADMIN_IMAGE_CONCURRENCY_LIMIT = 10
DEFAULT_USER_IMAGE_CONCURRENCY_LIMIT = 2
MAX_IMAGE_CONCURRENCY_LIMIT = 100


class ImageConcurrencyLimitExceeded(ValueError):
    def __init__(self, scope: str, limit: int):
        self.scope = scope
        self.limit = limit
        super().__init__(f"{scope} image concurrency limit of {limit} reached")


def normalize_image_concurrency_limit(value: object) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        normalized = DEFAULT_USER_IMAGE_CONCURRENCY_LIMIT
    return min(MAX_IMAGE_CONCURRENCY_LIMIT, max(1, normalized))


def image_owner_limit(identity: dict[str, object], global_limit: int) -> int:
    if str(identity.get("role") or "").strip().lower() == "admin":
        return min(ADMIN_IMAGE_CONCURRENCY_LIMIT, global_limit)
    return min(
        normalize_image_concurrency_limit(identity.get("image_concurrency_limit")),
        global_limit,
    )


@dataclass
class ImageConcurrencyLease:
    _release_callback: Callable[[], None]
    _released: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._release_callback()


class ImageConcurrencyGate:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_total = 0
        self._active_by_owner: dict[str, int] = {}

    @staticmethod
    def _owner_id(identity: dict[str, object]) -> str:
        role = str(identity.get("role") or "user").strip().lower()
        subject = str(identity.get("id") or identity.get("name") or role).strip()
        return f"{role}:{subject}"

    def try_acquire(
        self,
        identity: dict[str, object],
        *,
        global_limit: int | None = None,
        owner_limit: int | None = None,
        units: int = 1,
    ) -> ImageConcurrencyLease:
        normalized_global_limit = max(1, int(global_limit or config.image_global_concurrency))
        normalized_owner_limit = max(
            1,
            int(owner_limit or image_owner_limit(identity, normalized_global_limit)),
        )
        normalized_units = max(1, int(units))
        owner = self._owner_id(identity)
        if normalized_units > normalized_owner_limit:
            raise ImageConcurrencyLimitExceeded("owner", normalized_owner_limit)
        if normalized_units > normalized_global_limit:
            raise ImageConcurrencyLimitExceeded("request", normalized_global_limit)
        with self._lock:
            if self._active_total + normalized_units > normalized_global_limit:
                raise ImageConcurrencyLimitExceeded("global", normalized_global_limit)
            owner_active = self._active_by_owner.get(owner, 0)
            if owner_active + normalized_units > normalized_owner_limit:
                raise ImageConcurrencyLimitExceeded("owner", normalized_owner_limit)
            self._active_total += normalized_units
            self._active_by_owner[owner] = owner_active + normalized_units

        def release() -> None:
            with self._lock:
                self._active_total = max(0, self._active_total - normalized_units)
                remaining = self._active_by_owner.get(owner, 0) - normalized_units
                if remaining > 0:
                    self._active_by_owner[owner] = remaining
                else:
                    self._active_by_owner.pop(owner, None)

        return ImageConcurrencyLease(release)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "active_total": self._active_total,
                "active_by_owner": dict(self._active_by_owner),
            }


image_concurrency_gate = ImageConcurrencyGate()
