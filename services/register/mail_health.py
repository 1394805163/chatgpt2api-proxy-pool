from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class MailHealthTracker:
    def __init__(
        self,
        path: Path,
        *,
        min_risk_attempts: int = 5,
        min_success_rate: float = 0.2,
        cooldown_seconds: int = 24 * 60 * 60,
    ) -> None:
        self.path = path
        self.min_risk_attempts = max(1, int(min_risk_attempts))
        self.min_success_rate = min(1.0, max(0.0, float(min_success_rate)))
        self.cooldown_seconds = max(60, int(cooldown_seconds))
        self._lock = threading.RLock()

    @staticmethod
    def _domain(address: object) -> str:
        value = str(address or "").strip().lower()
        _, separator, domain = value.rpartition("@")
        return domain.strip(".") if separator else ""

    @staticmethod
    def _error_category(error: Exception | str | None) -> str:
        text = str(error or "").strip().lower()
        if not text:
            return ""
        if "registration_disallowed" in text:
            return "registration_disallowed"
        if "sentinel" in text:
            return "sentinel"
        if "otp" in text or "验证码" in text:
            return "email_otp"
        if any(token in text for token in ("timeout", "timed out", "proxy", "network", "connection")):
            return "network"
        return "other"

    def _load_locked(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            str(key): dict(value)
            for key, value in data.items()
            if isinstance(value, dict)
        }

    def _save_locked(self, state: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _key(provider_ref: str, domain: str) -> str:
        return f"{provider_ref.strip().lower()}|{domain.strip().lower()}"

    def record(
        self,
        mailbox: dict[str, Any],
        *,
        success: bool,
        error: Exception | str | None = None,
    ) -> None:
        provider = str(mailbox.get("provider") or "unknown").strip() or "unknown"
        provider_ref = str(mailbox.get("provider_ref") or provider).strip() or provider
        domain = self._domain(mailbox.get("address"))
        if not domain:
            return
        now = time.time()
        category = "" if success else self._error_category(error)
        key = self._key(provider_ref, domain)
        with self._lock:
            state = self._load_locked()
            item = dict(state.get(key) or {})
            item.update(
                {
                    "provider": provider,
                    "provider_ref": provider_ref,
                    "domain": domain,
                    "updated_at": now,
                }
            )
            item["attempts"] = int(item.get("attempts") or 0) + 1
            if success:
                item["success"] = int(item.get("success") or 0) + 1
            else:
                item["fail"] = int(item.get("fail") or 0) + 1
                item["last_error_category"] = category
                if category == "registration_disallowed":
                    item["registration_disallowed"] = int(item.get("registration_disallowed") or 0) + 1

            success_count = int(item.get("success") or 0)
            registration_disallowed = int(item.get("registration_disallowed") or 0)
            risk_attempts = success_count + registration_disallowed
            risk_success_rate = success_count / risk_attempts if risk_attempts else 0.0
            if risk_attempts >= self.min_risk_attempts and risk_success_rate < self.min_success_rate:
                item["disabled_until"] = max(
                    float(item.get("disabled_until") or 0.0),
                    now + self.cooldown_seconds,
                )
            state[key] = item
            self._save_locked(state)

    def is_disabled(self, provider_ref: str, domain: str, *, now: float | None = None) -> bool:
        key = self._key(str(provider_ref or ""), str(domain or ""))
        if not key.strip("|"):
            return False
        with self._lock:
            item = self._load_locked().get(key) or {}
        return float(item.get("disabled_until") or 0.0) > (time.time() if now is None else now)

    def snapshot(self) -> list[dict[str, Any]]:
        now = time.time()
        with self._lock:
            state = self._load_locked()
        result: list[dict[str, Any]] = []
        for item in state.values():
            success_count = int(item.get("success") or 0)
            registration_disallowed = int(item.get("registration_disallowed") or 0)
            risk_attempts = success_count + registration_disallowed
            result.append(
                {
                    "provider": str(item.get("provider") or ""),
                    "provider_ref": str(item.get("provider_ref") or ""),
                    "domain": str(item.get("domain") or ""),
                    "attempts": int(item.get("attempts") or 0),
                    "success": success_count,
                    "fail": int(item.get("fail") or 0),
                    "registration_disallowed": registration_disallowed,
                    "risk_attempts": risk_attempts,
                    "risk_success_rate": round(success_count * 100 / risk_attempts, 1) if risk_attempts else 0.0,
                    "last_error_category": str(item.get("last_error_category") or ""),
                    "disabled": float(item.get("disabled_until") or 0.0) > now,
                    "disabled_until": float(item.get("disabled_until") or 0.0),
                    "updated_at": float(item.get("updated_at") or 0.0),
                }
            )
        result.sort(
            key=lambda item: (
                not bool(item["disabled"]),
                -int(item["risk_attempts"]),
                str(item["provider_ref"]),
                str(item["domain"]),
            )
        )
        return result
