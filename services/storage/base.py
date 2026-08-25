from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

DEFAULT_LOG_RETENTION_DAYS = 10
DEFAULT_MAX_LOG_ITEMS = 50_000


class StorageBackend(ABC):
    """抽象存储后端基类"""

    @abstractmethod
    def load_accounts(self) -> list[dict[str, Any]]:
        """加载所有账号数据"""
        pass

    @abstractmethod
    def save_accounts(self, accounts: list[dict[str, Any]]) -> None:
        """保存所有账号数据"""
        pass

    @abstractmethod
    def load_auth_keys(self) -> list[dict[str, Any]]:
        """加载所有鉴权密钥数据"""
        pass

    @abstractmethod
    def save_auth_keys(self, auth_keys: list[dict[str, Any]]) -> None:
        """保存所有鉴权密钥数据"""
        pass

    def save_auth_key(self, auth_key: dict[str, Any]) -> bool:
        """Update one auth key when supported; file backends fall back to a full save."""
        return False

    def load_settings(self) -> dict[str, Any]:
        """Load service settings when the backend supports persistent settings."""
        return {}

    def save_settings(self, settings: dict[str, Any]) -> bool:
        """Persist service settings when supported by the backend."""
        return False

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """健康检查，返回存储后端状态"""
        pass

    @abstractmethod
    def get_backend_info(self) -> dict[str, Any]:
        """获取存储后端信息"""
        pass

    def save_log(self, item: dict[str, Any]) -> bool:
        return False

    def load_logs(self, limit: int | None = None, type: str = "") -> list[dict[str, Any]]:
        return []

    def load_logs_page(
        self,
        *,
        limit: int | None = None,
        type: str = "",
        before_time: str = "",
        before_id: str = "",
    ) -> list[dict[str, Any]]:
        """加载游标之前的一页日志。

        非数据库后端默认回退到旧接口；数据库后端可用索引直接截取，避免
        为管理页分页请求加载完整日志集合。
        """
        return self.load_logs(limit=limit, type=type)

    def delete_logs(self, ids: list[str]) -> int:
        return 0
