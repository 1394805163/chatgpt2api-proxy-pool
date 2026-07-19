from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Column, String, Text, create_engine, Integer, and_, or_, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from services.storage.base import DEFAULT_LOG_RETENTION_DAYS, DEFAULT_MAX_LOG_ITEMS, StorageBackend

Base = declarative_base()

LOG_PRUNE_INTERVAL = 100


class AccountModel(Base):
    """账号数据模型"""
    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    access_token = Column(String(2048), unique=True, nullable=False, index=True)
    data = Column(Text, nullable=False)  # JSON 格式存储完整账号数据


class AuthKeyModel(Base):
    """鉴权密钥数据模型"""
    __tablename__ = "auth_keys"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key_id = Column(String(255), unique=True, nullable=False, index=True)
    data = Column(Text, nullable=False)


class LogModel(Base):
    __tablename__ = "logs"

    id = Column(String(64), primary_key=True)
    time = Column(String(64), nullable=False, index=True)
    type = Column(String(64), nullable=False, index=True)
    data = Column(Text, nullable=False)


class SettingsModel(Base):
    __tablename__ = "settings"

    key_id = Column(String(64), primary_key=True)
    data = Column(Text, nullable=False)


class DatabaseStorageBackend(StorageBackend):
    """数据库存储后端（支持 SQLite、PostgreSQL、MySQL 等）"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,  # 自动检测连接是否有效
            pool_recycle=3600,   # 1小时回收连接
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self._log_writes_since_prune = 0
        self.prune_logs()

    def load_accounts(self) -> list[dict[str, Any]]:
        """从数据库加载账号数据"""
        session = self.Session()
        try:
            accounts = []
            for row in session.query(AccountModel).all():
                try:
                    account_data = json.loads(row.data)
                    if isinstance(account_data, dict):
                        accounts.append(account_data)
                except json.JSONDecodeError:
                    continue
            return accounts
        finally:
            session.close()

    def save_accounts(self, accounts: list[dict[str, Any]]) -> None:
        """保存账号数据到数据库"""
        self._save_rows(AccountModel, accounts, "access_token")

    def load_auth_keys(self) -> list[dict[str, Any]]:
        """从数据库加载鉴权密钥数据"""
        return self._load_rows(AuthKeyModel)

    def save_auth_keys(self, auth_keys: list[dict[str, Any]]) -> None:
        """保存鉴权密钥数据到数据库"""
        self._save_rows(AuthKeyModel, auth_keys, "id", "key_id")

    def save_auth_key(self, auth_key: dict[str, Any]) -> bool:
        key_id = str(auth_key.get("id") or "").strip()
        if not key_id:
            return False
        session = self.Session()
        try:
            row = session.query(AuthKeyModel).filter(AuthKeyModel.key_id == key_id).one_or_none()
            payload = json.dumps(auth_key, ensure_ascii=False)
            if row is None:
                session.add(AuthKeyModel(key_id=key_id, data=payload))
            else:
                row.data = payload
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def load_settings(self) -> dict[str, Any]:
        session = self.Session()
        try:
            row = session.query(SettingsModel).filter(SettingsModel.key_id == "service").one_or_none()
            if row is None:
                return {}
            try:
                data = json.loads(row.data)
            except json.JSONDecodeError:
                return {}
            return data if isinstance(data, dict) else {}
        finally:
            session.close()

    def save_settings(self, settings: dict[str, Any]) -> bool:
        session = self.Session()
        try:
            session.merge(
                SettingsModel(
                    key_id="service",
                    data=json.dumps(settings, ensure_ascii=False, separators=(",", ":")),
                )
            )
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save_log(self, item: dict[str, Any]) -> bool:
        log_id = str(item.get("id") or "").strip()
        if not log_id:
            return False
        session = self.Session()
        try:
            session.merge(
                LogModel(
                    id=log_id,
                    time=str(item.get("time") or ""),
                    type=str(item.get("type") or ""),
                    data=json.dumps(item, ensure_ascii=False, separators=(",", ":")),
                )
            )
            session.commit()
            self._log_writes_since_prune += 1
            if self._log_writes_since_prune >= LOG_PRUNE_INTERVAL:
                self._log_writes_since_prune = 0
                self.prune_logs()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def load_logs(self, limit: int | None = None, type: str = "") -> list[dict[str, Any]]:
        session = self.Session()
        try:
            query = session.query(LogModel)
            if type:
                query = query.filter(LogModel.type == type)
            query = query.order_by(LogModel.time.desc())
            if limit is not None:
                query = query.limit(max(0, int(limit)))
            items: list[dict[str, Any]] = []
            for row in query.all():
                try:
                    item = json.loads(row.data)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    items.append(item)
            return items
        finally:
            session.close()

    def delete_logs(self, ids: list[str]) -> int:
        target_ids = [str(item or "").strip() for item in ids if str(item or "").strip()]
        if not target_ids:
            return 0
        session = self.Session()
        try:
            removed = session.query(LogModel).filter(LogModel.id.in_(target_ids)).delete(synchronize_session=False)
            session.commit()
            return int(removed or 0)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def prune_logs(
        self,
        *,
        retention_days: int = DEFAULT_LOG_RETENTION_DAYS,
        max_items: int = DEFAULT_MAX_LOG_ITEMS,
    ) -> int:
        session = self.Session()
        try:
            removed = 0
            if retention_days >= 0:
                cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat().replace("+00:00", "Z")
                removed += int(
                    session.query(LogModel)
                    .filter(LogModel.time < cutoff)
                    .delete(synchronize_session=False)
                    or 0
                )

            if max_items <= 0:
                removed += int(session.query(LogModel).delete(synchronize_session=False) or 0)
            else:
                boundary = (
                    session.query(LogModel.time, LogModel.id)
                    .order_by(LogModel.time.desc(), LogModel.id.desc())
                    .offset(max_items - 1)
                    .first()
                )
                if boundary is not None:
                    removed += int(
                        session.query(LogModel)
                        .filter(
                            or_(
                                LogModel.time < boundary.time,
                                and_(LogModel.time == boundary.time, LogModel.id < boundary.id),
                            )
                        )
                        .delete(synchronize_session=False)
                        or 0
                    )
            session.commit()
            return removed
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _load_rows(self, model: type[AccountModel] | type[AuthKeyModel]) -> list[dict[str, Any]]:
        session = self.Session()
        try:
            items = []
            for row in session.query(model).all():
                try:
                    item_data = json.loads(row.data)
                    if isinstance(item_data, dict):
                        items.append(item_data)
                except json.JSONDecodeError:
                    continue
            return items
        finally:
            session.close()

    def _save_rows(
        self,
        model: type[AccountModel] | type[AuthKeyModel],
        items: list[dict[str, Any]],
        source_key: str,
        target_key: str | None = None,
    ) -> None:
        session = self.Session()
        try:
            session.query(model).delete()
            for item in items:
                if not isinstance(item, dict):
                    continue
                key_value = str(item.get(source_key) or "").strip()
                if not key_value:
                    continue
                session.add(
                    model(
                        **{target_key or source_key: key_value},
                        data=json.dumps(item, ensure_ascii=False),
                    )
                )
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def health_check(self) -> dict[str, Any]:
        """健康检查"""
        try:
            session = self.Session()
            try:
                # 尝试执行简单查询
                session.execute(text("SELECT 1"))
                count = session.query(AccountModel).count()
                auth_key_count = session.query(AuthKeyModel).count()
                return {
                    "status": "healthy",
                    "backend": "database",
                    "database_url": self._mask_password(self.database_url),
                    "account_count": count,
                    "auth_key_count": auth_key_count,
                }
            finally:
                session.close()
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": "database",
                "error": str(e),
            }

    def get_backend_info(self) -> dict[str, Any]:
        """获取存储后端信息"""
        db_type = "unknown"
        if "sqlite" in self.database_url:
            db_type = "sqlite"
        elif "postgresql" in self.database_url or "postgres" in self.database_url:
            db_type = "postgresql"
        elif "mysql" in self.database_url:
            db_type = "mysql"
        
        return {
            "type": "database",
            "db_type": db_type,
            "description": f"数据库存储 ({db_type})",
            "database_url": self._mask_password(self.database_url),
        }

    @staticmethod
    def _mask_password(url: str) -> str:
        """隐藏数据库连接字符串中的密码"""
        if "://" not in url:
            return url
        try:
            protocol, rest = url.split("://", 1)
            if "@" in rest:
                credentials, host = rest.split("@", 1)
                if ":" in credentials:
                    username, _ = credentials.split(":", 1)
                    return f"{protocol}://{username}:****@{host}"
            return url
        except Exception:
            return url
