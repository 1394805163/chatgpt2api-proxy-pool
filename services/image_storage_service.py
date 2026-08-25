from __future__ import annotations

import hashlib
import heapq
import io
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from urllib.parse import quote, urlparse

from curl_cffi import requests
from fastapi import HTTPException
from PIL import Image

from services.config import DATA_DIR, config
from services.time_utils import utc_now_iso, utc_timestamp_iso

IMAGE_INDEX_FILE = DATA_DIR / "image_index.json"
IMAGE_INDEX_LOCK = RLock()
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


class ImageStorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class StoredImage:
    rel: str
    url: str
    storage: str
    size: int


def _clean(value: object) -> str:
    return str(value or "").strip()


def _now_iso() -> str:
    return utc_now_iso()


def _positive_int(value: object, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _safe_relative_path(path: str) -> str:
    value = str(path or "").strip().replace("\\", "/").lstrip("/")
    if not value:
        raise HTTPException(status_code=404, detail="image not found")
    parts = Path(value).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise HTTPException(status_code=404, detail="image not found")
    return Path(*parts).as_posix()


def _image_dimensions(payload: bytes) -> tuple[int, int] | None:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            return image.size
    except Exception:
        return None


def _image_dimensions_path(path: Path) -> tuple[int, int] | None:
    """只读取图片头部获取尺寸，不把整张图片复制进 Python 堆。"""
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def _is_image_rel(path: str) -> bool:
    try:
        safe_rel = _safe_relative_path(path)
    except HTTPException:
        return False
    return Path(safe_rel).suffix.lower() in IMAGE_EXTENSIONS


def _local_image_path(relative_path: str) -> Path:
    rel = _safe_relative_path(relative_path)
    root = config.images_dir.resolve()
    path = (root / rel).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="image not found") from exc
    return path


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json_object(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


class WebDAVClient:
    def __init__(self, settings: dict[str, object]):
        self.url = _clean(settings.get("webdav_url")).rstrip("/")
        self.username = _clean(settings.get("webdav_username"))
        self.password = _clean(settings.get("webdav_password"))
        self.root_path = _clean(settings.get("webdav_root_path")).strip("/")
        self.session = requests.Session()

    def _auth_kwargs(self) -> dict[str, object]:
        return {"auth": (self.username, self.password)} if self.username or self.password else {}

    def _request(self, method: str, url: str, **kwargs):
        response = self.session.request(method, url, timeout=30, **self._auth_kwargs(), **kwargs)
        if response.status_code >= 400 and not (method == "MKCOL" and response.status_code in {405}):
            raise ImageStorageError(f"WebDAV {method} failed: HTTP {response.status_code}")
        return response

    def remote_url(self, rel: str = "") -> str:
        parts = [part for part in [self.root_path, _safe_relative_path(rel) if rel else ""] if part]
        encoded = "/".join(quote(part, safe="") for item in parts for part in item.split("/") if part)
        return f"{self.url}/{encoded}" if encoded else self.url

    def ensure_dirs(self, rel: str) -> None:
        parts = [part for part in [self.root_path, Path(_safe_relative_path(rel)).parent.as_posix()] if part and part != "."]
        current = self.url
        for item in "/".join(parts).split("/"):
            if not item:
                continue
            current = f"{current}/{quote(item, safe='')}"
            response = self.session.request("MKCOL", current, timeout=30, **self._auth_kwargs())
            if response.status_code in {201, 405}:
                continue
            if response.status_code >= 400:
                raise ImageStorageError(f"WebDAV MKCOL failed: HTTP {response.status_code}")

    def put(self, rel: str, payload: bytes, content_type: str = "image/png") -> str:
        self.ensure_dirs(rel)
        url = self.remote_url(rel)
        self._request("PUT", url, data=payload, headers={"Content-Type": content_type})
        return url

    def get(self, rel: str) -> bytes:
        response = self._request("GET", self.remote_url(rel))
        return bytes(response.content)

    def delete(self, rel: str) -> bool:
        response = self.session.request("DELETE", self.remote_url(rel), timeout=30, **self._auth_kwargs())
        if response.status_code in {200, 202, 204, 404}:
            return response.status_code != 404
        raise ImageStorageError(f"WebDAV DELETE failed: HTTP {response.status_code}")

    def test(self) -> dict[str, object]:
        if not self.url:
            return {"ok": False, "status": 0, "error": "WebDAV URL is required"}
        if urlparse(self.url).scheme not in {"http", "https"}:
            return {"ok": False, "status": 0, "error": "invalid WebDAV URL"}
        test_rel = ".chatgpt2api_webdav_test.txt"
        try:
            self.put(test_rel, b"chatgpt2api webdav test\n", content_type="text/plain")
            self.delete(test_rel)
            return {"ok": True, "status": 200, "error": None}
        except ImageStorageError as exc:
            return {"ok": False, "status": 0, "error": str(exc)}
        except Exception as exc:
            return {"ok": False, "status": 0, "error": str(exc) or exc.__class__.__name__}
        finally:
            self.session.close()


class ImageStorageService:
    def __init__(self, index_file: Path = IMAGE_INDEX_FILE):
        self.index_file = index_file
        self._index_lock = IMAGE_INDEX_LOCK

    def settings(self) -> dict[str, object]:
        return config.get_image_storage_settings()

    def mode(self) -> str:
        return _clean(self.settings().get("mode")) or "local"

    def _load_index(self) -> dict[str, dict[str, object]]:
        raw = _read_json_object(self.index_file)
        items = raw.get("items")
        if not isinstance(items, dict):
            return {}
        return {str(key): value for key, value in items.items() if isinstance(value, dict)}

    def _load_clean_index(self) -> dict[str, dict[str, object]]:
        items = self._load_index()
        return {rel: item for rel, item in items.items() if _is_image_rel(rel)}

    def _save_index(self, items: dict[str, dict[str, object]]) -> None:
        _write_json_object(self.index_file, {"items": items})

    def _public_url(self, rel: str, base_url: str | None = None) -> str:
        settings = self.settings()
        public_base_url = _clean(settings.get("public_base_url"))
        if public_base_url:
            return f"{public_base_url.rstrip('/')}/{_safe_relative_path(rel)}"
        return f"{(base_url or config.base_url).rstrip('/')}/images/{_safe_relative_path(rel)}"

    def make_relative_path(self, image_data: bytes) -> str:
        file_hash = hashlib.md5(image_data).hexdigest()
        filename = f"{int(time.time())}_{file_hash}.png"
        relative_dir = Path(time.strftime("%Y"), time.strftime("%m"), time.strftime("%d"))
        return f"{relative_dir.as_posix()}/{filename}"

    @staticmethod
    def _expires_at(item: dict[str, object]) -> float | None:
        try:
            expires_at = float(item.get("expires_at") or 0)
        except (TypeError, ValueError):
            return None
        return expires_at if expires_at > 0 else None

    @staticmethod
    def _created_at(item: dict[str, object]) -> float | None:
        value = str(item.get("created_at") or "").strip()
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.timestamp()
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _remove_thumbnail_files(relative_path: str) -> None:
        rel = _safe_relative_path(relative_path)
        thumbnails_root = config.image_thumbnails_dir.resolve()
        for path in (thumbnails_root / f"{rel}.png", thumbnails_root / rel):
            try:
                path.resolve().relative_to(thumbnails_root)
            except ValueError:
                continue
            path.unlink(missing_ok=True)

    @staticmethod
    def _remove_image_tags(relative_path: str) -> None:
        try:
            from services.image_tags_service import remove_tags

            remove_tags(relative_path)
        except Exception:
            pass

    def _retention_seconds(self, value: int | None) -> int:
        if value is not None:
            try:
                if int(value) > 0:
                    return int(value)
            except (TypeError, ValueError):
                pass
        return _positive_int(getattr(config, "image_retention_days", 30), 30) * 86400

    def save(
        self,
        image_data: bytes,
        base_url: str | None = None,
        *,
        retention_seconds: int | None = None,
        owner_id: str = "",
    ) -> StoredImage:
        rel = self.make_relative_path(image_data)
        mode = self.mode()
        if mode not in {"local", "webdav", "both"}:
            mode = "local"
        stored_local = False
        stored_webdav = False
        remote_url = ""

        if mode in {"local", "both"}:
            path = _local_image_path(rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(image_data)
            stored_local = True

        if mode in {"webdav", "both"}:
            remote_url = WebDAVClient(self.settings()).put(rel, image_data)
            stored_webdav = True

        dimensions = _image_dimensions(image_data)
        now = time.time()
        item = {
            "rel": rel,
            "path": rel,
            "name": Path(rel).name,
            "date": "-".join(rel.split("/")[:3]),
            "size": len(image_data),
            "created_at": utc_timestamp_iso(now),
            "expires_at": int(now) + self._retention_seconds(retention_seconds),
            "storage": "both" if stored_local and stored_webdav else ("webdav" if stored_webdav else "local"),
            "local": stored_local,
            "webdav": stored_webdav,
            "remote_url": remote_url,
        }
        if _clean(owner_id):
            item["owner_id"] = _clean(owner_id)
        if dimensions:
            item["width"], item["height"] = dimensions
        with self._index_lock:
            items = self._load_clean_index()
            items[rel] = item
            self._save_index(items)
        return StoredImage(rel=rel, url=self._public_url(rel, base_url), storage=str(item["storage"]), size=len(image_data))

    def is_expired(self, rel: str, now: float | None = None) -> bool:
        safe_rel = _safe_relative_path(rel)
        current = time.time() if now is None else float(now)
        with self._index_lock:
            item = self._load_clean_index().get(safe_rel)
        if item is None:
            path = _local_image_path(safe_rel)
            return path.is_file() and path.stat().st_mtime + self._retention_seconds(None) <= current
        expires_at = self._expires_at(item)
        if expires_at is not None:
            return expires_at <= current
        path = _local_image_path(safe_rel)
        created_at = self._created_at(item) or (path.stat().st_mtime if path.is_file() else None)
        return created_at is not None and created_at + self._retention_seconds(None) <= current

    def cache_max_age(self, rel: str, now: float | None = None) -> int:
        safe_rel = _safe_relative_path(rel)
        current = time.time() if now is None else float(now)
        with self._index_lock:
            item = self._load_clean_index().get(safe_rel)
        if item is not None and (expires_at := self._expires_at(item)) is not None:
            return max(0, int(expires_at - current))
        if item is not None:
            created_at = self._created_at(item)
            if created_at is not None:
                return max(0, int(created_at + self._retention_seconds(None) - current))
        return self._retention_seconds(None)

    def cleanup_expired(self, now: float | None = None) -> int:
        current = time.time() if now is None else float(now)
        removed = 0
        changed = False
        with self._index_lock:
            items = self._load_clean_index()
            for rel, item in list(items.items()):
                expires_at = self._expires_at(item)
                if expires_at is None:
                    path = _local_image_path(rel)
                    created_at = self._created_at(item) or (path.stat().st_mtime if path.is_file() else None)
                    if created_at is not None:
                        expires_at = created_at + self._retention_seconds(None)
                if expires_at is None or expires_at > current:
                    continue

                local_path = _local_image_path(rel)
                local_exists = local_path.is_file()
                if local_exists:
                    local_path.unlink()
                    local_exists = False

                webdav_exists = bool(item.get("webdav"))
                if webdav_exists:
                    try:
                        webdav_exists = not WebDAVClient(self.settings()).delete(rel)
                    except Exception:
                        webdav_exists = True

                self._remove_thumbnail_files(rel)
                self._remove_image_tags(rel)
                if local_exists or webdav_exists:
                    item = {
                        **item,
                        "local": local_exists,
                        "webdav": webdav_exists,
                        "storage": "both" if local_exists and webdav_exists else ("webdav" if webdav_exists else "local"),
                    }
                    items[rel] = item
                else:
                    items.pop(rel, None)
                    removed += 1
                changed = True

            fallback_cutoff = current - self._retention_seconds(None)
            for path in config.images_dir.rglob("*"):
                if not path.is_file() or not _is_image_rel(path.name):
                    continue
                rel = path.relative_to(config.images_dir).as_posix()
                if rel in items or path.stat().st_mtime > fallback_cutoff:
                    continue
                path.unlink()
                self._remove_thumbnail_files(rel)
                self._remove_image_tags(rel)
                removed += 1
                changed = True
            if changed:
                self._save_index(items)
        return removed

    def get_bytes(self, rel: str) -> bytes:
        safe_rel = _safe_relative_path(rel)
        if not _is_image_rel(safe_rel):
            raise HTTPException(status_code=404, detail="image not found")
        path = _local_image_path(safe_rel)
        if path.is_file():
            return path.read_bytes()
        item = self._load_clean_index().get(safe_rel, {})
        if item.get("webdav"):
            return WebDAVClient(self.settings()).get(safe_rel)
        raise HTTPException(status_code=404, detail="image not found")

    def exists(self, rel: str) -> bool:
        safe_rel = _safe_relative_path(rel)
        if not _is_image_rel(safe_rel):
            return False
        if _local_image_path(safe_rel).is_file():
            return True
        item = self._load_clean_index().get(safe_rel, {})
        return bool(item.get("webdav"))

    def has_local(self, rel: str) -> bool:
        safe_rel = _safe_relative_path(rel)
        return _is_image_rel(safe_rel) and _local_image_path(safe_rel).is_file()

    @staticmethod
    def _sort_key(item: dict[str, object]) -> tuple[str, str]:
        return str(item.get("created_at") or ""), str(item.get("rel") or "")

    @staticmethod
    def _cursor_value(cursor: str) -> tuple[str, str] | None:
        value = str(cursor or "").strip()
        if not value:
            return None
        try:
            created_at, rel = value.split("|", 1)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid image cursor") from exc
        if not created_at or not rel:
            raise HTTPException(status_code=400, detail="invalid image cursor")
        return created_at, rel

    @classmethod
    def _is_before_cursor(cls, item: dict[str, object], cursor: tuple[str, str] | None) -> bool:
        if cursor is None:
            return True
        created_at, rel = cls._sort_key(item)
        return created_at < cursor[0] or (created_at == cursor[0] and rel < cursor[1])

    @staticmethod
    def _encode_cursor(item: dict[str, object]) -> str:
        return f"{item.get('created_at') or ''}|{item.get('rel') or ''}"

    def _prepare_index(self) -> dict[str, dict[str, object]]:
        """同步索引元数据，始终只读取图片头部而不是完整图片。"""
        indexed = self._load_clean_index()
        root = config.images_dir
        changed = False
        for path in root.rglob("*"):
            if not path.is_file() or not _is_image_rel(path.name):
                continue
            rel = path.relative_to(root).as_posix()
            if rel in indexed:
                continue
            dimensions = _image_dimensions_path(path)
            stat = path.stat()
            indexed[rel] = {
                "rel": rel,
                "path": rel,
                "name": path.name,
                "date": "-".join(rel.split("/")[:3]) if len(rel.split("/")) >= 4 else datetime.fromtimestamp(stat.st_mtime, tz=UTC).strftime("%Y-%m-%d"),
                "size": stat.st_size,
                "created_at": utc_timestamp_iso(stat.st_mtime),
                "storage": "local",
                "local": True,
                "webdav": False,
                **({"width": dimensions[0], "height": dimensions[1]} if dimensions else {}),
            }
            changed = True

        for rel, item in list(indexed.items()):
            if not _is_image_rel(rel):
                indexed.pop(rel, None)
                changed = True
                continue
            local = _local_image_path(rel).is_file()
            webdav = bool(item.get("webdav"))
            if not local and not webdav:
                indexed.pop(rel, None)
                changed = True
                continue
            storage = "both" if local and webdav else ("webdav" if webdav else "local")
            if item.get("local") != local or item.get("storage") != storage:
                indexed[rel] = {**item, "local": local, "storage": storage}
                changed = True
        if changed:
            self._save_index(indexed)
        return indexed

    def list_items(self, base_url: str, start_date: str = "", end_date: str = "") -> list[dict[str, object]]:
        with self._index_lock:
            indexed = self._prepare_index()
            items: list[dict[str, object]] = []
            for rel, item in indexed.items():
                day = str(item.get("date") or "")
                if start_date and day < start_date:
                    continue
                if end_date and day > end_date:
                    continue
                items.append({
                    **item,
                    "rel": rel,
                    "path": rel,
                    "url": self._public_url(rel, base_url),
                })
        items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return items

    def list_items_page(
        self,
        base_url: str,
        *,
        start_date: str = "",
        end_date: str = "",
        limit: int = 50,
        cursor: str = "",
    ) -> dict[str, object]:
        page_limit = max(1, min(int(limit), 100))
        decoded = self._cursor_value(cursor)
        with self._index_lock:
            indexed = self._prepare_index()
            heap: list[tuple[tuple[str, str], dict[str, object]]] = []
            for rel, item in indexed.items():
                day = str(item.get("date") or "")
                if start_date and day < start_date:
                    continue
                if end_date and day > end_date:
                    continue
                candidate = {
                    **item,
                    "rel": rel,
                    "path": rel,
                    "url": self._public_url(rel, base_url),
                }
                if not self._is_before_cursor(candidate, decoded):
                    continue
                key = self._sort_key(candidate)
                heapq.heappush(heap, (key, candidate))
                if len(heap) > page_limit + 1:
                    heapq.heappop(heap)
            items = [item for _, item in sorted(heap, key=lambda entry: entry[0], reverse=True)]
        has_more = len(items) > page_limit
        items = items[:page_limit]
        return {
            "items": items,
            "next_cursor": self._encode_cursor(items[-1]) if has_more and items else None,
        }

    def matching_paths(self, *, start_date: str = "", end_date: str = "", maximum: int | None = None) -> list[str]:
        """按日期收集有限数量路径，用于有硬上限的批量删除。"""
        with self._index_lock:
            indexed = self._prepare_index()
            paths: list[str] = []
            for rel, item in indexed.items():
                day = str(item.get("date") or "")
                if start_date and day < start_date:
                    continue
                if end_date and day > end_date:
                    continue
                paths.append(rel)
                if maximum is not None and len(paths) >= maximum:
                    break
            return paths

    def delete_many(self, rels: list[str]) -> int:
        """一次加载、一次保存索引，避免逐张删除反复重写 image_index.json。"""
        safe_rels: list[str] = []
        for rel in rels:
            try:
                safe = _safe_relative_path(rel)
            except HTTPException:
                continue
            if _is_image_rel(safe) and safe not in safe_rels:
                safe_rels.append(safe)
        if not safe_rels:
            return 0
        removed = 0
        with self._index_lock:
            items = self._load_clean_index()
            webdav_client: WebDAVClient | None = None
            changed = False
            for safe_rel in safe_rels:
                path = _local_image_path(safe_rel)
                removed_item = False
                if path.is_file():
                    path.unlink()
                    removed_item = True
                item = items.get(safe_rel, {})
                if item.get("webdav"):
                    try:
                        if webdav_client is None:
                            webdav_client = WebDAVClient(self.settings())
                        removed_item = webdav_client.delete(safe_rel) or removed_item
                    except ImageStorageError:
                        if not removed_item:
                            raise
                if safe_rel in items:
                    items.pop(safe_rel, None)
                    changed = True
                if removed_item:
                    removed += 1
            if webdav_client is not None:
                webdav_client.session.close()
            if changed:
                self._save_index(items)
        return removed

    def delete(self, rel: str) -> bool:
        safe_rel = _safe_relative_path(rel)
        removed = False
        path = _local_image_path(safe_rel)
        if path.is_file():
            path.unlink()
            removed = True
        with self._index_lock:
            items = self._load_clean_index()
            item = items.get(safe_rel, {})
            if item.get("webdav"):
                try:
                    removed = WebDAVClient(self.settings()).delete(safe_rel) or removed
                except ImageStorageError:
                    if not removed:
                        raise
            if safe_rel in items:
                items.pop(safe_rel, None)
                self._save_index(items)
        return removed

    def sync_all(self) -> dict[str, int]:
        settings = self.settings()
        if self.mode() not in {"webdav", "both"}:
            raise ImageStorageError("WebDAV 图片存储未启用")
        uploaded = 0
        skipped = 0
        failed = 0
        with self._index_lock:
            items = self._load_clean_index()
            client = WebDAVClient(settings)
            for path in sorted(config.images_dir.rglob("*")):
                if not path.is_file() or not _is_image_rel(path.name):
                    continue
                rel = path.relative_to(config.images_dir).as_posix()
                item = items.get(rel, {})
                if item.get("webdav"):
                    skipped += 1
                    continue
                try:
                    payload = path.read_bytes()
                    remote_url = client.put(rel, payload)
                    dimensions = _image_dimensions(payload)
                    items[rel] = {
                        **item,
                        "rel": rel,
                        "path": rel,
                        "name": path.name,
                        "date": "-".join(rel.split("/")[:3]) if len(rel.split("/")) >= 4 else datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).strftime("%Y-%m-%d"),
                        "size": len(payload),
                        "created_at": str(item.get("created_at") or utc_timestamp_iso(path.stat().st_mtime)),
                        "storage": "both",
                        "local": True,
                        "webdav": True,
                        "remote_url": remote_url,
                        **({"width": dimensions[0], "height": dimensions[1]} if dimensions else {}),
                    }
                    uploaded += 1
                except Exception:
                    failed += 1
            self._save_index(items)
        return {"uploaded": uploaded, "skipped": skipped, "failed": failed}

    def test_webdav(self) -> dict[str, object]:
        return WebDAVClient(self.settings()).test()


image_storage_service = ImageStorageService()
