from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from services.image_storage_service import ImageStorageService, StoredImage
from services.image_service import delete_to_target, get_image_response
from services.protocol.conversation import format_image_result


def png_bytes() -> bytes:
    path = Path(tempfile.gettempdir()) / "chatgpt2api-test-image.png"
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(path, format="PNG")
    return path.read_bytes()


class FakeWebDAVClient:
    uploaded: dict[str, bytes] = {}
    deleted: list[str] = []

    def __init__(self, _settings):
        pass

    def put(self, rel: str, payload: bytes) -> str:
        self.uploaded[rel] = payload
        return f"https://dav.example.test/{rel}"

    def get(self, rel: str) -> bytes:
        return self.uploaded[rel]

    def delete(self, rel: str) -> bool:
        self.deleted.append(rel)
        self.uploaded.pop(rel, None)
        return True

    def test(self) -> dict[str, object]:
        self.put(".chatgpt2api_webdav_test.txt", b"chatgpt2api webdav test\n")
        self.delete(".chatgpt2api_webdav_test.txt")
        return {"ok": True, "status": 200, "error": None}


class ImageStorageServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_dir = Path(self.tmp.name)
        self.images_dir = self.data_dir / "images"
        self.settings = {
            "enabled": False,
            "mode": "local",
            "webdav_url": "",
            "webdav_username": "",
            "webdav_password": "",
            "webdav_root_path": "chatgpt2api/images",
            "public_base_url": "",
        }
        self.config_patcher = mock.patch("services.image_storage_service.config")
        self.mock_config = self.config_patcher.start()
        self.addCleanup(self.config_patcher.stop)
        self.mock_config.images_dir = self.images_dir
        self.mock_config.image_thumbnails_dir = self.data_dir / "image_thumbnails"
        self.mock_config.image_retention_days = 30
        self.mock_config.base_url = "http://app.test"
        self.mock_config.cleanup_old_images.return_value = 0
        self.mock_config.get_image_storage_settings.side_effect = lambda: dict(self.settings)
        FakeWebDAVClient.uploaded = {}
        FakeWebDAVClient.deleted = []

    def service(self) -> ImageStorageService:
        return ImageStorageService(self.data_dir / "image_index.json")

    def test_local_mode_saves_to_local_directory(self):
        stored = self.service().save(png_bytes(), "http://app.test")

        self.assertEqual(stored.storage, "local")
        self.assertTrue((self.images_dir / stored.rel).is_file())
        self.assertEqual(stored.url, f"http://app.test/images/{stored.rel}")

    def test_expired_key_image_removes_file_and_index(self):
        service = self.service()
        stored = service.save(
            png_bytes(),
            "http://app.test",
            retention_seconds=60,
            owner_id="user-key-1",
        )
        index = service._load_index()
        expires_at = int(index[stored.rel]["expires_at"])

        self.assertEqual(index[stored.rel]["owner_id"], "user-key-1")
        self.assertEqual(service.cleanup_expired(now=expires_at - 1), 0)
        self.assertTrue((self.images_dir / stored.rel).is_file())
        self.assertEqual(service.cleanup_expired(now=expires_at), 1)
        self.assertFalse((self.images_dir / stored.rel).exists())
        self.assertNotIn(stored.rel, service._load_index())

    def test_expired_key_image_removes_thumbnail(self):
        service = self.service()
        stored = service.save(png_bytes(), "http://app.test", retention_seconds=60)
        thumbnail = self.mock_config.image_thumbnails_dir / f"{stored.rel}.png"
        thumbnail.parent.mkdir(parents=True, exist_ok=True)
        thumbnail.write_bytes(png_bytes())
        expires_at = int(service._load_index()[stored.rel]["expires_at"])

        self.assertEqual(service.cleanup_expired(now=expires_at), 1)
        self.assertFalse(thumbnail.exists())

    def test_legacy_webdav_only_image_uses_created_at_for_global_expiry(self):
        service = self.service()
        rel = "2020/01/01/legacy.png"
        service._save_index({
            rel: {
                "rel": rel,
                "created_at": "2020-01-01T00:00:00+00:00",
                "storage": "webdav",
                "local": False,
                "webdav": True,
            }
        })
        FakeWebDAVClient.uploaded[rel] = png_bytes()

        with mock.patch("services.image_storage_service.WebDAVClient", FakeWebDAVClient):
            self.assertTrue(service.is_expired(rel))
            self.assertEqual(service.cleanup_expired(), 1)

        self.assertNotIn(rel, FakeWebDAVClient.uploaded)
        self.assertNotIn(rel, service._load_index())

    def test_cache_age_does_not_outlive_image_expiry(self):
        service = self.service()
        stored = service.save(png_bytes(), "http://app.test", retention_seconds=60)
        expires_at = int(service._load_index()[stored.rel]["expires_at"])

        self.assertEqual(service.cache_max_age(stored.rel, now=expires_at - 15), 15)
        self.assertEqual(service.cache_max_age(stored.rel, now=expires_at), 0)

    def test_image_result_passes_owner_and_retention_to_storage(self):
        stored = StoredImage("image.png", "http://app.test/images/image.png", "local", 10)
        with mock.patch("services.protocol.conversation.image_storage_service.save", return_value=stored) as save:
            result = format_image_result(
                [{"image_bytes": png_bytes()}],
                "prompt",
                "url",
                "http://app.test",
                retention_seconds=120,
                owner_id="key-1",
            )

        save.assert_called_once()
        self.assertEqual(save.call_args.kwargs["retention_seconds"], 120)
        self.assertEqual(save.call_args.kwargs["owner_id"], "key-1")
        self.assertEqual(result["data"][0]["url"], stored.url)

    def test_webdav_mode_uploads_without_local_file(self):
        self.settings.update({
            "enabled": True,
            "mode": "webdav",
            "webdav_url": "https://dav.example.test",
            "webdav_password": "secret",
        })
        with mock.patch("services.image_storage_service.WebDAVClient", FakeWebDAVClient):
            stored = self.service().save(png_bytes(), "http://app.test")
            payload = self.service().get_bytes(stored.rel)

        self.assertEqual(stored.storage, "webdav")
        self.assertFalse((self.images_dir / stored.rel).exists())
        self.assertIn(stored.rel, FakeWebDAVClient.uploaded)
        self.assertEqual(payload, FakeWebDAVClient.uploaded[stored.rel])

    def test_list_items_ignores_non_image_files(self):
        image = png_bytes()
        image_path = self.images_dir / "2026" / "05" / "07" / "sample.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image)
        (self.images_dir / ".DS_Store").write_text("not an image", encoding="utf-8")
        (self.images_dir / "2026" / ".DS_Store").write_text("not an image", encoding="utf-8")

        items = self.service().list_items("http://app.test")

        self.assertEqual([item["rel"] for item in items], ["2026/05/07/sample.png"])
        self.assertEqual(items[0]["storage"], "local")

    def test_both_mode_saves_to_local_and_webdav(self):
        self.settings.update({
            "enabled": True,
            "mode": "both",
            "webdav_url": "https://dav.example.test",
            "webdav_password": "secret",
            "public_base_url": "https://cdn.example.test/images",
        })
        with mock.patch("services.image_storage_service.WebDAVClient", FakeWebDAVClient):
            stored = self.service().save(png_bytes(), "http://app.test")

        self.assertEqual(stored.storage, "both")
        self.assertTrue((self.images_dir / stored.rel).is_file())
        self.assertIn(stored.rel, FakeWebDAVClient.uploaded)
        self.assertEqual(stored.url, f"https://cdn.example.test/images/{stored.rel}")

    def test_test_webdav_writes_and_deletes_probe_file(self):
        self.settings.update({
            "enabled": True,
            "mode": "webdav",
            "webdav_url": "https://dav.example.test",
            "webdav_password": "secret",
        })
        with mock.patch("services.image_storage_service.WebDAVClient", FakeWebDAVClient):
            result = self.service().test_webdav()

        self.assertTrue(result["ok"])
        self.assertIn(".chatgpt2api_webdav_test.txt", FakeWebDAVClient.deleted)

    def test_image_response_cache_ttl_uses_remaining_image_lifetime(self):
        image_path = self.images_dir / "cached.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(png_bytes())
        with mock.patch("services.image_service.image_storage_service.has_local", return_value=True), mock.patch(
            "services.image_service.image_storage_service.is_expired", return_value=False
        ), mock.patch("services.image_service.image_storage_service.cache_max_age", return_value=60), mock.patch(
            "services.image_service._safe_image_path", return_value=image_path
        ):
            response = get_image_response("cached.png")

        self.assertEqual(response.headers["cache-control"], "public, max-age=60, immutable")

    def test_low_disk_cleanup_keeps_images_younger_than_two_hours(self):
        image_path = self.images_dir / "recent.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(png_bytes())
        with mock.patch("services.image_service.config") as image_config, mock.patch(
            "services.image_service.shutil.disk_usage",
            return_value=mock.Mock(total=1024, used=1024, free=0),
        ):
            image_config.images_dir = self.images_dir
            image_config.image_thumbnails_dir = self.data_dir / "thumbnails"
            result = delete_to_target(100)

        self.assertTrue(image_path.exists())
        self.assertEqual(result["removed"], 0)
        self.assertEqual(result["protected"], 1)

if __name__ == "__main__":
    unittest.main()
