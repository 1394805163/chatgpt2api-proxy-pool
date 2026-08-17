from __future__ import annotations

import json

from scripts import migrate_storage
from services.storage.factory import create_storage_backend


def test_migrate_data_preserves_accounts_and_auth_keys(tmp_path, monkeypatch):
    accounts = [
        {
            "id": "account-1",
            "email": "sample@example.test",
            "access_token": "sample-access-token",
        }
    ]
    auth_keys = [{"id": "key-1", "name": "sample", "key": "masked"}]
    (tmp_path / "accounts.json").write_text(
        json.dumps(accounts), encoding="utf-8"
    )
    (tmp_path / "auth_keys.json").write_text(
        json.dumps({"items": auth_keys}), encoding="utf-8"
    )

    monkeypatch.setattr(migrate_storage, "DATA_DIR", tmp_path)
    database_path = (tmp_path / "accounts.db").as_posix()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.delenv("STORAGE_BACKEND", raising=False)

    migrate_storage.migrate_data("json", "sqlite")

    monkeypatch.setenv("STORAGE_BACKEND", "sqlite")
    storage = create_storage_backend(tmp_path)
    assert storage.load_accounts() == accounts
    assert storage.load_auth_keys() == auth_keys
