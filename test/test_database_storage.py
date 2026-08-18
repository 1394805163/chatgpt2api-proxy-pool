import json

import pytest

from services.storage.database_storage import AccountModel, AuthKeyModel, DatabaseStorageBackend


def _account_rows(backend: DatabaseStorageBackend) -> dict[str, AccountModel]:
    session = backend.Session()
    try:
        return {row.access_token: row for row in session.query(AccountModel).all()}
    finally:
        session.close()


def test_save_accounts_updates_incrementally_without_recreating_rows(tmp_path):
    backend = DatabaseStorageBackend(f"sqlite:///{tmp_path / 'accounts.db'}")
    backend.save_accounts([
        {"access_token": "token-a", "name": "A"},
        {"access_token": "token-b", "name": "B"},
    ])
    before = _account_rows(backend)

    backend.save_accounts([
        {"access_token": "token-b", "name": "B"},
        {"access_token": "token-a", "name": "A updated"},
        {"access_token": "token-c", "name": "C"},
    ])

    after = _account_rows(backend)
    assert set(after) == {"token-a", "token-b", "token-c"}
    assert after["token-a"].id == before["token-a"].id
    assert after["token-b"].id == before["token-b"].id
    assert after["token-b"].data == before["token-b"].data
    assert json.loads(after["token-a"].data)["name"] == "A updated"


def test_save_accounts_removes_rows_missing_from_snapshot(tmp_path):
    backend = DatabaseStorageBackend(f"sqlite:///{tmp_path / 'accounts.db'}")
    backend.save_accounts([
        {"access_token": "token-a"},
        {"access_token": "token-b"},
    ])

    backend.save_accounts([{ "access_token": "token-b" }])

    assert set(_account_rows(backend)) == {"token-b"}


def test_save_accounts_rejects_duplicate_tokens_without_partial_write(tmp_path):
    backend = DatabaseStorageBackend(f"sqlite:///{tmp_path / 'accounts.db'}")
    original = {"access_token": "token-a", "name": "A"}
    backend.save_accounts([original])

    with pytest.raises(ValueError, match="Duplicate access_token"):
        backend.save_accounts([
            {"access_token": "token-a", "name": "first"},
            {"access_token": "token-a", "name": "second"},
        ])

    assert backend.load_accounts() == [original]


def test_save_auth_keys_updates_by_public_key_id(tmp_path):
    backend = DatabaseStorageBackend(f"sqlite:///{tmp_path / 'auth-keys.db'}")
    backend.save_auth_keys([
        {"id": "key-a", "name": "A"},
        {"id": "key-b", "name": "B"},
    ])
    session = backend.Session()
    try:
        before = {row.key_id: row.id for row in session.query(AuthKeyModel).all()}
    finally:
        session.close()

    backend.save_auth_keys([
        {"id": "key-b", "name": "B"},
        {"id": "key-a", "name": "A updated"},
    ])

    session = backend.Session()
    try:
        rows = {row.key_id: row for row in session.query(AuthKeyModel).all()}
    finally:
        session.close()
    assert rows["key-a"].id == before["key-a"]
    assert json.loads(rows["key-a"].data)["name"] == "A updated"
