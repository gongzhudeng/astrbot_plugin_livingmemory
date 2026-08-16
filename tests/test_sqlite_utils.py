"""Focused tests for shared SQLite connection and lock helpers."""

import sqlite3
from unittest.mock import AsyncMock

import astrbot_plugin_livingmemory.storage.sqlite_utils as sqlite_utils
import pytest
from astrbot_plugin_livingmemory.storage.sqlite_utils import (
    get_sqlite_lock,
    open_sqlite_connection,
    sqlite_connection,
    with_sqlite_lock,
)


@pytest.mark.asyncio
async def test_reentrant_lock_and_exception_release(tmp_path):
    lock = get_sqlite_lock(tmp_path / "reentrant.db")

    with pytest.raises(RuntimeError, match="expected"):
        async with lock:
            async with lock:
                raise RuntimeError("expected")

    async with lock:
        pass


@pytest.mark.asyncio
async def test_decorator_path_getter_ignores_business_keyword_arguments(tmp_path):
    class Service:
        def __init__(self):
            self.db_path = tmp_path / "decorator.db"

        @with_sqlite_lock(lambda self: self.db_path)
        async def update(self, *, value: int) -> int:
            return value

    assert await Service().update(value=7) == 7


@pytest.mark.asyncio
async def test_connection_setup_retries_transient_lock(monkeypatch, tmp_path):
    first = AsyncMock()
    first.execute.side_effect = sqlite3.OperationalError("database is locked")
    second = AsyncMock()
    connect = AsyncMock(side_effect=[first, second])
    sleep = AsyncMock()
    monkeypatch.setattr(sqlite_utils.aiosqlite, "connect", connect)
    monkeypatch.setattr(sqlite_utils.asyncio, "sleep", sleep)

    connection = await open_sqlite_connection(tmp_path / "retry.db")

    assert connection is second
    assert connect.await_count == 2
    first.close.assert_awaited_once()
    sleep.assert_awaited_once_with(sqlite_utils._SQLITE_SETUP_RETRY_BASE_SECONDS)


@pytest.mark.asyncio
async def test_connection_setup_does_not_retry_non_lock_error(monkeypatch, tmp_path):
    connection = AsyncMock()
    connection.execute.side_effect = sqlite3.OperationalError("disk I/O error")
    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr(sqlite_utils.aiosqlite, "connect", connect)

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        await open_sqlite_connection(tmp_path / "io-error.db")

    connect.assert_awaited_once()
    connection.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_business_sql_lock_is_not_replayed(monkeypatch, tmp_path):
    connection = AsyncMock()
    connection.execute.side_effect = [
        None,
        None,
        sqlite3.OperationalError("database is locked"),
    ]
    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr(sqlite_utils.aiosqlite, "connect", connect)

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        async with sqlite_connection(tmp_path / "business.db") as db:
            await db.execute("INSERT INTO documents(text) VALUES ('once')")

    connect.assert_awaited_once()
    connection.close.assert_awaited_once()
