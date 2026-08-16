"""Shared SQLite connection and lock helpers for LivingMemory."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import aiosqlite

_SQLITE_BUSY_TIMEOUT_MS = 30_000
_SQLITE_SETUP_RETRIES = 4
_SQLITE_SETUP_RETRY_BASE_SECONDS = 0.05
_SQLITE_SETUP_RETRY_MAX_SECONDS = 0.8
_P = ParamSpec("_P")
_R = TypeVar("_R")


class ReentrantAsyncLock:
    """Task-reentrant lock used when one operation spans several stores."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[Any] | None = None
        self._depth = 0

    async def acquire(self) -> None:
        task = asyncio.current_task()
        if task is self._owner:
            self._depth += 1
            return
        await self._lock.acquire()
        self._owner = task
        self._depth = 1

    def release(self) -> None:
        task = asyncio.current_task()
        if task is not self._owner or self._depth <= 0:
            raise RuntimeError("ReentrantAsyncLock released by a non-owner task")
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
            self._lock.release()

    async def __aenter__(self) -> ReentrantAsyncLock:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


# One lock per database path keeps plugin-owned SQLite writers serialized while
# still allowing unrelated databases, such as conversations.db, to progress.
_locks: dict[str, ReentrantAsyncLock] = {}


def get_sqlite_lock(db_path: str | Path) -> ReentrantAsyncLock:
    """Return the process-local coordination lock for a database file."""
    key = str(Path(db_path).resolve()).casefold()
    lock = _locks.get(key)
    if lock is None:
        lock = ReentrantAsyncLock()
        _locks[key] = lock
    return lock


def _is_transient_sqlite_lock(error: BaseException) -> bool:
    """Return whether SQLite rejected only the connection setup phase."""
    return isinstance(error, sqlite3.OperationalError) and any(
        marker in str(error).lower()
        for marker in ("database is locked", "database table is locked")
    )


async def open_sqlite_connection(
    db_path: str | Path,
    *,
    journal_mode: str | None = None,
    foreign_keys: bool = False,
) -> aiosqlite.Connection:
    """Open and configure SQLite, retrying only transient setup contention.

    The caller owns the returned connection. Retries stop before any caller SQL
    runs, so a started business transaction is never replayed here.
    """
    lock = get_sqlite_lock(db_path)
    async with lock:
        last_error: BaseException | None = None
        for attempt in range(_SQLITE_SETUP_RETRIES + 1):
            db: aiosqlite.Connection | None = None
            try:
                db = await aiosqlite.connect(
                    str(db_path), timeout=_SQLITE_BUSY_TIMEOUT_MS / 1000
                )
                await db.execute(f"PRAGMA busy_timeout = {_SQLITE_BUSY_TIMEOUT_MS}")
                await db.execute("PRAGMA synchronous = NORMAL")
                if journal_mode is not None:
                    await db.execute(f"PRAGMA journal_mode = {journal_mode}")
                if foreign_keys:
                    await db.execute("PRAGMA foreign_keys = ON")
                return db
            except Exception as error:
                last_error = error
                if db is not None:
                    await db.close()
                if (
                    not _is_transient_sqlite_lock(error)
                    or attempt >= _SQLITE_SETUP_RETRIES
                ):
                    raise
                delay = min(
                    _SQLITE_SETUP_RETRY_BASE_SECONDS * (2**attempt),
                    _SQLITE_SETUP_RETRY_MAX_SECONDS,
                )
                await asyncio.sleep(delay)

    raise RuntimeError("SQLite connection setup exhausted retries") from last_error


@asynccontextmanager
async def sqlite_connection(db_path: str | Path):
    """Open a configured SQLite connection while holding the path lock."""
    lock = get_sqlite_lock(db_path)
    async with lock:
        db = await open_sqlite_connection(db_path)
        try:
            yield db
        finally:
            await db.close()


def with_sqlite_lock(
    path_getter: Callable[..., str | Path],
) -> Callable[[Callable[_P, Awaitable[_R]]], Callable[_P, Awaitable[_R]]]:
    """Serialize an async operation using the database path lock."""

    def decorator(
        function: Callable[_P, Awaitable[_R]],
    ) -> Callable[_P, Awaitable[_R]]:
        @wraps(function)
        async def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            # Path getters describe the receiver, not the business arguments.
            db_path = path_getter(*args[:1])
            async with get_sqlite_lock(db_path):
                return await function(*args, **kwargs)

        return wrapped

    return decorator


__all__ = [
    "get_sqlite_lock",
    "open_sqlite_connection",
    "sqlite_connection",
    "with_sqlite_lock",
]
