"""get_recent_memories 导出接口测试（会话/时间过滤）。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrbot_plugin_livingmemory.core.event_handler import EventHandler
from astrbot_plugin_livingmemory.storage.sqlite_utils import sqlite_connection


async def _make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "memories.db"
    now = time.time()
    rows = [
        # (session_id, text, create_time, status)
        ("private:umo", "上午聊了天台拍照的约定", now - 3600, "active"),
        ("private:umo", "中午说了想换一套家居服", now - 600, "active"),
        ("private:umo", "昨天之前的旧记忆", now - 90000, "active"),
        ("private:other", "别的会话的记忆", now - 100, "active"),
        ("private:umo", "已归档的记忆", now - 200, "archived"),
    ]
    async with sqlite_connection(db_path) as db:
        await db.execute(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                text TEXT,
                metadata TEXT,
                created_at REAL,
                updated_at REAL
            )
            """
        )
        for index, (session_id, text, create_time, status) in enumerate(rows):
            await db.execute(
                "INSERT INTO documents (doc_id, text, metadata, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    f"doc-{index}",
                    text,
                    json.dumps(
                        {"session_id": session_id, "status": status, "create_time": create_time}
                    ),
                    create_time,
                    create_time,
                ),
            )
        await db.commit()
    return db_path


def _handler(db_path: Path) -> EventHandler:
    handler = EventHandler.__new__(EventHandler)
    handler.memory_engine = SimpleNamespace(db_path=str(db_path))
    return handler


@pytest.mark.asyncio
async def test_recent_memories_filters_session_and_time(tmp_path: Path) -> None:
    db_path = await _make_db(tmp_path)
    handler = _handler(db_path)
    cutoff = time.time() - 7200  # 只取最近两小时

    results = await handler.get_recent_memories("private:umo", since=str(cutoff), limit=5)

    texts = [item["text"] for item in results]
    assert texts[0] == "中午说了想换一套家居服"  # 最新在前
    assert "上午聊了天台拍照的约定" in texts
    assert all("旧记忆" not in text for text in texts)
    assert all("别的会话" not in text for text in texts)
    assert all("已归档" not in text for text in texts)
    assert all(len(item["time"]) == 16 for item in results)  # YYYY-MM-DD HH:MM


@pytest.mark.asyncio
async def test_recent_memories_respects_limit_and_invalid_since(tmp_path: Path) -> None:
    db_path = await _make_db(tmp_path)
    handler = _handler(db_path)

    limited = await handler.get_recent_memories(
        "private:umo", since=str(time.time() - 7200), limit=1
    )
    assert len(limited) == 1

    assert await handler.get_recent_memories("private:umo", since="不是时间", limit=3) == []
    assert await handler.get_recent_memories("private:umo", since="", limit=3) == []
    assert await handler.get_recent_memories("", since=str(time.time()), limit=3) == []


@pytest.mark.asyncio
async def test_recent_memories_missing_db_returns_empty(tmp_path: Path) -> None:
    handler = _handler(tmp_path / "not_exists.db")
    assert await handler.get_recent_memories("private:umo", since=str(time.time()), limit=3) == []
