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
        # (session_id, text, create_time, status, consolidated)
        ("private:umo", "上午聊了天台拍照的约定", now - 3600, "active", False),
        ("private:umo", "中午说了想换一套家居服", now - 600, "active", False),
        ("private:umo", "零点整合出来的长期摘要", now - 300, "active", True),
        ("private:umo", "昨天之前的旧记忆", now - 90000, "active", False),
        ("private:other", "别的会话的记忆", now - 100, "active", False),
        ("private:umo", "已归档的记忆", now - 200, "archived", False),
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
        for index, (session_id, text, create_time, status, consolidated) in enumerate(
            rows
        ):
            metadata: dict = {
                "session_id": session_id,
                "status": status,
                "create_time": create_time,
            }
            if consolidated:
                metadata["consolidated_at"] = create_time
                metadata["consolidated_from"] = ["doc-a", "doc-b"]
            await db.execute(
                "INSERT INTO documents (doc_id, text, metadata, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (f"doc-{index}", text, json.dumps(metadata), create_time, create_time),
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
    assert all("整合" not in text for text in texts)
    assert all(len(item["time"]) == 16 for item in results)  # YYYY-MM-DD HH:MM


@pytest.mark.asyncio
async def test_recent_memories_respects_limit_and_unbounded_since(tmp_path: Path) -> None:
    db_path = await _make_db(tmp_path)
    handler = _handler(db_path)

    limited = await handler.get_recent_memories(
        "private:umo", since=str(time.time() - 7200), limit=1
    )
    assert len(limited) == 1
    assert limited[0]["text"] == "中午说了想换一套家居服"

    # 空/无效 since = 不限时间取最近 N 条（整合记忆仍被过滤）
    unbounded = await handler.get_recent_memories("private:umo", since="", limit=10)
    texts = [item["text"] for item in unbounded]
    assert "中午说了想换一套家居服" in texts
    assert "上午聊了天台拍照的约定" in texts
    assert "昨天之前的旧记忆" in texts
    assert all("整合" not in text for text in texts)

    invalid = await handler.get_recent_memories("private:umo", since="不是时间", limit=10)
    assert [item["text"] for item in invalid] == texts

    assert await handler.get_recent_memories("", since=str(time.time()), limit=3) == []


@pytest.mark.asyncio
async def test_recent_memories_consolidated_excluded_even_when_newest(tmp_path: Path) -> None:
    db_path = await _make_db(tmp_path)
    handler = _handler(db_path)

    # 整合记忆是库里最新的一条，但永远不该出现
    results = await handler.get_recent_memories("private:umo", since="", limit=2)
    texts = [item["text"] for item in results]
    assert "零点整合出来的长期摘要" not in texts
    assert len(texts) == 2


@pytest.mark.asyncio
async def test_recent_memories_missing_db_returns_empty(tmp_path: Path) -> None:
    handler = _handler(tmp_path / "not_exists.db")
    assert await handler.get_recent_memories("private:umo", since=str(time.time()), limit=3) == []
