"""记忆库整合（MemoryConsolidationManager）测试。"""

import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from astrbot_plugin_livingmemory.core.managers.consolidation_manager import (
    MemoryConsolidationManager,
)
from astrbot_plugin_livingmemory.core.managers.memory_engine import MemoryEngine


@dataclass
class _FakeRetrieveResult:
    similarity: float
    data: dict


class _FakeDocumentStorage:
    def __init__(self, db: "_FakeFaissDB"):
        self._db = db

    async def get_documents(self, metadata_filters, ids=None, limit=50, offset=0):
        docs = list(self._db.docs.values())
        if ids is not None:
            id_set = set(ids)
            docs = [doc for doc in docs if doc["id"] in id_set]
        for key, value in (metadata_filters or {}).items():
            docs = [doc for doc in docs if doc["metadata"].get(key) == value]
        docs = docs[offset : offset + limit]
        return [dict(doc) for doc in docs]

    async def count_documents(self, metadata_filters):
        docs = list(self._db.docs.values())
        for key, value in (metadata_filters or {}).items():
            docs = [doc for doc in docs if doc["metadata"].get(key) == value]
        return len(docs)


class _FakeFaissDB:
    def __init__(self):
        self.docs: dict[int, dict] = {}
        self._next_id = 1
        self.document_storage = _FakeDocumentStorage(self)

    async def insert(self, content: str, metadata: dict) -> int:
        doc_id = self._next_id
        self._next_id += 1
        self.docs[doc_id] = {
            "id": doc_id,
            "doc_id": f"uuid-{doc_id}",
            "text": content,
            "metadata": dict(metadata),
        }
        return doc_id

    async def retrieve(self, query, k, fetch_k, rerank, metadata_filters=None):
        return []

    async def delete(self, uuid_doc_id: str) -> None:
        for doc_id, doc in list(self.docs.items()):
            if doc["doc_id"] == uuid_doc_id:
                self.docs.pop(doc_id, None)

    async def close(self) -> None:
        return None


class _StubProcessor:
    """合并处理器桩：固定输出，不调用 LLM。"""

    def __init__(self):
        self.calls: list[list[dict]] = []

    async def merge_memories(self, memories: list[dict]) -> dict:
        self.calls.append(memories)
        return {
            "summary": "合并后的摘要",
            "key_facts": ["事实一", "事实二"],
            "topics": ["主题"],
            "importance": 0.4,
        }


class _StubConfig:
    def __init__(self, section: dict):
        self._section = section

    def get_section(self, name: str) -> dict:
        return self._section


def _make_engine(tmp_path: Path) -> MemoryEngine:
    return MemoryEngine(
        db_path=str(tmp_path / "memory.db"),
        faiss_db=_FakeFaissDB(),
        graph_vector_db=_FakeFaissDB(),
        config={"graph_memory_enabled": False, "fallback_enabled": True},
    )


async def _sync_documents_row(
    engine, memory_id: int, content: str, importance: float
) -> None:
    """测试桩不写 SQLite documents 表，这里手动同步一行（对齐真实存储行为）。"""
    import json as _json

    await engine.db_connection.execute(
        "INSERT OR REPLACE INTO documents "
        "(id, doc_id, text, metadata, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))",
        (
            memory_id,
            f"uuid-{memory_id}",
            content,
            _json.dumps(
                {
                    "importance": importance,
                    "create_time": time.time() - 1.0,
                    "session_id": "test:private:s1",
                    "key_facts": [],
                },
                ensure_ascii=False,
            ),
        ),
    )
    await engine.db_connection.commit()


@pytest.mark.asyncio
async def test_consolidation_session_groups_archive_originals(tmp_path: Path):
    engine = _make_engine(tmp_path)
    await engine.initialize()

    ids = []
    for i in range(3):
        mid = await engine.add_memory(
            content=f"零散记忆{i}",
            session_id="test:private:s1",
            persona_id="persona_1",
            importance=0.2,
            metadata={"key_facts": [f"事实{i}"], "topics": ["闲聊"]},
        )
        ids.append(mid)
        await _sync_documents_row(engine, mid, f"零散记忆{i}", 0.2)

    processor = _StubProcessor()
    manager = MemoryConsolidationManager(
        engine,
        processor,
        _StubConfig(
            {
                "enabled": True,
                "trigger": "daily",
                "granularity": "session",
                "keep_original": "archive",
                "min_memories_per_group": 3,
                "min_age_days": 0,
            }
        ),
    )

    stats = await manager.run_consolidation(force=True, include_disabled=True)
    assert stats.get("groups") == 1
    assert stats.get("merged") == 3
    assert stats.get("archived") == 3
    assert len(processor.calls) == 1

    # 新记忆写入，包含溯源信息（测试桩走 fake FAISS 存储）
    consolidated = [
        d
        for d in engine.faiss_db.docs.values()
        if d["metadata"].get("consolidated_from")
    ]
    assert len(consolidated) == 1
    assert set(consolidated[0]["metadata"]["consolidated_from"]) == set(ids)

    # 旧记忆已归档：仍可读取原文，但 status 为 archived
    for old_id in ids:
        memory = await engine.get_memory(old_id)
        assert memory is not None
        assert memory["metadata"]["status"] == "archived"

    await engine.close()


@pytest.mark.asyncio
async def test_consolidation_delete_mode_removes_originals(tmp_path: Path):
    engine = _make_engine(tmp_path)
    await engine.initialize()

    ids = []
    for i in range(3):
        mid = await engine.add_memory(
            content=f"待删除记忆{i}",
            session_id="test:private:s1",
            persona_id="persona_1",
            importance=0.2,
            metadata={},
        )
        ids.append(mid)
        await _sync_documents_row(engine, mid, f"待删除记忆{i}", 0.2)

    manager = MemoryConsolidationManager(
        engine,
        _StubProcessor(),
        _StubConfig(
            {
                "enabled": True,
                "granularity": "session",
                "keep_original": "delete",
                "min_memories_per_group": 3,
                "min_age_days": 0,
            }
        ),
    )

    stats = await manager.run_consolidation(force=True, include_disabled=True)
    assert stats.get("deleted") == 3

    for old_id in ids:
        assert await engine.get_memory(old_id) is None

    await engine.close()


@pytest.mark.asyncio
async def test_consolidation_disabled_skips(tmp_path: Path):
    engine = _make_engine(tmp_path)
    await engine.initialize()

    manager = MemoryConsolidationManager(
        engine, _StubProcessor(), _StubConfig({"enabled": False})
    )
    stats = await manager.maybe_run("daily")
    assert stats.get("skipped") is True
    await engine.close()
