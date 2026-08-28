"""
索引一致性验证器 - 检测并修复索引与数据库的不一致问题
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from astrbot.api import logger

from ...storage.sqlite_utils import sqlite_connection


@dataclass
class IndexStatus:
    """索引状态信息"""

    is_consistent: bool  # 是否一致
    documents_count: int  # documents表中的文档数
    bm25_count: int  # BM25索引中的文档数
    vector_count: int  # 向量索引中的文档数
    missing_in_bm25: int  # documents中有但BM25中缺失的数量
    missing_in_vector: int  # documents中有但向量索引中缺失的数量
    needs_rebuild: bool  # 是否需要重建
    reason: str  # 不一致的原因描述


class IndexValidator:
    """
    索引一致性验证器

    检测documents表与BM25索引、向量索引之间的一致性
    """

    def __init__(self, db_path: str, faiss_db: Any):
        """
        初始化验证器

        Args:
            db_path: SQLite数据库路径
            faiss_db: FaissVecDB实例
        """
        self.db_path = db_path
        self.faiss_db = faiss_db
        # 防止后台维护与手动 /lmem rebuild-index 并发重建
        self._rebuild_lock = asyncio.Lock()

    DEFAULT_REBUILD_BATCH_SIZE = 50
    DEFAULT_EMBEDDING_BATCH_SIZE = 8
    DEFAULT_TASKS_LIMIT = 1
    DEFAULT_MAX_RETRIES = 5
    DEFAULT_RETRY_BASE_DELAY = 30.0
    DEFAULT_BATCH_DELAY = 5.0
    DEFAULT_REQUEST_DELAY = 5.0
    RATE_LIMIT_RETRY_MIN_DELAY = 30.0
    DEFAULT_MAX_FAILURE_RATIO = 0.02

    async def check_consistency(self) -> IndexStatus:
        """
        检查索引一致性

        Returns:
            IndexStatus: 索引状态信息
        """
        try:
            async with sqlite_connection(self.db_path) as db:
                # 1. 获取documents表中的文档数和ID集合
                cursor = await db.execute("SELECT COUNT(*) FROM documents")
                count_result = await cursor.fetchone()
                documents_count = count_result[0] if count_result else 0

                cursor = await db.execute("SELECT id FROM documents")
                doc_ids = {row[0] for row in await cursor.fetchall()}

                # 2. 检查BM25索引（livingmemory_memories_fts表）
                cursor = await db.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='livingmemory_memories_fts'
                """)
                has_fts_table = await cursor.fetchone()

                if has_fts_table:
                    cursor = await db.execute(
                        "SELECT COUNT(DISTINCT doc_id) FROM livingmemory_memories_fts"
                    )
                    bm25_result = await cursor.fetchone()
                    bm25_count = bm25_result[0] if bm25_result else 0

                    # 直接在 SQL 层计算 BM25 缺失数量，避免加载全量 ID 集合到内存。
                    cursor = await db.execute(
                        """
                        SELECT COUNT(*) FROM documents d
                        WHERE NOT EXISTS (
                            SELECT 1 FROM livingmemory_memories_fts f
                            WHERE f.doc_id = d.id
                        )
                        """
                    )
                    missing_result = await cursor.fetchone()
                    missing_in_bm25 = missing_result[0] if missing_result else 0
                else:
                    bm25_count = 0
                    missing_in_bm25 = 0

                # 3. 检查向量索引
                vector_count = 0
                vector_ids = set()

                try:
                    embedding_storage = getattr(
                        self.faiss_db, "embedding_storage", None
                    )
                    index = getattr(embedding_storage, "index", None)
                    if index is not None:
                        vector_count = int(getattr(index, "ntotal", 0))
                        # Try to get concrete vector IDs from IndexIDMap.
                        try:
                            import faiss

                            if hasattr(index, "id_map"):
                                vector_to_array = getattr(
                                    faiss, "vector_to_array", None
                                )
                                if callable(vector_to_array):
                                    raw_ids = cast(Any, vector_to_array(index.id_map))
                                    vector_ids = {int(i) for i in raw_ids}
                        except Exception as e:
                            logger.debug(f"读取向量ID失败，使用计数模式: {e}")
                except Exception as e:
                    logger.warning(f"检查向量索引失败: {e}")

                # 4. 计算差异
                if vector_ids:
                    missing_in_vector = len(doc_ids - vector_ids)
                else:
                    missing_in_vector = max(0, documents_count - vector_count)

                # 5. 判断是否需要重建
                needs_rebuild = False
                reason = ""

                if documents_count == 0:
                    reason = "数据库为空"
                    is_consistent = True
                elif missing_in_bm25 > 0 or missing_in_vector > 0:
                    needs_rebuild = True
                    is_consistent = False
                    reasons = []
                    if missing_in_bm25 > 0:
                        reasons.append(f"BM25索引缺失{missing_in_bm25}条文档")
                    if missing_in_vector > 0:
                        reasons.append(f"向量索引缺失{missing_in_vector}条文档")
                    reason = "；".join(reasons)
                elif bm25_count > documents_count:
                    needs_rebuild = True
                    is_consistent = False
                    reason = "BM25索引中存在冗余数据"
                elif vector_count > documents_count:
                    # FAISS ntotal 包含逻辑删除的槽位，冗余向量不影响召回正确性，
                    # 不触发全量重建（否则每次启动都会重建）
                    is_consistent = True
                    reason = f"向量索引含{vector_count - documents_count}条冗余槽位（正常，不影响召回）"
                else:
                    is_consistent = True
                    reason = "索引状态正常"

                return IndexStatus(
                    is_consistent=is_consistent,
                    documents_count=documents_count,
                    bm25_count=bm25_count,
                    vector_count=vector_count,
                    missing_in_bm25=missing_in_bm25,
                    missing_in_vector=missing_in_vector,
                    needs_rebuild=needs_rebuild,
                    reason=reason,
                )

        except Exception as e:
            logger.error(f"检查索引一致性失败: {e}", exc_info=True)
            return IndexStatus(
                is_consistent=False,
                documents_count=0,
                bm25_count=0,
                vector_count=0,
                missing_in_bm25=0,
                missing_in_vector=0,
                needs_rebuild=True,
                reason=f"检查失败: {str(e)}",
            )

    async def get_migration_status(self) -> tuple[bool, int]:
        """
        获取v1迁移状态

        Returns:
            Tuple[bool, int]: (是否需要重建, 待处理文档数)
        """
        try:
            async with sqlite_connection(self.db_path) as db:
                # 检查migration_status表
                cursor = await db.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='migration_status'
                """)
                has_table = await cursor.fetchone()

                if not has_table:
                    return False, 0

                # 检查是否需要重建
                cursor = await db.execute("""
                    SELECT value FROM migration_status
                    WHERE key='needs_index_rebuild'
                """)
                row = await cursor.fetchone()

                if not row or len(row) == 0 or row[0] != "true":
                    return False, 0

                # 获取待处理文档数
                cursor = await db.execute("""
                    SELECT value FROM migration_status
                    WHERE key='pending_documents_count'
                """)
                count_row = await cursor.fetchone()
                pending_count = (
                    int(count_row[0])
                    if count_row and len(count_row) > 0 and count_row[0]
                    else 0
                )

                return True, pending_count

        except Exception as e:
            logger.error(f"获取迁移状态失败: {e}", exc_info=True)
            return False, 0

    def _get_rebuild_options(self, memory_engine: Any) -> dict[str, Any]:
        config = getattr(memory_engine, "config", {}) or {}

        def read_int(key: str, default: int, minimum: int, maximum: int) -> int:
            try:
                value = int(config.get(key, default))
            except (TypeError, ValueError):
                value = default
            return max(minimum, min(maximum, value))

        def read_float(
            key: str, default: float, minimum: float, maximum: float
        ) -> float:
            try:
                value = float(config.get(key, default))
            except (TypeError, ValueError):
                value = default
            return max(minimum, min(maximum, value))

        return {
            "batch_size": read_int(
                "index_rebuild_batch_size", self.DEFAULT_REBUILD_BATCH_SIZE, 1, 500
            ),
            "embedding_batch_size": read_int(
                "index_rebuild_embedding_batch_size",
                self.DEFAULT_EMBEDDING_BATCH_SIZE,
                1,
                256,
            ),
            "tasks_limit": read_int(
                "index_rebuild_tasks_limit", self.DEFAULT_TASKS_LIMIT, 1, 8
            ),
            "max_retries": read_int(
                "index_rebuild_max_retries", self.DEFAULT_MAX_RETRIES, 1, 8
            ),
            "retry_base_delay": read_float(
                "index_rebuild_retry_base_delay",
                self.DEFAULT_RETRY_BASE_DELAY,
                0.0,
                60.0,
            ),
            "batch_delay": read_float(
                "index_rebuild_batch_delay", self.DEFAULT_BATCH_DELAY, 0.0, 10.0
            ),
            "request_delay": read_float(
                "index_rebuild_request_delay", self.DEFAULT_REQUEST_DELAY, 0.0, 60.0
            ),
            "max_failure_ratio": read_float(
                "index_rebuild_max_failure_ratio",
                self.DEFAULT_MAX_FAILURE_RATIO,
                0.0,
                1.0,
            ),
        }

    @staticmethod
    def _failure_ratio(errors: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return errors / total

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        message = str(error).lower()
        return (
            "429" in message
            or "rate limit" in message
            or "tpm limit" in message
            or "too many requests" in message
        )

    async def _get_document_count(self) -> int:
        async with sqlite_connection(self.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM documents")
            row = await cursor.fetchone()
            return int(row[0]) if row else 0

    async def _get_document_ids(self) -> set[int]:
        async with sqlite_connection(self.db_path) as db:
            cursor = await db.execute("SELECT id FROM documents")
            return {int(row[0]) for row in await cursor.fetchall()}

    async def _iter_document_batches(
        self,
        batch_size: int,
        document_ids: set[int] | None = None,
        after_id: int = 0,
    ):
        if document_ids is not None:
            sorted_ids = sorted(int(doc_id) for doc_id in document_ids)
            for start in range(0, len(sorted_ids), batch_size):
                chunk = sorted_ids[start : start + batch_size]
                placeholders = ",".join("?" for _ in chunk)
                async with sqlite_connection(self.db_path) as db:
                    await db.execute("PRAGMA busy_timeout = 10000")
                    cursor = await db.execute(
                        f"""
                        SELECT id, doc_id, text, metadata
                        FROM documents
                        WHERE id IN ({placeholders})
                        ORDER BY id
                        """,
                        chunk,
                    )
                    yield await cursor.fetchall()
            return

        last_id = after_id
        while True:
            async with sqlite_connection(self.db_path) as db:
                await db.execute("PRAGMA busy_timeout = 10000")
                cursor = await db.execute(
                    """
                    SELECT id, doc_id, text, metadata
                    FROM documents
                    WHERE id > ?
                    ORDER BY id
                    LIMIT ?
                    """,
                    (last_id, batch_size),
                )
                rows = await cursor.fetchall()

            if not rows:
                break
            last_id = int(rows[-1][0])
            yield rows

    def _get_vector_count(self) -> int:
        embedding_storage = getattr(self.faiss_db, "embedding_storage", None)
        index = getattr(embedding_storage, "index", None)
        if index is None:
            return 0
        return int(getattr(index, "ntotal", 0))

    def _get_vector_ids(self) -> set[int] | None:
        embedding_storage = getattr(self.faiss_db, "embedding_storage", None)
        index = getattr(embedding_storage, "index", None)
        if index is None:
            return set()
        try:
            import faiss

            if hasattr(index, "id_map"):
                vector_to_array = getattr(faiss, "vector_to_array", None)
                if callable(vector_to_array):
                    raw_ids = cast(Any, vector_to_array(index.id_map))
                    return {int(i) for i in raw_ids}
        except Exception as e:
            logger.debug(f"读取向量ID失败: {e}")
        return None

    async def _rebuild_bm25_index(
        self,
        memory_engine: Any,
        total: int,
        options: dict[str, Any],
        progress_callback=None,
        document_ids: set[int] | None = None,
    ) -> dict[str, Any]:
        """重建 BM25 索引。

        全量模式（document_ids=None）写入影子表，完成后与线上表在同一事务内
        DROP+RENAME 原子切换，重建期间线上检索不中断；修复模式直接写入线上表。
        """
        bm25_retriever = getattr(memory_engine, "bm25_retriever", None)
        text_processor = getattr(bm25_retriever, "text_processor", None)
        if text_processor is None:
            text_processor = getattr(memory_engine, "text_processor", None)
        if text_processor is None:
            raise RuntimeError("无法重建 BM25：TextProcessor 未初始化")

        table_name = getattr(bm25_retriever, "fts_table", "livingmemory_memories_fts")
        batch_size = int(options["batch_size"])
        max_failure_ratio = float(options["max_failure_ratio"])

        full_rebuild = document_ids is None
        shadow_table = f"{table_name}_rebuild_shadow"
        if full_rebuild:
            # 准备影子表：遗留的半成品直接重建
            async with sqlite_connection(self.db_path) as db:
                await db.execute("PRAGMA busy_timeout = 10000")
                await db.execute(f"DROP TABLE IF EXISTS {shadow_table}")
                await db.execute(
                    f"""
                    CREATE VIRTUAL TABLE {shadow_table}
                    USING fts5(
                        content,
                        doc_id UNINDEXED,
                        tokenize='unicode61'
                    )
                    """
                )
                await db.commit()
        target_table = shadow_table if full_rebuild else table_name

        processed = 0
        failed_ids: set[int] = set()

        async for batch in self._iter_document_batches(batch_size, document_ids):
            rows_to_insert: list[tuple[int, str]] = []
            for doc_id, _doc_uuid, text, _metadata_json in batch:
                try:
                    if hasattr(text_processor, "preprocess_for_bm25"):
                        processed_content = text_processor.preprocess_for_bm25(
                            text or ""
                        )
                    else:
                        tokens = text_processor.tokenize(text or "", True)
                        processed_content = " ".join(tokens)
                    rows_to_insert.append((int(doc_id), processed_content))
                except Exception as e:
                    failed_ids.add(int(doc_id))
                    logger.error(f"BM25 预处理失败 doc_id={doc_id}: {e}")

            if rows_to_insert:
                try:
                    async with sqlite_connection(self.db_path) as db:
                        await db.execute("PRAGMA busy_timeout = 10000")
                        await db.executemany(
                            f"INSERT INTO {target_table}(doc_id, content) VALUES (?, ?)",
                            rows_to_insert,
                        )
                        await db.commit()
                    processed += len(rows_to_insert)
                except Exception as batch_error:
                    logger.warning(f"BM25 批量写入失败，将逐条重试: {batch_error}")
                    for row_doc_id, processed_content in rows_to_insert:
                        try:
                            async with sqlite_connection(self.db_path) as db:
                                await db.execute("PRAGMA busy_timeout = 10000")
                                await db.execute(
                                    f"INSERT INTO {target_table}(doc_id, content) VALUES (?, ?)",
                                    (row_doc_id, processed_content),
                                )
                                await db.commit()
                            processed += 1
                        except Exception as e:
                            failed_ids.add(int(row_doc_id))
                            logger.error(f"BM25 写入失败 doc_id={row_doc_id}: {e}")

            if progress_callback:
                await progress_callback(
                    processed,
                    total,
                    f"BM25 已处理 {processed}/{total} 条",
                )

            if self._failure_ratio(len(failed_ids), total) > max_failure_ratio:
                logger.error(
                    f"BM25 重建失败率过高: {len(failed_ids)}/{total}，停止后续重建"
                )
                break

        if full_rebuild:
            ratio = self._failure_ratio(len(failed_ids), total)
            if ratio <= max_failure_ratio and processed > 0:
                # 原子切换：DROP 旧表 + RENAME 影子表在同一事务内完成
                async with sqlite_connection(self.db_path) as db:
                    await db.execute("PRAGMA busy_timeout = 10000")
                    await db.execute("BEGIN IMMEDIATE")
                    try:
                        await db.execute(f"DROP TABLE IF EXISTS {table_name}")
                        await db.execute(
                            f"ALTER TABLE {shadow_table} RENAME TO {table_name}"
                        )
                        await db.commit()
                    except Exception:
                        await db.rollback()
                        raise
                logger.info(f"BM25 影子表已原子切换: {table_name}")
            else:
                # 失败：丢弃影子表，线上表保持不变
                async with sqlite_connection(self.db_path) as db:
                    await db.execute("PRAGMA busy_timeout = 10000")
                    await db.execute(f"DROP TABLE IF EXISTS {shadow_table}")
                    await db.commit()
                logger.warning("BM25 影子表重建失败，已保留线上索引")

        return {
            "processed": processed,
            "errors": len(failed_ids),
            "failed_ids": failed_ids,
        }

    async def _embed_batch_with_retry(
        self,
        provider: Any,
        contents: list[str],
        options: dict[str, Any],
    ) -> list[Any]:
        if not contents:
            return []

        max_retries = int(options["max_retries"])
        retry_base_delay = float(options["retry_base_delay"])
        embedding_batch_size = int(options["embedding_batch_size"])
        request_delay = float(options["request_delay"])
        vectors: list[Any] = []

        for start in range(0, len(contents), embedding_batch_size):
            chunk = contents[start : start + embedding_batch_size]
            logger.debug(
                "Embedding 子请求: "
                f"offset={start}, size={len(chunk)}, total={len(contents)}"
            )
            vectors.extend(
                await self._embed_request_with_retry(
                    provider,
                    chunk,
                    max_retries=max_retries,
                    retry_base_delay=retry_base_delay,
                )
            )
            if request_delay > 0 and start + embedding_batch_size < len(contents):
                await asyncio.sleep(request_delay)

        return vectors

    async def _embed_request_with_retry(
        self,
        provider: Any,
        contents: list[str],
        *,
        max_retries: int,
        retry_base_delay: float,
    ) -> list[Any]:
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                get_embeddings = getattr(provider, "get_embeddings", None)
                if callable(get_embeddings):
                    return await get_embeddings(contents)

                if hasattr(provider, "get_embeddings_batch"):
                    try:
                        return await provider.get_embeddings_batch(
                            contents,
                            batch_size=len(contents),
                            tasks_limit=1,
                            max_retries=1,
                        )
                    except TypeError:
                        return await provider.get_embeddings_batch(contents)

                vectors = []
                for content in contents:
                    vectors.append(await provider.get_embedding(content))
                return vectors
            except Exception as e:
                last_error = e
                if attempt >= max_retries - 1:
                    break
                wait_seconds = retry_base_delay * (2**attempt)
                if self._is_rate_limit_error(e):
                    wait_seconds = max(wait_seconds, self.RATE_LIMIT_RETRY_MIN_DELAY)
                logger.warning(
                    f"Embedding 批次失败，{wait_seconds:.1f}s 后重试 "
                    f"({attempt + 1}/{max_retries}): {e}"
                )
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)

        raise RuntimeError(f"Embedding 批次重试失败: {last_error}") from last_error

    async def _repair_missing_vectors(
        self,
        memory_engine: Any,
        missing_ids: set[int],
        options: dict[str, Any],
        progress_callback=None,
    ) -> dict[str, Any]:
        import numpy as np

        faiss_db = getattr(memory_engine, "faiss_db", None)
        embedding_storage = getattr(faiss_db, "embedding_storage", None)
        provider = getattr(faiss_db, "embedding_provider", None)
        if embedding_storage is None or provider is None:
            raise RuntimeError("无法修复向量索引：Embedding 组件未初始化")

        total = len(missing_ids)
        processed = 0
        failed_ids: set[int] = set()
        batch_delay = float(options["batch_delay"])
        max_failure_ratio = float(options["max_failure_ratio"])
        batch_index = 0

        async for batch in self._iter_document_batches(
            int(options["batch_size"]), missing_ids
        ):
            batch_index += 1
            ids = [int(row[0]) for row in batch]
            contents = [row[2] or "" for row in batch]
            logger.info(
                "向量补写批次开始: "
                f"batch={batch_index}, size={len(ids)}, "
                f"id_range={ids[0]}-{ids[-1]}, processed={processed}/{total}, "
                f"failed={len(failed_ids)}"
            )
            try:
                vectors = await self._embed_batch_with_retry(
                    provider, contents, options
                )
                vectors_array = np.asarray(vectors, dtype=np.float32)
                if vectors_array.ndim != 2 or len(vectors_array) != len(ids):
                    raise ValueError(
                        f"Embedding 返回数量不匹配: 期望 {len(ids)}，实际 {len(vectors_array)}"
                    )
                await embedding_storage.insert_batch(vectors_array, ids)
                processed += len(ids)
            except Exception as e:
                failed_ids.update(ids)
                logger.error(f"向量补写批次失败 ids={ids[:3]}...: {e}", exc_info=True)

            if progress_callback:
                await progress_callback(
                    processed,
                    total,
                    f"向量补写已处理 {processed}/{total} 条",
                )

            logger.info(
                "向量补写进度: "
                f"processed={processed}/{total}, failed={len(failed_ids)}, "
                f"failure_ratio={self._failure_ratio(len(failed_ids), total):.2%}"
            )

            if self._failure_ratio(len(failed_ids), total) > max_failure_ratio:
                break
            if batch_delay > 0:
                await asyncio.sleep(batch_delay)

        return {
            "mode": "repair",
            "processed": processed,
            "errors": len(failed_ids),
            "failed_ids": failed_ids,
            "switched": False,
            "partial": len(failed_ids) > 0,
        }

    async def _rebuild_vector_index_full(
        self,
        memory_engine: Any,
        total: int,
        options: dict[str, Any],
        progress_callback=None,
    ) -> dict[str, Any]:
        import faiss
        import numpy as np

        faiss_db = getattr(memory_engine, "faiss_db", None)
        embedding_storage = getattr(faiss_db, "embedding_storage", None)
        provider = getattr(faiss_db, "embedding_provider", None)
        if embedding_storage is None or provider is None:
            raise RuntimeError("无法重建向量索引：Embedding 组件未初始化")

        dimension = int(getattr(embedding_storage, "dimension", 0) or 0)
        if dimension <= 0:
            raise RuntimeError("无法重建向量索引：索引维度无效")

        index_path = getattr(embedding_storage, "path", None)
        # 影子索引与断点进度文件：重建期间持久化，进程中断后可续跑
        shadow_path = f"{index_path}.rebuild.shadow" if index_path else None
        progress_path = f"{index_path}.rebuild.progress.json" if index_path else None

        temp_index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
        processed = 0
        failed_ids: set[int] = set()
        after_id = 0

        # 断点续跑：加载上次中断留下的影子索引与进度
        if shadow_path and progress_path and os.path.exists(shadow_path):
            try:
                saved_index = faiss.read_index(shadow_path)
                saved_progress: dict[str, Any] = {}
                if os.path.exists(progress_path):
                    saved_progress = json.loads(
                        Path(progress_path).read_text(encoding="utf-8")
                    )
                if int(getattr(saved_index, "d", 0)) == dimension:
                    temp_index = saved_index
                    after_id = int(saved_progress.get("last_id", 0))
                    processed = int(saved_progress.get("processed", 0))
                    failed_ids = {
                        int(doc_id) for doc_id in saved_progress.get("failed_ids", [])
                    }
                    logger.info(
                        f"向量重建断点续跑: 已处理 {processed}/{total}, "
                        f"从 id>{after_id} 继续"
                    )
                else:
                    logger.warning("影子索引维度与当前模型不一致，忽略并从头重建")
            except Exception as e:
                logger.warning(f"加载影子索引失败，从头重建: {e}")
                temp_index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
                processed = 0
                failed_ids = set()
                after_id = 0

        def _persist_shadow_progress(last_id: int) -> None:
            if not shadow_path or not progress_path:
                return
            try:
                faiss.write_index(temp_index, shadow_path)
                Path(progress_path).write_text(
                    json.dumps(
                        {
                            "last_id": last_id,
                            "processed": processed,
                            "failed_ids": sorted(failed_ids),
                        }
                    ),
                    encoding="utf-8",
                )
            except Exception as e:
                # 进度持久化失败只影响续跑能力，不中断重建
                logger.debug(f"保存向量重建进度失败: {e}")

        failed_ratio_exceeded = False
        batch_delay = float(options["batch_delay"])
        max_failure_ratio = float(options["max_failure_ratio"])
        batch_index = 0

        async for batch in self._iter_document_batches(
            int(options["batch_size"]), after_id=after_id
        ):
            batch_index += 1
            ids = [int(row[0]) for row in batch]
            contents = [row[2] or "" for row in batch]
            logger.info(
                "向量重建批次开始: "
                f"batch={batch_index}, size={len(ids)}, "
                f"id_range={ids[0]}-{ids[-1]}, processed={processed}/{total}, "
                f"failed={len(failed_ids)}"
            )
            batch_succeeded = False
            try:
                vectors = await self._embed_batch_with_retry(
                    provider, contents, options
                )
                vectors_array = np.asarray(vectors, dtype=np.float32)
                if vectors_array.ndim != 2 or len(vectors_array) != len(ids):
                    raise ValueError(
                        f"Embedding 返回数量不匹配: 期望 {len(ids)}，实际 {len(vectors_array)}"
                    )
                if vectors_array.shape[1] != dimension:
                    raise ValueError(
                        f"Embedding 维度不匹配: 期望 {dimension}，实际 {vectors_array.shape[1]}"
                    )
                temp_index.add_with_ids(vectors_array, np.asarray(ids, dtype=np.int64))
                processed += len(ids)
                batch_succeeded = True
            except Exception as e:
                failed_ids.update(ids)
                logger.error(f"向量重建批次失败 ids={ids[:3]}...: {e}", exc_info=True)

            if progress_callback:
                await progress_callback(
                    processed,
                    total,
                    f"向量索引已处理 {processed}/{total} 条",
                )

            logger.info(
                "向量重建进度: "
                f"processed={processed}/{total}, failed={len(failed_ids)}, "
                f"failure_ratio={self._failure_ratio(len(failed_ids), total):.2%}"
            )

            if batch_succeeded:
                # 进度与影子索引内容严格对齐，续跑不会重复添加向量
                _persist_shadow_progress(ids[-1])

            if self._failure_ratio(len(failed_ids), total) > max_failure_ratio:
                logger.error(
                    f"向量重建失败率过高: {len(failed_ids)}/{total}，不会切换新索引"
                )
                failed_ratio_exceeded = True
                break
            if batch_delay > 0:
                await asyncio.sleep(batch_delay)

        if failed_ratio_exceeded:
            # 保留影子索引与进度，下次重建可断点续跑；线上索引不受影响
            return {
                "mode": "full",
                "processed": processed,
                "errors": len(failed_ids),
                "failed_ids": failed_ids,
                "switched": False,
                "partial": True,
            }

        if total > 0 and processed == 0:
            return {
                "mode": "full",
                "processed": 0,
                "errors": max(total, len(failed_ids)),
                "failed_ids": failed_ids,
                "switched": False,
                "partial": True,
            }

        if index_path:
            if shadow_path and os.path.exists(shadow_path):
                # 影子索引在重建期间已随批次持久化，完成后原子替换线上索引
                os.replace(shadow_path, index_path)
            else:
                temp_path = f"{index_path}.rebuild.tmp"
                try:
                    faiss.write_index(temp_index, temp_path)
                    os.replace(temp_path, index_path)
                finally:
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass
            if progress_path and os.path.exists(progress_path):
                try:
                    os.remove(progress_path)
                except OSError:
                    pass

        embedding_storage.index = temp_index
        return {
            "mode": "full",
            "processed": processed,
            "errors": len(failed_ids),
            "failed_ids": failed_ids,
            "switched": True,
            "partial": len(failed_ids) > 0,
        }

    async def _rebuild_or_repair_vector_index(
        self,
        memory_engine: Any,
        total: int,
        options: dict[str, Any],
        progress_callback=None,
        force_full: bool = False,
    ) -> dict[str, Any]:
        document_ids = await self._get_document_ids()
        if not document_ids:
            return {
                "mode": "skip",
                "processed": 0,
                "errors": 0,
                "failed_ids": set(),
                "switched": False,
                "partial": False,
            }

        if not force_full:
            vector_ids = self._get_vector_ids()
            vector_count = self._get_vector_count()
            if vector_ids is not None:
                missing_ids = document_ids - vector_ids
                if not missing_ids:
                    return {
                        "mode": "skip",
                        "processed": 0,
                        "errors": 0,
                        "failed_ids": set(),
                        "switched": False,
                        "partial": False,
                    }
                if vector_ids:
                    logger.info(f"检测到 {len(missing_ids)} 条向量缺失，执行增量补写")
                    return await self._repair_missing_vectors(
                        memory_engine, missing_ids, options, progress_callback
                    )

            if vector_ids is None and vector_count >= total:
                logger.info("向量索引计数不小于 documents 数量，跳过全量向量重建")
                return {
                    "mode": "skip",
                    "processed": 0,
                    "errors": 0,
                    "failed_ids": set(),
                    "switched": False,
                    "partial": False,
                }

        logger.info("向量索引缺失或为空，执行安全全量重建")
        return await self._rebuild_vector_index_full(
            memory_engine, total, options, progress_callback
        )

    async def _update_migration_rebuild_status(
        self, completed_value: str = "true"
    ) -> None:
        from datetime import datetime, timezone

        try:
            async with sqlite_connection(self.db_path) as status_db:
                await status_db.execute("""
                    CREATE TABLE IF NOT EXISTS migration_status (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at TEXT
                    )
                """)
                await status_db.execute(
                    """
                    INSERT OR REPLACE INTO migration_status (key, value, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (
                        "needs_index_rebuild",
                        "false",
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                await status_db.execute(
                    """
                    INSERT OR REPLACE INTO migration_status (key, value, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (
                        "index_rebuild_completed",
                        completed_value,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                await status_db.commit()
        except Exception as e:
            logger.warning(f"更新迁移状态失败: {e}")

    async def rebuild_indexes(
        self,
        memory_engine: Any,
        progress_callback=None,
        force_vector_rebuild: bool = False,
    ) -> dict[str, Any]:
        """
        分批安全重建索引

        安全策略：
        1. documents 表只读，始终作为原始数据源。
        2. BM25 写入影子表，完成后原子切换，重建期间检索不中断。
        3. 向量索引优先增量补缺；全量重建写入影子索引（支持断点续跑），
           完成后原子替换；失败率超过阈值时不切换。
        4. 重建完成后对并发写入的新文档做一次有上限的补偿修复。

        Args:
            memory_engine: MemoryEngine实例
            progress_callback: 进度回调函数 (current, total, message)
            force_vector_rebuild: 跳过向量增量判断，强制全量重建
                （嵌入模型更换时由调用方传入）

        Returns:
            Dict: 重建结果
        """
        if self._rebuild_lock.locked():
            return {
                "success": False,
                "message": "索引重建已在进行中，请等待当前重建完成。",
                "processed": 0,
                "errors": 0,
                "total": 0,
                "partial": False,
                "switched": False,
                "already_running": True,
            }
        async with self._rebuild_lock:
            return await self._rebuild_indexes_locked(
                memory_engine, progress_callback, force_vector_rebuild
            )

    async def _rebuild_indexes_locked(
        self,
        memory_engine: Any,
        progress_callback=None,
        force_vector_rebuild: bool = False,
    ) -> dict[str, Any]:
        try:
            logger.info("开始分批安全重建索引。")
            options = self._get_rebuild_options(memory_engine)
            total = await self._get_document_count()

            if total <= 0:
                return {
                    "success": True,
                    "message": "没有需要重建的文档",
                    "processed": 0,
                    "errors": 0,
                    "total": 0,
                    "partial": False,
                    "switched": False,
                }

            logger.info(
                "重建参数: "
                f"total={total}, batch_size={options['batch_size']}, "
                f"embedding_batch_size={options['embedding_batch_size']}, "
                f"tasks_limit={options['tasks_limit']}, "
                f"request_delay={options['request_delay']}, "
                f"batch_delay={options['batch_delay']}, "
                f"max_failure_ratio={options['max_failure_ratio']}"
            )

            bm25_result = await self._rebuild_bm25_index(
                memory_engine, total, options, progress_callback
            )
            bm25_failed_ids = set(bm25_result["failed_ids"])
            if self._failure_ratio(len(bm25_failed_ids), total) > float(
                options["max_failure_ratio"]
            ):
                message = (
                    f"BM25 重建失败率过高: {len(bm25_failed_ids)}/{total}。"
                    "documents 原始数据未被删除，已停止向量重建。"
                )
                logger.error(message)
                return {
                    "success": False,
                    "message": message,
                    "processed": total - len(bm25_failed_ids),
                    "errors": len(bm25_failed_ids),
                    "total": total,
                    "partial": True,
                    "switched": False,
                    "bm25_processed": bm25_result["processed"],
                    "bm25_errors": bm25_result["errors"],
                    "vector_processed": 0,
                    "vector_errors": 0,
                    "failure_ratio": self._failure_ratio(len(bm25_failed_ids), total),
                }

            vector_result = await self._rebuild_or_repair_vector_index(
                memory_engine,
                total,
                options,
                progress_callback,
                force_full=force_vector_rebuild,
            )
            vector_failed_ids = set(vector_result["failed_ids"])
            failed_ids = bm25_failed_ids | vector_failed_ids
            failure_ratio = self._failure_ratio(len(failed_ids), total)
            accepted = failure_ratio <= float(options["max_failure_ratio"])
            partial = bool(failed_ids)

            if accepted:
                await self._update_migration_rebuild_status(
                    "partial" if partial else "true"
                )
                message = (
                    "索引重建完成"
                    if not partial
                    else (
                        "索引已按失败率阈值完成可接受切换，"
                        f"仍有 {len(failed_ids)} 条需后续重试"
                    )
                )
            else:
                message = (
                    f"索引重建失败率过高: {len(failed_ids)}/{total}。"
                    "全量向量重建未切换新索引，documents 原始数据未被删除。"
                )

            compensated = 0
            if accepted:
                # 并发写入收尾补偿：重建期间新增/变更的文档补齐派生索引
                compensated = await self._repair_rebuild_delta(
                    memory_engine, options, progress_callback
                )

            logger.info(
                "索引重建结果: "
                f"accepted={accepted}, partial={partial}, "
                f"bm25={bm25_result['processed']}/{total}, "
                f"vector={vector_result['processed']}/{total}, "
                f"errors={len(failed_ids)}, vector_mode={vector_result['mode']}, "
                f"compensated={compensated}"
            )

            return {
                "success": accepted,
                "message": message,
                "processed": max(0, total - len(failed_ids)),
                "errors": len(failed_ids),
                "total": total,
                "partial": partial,
                "switched": bool(vector_result["switched"]),
                "bm25_processed": bm25_result["processed"],
                "bm25_errors": bm25_result["errors"],
                "vector_processed": vector_result["processed"],
                "vector_errors": vector_result["errors"],
                "vector_mode": vector_result["mode"],
                "failure_ratio": failure_ratio,
                "compensated": compensated,
            }

        except Exception as e:
            logger.error(f"重建索引失败: {e}", exc_info=True)
            return {
                "success": False,
                "message": (
                    f"重建索引失败: {str(e)}。documents 原始数据未被删除，"
                    "请查看日志后重试 /lmem rebuild-index。"
                ),
                "error": str(e),
            }

    _MAX_REBUILD_DELTA_REPAIR = 200

    async def _get_missing_bm25_ids(self, limit: int | None = None) -> list[int]:
        """返回 documents 中存在但 BM25 索引缺失的文档 ID。"""
        async with sqlite_connection(self.db_path) as db:
            query = """
                SELECT d.id FROM documents d
                WHERE NOT EXISTS (
                    SELECT 1 FROM livingmemory_memories_fts f
                    WHERE f.doc_id = d.id
                )
                ORDER BY d.id
            """
            params: list[Any] = []
            if limit is not None:
                query += " LIMIT ?"
                params.append(int(limit))
            cursor = await db.execute(query, params)
            return [int(row[0]) for row in await cursor.fetchall()]

    async def _repair_rebuild_delta(
        self,
        memory_engine: Any,
        options: dict[str, Any],
        progress_callback=None,
    ) -> int:
        """对重建期间并发写入的文档做一次有上限的补偿修复。

        Args:
            memory_engine: MemoryEngine实例
            options: 重建参数
            progress_callback: 进度回调函数

        Returns:
            int: 补偿修复的文档数量
        """
        repaired = 0
        try:
            missing_bm25 = await self._get_missing_bm25_ids(
                limit=self._MAX_REBUILD_DELTA_REPAIR
            )
            if missing_bm25:
                logger.info(f"重建后补偿: 补写 {len(missing_bm25)} 条 BM25 缺失")
                result = await self._rebuild_bm25_index(
                    memory_engine,
                    len(missing_bm25),
                    options,
                    progress_callback,
                    document_ids=set(missing_bm25),
                )
                repaired += int(result["processed"])

            document_ids = await self._get_document_ids()
            vector_ids = self._get_vector_ids()
            if vector_ids is not None:
                missing_vectors = document_ids - vector_ids
                if missing_vectors:
                    capped = set(
                        sorted(missing_vectors)[: self._MAX_REBUILD_DELTA_REPAIR]
                    )
                    logger.info(f"重建后补偿: 补写 {len(capped)} 条向量缺失")
                    result = await self._repair_missing_vectors(
                        memory_engine, capped, options, progress_callback
                    )
                    repaired += int(result["processed"])
        except Exception as e:
            logger.warning(f"重建后补偿修复失败（可在下次重建时补齐）: {e}")
        return repaired

    def _embedding_fingerprint_path(self) -> str:
        return f"{self.db_path}.embedding_fingerprint.json"

    @staticmethod
    def _compute_embedding_fingerprint(provider: Any) -> str:
        """Compute a stable identity string for the embedding provider."""
        config = getattr(provider, "provider_config", None) or {}
        provider_id = (
            str(config.get("id", "") or "") if isinstance(config, dict) else ""
        )
        get_model = getattr(provider, "get_model", None)
        model = str(get_model()) if callable(get_model) else ""
        get_dim = getattr(provider, "get_dim", None)
        try:
            dimension = int(get_dim()) if callable(get_dim) else 0
        except (TypeError, ValueError):
            dimension = 0
        return f"{provider_id}:{model}:{dimension}"

    async def check_embedding_fingerprint(self, provider: Any) -> tuple[bool, str]:
        """Compare the provider identity against the stored fingerprint.

        Returns:
            tuple[bool, str]: (指纹是否变化, 当前指纹)
        """
        current = self._compute_embedding_fingerprint(provider)
        stored: str | None = None
        path = self._embedding_fingerprint_path()
        if os.path.exists(path):
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                stored = data.get("fingerprint")
            except Exception as e:
                logger.debug(f"读取嵌入模型指纹失败: {e}")
        return (stored is not None and stored != current), current

    async def save_embedding_fingerprint(self, provider: Any) -> None:
        """Persist the current embedding provider identity."""
        try:
            path = self._embedding_fingerprint_path()
            Path(path).write_text(
                json.dumps(
                    {
                        "fingerprint": self._compute_embedding_fingerprint(provider),
                        "updated_at": time.time(),
                    }
                ),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug(f"保存嵌入模型指纹失败: {e}")
