"""Manage graph-memory indexing and synchronization."""

from __future__ import annotations

from typing import Any

from ...storage.graph_store import GraphStore
from ..models.graph_models import GraphEntry
from ..processors.graph_extractor import GraphExtractor
from ..retrieval.graph_vector_retriever import GraphVectorRetriever


class GraphMemoryManager:
    """Synchronize graph-memory artifacts with the document memory store.

    Graph vectors are compressed to one aggregated semantic vector per
    source memory (instead of one vector per graph entry), keeping the
    graph FAISS index proportional to the number of memories.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        graph_vector_retriever: GraphVectorRetriever,
        graph_extractor: GraphExtractor,
    ):
        self.graph_store = graph_store
        self.graph_vector_retriever = graph_vector_retriever
        self.graph_extractor = graph_extractor

    async def index_memory(
        self,
        source_memory_id: int,
        content: str,
        metadata: dict[str, Any] | None,
        atoms: list | None = None,
    ) -> None:
        """Rebuild graph artifacts for one source memory.

        When atoms are provided, each atom independently contributes
        nodes/edges/entries with per-atom confidence scores.
        """
        await self._delete_memory_now(source_memory_id)

        entries, entry_ids = await self._store_graph_structure(
            source_memory_id,
            content,
            metadata,
            atoms,
        )
        if not entries:
            return

        # 以单条来源记忆为边界批量写入图向量：一次聚合嵌入 + 一次 FAISS 快照
        vector_doc_id = await self.graph_vector_retriever.add_memory_entries(
            [(entry.content, dict(entry.metadata)) for entry in entries]
        )
        await self.graph_store.update_entry_vector_doc_ids(
            {entry_ids[0]: vector_doc_id}
        )

    async def _store_graph_structure(
        self,
        source_memory_id: int,
        content: str,
        metadata: dict[str, Any] | None,
        atoms: list | None = None,
    ) -> tuple[list[GraphEntry], list[int]]:
        """Persist graph structure without touching the vector index."""
        extracted = self.graph_extractor.extract(
            source_memory_id, content, metadata, atoms
        )
        if not extracted.entries:
            return [], []

        node_key_to_id = await self.graph_store.upsert_nodes(extracted.nodes)

        edge_key_to_id = await self.graph_store.add_edges(
            extracted.edges,
            node_key_to_id,
        )

        entry_ids = await self.graph_store.add_entries(
            extracted.entries,
            node_key_to_id,
            edge_key_to_id,
        )
        if len(entry_ids) != len(extracted.entries):
            raise RuntimeError(
                "graph entry id count mismatch: "
                f"ids={len(entry_ids)}, entries={len(extracted.entries)}"
            )
        return extracted.entries, entry_ids

    async def delete_memory(self, source_memory_id: int) -> None:
        """Delete graph artifacts belonging to one source memory."""
        await self._delete_memory_now(source_memory_id)

    async def _delete_memory_now(self, source_memory_id: int) -> None:
        vector_doc_ids = await self.graph_store.delete_memory(source_memory_id)
        await self.graph_vector_retriever.delete_entries(
            source_memory_id, vector_doc_ids
        )

    async def batch_delete_memories(self, source_memory_ids: list[int]) -> None:
        """Batch delete graph artifacts for multiple source memories."""
        if not source_memory_ids:
            return
        memory_vec_map = await self.graph_store.batch_delete_memories(source_memory_ids)
        await self.graph_vector_retriever.delete_entries_batch(memory_vec_map)


__all__ = ["GraphMemoryManager"]
