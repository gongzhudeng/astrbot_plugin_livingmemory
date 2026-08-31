"""供 Agent 主动调用的长期记忆写入工具。"""

import asyncio
import json
from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass

from astrbot.api import logger
from astrbot.api.platform import MessageType
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext

from ..utils import get_persona_id


def _json_result(data: dict[str, Any]) -> str:
    """将工具结果稳定序列化为 JSON 文本。"""
    return json.dumps(data, ensure_ascii=False, default=str)


def _normalize_list(value: Any, limit: int = 5) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()][:limit]
    if isinstance(value, str) and value.strip():
        return [value.strip()][:limit]
    return []


@dataclass
class MemoryMemorizeTool(FunctionTool[AstrAgentContext]):
    """长期记忆主动写入工具。"""

    __pydantic_config__ = {"arbitrary_types_allowed": True}

    context: Any = None
    memory_engine: Any = None
    memory_processor: Any = None

    name: str = "memorize_long_term_memory"
    description: str = (
        "用户明确要求记住，或出现稳定偏好、身份信息、约定或项目背景时，写入长期记忆。"
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "memory": {
                    "type": "string",
                    "description": "要保存的精炼、可长期复用的事实记忆。只写稳定偏好、身份信息、约定、承诺或项目背景，不要复制整段对话，也不要写临时闲聊或冗长推理。",
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "可选的简短主题标签，最多 5 个；使用便于后续检索的实体或主题词。",
                    "default": [],
                },
                "key_facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "可选的关键事实，最多 5 条；保留支撑该记忆的重要原始事实，不要复述完整对话。",
                    "default": [],
                },
                "sentiment": {
                    "type": "string",
                    "description": "记忆情感：positive、neutral 或 negative。",
                    "default": "neutral",
                },
                "importance": {
                    "type": "number",
                    "description": "重要度 0.0 至 1.0。稳定偏好、身份事实、明确承诺或长期项目背景使用较高值；普通补充信息使用较低值。",
                    "default": 0.7,
                },
                "reason": {
                    "type": "string",
                    "description": "可选；简短说明为何值得长期记住，不要重复 memory 正文。",
                    "default": "",
                },
            },
            "required": ["memory"],
        }
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        memory: str,
        topics: list[str] | None = None,
        key_facts: list[str] | None = None,
        sentiment: str = "neutral",
        importance: float = 0.7,
        reason: str = "",
    ) -> ToolExecResult:
        """执行长期记忆写入。"""
        cleaned_memory = (memory or "").strip()
        if not cleaned_memory:
            return _json_result({"memorized": False, "error": "memory is empty"})

        normalized_sentiment = str(sentiment or "neutral").strip().lower()
        if normalized_sentiment not in {"positive", "neutral", "negative"}:
            normalized_sentiment = "neutral"

        if (
            self.context is None
            or self.memory_engine is None
            or self.memory_processor is None
        ):
            return _json_result(
                {
                    "memorized": False,
                    "error": "memory memorize tool is not initialized",
                }
            )

        try:
            event = context.context.event
            session_id = event.unified_msg_origin
            persona_id = await get_persona_id(self.context, event)
            is_group_chat = event.get_message_type() == MessageType.GROUP_MESSAGE

            structured_data = {
                "summary": cleaned_memory,
                "topics": _normalize_list(topics),
                "key_facts": _normalize_list(key_facts),
                "sentiment": normalized_sentiment,
                "importance": importance,
            }

            content, metadata, normalized_importance = (
                self.memory_processor.build_memory_from_structured_data(
                    structured_data=structured_data,
                    is_group_chat=is_group_chat,
                    fallback_excerpt=cleaned_memory,
                )
            )
            metadata["source_window"] = {
                "session_id": session_id,
                "triggered_by": "agent_tool",
                "tool_name": self.name,
            }
            metadata["memory_origin"] = "agent_memorize_tool"
            cleaned_reason = (reason or "").strip()
            if cleaned_reason:
                metadata["memorize_reason"] = cleaned_reason

            memory_id = await self.memory_engine.add_memory(
                content=content,
                session_id=session_id,
                persona_id=persona_id,
                importance=normalized_importance,
                metadata=metadata,
            )

            return _json_result(
                {
                    "memorized": True,
                    "id": memory_id,
                    "content": content,
                    "importance": normalized_importance,
                    "session_id": session_id,
                    "persona_id": persona_id,
                }
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"记忆工具写入失败: {e}", exc_info=True)
            return _json_result({"memorized": False, "error": "internal_error"})
