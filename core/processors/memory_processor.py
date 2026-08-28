"""
记忆处理器 - 使用LLM将对话历史处理为结构化记忆
"""

import asyncio
import inspect
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from astrbot.api import logger

from ..models.conversation_models import Message
from ..models.memory_atom import MemoryAtom
from .atom_classifier import classify_atoms

EMOTIONAL_FACT_MAX_CHARS = 600


def _bound_complete_text(text: str, max_chars: int) -> str:
    """Bound generated text at a readable punctuation boundary."""
    clean = str(text or "").strip()
    limit = max(2, int(max_chars))
    if len(clean) <= limit:
        return clean

    window = clean[: limit - 1].rstrip()
    minimum_boundary = min(40, max(1, limit // 3))
    for pattern, strip_chars in (
        (re.compile(r"[。！？!?；;](?:[\"'”’」』】）)]*)"), ""),
        (re.compile(r"[，,、：:](?:[\"'”’」』】）)]*)"), "，,、：:"),
    ):
        matches = list(pattern.finditer(window))
        if matches and matches[-1].end() >= minimum_boundary:
            prefix = window[: matches[-1].end()].rstrip(strip_chars).rstrip()
            return f"{prefix}…"
    return f"{window}…"


class MemoryProcessor:
    """
    记忆处理器

    使用LLM将对话历史转换为结构化记忆。
    支持私聊和群聊两种场景的不同处理策略。
    """

    def __init__(
        self,
        context=None,
        llm_provider: Any = None,
        fallback_provider_ids: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        初始化记忆处理器

        Args:
            context: AstrBot上下文,用于获取人格管理器
            llm_provider: LLM Provider 实例或 Provider ID 字符串。
                          传入实例时直接使用（测试用）；传入字符串时动态解析。
                          留空则使用AstrBot默认Provider。
            fallback_provider_ids: 首选 Provider 失败后按顺序尝试的 Provider ID。
            config: 记忆处理器配置。
        """
        self.context = context
        self._llm_provider = llm_provider
        self._fallback_provider_ids = [
            str(provider_id).strip()
            for provider_id in (fallback_provider_ids or [])
            if str(provider_id).strip()
        ]
        self.config = config or {}

        # 加载提示词模板
        self._load_prompts()

    @staticmethod
    def _provider_name(provider: Any) -> str:
        try:
            return str(provider.meta().id)
        except Exception:
            return type(provider).__name__

    def _summary_model_limits(self) -> tuple[int, float]:
        """Return additional retries and per-call timeout for memory summaries."""
        raw_retries = self.config.get("summary_model_max_retries", 1)
        try:
            retries = int(raw_retries)
        except (TypeError, ValueError):
            retries = 1
        raw_timeout = self.config.get("summary_model_timeout_seconds", 120)
        try:
            timeout = float(raw_timeout)
        except (TypeError, ValueError):
            timeout = 120.0
        return max(0, min(10, retries)), max(1.0, min(3600.0, timeout))

    @staticmethod
    def _supports_request_max_retries(provider: Any) -> bool:
        """Detect lightweight test providers that expose no transport option."""
        target = getattr(provider, "text_chat", None)
        side_effect = getattr(target, "side_effect", None)
        if callable(side_effect):
            target = side_effect
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            return True
        return "request_max_retries" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    async def _text_chat_once(
        self,
        provider: Any,
        *,
        prompt: str,
        system_prompt: str,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "system_prompt": system_prompt,
        }
        if self._supports_request_max_retries(provider):
            kwargs["request_max_retries"] = 1
        return await provider.text_chat(**kwargs)

    def _get_current_llm_providers(self) -> list[Any]:
        """Resolve the ordered provider chain without retaining stale instances."""
        if not self.context:
            if self._llm_provider is not None and not isinstance(
                self._llm_provider, str
            ):
                return [self._llm_provider]
            return []

        providers: list[Any] = []
        seen: set[int] = set()

        def append_provider(provider: Any) -> None:
            if provider is None or id(provider) in seen:
                return
            seen.add(id(provider))
            providers.append(provider)

        if self._llm_provider is not None and not isinstance(self._llm_provider, str):
            append_provider(self._llm_provider)
        elif isinstance(self._llm_provider, str) and self._llm_provider:
            try:
                append_provider(self.context.get_provider_by_id(self._llm_provider))
            except Exception as e:
                logger.warning(
                    f"[MemoryProcessor] 首选 LLM Provider 不可用 "
                    f"({self._llm_provider}): {e}"
                )
        else:
            try:
                append_provider(self.context.get_using_provider())
            except Exception as e:
                logger.warning(f"[MemoryProcessor] 默认 LLM Provider 不可用: {e}")

        for provider_id in self._fallback_provider_ids:
            try:
                provider = self.context.get_provider_by_id(provider_id)
            except Exception as e:
                logger.warning(
                    f"[MemoryProcessor] 兜底 LLM Provider 不可用 ({provider_id}): {e}"
                )
                continue
            if provider is None:
                logger.warning(
                    f"[MemoryProcessor] 未找到兜底 LLM Provider: {provider_id}"
                )
                continue
            append_provider(provider)

        return providers

    def _get_current_llm_provider(self):
        """Return the first provider for backward-compatible internal callers."""
        providers = self._get_current_llm_providers()
        return providers[0] if providers else None

    def _load_prompts(self) -> None:
        """从外部文件加载提示词模板"""
        prompt_dir = Path(__file__).parent.parent / "prompts"

        try:
            # 加载私聊提示词
            private_prompt_file = prompt_dir / "private_chat_prompt.txt"
            with open(private_prompt_file, encoding="utf-8") as f:
                self.private_chat_prompt = f.read()

            # 加载群聊提示词
            group_prompt_file = prompt_dir / "group_chat_prompt.txt"
            with open(group_prompt_file, encoding="utf-8") as f:
                self.group_chat_prompt = f.read()

            logger.info("[MemoryProcessor] 提示词模板加载成功")

        except Exception as e:
            logger.error(f"[MemoryProcessor] 加载提示词模板失败: {e}")
            # 使用简单的后备提示词（注意：使用 replace 替换，无需转义大括号）
            self.private_chat_prompt = """分析以下对话并生成JSON格式的记忆:
{conversation}

输出格式:
{"summary": "摘要", "topics": ["主题"], "key_facts": ["事实"], "sentiment": "neutral", "importance": 0.5}
"""
            self.group_chat_prompt = """分析以下群聊对话并生成JSON格式的记忆:
{conversation}

输出格式:
{"summary": "摘要", "topics": ["主题"], "key_facts": ["事实"], "participants": ["参与者"], "sentiment": "neutral", "importance": 0.5}
"""

    async def _build_system_prompt_with_persona(self, persona_id: str | None) -> str:
        """
        构建包含人格提示的 system_prompt

        Args:
            persona_id: 人格ID

        Returns:
            str: 包含人格提示的 system_prompt
        """
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        base_prompt = (
            "你正在总结对话记忆。请严格按照JSON格式输出。\n"
            f"当前日期时间: {current_date}\n"
            "重要: 请将对话中出现的相对时间表达（如\u201c今天\u201d、\u201c明天\u201d、\u201c昨天\u201d、\u201c下周\u201d、\u201c上个月\u201d等）"
            "转换为具体日期后再写入记忆，以便未来查阅时仍能准确理解时间信息。"
        )

        if not persona_id:
            logger.debug("[MemoryProcessor] 未指定人格ID，使用基础提示词")
            return base_prompt

        if not self.context:
            logger.debug("[MemoryProcessor] Context 未设置，使用基础提示词")
            return base_prompt

        try:
            persona_manager = getattr(self.context, "persona_manager", None)
            if not persona_manager:
                logger.warning(
                    "[MemoryProcessor] persona_manager 不可用，使用基础提示词"
                )
                return base_prompt

            persona = await persona_manager.get_persona(persona_id)
            if not persona:
                logger.warning(
                    f"[MemoryProcessor] 人格 '{persona_id}' 不存在，使用基础提示词"
                )
                return base_prompt

            if not persona.system_prompt:
                logger.debug(
                    f"[MemoryProcessor] 人格 '{persona_id}' 无 system_prompt，使用基础提示词"
                )
                return base_prompt

            persona_prompt = persona.system_prompt.strip()
            if not persona_prompt:
                logger.debug(
                    f"[MemoryProcessor] 人格 '{persona_id}' 的 system_prompt 为空，使用基础提示词"
                )
                return base_prompt

            logger.info(
                f"[MemoryProcessor] 成功加载人格 '{persona_id}' 的提示词 "
                f"(长度={len(persona_prompt)}字符)"
            )
            logger.debug(f"[MemoryProcessor] 人格提示词预览: {persona_prompt[:100]}...")

            enhanced_prompt = (
                f"{base_prompt}\n\n"
                f"## 你的人格设定\n"
                f"{persona_prompt}\n\n"
                f"## 记忆总结要求\n"
                f"在总结对话记忆时,你需要:\n"
                f"1. **保持你的人格特色**: 使用符合上述人格设定的语气、用词习惯和表达方式\n"
                f'2. **第一人称视角**: 以"我"的视角回顾对话,不要说"bot"、"助手"等第三人称\n'
                f"3. **体现你的关注点**: 根据你的人格特点,侧重记录你会关注的信息\n"
                f"4. **自然真实**: 让记忆读起来像是你本人在回忆这段对话,而不是机械的客观描述\n"
                f"5. **时间转换**: 将对话中的相对时间（今天、明天、下周等）转换为具体日期（当前日期: {current_date}）\n\n"
                f"例如:\n"
                f'- 如果你是活泼可爱的性格,记忆中可以使用"呀"、"呢"、"~"等语气词\n'
                f"- 如果你是专业严谨的性格,记忆应该用词准确、逻辑清晰、格式规范\n"
                f"- 如果你是幽默风趣的性格,记忆中可以包含轻松的表达和有趣的观察"
            )

            return enhanced_prompt

        except ValueError as e:
            logger.warning(f"[MemoryProcessor] 人格 '{persona_id}' 不存在: {e}")
            return base_prompt
        except Exception as e:
            logger.error(
                f"[MemoryProcessor] 获取人格提示词时发生错误: {e}", exc_info=True
            )
            return base_prompt

    @staticmethod
    def _response_has_structured_payload(response_text: str) -> bool:
        """Return whether a response contains a recognizable memory object."""
        if not isinstance(response_text, str) or not response_text.strip():
            return False

        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        candidates = [cleaned, *re.findall(r"\{.*?\}", cleaned, re.DOTALL)]
        expected_fields = {"summary", "topics", "key_facts", "importance"}
        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(data, dict) and expected_fields.intersection(data):
                return True
        return False

    async def _call_provider_with_retry(
        self,
        provider: Any,
        prompt: str,
        system_prompt: str,
        max_retries: int | None = None,
    ) -> str:
        provider_name = self._provider_name(provider)
        last_error: Exception | None = None
        if max_retries is None:
            additional_retries, timeout_seconds = self._summary_model_limits()
        else:
            # Preserve the old direct-call API: an explicit value represented
            # total attempts. Plugin config uses additional retry semantics.
            try:
                additional_retries = max(0, int(max_retries) - 1)
            except (TypeError, ValueError):
                additional_retries = 0
            _, timeout_seconds = self._summary_model_limits()
        max_attempts = additional_retries + 1
        for attempt in range(max_attempts):
            started = asyncio.get_running_loop().time()
            try:
                response = await asyncio.wait_for(
                    self._text_chat_once(
                        provider,
                        prompt=prompt,
                        system_prompt=system_prompt,
                    ),
                    timeout=timeout_seconds,
                )
                response_text = getattr(response, "completion_text", "")
                if not self._response_has_structured_payload(response_text):
                    raise ValueError("模型未返回可识别的结构化记忆")
                logger.info(
                    f"[MemoryProcessor] LLM Provider {provider_name} 总结成功，"
                    f"尝试 {attempt + 1}/{max_attempts}，耗时 "
                    f"{asyncio.get_running_loop().time() - started:.1f}s"
                )
                return response_text
            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"Provider {provider_name} 总结调用超过 {timeout_seconds:.1f}s"
                )
            except Exception as e:
                last_error = e
            if attempt == max_attempts - 1:
                break
            wait_time = (2**attempt) + random.uniform(0, 1)
            logger.warning(
                f"[MemoryProcessor] LLM Provider {provider_name} 调用失败，"
                f"{wait_time:.1f}s 后重试 ({attempt + 1}/{additional_retries}): {last_error}"
            )
            await asyncio.sleep(wait_time)
        raise last_error or RuntimeError("LLM 调用失败，未捕获到具体异常")

    async def _call_llm_with_retry(
        self, prompt: str, system_prompt: str, max_retries: int | None = None
    ) -> str:
        """Try each configured provider in order, with configured retries per provider."""
        providers = self._get_current_llm_providers()
        if not providers:
            raise RuntimeError("LLM Provider 不可用")

        last_error: Exception | None = None
        for index, provider in enumerate(providers):
            provider_name = self._provider_name(provider)
            try:
                response_text = await self._call_provider_with_retry(
                    provider,
                    prompt,
                    system_prompt,
                    max_retries,
                )
                if index > 0:
                    logger.info(
                        f"[MemoryProcessor] 兜底 LLM Provider {provider_name} 总结成功"
                    )
                return response_text
            except Exception as e:
                last_error = e
                if index < len(providers) - 1:
                    logger.warning(
                        f"[MemoryProcessor] LLM Provider {provider_name} 总结失败，"
                        f"尝试下一个兜底模型: {e}"
                    )
                else:
                    logger.error(
                        f"[MemoryProcessor] LLM Provider {provider_name} 总结失败，"
                        f"已无可用兜底模型: {e}"
                    )

        raise last_error or RuntimeError("所有 LLM Provider 均总结失败")

    def _try_fix_json(self, text: str) -> str:
        """
        尝试修复损坏的 JSON 字符串

        Args:
            text: 可能损坏的 JSON 字符串

        Returns:
            修复后的 JSON 字符串
        """
        fixed = text.strip()

        # 移除 markdown 代码块标记
        if fixed.startswith("```json"):
            fixed = fixed[7:]
        elif fixed.startswith("```"):
            fixed = fixed[3:]
        if fixed.endswith("```"):
            fixed = fixed[:-3]
        fixed = fixed.strip()

        # 修复未闭合的字符串（截断的 JSON）
        open_quotes = fixed.count('"') - fixed.count('\\"')
        if open_quotes % 2 != 0:
            fixed += '"'

        # 修复未闭合的数组
        open_brackets = fixed.count("[") - fixed.count("]")
        if open_brackets > 0:
            fixed += "]" * open_brackets

        # 修复未闭合的对象
        open_braces = fixed.count("{") - fixed.count("}")
        if open_braces > 0:
            fixed += "}" * open_braces

        # 移除尾部逗号（JSON 不允许）
        fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)

        # 修复常见的转义问题
        fixed = fixed.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

        return fixed

    async def process_conversation(
        self,
        messages: list[Message],
        is_group_chat: bool = False,
        persona_id: str | None = None,
        emotion_review_context: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any], float]:
        """
        处理对话历史,生成结构化记忆

        Args:
            messages: 消息列表(Message对象)
            is_group_chat: 是否为群聊
            persona_id: 人格ID,用于获取人格提示词

        Returns:
            tuple: (content, metadata, importance)
                - content: 格式化的记忆内容字符串
                - metadata: 包含结构化信息的字典
                - importance: 重要性评分(0-1)

        Raises:
            Exception: 处理失败时抛出异常
        """
        if not messages:
            raise ValueError("消息列表不能为空")

        # 1. 格式化对话历史
        conversation_text = self._format_conversation(messages)
        emotion_source_has_text = self._has_emotion_source_text(messages)

        # 2. 选择合适的提示词模板
        # 使用 replace 而非 format，避免对话内容中的大括号导致解析错误
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        if is_group_chat:
            prompt = self.group_chat_prompt.replace("{conversation}", conversation_text)
        else:
            prompt = self.private_chat_prompt.replace(
                "{conversation}", conversation_text
            )
        if not is_group_chat and emotion_review_context:
            review_json = json.dumps(
                emotion_review_context, ensure_ascii=False, default=str
            )[:6000]
            prompt += (
                "\n\n# 当前角色心理事件与待关注事项复核目录（只读）\n"
                "请优先复核或合并同主题情绪事件；普通生活和互动可作为episodic近期片段。"
                "复核既有情绪事件必须原样使用 event_id 与 event_version，"
                "复核既有待关注事项必须原样使用 item_id 与 item_version，"
                "不要凭空创造任何ID。"
                "对已有事项的询问、引用、复述或纠错不能创建新事项；"
                "用户明确指出事项记错、已经完成或不再需要时，必须对目录中的原ID执行"
                "cancel、complete或supersede。AI自己的错误建议和未获用户确认的临时承诺"
                "不能成为新事项。\n"
                f"{review_json}"
            )
        # 注入当前日期，让 LLM 能将相对时间转换为绝对日期
        prompt = prompt.replace("{current_date}", current_date)

        # 3. 调用LLM生成结构化记忆
        conversation_type = "群聊" if is_group_chat else "私聊"
        try:
            logger.info(
                f"[MemoryProcessor] 准备调用 LLM，对话类型={conversation_type}, 消息数={len(messages)}"
            )
            logger.debug(f"[MemoryProcessor] Prompt 模板长度={len(prompt)}")
            logger.debug(
                f"[MemoryProcessor] 发送给LLM的对话内容（前500字符）:\n{conversation_text[:500]}"
            )

            # 构建 system_prompt，嵌入人格提示
            system_prompt = await self._build_system_prompt_with_persona(persona_id)
            logger.debug(f"[MemoryProcessor] System Prompt: {system_prompt[:200]}...")

            llm_response_text = await self._call_llm_with_retry(
                prompt=prompt,
                system_prompt=system_prompt,
            )

            logger.info(
                f"[MemoryProcessor]  LLM 响应成功，响应长度={len(llm_response_text)}"
            )
            logger.debug(f"[MemoryProcessor] LLM 原始响应内容:\n{llm_response_text}")

            # 4. 解析LLM响应
            structured_data = self._parse_llm_response(llm_response_text, is_group_chat)
            if not is_group_chat:
                structured_data["emotional_observations"] = (
                    self._filter_emotional_observations_by_source(
                        structured_data.get("emotional_observations", []), messages
                    )
                )
                structured_data["attention_observations"] = (
                    self._filter_attention_observations_by_source(
                        structured_data.get("attention_observations", []), messages
                    )
                )

            # 4.5 质量校验
            quality = self._validate_summary_quality(structured_data)
            if quality == "low":
                logger.warning(
                    "[MemoryProcessor] 总结质量不达标（low），将标记但仍写入"
                )
            structured_data["_quality"] = quality

            # 5. 构建存储格式
            fallback_excerpt = (
                conversation_text[:200] + "..."
                if len(conversation_text) > 200
                else conversation_text
            )
            content, metadata = self._build_storage_format(
                fallback_excerpt, structured_data, is_group_chat
            )
            # 稳定人物身份：以平台+发送者ID写入参与者身份，图谱据此复用节点
            metadata["participant_identities"] = self._extract_participant_identities(
                messages
            )
            # 将质量标记写入 metadata
            metadata["summary_quality"] = structured_data.get("_quality", "normal")
            metadata["emotion_source_has_text"] = emotion_source_has_text

            importance = float(structured_data.get("importance", 0.5))

            logger.info(
                f"[MemoryProcessor]  成功生成结构化记忆: 摘要={structured_data.get('summary', '')[:50]}..., "
                f"主题={structured_data.get('topics', [])}, "
                f"重要性={importance}, 类型={conversation_type}"
            )
            logger.debug(
                f"[MemoryProcessor] 生成的记忆内容（前200字符）:\n{content[:200]}"
            )

            return content, metadata, importance

        except Exception as e:
            logger.error(f"[MemoryProcessor] 处理对话历史失败: {e}", exc_info=True)
            # 不再降级处理，直接向上抛出异常，由调用方处理重试逻辑
            raise

    def _format_conversation(self, messages: list[Message]) -> str:
        """
        格式化对话历史为文本

        Args:
            messages: 消息列表(Message对象)

        Returns:
            格式化后的对话文本
        """

        formatted_lines = []
        for i, msg in enumerate(messages):
            logger.debug(
                f"[_format_conversation] 消息#{i}: "
                f"sender_id={msg.sender_id}, sender_name={msg.sender_name}, "
                f"role={msg.role}, group_id={msg.group_id}"
            )

            content_text = self._message_content_to_text(msg.content)
            sender_info = self._format_sender_info(msg)
            formatted_line = f"{sender_info} {content_text}".rstrip()
            formatted_lines.append(formatted_line)
            if msg.group_id:
                logger.debug(
                    f"[_format_conversation] 消息#{i} 格式化结果(群聊): {formatted_line[:100]}..."
                )
            else:
                logger.debug(
                    f"[_format_conversation] 消息#{i} 格式化结果(私聊): {sender_info[:50]}..."
                )
        return "\n".join(formatted_lines)

    @staticmethod
    def _is_attack_observation(observation: dict[str, Any]) -> bool:
        tags = {
            str(tag).strip().lower()
            for tag in observation.get("tags", [])
            if str(tag).strip()
        }
        text = " ".join(
            str(observation.get(field, ""))
            for field in ("fact", "emotional_meaning", "evidence_quote")
        )
        return bool(
            tags & {"abuse", "attack", "insult", "negative_attack"}
            or re.search(r"恶心|滚开|去死|废物|臭烘烘|臭死|烦死|讨厌", text)
        )

    @staticmethod
    def _infer_attack_target(evidence_quote: str) -> tuple[str, str]:
        clean = re.sub(r"\s+", " ", str(evidence_quote or "")).strip()
        if not re.search(r"恶心|滚开|去死|废物|臭烘烘|臭死|烦死|讨厌", clean):
            return "unknown", "no_attack_evidence"
        third_party = re.compile(
            r"(?:地铁|车站|公交|路上|公司|学校|医院|店里|网上|评论区|新闻|"
            r"群里|视频里|照片里|别人|他人|某个|一个|那个|这位|那位).{0,32}"
            r"(?:老头|老太太|男人|女人|男的|女的|女生|男生|那个人|某人|路人|"
            r"乘客|店员|同事|老板|孩子|小孩|人)"
            r"|(?:老头|老太太|男人|女人|男的|女的|女生|男生|那个人|某人|路人|"
            r"乘客|店员|同事|老板|孩子|小孩).{0,24}(?:身上|很|太|真|特别|恶心|臭|烦|讨厌)"
            r"|(?:碰到|遇到|看见|看到|闻到).{0,32}(?:恶心|臭烘烘|臭死|讨厌)"
        )
        if third_party.search(clean):
            return "third_party", "explicit_third_party_subject"
        if re.search(
            r"(?:^|[\s，。！？!?；;])你(?:这个|这|真|太|好|怎么|可真|真的)?",
            clean,
        ):
            return "user", "explicit_user_subject"
        return "unknown", "no_explicit_subject"

    @classmethod
    def _filter_emotional_observations_by_source(
        cls,
        observations: Any,
        messages: list[Message],
    ) -> list[dict[str, Any]]:
        """Accept hostile relationship events only with exact direct user evidence."""
        if not isinstance(observations, list):
            return []
        user_messages = [
            message
            for message in messages
            if message.role != "assistant"
            and not message.metadata.get("is_bot_message", False)
        ]
        accepted: list[dict[str, Any]] = []
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            if not cls._is_attack_observation(observation):
                accepted.append(observation)
                continue

            quote = str(observation.get("evidence_quote", "")).strip()
            speaker = str(observation.get("evidence_speaker", "")).strip().lower()
            target = str(observation.get("target", "unknown")).strip().lower()
            if not quote or speaker != "user" or target != "user":
                continue
            direct_evidence = False
            for message in user_messages:
                text = cls._message_content_to_text(message.content).strip()
                quoted_texts = [
                    str(item).strip()
                    for item in message.metadata.get("quoted_texts", [])
                    if str(item).strip()
                ]
                direct_text = text
                for quoted in quoted_texts:
                    direct_text = direct_text.replace(quoted, " ")
                direct_text = re.sub(r"\[引用(?::|消息).*?\]", " ", direct_text)
                if quote in direct_text:
                    direct_evidence = True
                    break
            if not direct_evidence:
                continue
            inferred_target, target_basis = cls._infer_attack_target(quote)
            if inferred_target != "user":
                continue
            normalized = dict(observation)
            normalized["target"] = "user"
            normalized["target_basis"] = target_basis
            accepted.append(normalized)
        return accepted

    @classmethod
    def _filter_attention_observations_by_source(
        cls,
        observations: Any,
        messages: list[Message],
    ) -> list[dict[str, Any]]:
        """Require exact user evidence and reject clearly short-lived or quoted tasks."""
        if not isinstance(observations, list):
            return []
        speaker_texts = {
            "user": [
                cls._message_content_to_text(message.content).strip()
                for message in messages
                if message.role != "assistant"
                and not message.metadata.get("is_bot_message", False)
            ],
            "assistant": [
                cls._message_content_to_text(message.content).strip()
                for message in messages
                if message.role == "assistant"
                or message.metadata.get("is_bot_message", False)
            ],
        }
        speaker_texts["both"] = speaker_texts["user"] + speaker_texts["assistant"]
        immediate = re.compile(
            r"(?:等一下|一会儿|待会儿|马上|现在|正在).{0,28}"
            r"(?:发|拍|聊|说|做|去|给|看|回复)"
        )
        quoted_or_corrected = re.compile(
            r"(?:有个|这个|那个).{0,12}(?:计划|约定|事项).{0,20}"
            r"(?:能看见|看得到|还在吗)|"
            r"(?:插件|模型|你).{0,16}(?:记错|说错|弄错|发癫)|"
            r"(?:不是|并非).{0,8}(?:计划|约定)"
        )
        assistant_noise = re.compile(
            r"(?:记错|说错|弄错|发癫|猜测|猜想|可能|也许|不确定|临时承诺)"
        )
        accepted: list[dict[str, Any]] = []
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            quote = str(observation.get("evidence_quote", "")).strip()
            speaker = str(observation.get("evidence_speaker", "")).strip().lower()
            if speaker not in speaker_texts or not quote:
                continue
            if not any(quote in text for text in speaker_texts[speaker]):
                continue
            if observation.get("action") == "create":
                status = str(observation.get("status", "open")).strip().lower()
                evidence = f"{quote} {observation.get('content', '')}"
                if immediate.search(evidence) or quoted_or_corrected.search(evidence):
                    continue
                if status == "proposed":
                    if speaker == "assistant" and assistant_noise.search(evidence):
                        continue
                elif speaker != "user":
                    continue
            elif observation.get("action") == "complete":
                if speaker not in {"user", "assistant", "both"}:
                    continue
            elif speaker != "user":
                continue
            accepted.append(observation)
        return accepted

    @classmethod
    def _has_emotion_source_text(cls, messages: list[Message]) -> bool:
        placeholders = re.compile(r"\[(?:图片|视频|语音|文件)(?:消息|:[^\]]*)?\]", re.I)
        for message in messages:
            is_bot = message.role == "assistant" or message.metadata.get(
                "is_bot_message", False
            )
            if is_bot:
                continue
            clean = cls._message_content_to_text(message.content)
            clean = re.sub(
                r"<!--\s*astrbot-chat-merger:image-context(?::[^>]*)?-->.*?"
                r"<!--\s*/?astrbot-chat-merger:image-context(?::[^>]*)?-->",
                " ",
                clean,
                flags=re.DOTALL | re.I,
            )
            clean = re.sub(
                r"<!--\s*astrbot-chat-merger:image-context(?::[^>]*)?-->.*$",
                " ",
                clean,
                flags=re.DOTALL | re.I,
            )
            clean = re.sub(
                r"<image_context\b[^>]*>.*?</image_context>",
                " ",
                clean,
                flags=re.DOTALL | re.I,
            )
            clean = re.sub(
                r"<image_context\b[^>]*>.*$",
                " ",
                clean,
                flags=re.DOTALL | re.I,
            )
            clean = placeholders.sub(" ", clean)
            if re.sub(r"[\W_]+", "", clean, flags=re.UNICODE):
                return True
        return False

    @staticmethod
    def _format_sender_info(msg: Message) -> str:
        time_str = datetime.fromtimestamp(msg.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        display_name = msg.sender_name if msg.sender_name else msg.sender_id or "未知"
        is_bot = msg.metadata.get("is_bot_message", False) or msg.role == "assistant"
        if is_bot:
            return f"[Bot: {display_name} | ID: {msg.sender_id} | {time_str}]"
        return f"[{display_name} | ID: {msg.sender_id} | {time_str}]"

    @staticmethod
    def _extract_participant_identities(
        messages: list[Message],
    ) -> list[dict[str, Any]]:
        """Build stable graph identities from message sender IDs, not LLM names."""
        identities: dict[str, dict[str, Any]] = {}
        for message in messages:
            if message.role == "system":
                continue
            sender_id = str(message.sender_id or "").strip()
            if not sender_id:
                continue
            platform = str(message.platform or "unknown").strip().lower() or "unknown"
            identity_key = f"{platform}:{sender_id}"
            display_name = str(message.sender_name or sender_id).strip() or sender_id
            is_bot = bool(
                message.metadata.get("is_bot_message", False)
                or message.role == "assistant"
            )

            identity = identities.setdefault(
                identity_key,
                {
                    "identity_key": identity_key,
                    "sender_id": sender_id,
                    "platform": platform,
                    "display_name": display_name,
                    "aliases": [],
                    "is_bot": is_bot,
                },
            )
            identity["display_name"] = display_name
            identity["is_bot"] = bool(identity["is_bot"] or is_bot)
            if display_name not in identity["aliases"]:
                identity["aliases"].append(display_name)

        return list(identities.values())

    @classmethod
    def _message_content_to_text(cls, content: Any) -> str:
        return Message.content_to_text(content)

    @classmethod
    def _message_part_to_text(cls, part: Any) -> tuple[str, bool]:
        return Message._content_part_to_text(part)

    def _parse_llm_response(
        self, response_text: str, is_group_chat: bool
    ) -> dict[str, Any]:
        """
        解析LLM响应,提取JSON数据

        Args:
            response_text: LLM响应文本
            is_group_chat: 是否为群聊

        Returns:
            解析后的字典数据
        """
        logger.debug(f"[MemoryProcessor] 开始解析 LLM 响应，长度={len(response_text)}")

        try:
            # 尝试直接解析JSON
            # 先清理可能的markdown代码块标记
            cleaned_text = response_text.strip()
            logger.debug(
                f"[MemoryProcessor] 清理前的响应文本（前100字符）: {response_text[:100]}"
            )

            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
                logger.debug("[MemoryProcessor] 移除了 ```json 标记")
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
                logger.debug("[MemoryProcessor] 移除了 ``` 标记")
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
                logger.debug("[MemoryProcessor] 移除了结尾 ``` 标记")
            cleaned_text = cleaned_text.strip()

            logger.debug(
                f"[MemoryProcessor] 清理后准备解析的 JSON（前500字符）:\n{cleaned_text[:500]}"
            )

            # 解析JSON
            data = json.loads(cleaned_text)

            # 类型检查：确保解析结果是 dict
            if not isinstance(data, dict):
                logger.warning(
                    f"[MemoryProcessor] JSON 解析结果不是 dict，类型为 {type(data).__name__}"
                )
                raise ValueError(f"期望 dict 类型，实际为 {type(data).__name__}")

            logger.info("[MemoryProcessor] JSON 解析成功")
            logger.debug(f"[MemoryProcessor] 解析得到的字段: {list(data.keys())}")

            # 验证必需字段 - 简化后的字段列表
            required_fields = [
                "summary",
                "topics",
                "key_facts",
                "sentiment",
                "importance",
            ]
            if is_group_chat:
                required_fields.append("participants")

            for field in required_fields:
                if field not in data:
                    logger.warning(
                        f"[MemoryProcessor] LLM 响应缺少字段: {field}, 使用默认值"
                    )
                    data[field] = self._get_default_value(field)

            # 数据类型校验和规范化
            data["summary"] = str(data.get("summary", ""))
            logger.debug(f"[MemoryProcessor] 提取 summary: {data['summary'][:100]}...")

            data["topics"] = self._ensure_list(data.get("topics", []))[:5]
            logger.debug(
                f"[MemoryProcessor] 提取 topics ({len(data['topics'])} 个): {data['topics']}"
            )

            data["key_facts"] = self._ensure_list(data.get("key_facts", []))[:5]
            logger.debug(
                f"[MemoryProcessor] 提取 key_facts ({len(data['key_facts'])} 个): {data['key_facts']}"
            )

            data["sentiment"] = self._validate_sentiment(
                data.get("sentiment", "neutral")
            )
            logger.debug(f"[MemoryProcessor] 提取 sentiment: {data['sentiment']}")

            data["importance"] = self._validate_importance(data.get("importance", 0.5))
            logger.debug(f"[MemoryProcessor] 提取 importance: {data['importance']}")

            data["emotional_observations"] = self._normalize_emotional_observations(
                data.get("emotional_observations", []), is_group_chat
            )
            data["attention_observations"] = self._normalize_attention_observations(
                data.get("attention_observations", []), is_group_chat
            )
            data["mood_adjustment"] = self._normalize_mood_adjustment(
                data.get("mood_adjustment", {}), is_group_chat
            )

            if is_group_chat:
                data["participants"] = self._ensure_list(data.get("participants", []))
                logger.debug(
                    f"[MemoryProcessor] 提取 participants ({len(data['participants'])} 个): {data['participants']}"
                )

            return data

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[MemoryProcessor]  JSON 解析失败: {e}")
            logger.debug(
                f"[MemoryProcessor] 解析失败的内容（前200字符）: {response_text[:200]}"
            )

            # 尝试修复 JSON 后再解析
            logger.info("[MemoryProcessor] 尝试修复 JSON 后重新解析")
            try:
                fixed_text = self._try_fix_json(response_text)
                data = json.loads(fixed_text)
                if isinstance(data, dict):
                    logger.info("[MemoryProcessor] JSON 修复后解析成功")
                    return self._normalize_parsed_data(data, is_group_chat)
            except (json.JSONDecodeError, ValueError) as fix_err:
                logger.debug(f"[MemoryProcessor] JSON 修复后仍无法解析: {fix_err}")

            logger.info("[MemoryProcessor] 尝试使用正则表达式提取 JSON")
            # 尝试正则提取
            return self._extract_by_regex(response_text, is_group_chat)
        except Exception as e:
            logger.error(
                f"[MemoryProcessor]  解析 LLM 响应时发生异常: {e}", exc_info=True
            )
            logger.debug(
                f"[MemoryProcessor] 异常发生时的响应内容: {response_text[:200]}"
            )
            return self._get_default_structured_data(is_group_chat)

    def _extract_by_regex(self, text: str, is_group_chat: bool) -> dict[str, Any]:
        """
        使用正则表达式从文本中提取结构化数据(备用方案)

        Args:
            text: 响应文本
            is_group_chat: 是否为群聊

        Returns:
            提取的结构化数据
        """
        logger.debug("[MemoryProcessor] 开始使用正则表达式提取结构化数据")
        data = self._get_default_structured_data(is_group_chat)

        try:
            # 先尝试找到完整的 JSON 块
            json_matches = re.findall(
                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL
            )
            logger.debug(
                f"[MemoryProcessor] 正则匹配到 {len(json_matches)} 个可能的 JSON 块"
            )

            for i, match in enumerate(json_matches):
                logger.debug(
                    f"[MemoryProcessor] JSON 块 #{i + 1} (前200字符): {match[:200]}..."
                )
                try:
                    # 尝试解析每个匹配的块
                    parsed = json.loads(match)
                    if "summary" in parsed:
                        logger.info(
                            f"[MemoryProcessor]  成功从第 {i + 1} 个 JSON 块中解析数据"
                        )
                        data = parsed
                        break
                except json.JSONDecodeError:
                    continue

            # 如果没有找到完整的 JSON，尝试单独提取字段
            if data == self._get_default_structured_data(is_group_chat):
                logger.debug("[MemoryProcessor] 未找到完整 JSON，尝试提取单独字段")

                # 提取summary
                summary_match = re.search(r'"summary"\s*:\s*"([^"]+)"', text)
                if summary_match:
                    data["summary"] = summary_match.group(1)
                    logger.debug(
                        f"[MemoryProcessor] 正则提取 summary: {data['summary'][:50]}..."
                    )

                # 提取importance
                importance_match = re.search(r'"importance"\s*:\s*([0-9.]+)', text)
                if importance_match:
                    data["importance"] = float(importance_match.group(1))
                    logger.debug(
                        f"[MemoryProcessor] 正则提取 importance: {data['importance']}"
                    )

                # 提取sentiment
                sentiment_match = re.search(r'"sentiment"\s*:\s*"(\w+)"', text)
                if sentiment_match:
                    data["sentiment"] = sentiment_match.group(1)
                    logger.debug(
                        f"[MemoryProcessor] 正则提取 sentiment: {data['sentiment']}"
                    )

                # 提取 topics 数组
                topics_match = re.search(r'"topics"\s*:\s*\[(.*?)\]', text, re.DOTALL)
                if topics_match:
                    topics_str = topics_match.group(1)
                    topics = re.findall(r'"([^"]+)"', topics_str)
                    data["topics"] = topics[:5]
                    logger.debug(f"[MemoryProcessor] 正则提取 topics: {data['topics']}")

                # 提取 key_facts 数组
                facts_match = re.search(r'"key_facts"\s*:\s*\[(.*?)\]', text, re.DOTALL)
                if facts_match:
                    facts_str = facts_match.group(1)
                    facts = re.findall(r'"([^"]+)"', facts_str)
                    data["key_facts"] = facts[:5]
                    logger.debug(
                        f"[MemoryProcessor] 正则提取 key_facts: {data['key_facts']}"
                    )

            logger.info(
                f"[MemoryProcessor] 正则提取完成，提取到的字段: {list(data.keys())}"
            )

        except Exception as e:
            logger.error(f"[MemoryProcessor]  正则提取失败: {e}", exc_info=True)

        return self._normalize_parsed_data(data, is_group_chat)

    def _build_storage_format(
        self,
        fallback_excerpt: str,
        structured_data: dict[str, Any],
        is_group_chat: bool,
    ) -> tuple[str, dict[str, Any]]:
        """
        构建存储格式

        Args:
            fallback_excerpt: 当摘要为空时使用的对话摘录
            structured_data: 结构化数据
            is_group_chat: 是否为群聊

        Returns:
            (content, metadata) 元组
        """
        summary = str(structured_data.get("summary", "")).strip()
        key_facts = structured_data.get("key_facts", [])

        # content is both vector/BM25 corpus and tool recall output, so keep the
        # personality summary and concrete facts together instead of competing.
        rich_parts = [summary] if summary else []
        if key_facts:
            rich_parts.append("；".join(str(f) for f in key_facts[:5] if f))
        rich_content = " | ".join(rich_parts)
        content = rich_content if rich_content else fallback_excerpt

        # Custom prompts may still provide a neutral summary for graph consumers.
        # Built-in prompts omit it, preserving the v2 field with rich text fallback.
        canonical_summary = str(structured_data.get("canonical_summary") or "").strip()
        if not canonical_summary:
            canonical_summary = rich_content

        # metadata字段:存储结构化信息
        # 注意：不要在这里设置 create_time 和 last_access_time
        # 这些字段会由 MemoryEngine.add_memory() 自动添加
        metadata = {
            "topics": structured_data.get("topics", []),
            "key_facts": key_facts,
            "sentiment": structured_data.get("sentiment", "neutral"),
            "interaction_type": "group_chat" if is_group_chat else "private_chat",
            # canonical_summary is retained for neutral-text consumers such as
            # graph extraction; persona_summary preserves the styled memory.
            "canonical_summary": canonical_summary,
            "persona_summary": summary,
            "summary_schema_version": "v2",
            "emotional_observations": self._normalize_emotional_observations(
                structured_data.get("emotional_observations", []), is_group_chat
            ),
            "attention_observations": self._normalize_attention_observations(
                structured_data.get("attention_observations", []), is_group_chat
            ),
            "mood_adjustment": self._normalize_mood_adjustment(
                structured_data.get("mood_adjustment", {}), is_group_chat
            ),
            # summary_quality 由 process_conversation 中的 SummaryValidator 覆盖写入
        }

        if is_group_chat and "participants" in structured_data:
            metadata["participants"] = structured_data["participants"]

        return content, metadata

    def _normalize_parsed_data(self, data: dict, is_group_chat: bool) -> dict[str, Any]:
        """
        规范化解析后的数据（补充缺失字段、类型转换）

        Args:
            data: 解析后的原始字典
            is_group_chat: 是否为群聊

        Returns:
            规范化后的字典
        """
        required_fields = ["summary", "topics", "key_facts", "sentiment", "importance"]
        if is_group_chat:
            required_fields.append("participants")

        for field in required_fields:
            if field not in data:
                data[field] = self._get_default_value(field)

        data["summary"] = str(data.get("summary", ""))
        data["topics"] = self._ensure_list(data.get("topics", []))[:5]
        data["key_facts"] = self._ensure_list(data.get("key_facts", []))[:5]
        data["sentiment"] = self._validate_sentiment(data.get("sentiment", "neutral"))
        data["importance"] = self._validate_importance(data.get("importance", 0.5))
        data["emotional_observations"] = self._normalize_emotional_observations(
            data.get("emotional_observations", []), is_group_chat
        )
        data["attention_observations"] = self._normalize_attention_observations(
            data.get("attention_observations", []), is_group_chat
        )
        data["mood_adjustment"] = self._normalize_mood_adjustment(
            data.get("mood_adjustment", {}), is_group_chat
        )

        if is_group_chat:
            data["participants"] = self._ensure_list(data.get("participants", []))

        return data

    @staticmethod
    def _normalize_emotional_observations(
        value: Any, is_group_chat: bool
    ) -> list[dict[str, Any]]:
        """Keep optional private emotional suggestions isolated from memory parsing."""
        if is_group_chat or not isinstance(value, list):
            return []
        actions = {
            "create",
            "merge",
            "intensify",
            "ease",
            "dormant",
            "archive",
            "retain",
        }
        categories = {"transient", "episodic", "psychological", "concrete"}
        result: list[dict[str, Any]] = []
        for raw in value[:6]:
            if not isinstance(raw, dict):
                continue
            action = str(raw.get("action", "create")).strip().lower()
            event_id = str(raw.get("event_id", "")).strip()[:80]
            try:
                event_version = (
                    int(raw["event_version"])
                    if raw.get("event_version") is not None
                    else None
                )
            except (TypeError, ValueError):
                event_version = None
            fact = _bound_complete_text(raw.get("fact", ""), EMOTIONAL_FACT_MAX_CHARS)
            if action not in actions:
                continue
            if action == "create" and not fact:
                continue
            if action != "create" and (not event_id or event_version is None):
                continue
            try:
                valence = min(1.0, max(-1.0, float(raw.get("valence", 0.0))))
                intensity = min(1.0, max(0.0, float(raw.get("intensity", 0.35))))
                confidence = min(1.0, max(0.0, float(raw.get("confidence", 0.5))))
            except (TypeError, ValueError):
                continue
            category = str(raw.get("category", "concrete")).strip().lower()
            if category not in categories:
                category = "concrete"
            result.append(
                {
                    "action": action,
                    "event_id": event_id,
                    "event_version": event_version,
                    "fact": fact,
                    "emotional_meaning": str(
                        raw.get("emotional_meaning", "这段互动可能影响当前心境")
                    )[:240],
                    "target": str(raw.get("target", "unknown"))[:80],
                    "target_basis": str(raw.get("target_basis", ""))[:120],
                    "evidence_quote": str(raw.get("evidence_quote", ""))[:240],
                    "evidence_speaker": str(raw.get("evidence_speaker", ""))
                    .strip()
                    .lower()[:24],
                    "category": category,
                    "valence": valence,
                    "intensity": intensity,
                    "confidence": confidence,
                    "uncertain": bool(raw.get("uncertain", confidence < 0.62)),
                    "note": str(raw.get("note", ""))[:240],
                    "tags": [str(tag)[:40] for tag in raw.get("tags", [])[:5]]
                    if isinstance(raw.get("tags", []), list)
                    else [],
                }
            )
        return result

    @staticmethod
    def _normalize_attention_observations(
        value: Any, is_group_chat: bool
    ) -> list[dict[str, Any]]:
        """Normalize explicit, durable private attention-item suggestions."""
        if is_group_chat or not isinstance(value, list):
            return []
        actions = {"create", "confirm", "update", "complete", "cancel", "supersede"}
        kinds = {"commitment", "plan", "remember", "follow_up"}
        statuses = {"proposed", "open", "completed", "cancelled", "superseded"}
        result: list[dict[str, Any]] = []
        for raw in value[:6]:
            if not isinstance(raw, dict):
                continue
            action = str(raw.get("action", "create")).strip().lower()
            item_id = str(raw.get("item_id", "")).strip()[:80]
            content = str(raw.get("content", "")).strip()[:240]
            evidence_quote = str(raw.get("evidence_quote", "")).strip()[:240]
            evidence_speaker = str(raw.get("evidence_speaker", "")).strip().lower()[:24]
            explicit = bool(raw.get("explicit", False))
            try:
                item_version = (
                    int(raw["item_version"])
                    if raw.get("item_version") is not None
                    else None
                )
                confidence = min(1.0, max(0.0, float(raw.get("confidence", 0.0))))
            except (TypeError, ValueError):
                continue
            if action not in actions or not evidence_quote:
                continue
            kind = str(raw.get("kind", "follow_up")).strip().lower()
            status = str(raw.get("status", "open")).strip().lower()
            if status not in statuses:
                status = "open"
            if action == "create" and (
                not content
                or not evidence_quote
                or confidence < (0.62 if status == "proposed" else 0.82)
                or (
                    status != "proposed"
                    and (not explicit or evidence_speaker != "user")
                )
                or (
                    status == "proposed"
                    and evidence_speaker not in {"user", "assistant", "both"}
                )
            ):
                continue
            if action != "create" and (
                not item_id
                or item_version is None
                or confidence < 0.78
                or (action != "complete" and evidence_speaker != "user")
                or (
                    action == "complete"
                    and evidence_speaker not in {"user", "assistant", "both"}
                )
            ):
                continue
            result.append(
                {
                    "action": action,
                    "item_id": item_id,
                    "item_version": item_version,
                    "content": content,
                    "kind": kind if kind in kinds else "follow_up",
                    "status": status if status in statuses else "open",
                    "actor": str(raw.get("actor", "both"))[:40],
                    "time_hint": str(raw.get("time_hint", ""))[:80],
                    "due_at": str(raw.get("due_at", ""))[:64],
                    "confidence": confidence,
                    "explicit": explicit,
                    "evidence_quote": evidence_quote,
                    "evidence_speaker": evidence_speaker,
                    "note": str(raw.get("note", ""))[:240],
                }
            )
        return result

    @staticmethod
    def _normalize_mood_adjustment(value: Any, is_group_chat: bool) -> dict[str, Any]:
        """Normalize an optional private mood proposal without affecting memory storage."""
        if is_group_chat or not isinstance(value, dict):
            return {}
        try:
            confidence = min(1.0, max(0.0, float(value.get("confidence", 0.0))))
            if confidence <= 0.0:
                return {}
            return {
                "valence": min(1.0, max(-1.0, float(value.get("valence", 0.0)))),
                "energy": min(1.0, max(0.0, float(value.get("energy", 0.45)))),
                "tension": min(1.0, max(0.0, float(value.get("tension", 0.2)))),
                "label": str(value.get("label", "")).strip()[:40],
                "confidence": confidence,
            }
        except (TypeError, ValueError):
            return {}

    def _ensure_list(self, value: Any) -> list[str]:
        """确保值是字符串列表"""
        if isinstance(value, list):
            return [str(item) for item in value if item]
        elif isinstance(value, str):
            return [value] if value else []
        else:
            return []

    def _validate_sentiment(self, sentiment: str) -> str:
        """验证情感值"""
        valid_sentiments = ["positive", "neutral", "negative"]
        sentiment = sentiment.lower()
        return sentiment if sentiment in valid_sentiments else "neutral"

    def _validate_importance(self, importance: Any) -> float:
        """验证重要性评分"""
        try:
            score = float(importance)
            return max(0.0, min(1.0, score))  # 限制在0-1之间
        except (ValueError, TypeError):
            return 0.5

    async def merge_memories(self, memories: list[dict]) -> dict[str, Any]:
        """把一组零散记忆合并为一条精炼记忆（供记忆库整合使用）。

        Args:
            memories: 待合并的记忆列表，每条为 {"content": str, "metadata": dict}。

        Returns:
            包含 summary/key_facts/topics/importance 的字典。

        Raises:
            RuntimeError: LLM 不可用或解析失败时抛出。
        """
        items: list[dict[str, Any]] = []
        for i, mem in enumerate(memories, 1):
            metadata = mem.get("metadata") or {}
            summary = str(
                metadata.get("persona_summary") or str(mem.get("content", "")).strip()
            ).strip()
            items.append(
                {
                    "id": i,
                    "summary": summary,
                    "key_facts": metadata.get("key_facts") or [],
                    "topics": metadata.get("topics") or [],
                }
            )

        system_prompt = (
            "你是记忆整理助手。把多条关于同一主题或会话的零散记忆合并为一条精炼、"
            "信息无损的记忆摘要。保留所有关键事实与具体细节，去重并消除相互矛盾，"
            "避免泛化和丢失专有名词。只输出 JSON，不要输出任何其他内容。"
        )
        prompt = (
            f"以下是一组需要合并的记忆（共 {len(items)} 条）：\n"
            f"{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
            "请将它们合并为一条记忆，按如下 JSON 格式输出：\n"
            '{"summary": "合并后的精炼摘要", "key_facts": ["事实1", "事实2"], '
            '"topics": ["主题1"], "importance": 0.5}'
        )

        text = await self._call_llm_with_retry(prompt, system_prompt)
        data = self._parse_merge_response(text)

        summary = str(data.get("summary", "")).strip()
        if not summary:
            raise RuntimeError("合并结果缺少 summary")

        return {
            "summary": summary,
            "key_facts": self._ensure_list(data.get("key_facts", []))[:5],
            "topics": self._ensure_list(data.get("topics", []))[:5],
            "importance": self._validate_importance(data.get("importance", 0.5)),
        }

    def _parse_merge_response(self, text: str) -> dict[str, Any]:
        """解析合并 LLM 响应中的 JSON，失败时抛出异常。"""
        candidates = [text]
        fixed = self._try_fix_json(text)
        if fixed != text.strip():
            candidates.append(fixed)

        from ..utils import extract_json_from_response

        extracted = extract_json_from_response(text)
        if extracted != text.strip():
            candidates.append(extracted)

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                data = json.loads(self._try_fix_json(candidate))
            except (json.JSONDecodeError, TypeError) as e:
                last_error = e
                continue
            if isinstance(data, dict):
                return data
        raise RuntimeError(f"无法解析合并结果 JSON: {last_error}")

    def build_memory_from_structured_data(
        self,
        structured_data: dict[str, Any],
        is_group_chat: bool = False,
        fallback_excerpt: str = "",
    ) -> tuple[str, dict[str, Any], float]:
        """复用自动总结流程，将结构化数据转换为标准记忆存储格式。"""
        # 与自动总结路径保持一致：先校验质量，再规范化。
        # 这样原始 importance 越界等异常仍会被判为 low quality。
        quality = self._validate_summary_quality(structured_data)
        normalized = self._normalize_parsed_data(structured_data, is_group_chat)
        normalized["_quality"] = quality

        content, metadata = self._build_storage_format(
            fallback_excerpt or normalized.get("summary", ""),
            normalized,
            is_group_chat,
        )
        metadata["summary_quality"] = quality
        return (
            content,
            metadata,
            self._validate_importance(normalized.get("importance")),
        )

    def _get_default_value(self, field: str) -> Any:
        """获取字段的默认值"""
        defaults = {
            "summary": "",
            "topics": [],
            "key_facts": [],
            "participants": [],
            "sentiment": "neutral",
            "importance": 0.5,
        }
        return defaults.get(field, "")

    def _get_default_structured_data(self, is_group_chat: bool) -> dict[str, Any]:
        """获取默认的结构化数据"""
        data = {
            "summary": "对话记录",
            "topics": [],
            "key_facts": [],
            "sentiment": "neutral",
            "importance": 0.5,
        }
        if is_group_chat:
            data["participants"] = []
        return data

    def _validate_summary_quality(self, structured_data: dict[str, Any]) -> str:
        """
        校验总结质量，返回质量等级。

        检查规则：
        1. summary 不能为空或过短（< 10 字符）
        2. key_facts 至少有 1 条
        3. importance 在合法范围内
        4. summary 不含泛化词（"某用户"、"有人"等）

        Returns:
            "normal" 或 "low"
        """
        summary = structured_data.get("summary", "")
        key_facts = structured_data.get("key_facts", [])
        importance = structured_data.get("importance", 0.5)

        if not summary or len(summary.strip()) < 10:
            return "low"
        if not key_facts:
            return "low"
        if not isinstance(importance, (int, float)) or not (0.0 <= importance <= 1.0):
            return "low"

        # 泛化词检测
        generic_terms = [
            "某用户",
            "有人",
            "某人",
            "用户说",
            "对方说",
            "群成员",
            "某群成员",
        ]
        if any(term in summary for term in generic_terms):
            return "low"

        return "normal"

    def classify_atoms_from_metadata(
        self,
        metadata: dict[str, Any],
        parent_importance: float = 0.5,
        session_id: str | None = None,
        persona_id: str | None = None,
    ) -> list[MemoryAtom]:
        """Generate time-aware memory atoms from key_facts in metadata.

        This is a post-processing step after process_conversation().
        It does NOT make additional LLM calls — classification is rule-based.
        """
        if not self.config.get("atom_enabled", True):
            return []
        key_facts: list[str] = metadata.get("key_facts", [])
        if not key_facts:
            return []
        topics = metadata.get("topics", [])
        participants = metadata.get("participants", [])
        return classify_atoms(
            key_facts=key_facts,
            topics=topics,
            participants=participants,
            parent_importance=parent_importance,
            session_id=session_id,
            persona_id=persona_id,
        )
