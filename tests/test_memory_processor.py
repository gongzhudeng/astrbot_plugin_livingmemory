"""
Tests for MemoryProcessor.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from astrbot_plugin_livingmemory.core.models.conversation_models import Message
from astrbot_plugin_livingmemory.core.processors.memory_processor import MemoryProcessor


class _DummyLLMProvider:
    def __init__(self, completion_text: str):
        self._completion_text = completion_text
        self.text_chat = AsyncMock(side_effect=self._chat)

    async def _chat(self, prompt: str, system_prompt: str):
        return SimpleNamespace(completion_text=self._completion_text)


def _make_messages():
    return [
        Message(
            id=1,
            session_id="s1",
            role="user",
            content="明天下午三点开会",
            sender_id="u1",
            sender_name="张三",
            group_id=None,
            platform="test",
            metadata={},
        ),
        Message(
            id=2,
            session_id="s1",
            role="assistant",
            content="收到，我会提醒你",
            sender_id="bot",
            sender_name="Bot",
            group_id=None,
            platform="test",
            metadata={"is_bot_message": True},
        ),
    ]


@pytest.mark.asyncio
async def test_process_conversation_success():
    llm = _DummyLLMProvider(
        """{
            "summary":"我记录了张三明天下午三点开会，并给出提醒",
            "topics":["会议提醒"],
            "key_facts":["张三明天下午三点开会"],
            "sentiment":"neutral",
            "importance":0.8
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    content, metadata, importance = await processor.process_conversation(
        messages=_make_messages(),
        is_group_chat=False,
        persona_id=None,
    )

    assert "张三" in content
    assert metadata["interaction_type"] == "private_chat"
    assert "会议提醒" in metadata["topics"]
    assert importance == 0.8


@pytest.mark.asyncio
async def test_private_summary_receives_event_catalog_and_preserves_review_fields():
    llm = _DummyLLMProvider(
        """{
            "summary":"我回顾了张三带来的持续影响并完成复核",
            "topics":["情绪复核"],
            "key_facts":["张三此前认真安慰了我"],
            "sentiment":"positive",
            "importance":0.7,
            "emotional_observations":[
                {
                    "action":"merge",
                    "event_id":"event-123",
                    "event_version":4,
                    "fact":"张三的安慰让我逐渐放松下来",
                    "emotional_meaning":"这件事正在缓解",
                    "target":"user",
                    "category":"psychological",
                    "valence":0.2,
                    "intensity":0.6,
                    "confidence":0.8,
                    "uncertain":false,
                    "tags":["reviewed"]
                }
            ],
            "mood_adjustment": {
                "valence": 1.4,
                "energy": 0.62,
                "tension": -0.2,
                "label": "轻松了一些",
                "confidence": 0.83
            }
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)
    review_context = {
        "state_version": 7,
        "message_watermark": 12,
        "events": [
            {
                "event_id": "event-123",
                "category": "psychological",
                "fact": "张三认真安慰了我",
                "intensity": 0.7,
                "lifecycle": "active",
            }
        ],
        "attention_items": [
            {
                "item_id": "attention-456",
                "item_version": 2,
                "kind": "plan",
                "content": "等一下玩角色扮演",
                "status": "open",
            }
        ],
    }

    _, metadata, _ = await processor.process_conversation(
        messages=_make_messages(),
        is_group_chat=False,
        persona_id=None,
        emotion_review_context=review_context,
    )

    prompt = llm.text_chat.call_args.kwargs["prompt"]
    observation = metadata["emotional_observations"][0]
    assert "event-123" in prompt
    assert "attention-456" in prompt
    assert "当前角色心理事件与待关注事项复核目录" in prompt
    assert "普通生活和互动可作为episodic近期片段" in prompt
    assert "复核既有情绪事件必须原样使用 event_id 与 event_version" in prompt
    assert "复核既有待关注事项必须原样使用 item_id 与 item_version" in prompt
    assert observation["event_id"] == "event-123"
    assert observation["event_version"] == 4
    assert observation["action"] == "merge"
    assert observation["category"] == "psychological"
    assert observation["tags"] == ["reviewed"]
    assert metadata["mood_adjustment"] == {
        "valence": 1.0,
        "energy": 0.62,
        "tension": 0.0,
        "label": "轻松了一些",
        "confidence": 0.83,
    }


def test_group_summary_discards_private_emotion_fields():
    processor = MemoryProcessor(llm_provider=None, context=None)
    normalized = processor._normalize_parsed_data(
        {
            "summary": "群聊成员讨论了日常安排",
            "topics": ["日常"],
            "key_facts": ["成员准备去上班"],
            "participants": ["张三"],
            "sentiment": "neutral",
            "importance": 0.5,
            "emotional_observations": [
                {"action": "create", "fact": "不应进入私聊情绪"}
            ],
            "mood_adjustment": {"valence": 1.0, "confidence": 1.0},
        },
        is_group_chat=True,
    )

    assert normalized["emotional_observations"] == []
    assert normalized["attention_observations"] == []
    assert normalized["mood_adjustment"] == {}


def test_attention_observation_requires_versioned_terminal_evidence():
    processor = MemoryProcessor(llm_provider=None, context=None)

    normalized = processor._normalize_attention_observations(
        [
            {"action": "complete", "content": "模型猜测已经完成"},
            {
                "action": "confirm",
                "item_id": "attention-1",
                "item_version": 3,
                "kind": "commitment",
                "confidence": 0.9,
                "evidence_quote": "你一定要记住",
                "evidence_speaker": "user",
            },
            {
                "action": "create",
                "content": "下周三晚上一起去看电影",
                "kind": "plan",
                "status": "open",
                "confidence": 0.85,
                "explicit": True,
                "evidence_quote": "下周三晚上一起去看电影",
                "evidence_speaker": "user",
            },
        ],
        is_group_chat=False,
    )

    assert [item["action"] for item in normalized] == ["confirm", "create"]
    assert normalized[0]["item_id"] == "attention-1"
    assert normalized[0]["item_version"] == 3
    assert normalized[1]["kind"] == "plan"


def test_attention_completion_accepts_assistant_evidence_but_not_other_reviews():
    processor = MemoryProcessor(llm_provider=None, context=None)
    normalized = processor._normalize_attention_observations(
        [
            {
                "action": "complete",
                "item_id": "attention-1",
                "item_version": 3,
                "confidence": 0.9,
                "evidence_quote": "我已经把语音发给你了",
                "evidence_speaker": "assistant",
            },
            {
                "action": "cancel",
                "item_id": "attention-2",
                "item_version": 2,
                "confidence": 0.9,
                "evidence_quote": "我不做了",
                "evidence_speaker": "assistant",
            },
        ],
        is_group_chat=False,
    )
    filtered = processor._filter_attention_observations_by_source(
        normalized,
        [
            Message(
                id=1,
                session_id="private:test",
                role="assistant",
                content="我已经把语音发给你了",
                sender_id="bot",
            ),
            Message(
                id=2,
                session_id="private:test",
                role="assistant",
                content="我不做了",
                sender_id="bot",
            ),
        ],
    )
    assert [item["action"] for item in filtered] == ["complete"]


def test_attention_completion_accepts_matching_image_and_voice_records():
    processor = MemoryProcessor(llm_provider=None, context=None)
    observations = processor._normalize_attention_observations(
        [
            {
                "action": "complete",
                "item_id": "photo-item",
                "item_version": 1,
                "confidence": 0.95,
                "evidence_quote": "[图片消息]",
                "evidence_speaker": "user",
            },
            {
                "action": "complete",
                "item_id": "voice-item",
                "item_version": 2,
                "confidence": 0.95,
                "evidence_quote": "[语音消息]",
                "evidence_speaker": "assistant",
            },
        ],
        is_group_chat=False,
    )
    messages = [
        Message(
            id=1,
            session_id="private:test",
            role="user",
            content=[{"type": "image", "url": "photo.jpg"}],
            sender_id="user",
        ),
        Message(
            id=2,
            session_id="private:test",
            role="assistant",
            content=[{"type": "record", "file": "voice.amr"}],
            sender_id="bot",
            metadata={"is_bot_message": True},
        ),
    ]

    filtered = processor._filter_attention_observations_by_source(
        observations, messages
    )

    assert [item["item_id"] for item in filtered] == ["photo-item", "voice-item"]


def test_proposed_attention_accepts_exact_combined_speaker_evidence():
    processor = MemoryProcessor(llm_provider=None, context=None)
    observation = {
        "action": "create",
        "content": "以后由我提醒开会",
        "kind": "follow_up",
        "status": "proposed",
        "confidence": 0.8,
        "explicit": False,
        "evidence_quote": "我会提醒你",
        "evidence_speaker": "both",
    }

    normalized = processor._normalize_attention_observations(
        [observation], is_group_chat=False
    )
    filtered = processor._filter_attention_observations_by_source(
        normalized, _make_messages()
    )

    assert filtered == normalized
    assert filtered[0]["status"] == "proposed"
    assert filtered[0]["evidence_speaker"] == "both"


def test_review_observation_requires_existing_event_id_shape():
    processor = MemoryProcessor(llm_provider=None, context=None)

    normalized = processor._normalize_emotional_observations(
        [
            {"action": "ease", "fact": "模型重写的旧事实"},
            {
                "action": "retain",
                "event_id": "known-event",
                "event_version": 2,
                "confidence": 0.8,
            },
            {"action": "create", "fact": "本段对话中的具体新事实"},
        ],
        is_group_chat=False,
    )

    assert [item["action"] for item in normalized] == ["retain", "create"]
    assert normalized[0]["event_id"] == "known-event"
    assert normalized[0]["event_version"] == 2
    assert normalized[1]["event_id"] == ""


@pytest.mark.asyncio
async def test_process_conversation_rejects_unstructured_response():
    llm = _DummyLLMProvider("summary=测试, importance=0.6")
    processor = MemoryProcessor(llm_provider=llm, context=None)

    with pytest.raises(ValueError, match="结构化记忆"):
        await processor._call_llm_with_retry(
            prompt="prompt",
            system_prompt="system",
            max_retries=1,
        )


@pytest.mark.asyncio
async def test_llm_fallback_uses_next_provider_after_primary_failure():
    primary = _DummyLLMProvider("")
    fallback = _DummyLLMProvider(
        """{
            "summary":"用户告知明天下午三点有重要会议需要参加",
            "topics":["会议"],
            "key_facts":["明天下午三点开会"],
            "sentiment":"neutral",
            "importance":0.8
        }"""
    )
    context = Mock()
    context.get_provider_by_id = Mock(
        side_effect=lambda provider_id: {
            "primary": primary,
            "fallback": fallback,
        }.get(provider_id)
    )
    processor = MemoryProcessor(
        context=context,
        llm_provider="primary",
        fallback_provider_ids=["fallback"],
    )

    response = await processor._call_llm_with_retry(
        prompt="prompt",
        system_prompt="system",
        max_retries=1,
    )

    assert "明天下午三点" in response
    primary.text_chat.assert_awaited_once()
    fallback.text_chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_low_quality_structured_response_does_not_use_fallback():
    primary = _DummyLLMProvider(
        '{"summary":"普通闲聊","topics":[],"key_facts":[],"sentiment":"neutral","importance":0.2}'
    )
    fallback = _DummyLLMProvider(
        '{"summary":"不应使用兜底模型","topics":["测试"],"key_facts":["兜底被调用"],"sentiment":"neutral","importance":0.8}'
    )
    context = Mock()
    context.get_provider_by_id = Mock(
        side_effect=lambda provider_id: {
            "primary": primary,
            "fallback": fallback,
        }.get(provider_id)
    )
    processor = MemoryProcessor(
        context=context,
        llm_provider="primary",
        fallback_provider_ids=["fallback"],
    )

    _, metadata, _ = await processor.process_conversation(
        messages=_make_messages(),
        is_group_chat=False,
        persona_id=None,
    )

    assert metadata["summary_quality"] == "low"
    primary.text_chat.assert_awaited_once()
    fallback.text_chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_persona_prompt_is_included_when_available():
    llm = _DummyLLMProvider(
        """{
            "summary":"我愉快地记录了这次交流",
            "topics":["闲聊"],
            "key_facts":["用户问候"],
            "sentiment":"positive",
            "importance":0.5
        }"""
    )
    context = Mock()
    context.persona_manager = Mock()
    context.persona_manager.get_persona = AsyncMock(
        return_value=SimpleNamespace(system_prompt="你是活泼助手")
    )

    processor = MemoryProcessor(llm_provider=llm, context=context)

    system_prompt = await processor._build_system_prompt_with_persona("persona_1")
    assert "人格设定" in system_prompt
    assert "活泼助手" in system_prompt


# ── New tests for dual-channel summary and quality validator ──────────────────


@pytest.mark.asyncio
async def test_dual_channel_summary_stores_canonical_and_persona():
    """Custom neutral summary is retained without thinning retrieval content."""
    llm = _DummyLLMProvider(
        """{
            "summary":"我记录了张三明天下午三点开会呀，并认真给出了提醒",
            "canonical_summary":"张三明天下午三点开会，Bot 已确认提醒",
            "topics":["会议提醒"],
            "key_facts":["张三明天下午三点开会"],
            "sentiment":"neutral",
            "importance":0.8
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    content, metadata, importance = await processor.process_conversation(
        messages=_make_messages(),
        is_group_chat=False,
        persona_id=None,
    )

    assert metadata["canonical_summary"] == "张三明天下午三点开会，Bot 已确认提醒"
    assert "呀" not in metadata["canonical_summary"]
    assert metadata["persona_summary"] == (
        "我记录了张三明天下午三点开会呀，并认真给出了提醒"
    )
    assert content == (
        "我记录了张三明天下午三点开会呀，并认真给出了提醒 | 张三明天下午三点开会"
    )
    assert importance == 0.8
    assert metadata.get("summary_schema_version") == "v2"


@pytest.mark.asyncio
async def test_canonical_summary_includes_key_facts():
    """canonical_summary 应将 key_facts 拼接到摘要中，提升检索覆盖率。"""
    llm = _DummyLLMProvider(
        """{
            "summary":"用户提到了一个重要事项",
            "topics":["备忘"],
            "key_facts":["明天下午三点开会", "需要准备PPT"],
            "sentiment":"neutral",
            "importance":0.7
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    content, metadata, _ = await processor.process_conversation(
        messages=_make_messages(),
        is_group_chat=False,
        persona_id=None,
    )

    # canonical_summary 应包含 key_facts 内容
    assert "明天下午三点开会" in metadata["canonical_summary"]
    assert "需要准备PPT" in metadata["canonical_summary"]


@pytest.mark.asyncio
async def test_summary_quality_normal_for_valid_response():
    """有效的 LLM 响应应标记为 summary_quality=normal。"""
    llm = _DummyLLMProvider(
        """{
            "summary":"用户告知明天下午三点有重要会议需要参加",
            "topics":["会议"],
            "key_facts":["明天下午三点开会"],
            "sentiment":"neutral",
            "importance":0.8
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    _, metadata, _ = await processor.process_conversation(
        messages=_make_messages(),
        is_group_chat=False,
        persona_id=None,
    )

    assert metadata.get("summary_quality") == "normal"


@pytest.mark.asyncio
async def test_summary_quality_low_for_empty_summary():
    """summary 为空时应标记为 summary_quality=low。"""
    llm = _DummyLLMProvider(
        """{
            "summary":"",
            "topics":["闲聊"],
            "key_facts":["用户问候"],
            "sentiment":"neutral",
            "importance":0.5
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    _, metadata, _ = await processor.process_conversation(
        messages=_make_messages(),
        is_group_chat=False,
        persona_id=None,
    )

    assert metadata.get("summary_quality") == "low"


@pytest.mark.asyncio
async def test_summary_quality_low_for_missing_key_facts():
    """key_facts 为空时应标记为 summary_quality=low。"""
    llm = _DummyLLMProvider(
        """{
            "summary":"用户进行了一次普通对话",
            "topics":["闲聊"],
            "key_facts":[],
            "sentiment":"neutral",
            "importance":0.5
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    _, metadata, _ = await processor.process_conversation(
        messages=_make_messages(),
        is_group_chat=False,
        persona_id=None,
    )

    assert metadata.get("summary_quality") == "low"


@pytest.mark.asyncio
async def test_summary_quality_low_for_generic_terms():
    """summary 包含泛化词（某用户、有人等）时应标记为 summary_quality=low。"""
    llm = _DummyLLMProvider(
        """{
            "summary":"某用户提到了一些事情",
            "topics":["闲聊"],
            "key_facts":["某用户说了话"],
            "sentiment":"neutral",
            "importance":0.5
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    _, metadata, _ = await processor.process_conversation(
        messages=_make_messages(),
        is_group_chat=False,
        persona_id=None,
    )

    assert metadata.get("summary_quality") == "low"


def test_validate_summary_quality_directly():
    """直接测试 _validate_summary_quality 的各种边界情况。"""
    from unittest.mock import MagicMock

    processor = MemoryProcessor(llm_provider=MagicMock(), context=None)

    # 正常情况
    assert (
        processor._validate_summary_quality(
            {
                "summary": "用户明确表示喜欢吃寿司",
                "key_facts": ["用户喜欢寿司"],
                "importance": 0.7,
            }
        )
        == "normal"
    )

    # summary 过短
    assert (
        processor._validate_summary_quality(
            {
                "summary": "短",
                "key_facts": ["fact"],
                "importance": 0.5,
            }
        )
        == "low"
    )

    # importance 超出范围
    assert (
        processor._validate_summary_quality(
            {
                "summary": "用户明确表示喜欢吃寿司",
                "key_facts": ["用户喜欢寿司"],
                "importance": 1.5,
            }
        )
        == "low"
    )

    # 泛化词检测
    assert (
        processor._validate_summary_quality(
            {
                "summary": "有人提到了一些事情",
                "key_facts": ["有人说话"],
                "importance": 0.5,
            }
        )
        == "low"
    )


def test_build_memory_from_structured_data_uses_standard_storage_format():
    processor = MemoryProcessor(llm_provider=Mock(), context=None)

    content, metadata, importance = processor.build_memory_from_structured_data(
        {
            "summary": "用户希望主动记忆工具复用自动总结格式",
            "topics": ["LivingMemory", "主动记忆"],
            "key_facts": ["主动记忆应复用 MemoryProcessor 格式化流程"],
            "sentiment": "neutral",
            "importance": 0.8,
        },
        is_group_chat=False,
        fallback_excerpt="fallback",
    )

    assert content == metadata["canonical_summary"]
    assert metadata["persona_summary"] == "用户希望主动记忆工具复用自动总结格式"
    assert metadata["topics"] == ["LivingMemory", "主动记忆"]
    assert metadata["key_facts"] == ["主动记忆应复用 MemoryProcessor 格式化流程"]
    assert metadata["sentiment"] == "neutral"
    assert metadata["interaction_type"] == "private_chat"
    assert metadata["summary_schema_version"] == "v2"
    assert metadata["summary_quality"] == "normal"
    assert importance == 0.8


def test_build_memory_from_structured_data_flags_low_quality_for_out_of_range_importance():
    """与自动总结路径一致：原始 importance 越界时应判为 low quality。"""
    processor = MemoryProcessor(llm_provider=Mock(), context=None)

    _, metadata, importance = processor.build_memory_from_structured_data(
        {
            "summary": "用户希望主动记忆工具复用自动总结格式",
            "topics": ["测试"],
            "key_facts": ["importance 越界"],
            "sentiment": "neutral",
            "importance": 1.5,
        },
        is_group_chat=False,
        fallback_excerpt="fallback",
    )

    assert metadata["summary_quality"] == "low"
    assert importance == 1.0


# ── 群聊路径测试 ──────────────────────────────────────────────────────────────


def _make_group_messages():
    """构造一组群聊消息（含 group_id）"""
    return [
        Message(
            id=1,
            session_id="aiocqhttp:GroupMessage:88888",
            role="user",
            content="大家觉得 AI 工具怎么样？",
            sender_id="10001",
            sender_name="张三",
            group_id="88888",
            platform="aiocqhttp",
            metadata={},
        ),
        Message(
            id=2,
            session_id="aiocqhttp:GroupMessage:88888",
            role="user",
            content="我觉得 ChatGPT 写代码效率提升了 30%",
            sender_id="10002",
            sender_name="李四",
            group_id="88888",
            platform="aiocqhttp",
            metadata={},
        ),
        Message(
            id=3,
            session_id="aiocqhttp:GroupMessage:88888",
            role="assistant",
            content="AI 工具确实能提升效率，但需要仔细审查生成的代码",
            sender_id="bot",
            sender_name="Bot",
            group_id="88888",
            platform="aiocqhttp",
            metadata={"is_bot_message": True},
        ),
    ]


@pytest.mark.asyncio
async def test_process_group_chat_sets_interaction_type():
    """群聊路径应将 interaction_type 设置为 group_chat。"""
    llm = _DummyLLMProvider(
        """{
            "summary":"群聊讨论了 AI 工具的使用效果",
            "topics":["AI工具","工作效率"],
            "key_facts":["张三认为 ChatGPT 效率提升 30%","需要仔细审查 AI 生成代码"],
            "participants":["张三","李四"],
            "sentiment":"positive",
            "importance":0.75
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    content, metadata, importance = await processor.process_conversation(
        messages=_make_group_messages(),
        is_group_chat=True,
        persona_id=None,
    )

    assert metadata["interaction_type"] == "group_chat"
    assert importance == 0.75


@pytest.mark.asyncio
async def test_process_group_chat_extracts_participants():
    """群聊路径应正确提取 participants 字段。"""
    llm = _DummyLLMProvider(
        """{
            "summary":"群聊讨论了 AI 工具的使用效果",
            "topics":["AI工具"],
            "key_facts":["张三认为 ChatGPT 效率提升 30%"],
            "participants":["张三","李四","王五"],
            "sentiment":"positive",
            "importance":0.7
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    _, metadata, _ = await processor.process_conversation(
        messages=_make_group_messages(),
        is_group_chat=True,
        persona_id=None,
    )

    assert "participants" in metadata
    assert "张三" in metadata["participants"]
    assert "李四" in metadata["participants"]
    assert "王五" in metadata["participants"]


@pytest.mark.asyncio
async def test_process_group_chat_dual_channel_summary():
    """群聊路径也应生成双通道摘要（canonical_summary + persona_summary）。"""
    llm = _DummyLLMProvider(
        """{
            "summary":"群聊讨论了 AI 工具的使用效果，建议内部部署私有化 LLM",
            "topics":["AI工具","数据安全"],
            "key_facts":["建议公司内部部署私有化 LLM","注意数据安全"],
            "participants":["张三","李四"],
            "sentiment":"positive",
            "importance":0.8
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    content, metadata, _ = await processor.process_conversation(
        messages=_make_group_messages(),
        is_group_chat=True,
        persona_id=None,
    )

    assert "canonical_summary" in metadata
    assert "persona_summary" in metadata
    assert metadata.get("summary_schema_version") == "v2"
    # canonical_summary 应包含 key_facts
    assert "私有化 LLM" in metadata["canonical_summary"]
    # content 应等于 canonical_summary
    assert content == metadata["canonical_summary"]


@pytest.mark.asyncio
async def test_process_group_chat_missing_participants_uses_default():
    """群聊 LLM 响应缺少 participants 字段时，应使用空列表默认值。"""
    llm = _DummyLLMProvider(
        """{
            "summary":"群聊讨论了一些话题",
            "topics":["闲聊"],
            "key_facts":["大家聊了很多"],
            "sentiment":"neutral",
            "importance":0.5
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    _, metadata, _ = await processor.process_conversation(
        messages=_make_group_messages(),
        is_group_chat=True,
        persona_id=None,
    )

    # 缺少 participants 时应补充默认空列表
    assert "participants" in metadata
    assert isinstance(metadata["participants"], list)


@pytest.mark.asyncio
async def test_process_private_chat_no_participants_field():
    """私聊路径不应在 metadata 中包含 participants 字段。"""
    llm = _DummyLLMProvider(
        """{
            "summary":"用户告知明天下午三点有重要会议",
            "topics":["会议"],
            "key_facts":["明天下午三点开会"],
            "sentiment":"neutral",
            "importance":0.8
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    _, metadata, _ = await processor.process_conversation(
        messages=_make_messages(),
        is_group_chat=False,
        persona_id=None,
    )

    assert "participants" not in metadata
    assert metadata["interaction_type"] == "private_chat"


@pytest.mark.asyncio
async def test_process_group_chat_long_content():
    """群聊长内容（多条消息）应正常处理，不崩溃。"""
    long_messages = []
    for i in range(20):
        long_messages.append(
            Message(
                id=i + 1,
                session_id="aiocqhttp:GroupMessage:99999",
                role="user",
                content=f"成员{i % 5} 说：这是第 {i + 1} 条消息，内容比较详细，包含了很多信息。"
                * 3,
                sender_id=str(10000 + i % 5),
                sender_name=f"成员{i % 5}",
                group_id="99999",
                platform="aiocqhttp",
                metadata={},
            )
        )

    llm = _DummyLLMProvider(
        """{
            "summary":"群聊成员进行了多轮讨论，涉及多个话题",
            "topics":["群聊","讨论"],
            "key_facts":["多名成员参与讨论","讨论内容丰富"],
            "participants":["成员0","成员1","成员2","成员3","成员4"],
            "sentiment":"neutral",
            "importance":0.6
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    content, metadata, importance = await processor.process_conversation(
        messages=long_messages,
        is_group_chat=True,
        persona_id=None,
    )

    assert isinstance(content, str) and len(content) > 0
    assert metadata["interaction_type"] == "group_chat"
    assert len(metadata["participants"]) == 5
    assert 0.0 <= importance <= 1.0


@pytest.mark.asyncio
async def test_process_group_chat_quality_low_for_generic_terms():
    """群聊总结包含泛化词时，summary_quality 应为 low。"""
    llm = _DummyLLMProvider(
        """{
            "summary":"某用户在群里说了一些话",
            "topics":["闲聊"],
            "key_facts":["有人说话了"],
            "participants":["某用户"],
            "sentiment":"neutral",
            "importance":0.4
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    _, metadata, _ = await processor.process_conversation(
        messages=_make_group_messages(),
        is_group_chat=True,
        persona_id=None,
    )

    assert metadata.get("summary_quality") == "low"


def test_format_conversation_sanitizes_multimodal_private_message():
    processor = MemoryProcessor(llm_provider=None, context=None)
    message = Message(
        id=1,
        session_id="s1",
        role="user",
        content=[
            {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
            {"type": "text", "text": "这张图里有会议安排"},
        ],
        sender_id="u1",
        sender_name="张三",
        group_id=None,
        platform="test",
        metadata={},
    )

    formatted = processor._format_conversation([message])

    assert "这张图里有会议安排" in formatted
    assert "image_url" not in formatted
    assert "example.test" not in formatted


def test_emotion_source_text_ignores_media_placeholders():
    processor = MemoryProcessor(llm_provider=None, context=None)
    pure_media = Message(
        id=1,
        session_id="s1",
        role="user",
        content=[
            {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}
        ],
        sender_id="u1",
        sender_name="张三",
        group_id=None,
        platform="test",
        metadata={},
    )
    media_with_text = Message(
        id=2,
        session_id="s1",
        role="user",
        content=[
            {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
            {"type": "text", "text": "我要去上班了"},
        ],
        sender_id="u1",
        sender_name="张三",
        group_id=None,
        platform="test",
        metadata={},
    )

    assistant_reply = Message(
        id=3,
        session_id="s1",
        role="assistant",
        content="我看到你发来的图片了",
        sender_id="bot",
        sender_name="Bot",
        group_id=None,
        platform="test",
        metadata={"is_bot_message": True},
    )
    truncated_media_context = Message(
        id=4,
        session_id="s1",
        role="user",
        content=(
            "<!-- astrbot-chat-merger:image-context:v1:start -->"
            '<image_context id="图1">画面描述未闭合'
        ),
        sender_id="u1",
        sender_name="张三",
        group_id=None,
        platform="test",
        metadata={},
    )

    assert processor._has_emotion_source_text([pure_media]) is False
    assert processor._has_emotion_source_text([pure_media, assistant_reply]) is False
    assert processor._has_emotion_source_text([truncated_media_context]) is False
    assert processor._has_emotion_source_text([media_with_text]) is True


def test_format_conversation_uses_placeholder_for_image_only_group_message():
    processor = MemoryProcessor(llm_provider=None, context=None)
    message = Message(
        id=1,
        session_id="g1",
        role="user",
        content=[
            {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}
        ],
        sender_id="u1",
        sender_name="张三",
        group_id="group1",
        platform="test",
        metadata={},
    )

    formatted = processor._format_conversation([message])

    assert "张三" in formatted
    assert "[图片消息]" in formatted
    assert "image_url" not in formatted


def test_emotional_observation_source_filter_requires_direct_unquoted_attack_evidence():
    processor = MemoryProcessor(llm_provider=None, context=None)
    direct = Message(
        id=1,
        session_id="s1",
        role="user",
        content="你真恶心，滚开",
        sender_id="u1",
        metadata={},
    )
    ambiguous = Message(
        id=2,
        session_id="s1",
        role="user",
        content="恶心死了",
        sender_id="u1",
        metadata={},
    )
    third_party = Message(
        id=3,
        session_id="s1",
        role="user",
        content="地铁站老头身上臭烘烘的，恶心死了",
        sender_id="u1",
        metadata={},
    )
    quoted = Message(
        id=4,
        session_id="s1",
        role="user",
        content="你真恶心，滚开",
        sender_id="u1",
        metadata={"has_reply": True, "quoted_texts": ["你真恶心，滚开"]},
    )

    def hostile(quote: str) -> dict[str, object]:
        return {
            "action": "create",
            "fact": quote,
            "emotional_meaning": "受到攻击",
            "target": "user",
            "evidence_quote": quote,
            "evidence_speaker": "user",
            "category": "psychological",
            "confidence": 0.9,
            "intensity": 0.8,
            "tags": ["abuse"],
        }

    assert (
        processor._filter_emotional_observations_by_source(
            [hostile("你真恶心，滚开")], [direct]
        )[0]["target_basis"]
        == "explicit_user_subject"
    )
    assert (
        processor._filter_emotional_observations_by_source(
            [hostile("恶心死了")], [ambiguous]
        )
        == []
    )
    assert (
        processor._filter_emotional_observations_by_source(
            [hostile("地铁站老头身上臭烘烘的，恶心死了")], [third_party]
        )
        == []
    )
    assert (
        processor._filter_emotional_observations_by_source(
            [hostile("你真恶心，滚开")], [quoted]
        )
        == []
    )


def test_emotional_observation_normalization_preserves_attribution_fields():
    processor = MemoryProcessor(llm_provider=None, context=None)

    normalized = processor._normalize_emotional_observations(
        [
            {
                "action": "create",
                "fact": "你真恶心，滚开",
                "emotional_meaning": "受到攻击",
                "target": "user",
                "target_basis": "explicit_user_subject",
                "evidence_quote": "你真恶心，滚开",
                "evidence_speaker": "user",
                "category": "psychological",
                "confidence": 0.9,
                "tags": ["abuse"],
            }
        ],
        is_group_chat=False,
    )

    assert normalized[0]["target"] == "user"
    assert normalized[0]["target_basis"] == "explicit_user_subject"
    assert normalized[0]["evidence_quote"] == "你真恶心，滚开"
    assert normalized[0]["evidence_speaker"] == "user"


def test_emotional_observation_preserves_complete_fact_past_240_chars():
    processor = MemoryProcessor(llm_provider=None, context=None)
    fact = "这是一段已经完整写完的情绪事实。" * 16
    assert 240 < len(fact) < 600

    normalized = processor._normalize_emotional_observations(
        [{"action": "create", "fact": fact, "confidence": 0.9}],
        is_group_chat=False,
    )

    assert normalized[0]["fact"] == fact


def test_emotional_observation_overflow_uses_punctuation_boundary():
    processor = MemoryProcessor(llm_provider=None, context=None)
    fact = "这是一段完整的情绪事实。" * 70 + "我当这个残句不应该裸露"

    normalized = processor._normalize_emotional_observations(
        [{"action": "create", "fact": fact, "confidence": 0.9}],
        is_group_chat=False,
    )

    bounded = normalized[0]["fact"]
    assert len(bounded) <= 600
    assert bounded.endswith("。…")
    assert "我当" not in bounded


def test_private_prompt_requires_complete_bounded_emotional_facts():
    processor = MemoryProcessor(llm_provider=None, context=None)

    assert "控制在320个中文字符以内" in processor.private_chat_prompt
    assert "不能在半句话中间截断" in processor.private_chat_prompt


def test_memory_language_directive_modes():
    """memory_language 配置应生成对应的语言指令（zh 默认 / mixed 照录英文）。"""
    from astrbot_plugin_livingmemory.core.processors.memory_processor import (
        MemoryProcessor,
    )

    processor = MemoryProcessor(context=None, config={"memory_language": "mixed"})
    mixed = processor._memory_language_directive()
    assert "中文" in mixed
    assert "原样保留英文" in mixed

    processor_default = MemoryProcessor(context=None, config={})
    zh = processor_default._memory_language_directive()
    assert "中文" in zh
    assert "原样保留英文" not in zh

    processor_bad = MemoryProcessor(context=None, config={"memory_language": "xx"})
    assert "中文" in processor_bad._memory_language_directive()
