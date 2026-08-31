"""Tests for plugin LLM tool registration."""

from unittest.mock import Mock

from astrbot_plugin_livingmemory.core.base.config_manager import ConfigManager
from astrbot_plugin_livingmemory.core.tools import MemoryMemorizeTool, MemorySearchTool
from astrbot_plugin_livingmemory.main import (
    LivingMemoryPlugin,
    _parse_version,
    _version_lt,
)


def test_parse_version_accepts_source_and_prerelease_versions():
    assert _parse_version("4.25.2") == (4, 25, 2)
    assert _parse_version("v4.25.2-beta.1") == (4, 25, 2)
    assert _parse_version("not-a-version") == ()


def test_version_lt_pads_version_segments():
    assert _version_lt("4.24.1", "4.24.2") is True
    assert _version_lt("4.25", "4.25.0") is False
    assert _version_lt("4.25.2", "4.24.2") is False
    assert _version_lt("unknown", "4.24.2") is False


def test_register_llm_tools_is_idempotent():
    plugin = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    plugin.context = Mock()
    plugin.config_manager = ConfigManager(
        {"agent_tools": {"enable_recall_tool": True, "enable_memorize_tool": True}}
    )
    plugin.initializer = Mock()
    plugin.initializer.memory_engine = Mock()
    plugin.initializer.memory_processor = Mock()
    plugin._llm_tools_registered = False

    plugin._register_agent_tools_if_needed()
    plugin._register_agent_tools_if_needed()

    plugin.context.add_llm_tools.assert_called_once()
    tools = plugin.context.add_llm_tools.call_args.args
    tools_by_name = {tool.name: tool for tool in tools}
    assert set(tools_by_name) == {
        "recall_long_term_memory",
        "memorize_long_term_memory",
    }
    assert isinstance(tools_by_name["recall_long_term_memory"], MemorySearchTool)
    assert isinstance(tools_by_name["memorize_long_term_memory"], MemoryMemorizeTool)
    assert plugin._llm_tools_registered is True


def test_register_llm_tools_defaults_only_recall():
    plugin = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    plugin.context = Mock()
    plugin.config_manager = ConfigManager()
    plugin.initializer = Mock()
    plugin.initializer.memory_engine = Mock()
    plugin.initializer.memory_processor = Mock()
    plugin._llm_tools_registered = False

    plugin._register_agent_tools_if_needed()

    plugin.context.add_llm_tools.assert_called_once()
    tools = plugin.context.add_llm_tools.call_args.args
    assert [tool.name for tool in tools] == ["recall_long_term_memory"]
    assert isinstance(tools[0], MemorySearchTool)
    assert plugin._llm_tools_registered is True


def test_recall_tool_layers_routing_and_query_rules_without_changing_contract():
    tool = MemorySearchTool()

    assert tool.name == "recall_long_term_memory"
    assert "过去的事实" in tool.description
    assert "完整用户消息" not in tool.description
    assert len(tool.description) < 60
    assert tool.parameters["required"] == ["query"]
    assert set(tool.parameters["properties"]) == {"query", "k"}
    assert tool.parameters["properties"]["query"]["type"] == "string"
    assert tool.parameters["properties"]["k"]["type"] == "integer"
    assert tool.parameters["properties"]["k"]["default"] == 5
    query_description = tool.parameters["properties"]["query"]["description"]
    assert "完整用户消息" in query_description
    assert "再次调用" in query_description


def test_memorize_tool_layers_memory_format_and_importance_rules():
    tool = MemoryMemorizeTool()

    assert tool.name == "memorize_long_term_memory"
    assert len(tool.description) < 70
    assert "稳定偏好" in tool.description
    assert tool.parameters["required"] == ["memory"]
    memory_description = tool.parameters["properties"]["memory"]["description"]
    importance_description = tool.parameters["properties"]["importance"]["description"]
    assert "不要复制整段对话" in memory_description
    assert "临时闲聊" in memory_description
    assert "身份事实" in importance_description


def test_register_llm_tools_no_memory_engine():
    plugin = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    plugin.context = Mock()
    plugin.config_manager = ConfigManager()
    plugin.initializer = Mock()
    plugin.initializer.memory_engine = None
    plugin.initializer.memory_processor = Mock()
    plugin._llm_tools_registered = False

    plugin._register_agent_tools_if_needed()

    plugin.context.add_llm_tools.assert_not_called()
    assert plugin._llm_tools_registered is False


def test_register_llm_tools_no_memory_processor():
    plugin = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    plugin.context = Mock()
    plugin.config_manager = ConfigManager()
    plugin.initializer = Mock()
    plugin.initializer.memory_engine = Mock()
    plugin.initializer.memory_processor = None
    plugin._llm_tools_registered = False

    plugin._register_agent_tools_if_needed()

    plugin.context.add_llm_tools.assert_not_called()
    assert plugin._llm_tools_registered is False


def test_register_llm_tools_respects_recall_tool_disabled():
    plugin = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    plugin.context = Mock()
    plugin.config_manager = ConfigManager(
        {"agent_tools": {"enable_recall_tool": False, "enable_memorize_tool": True}}
    )
    plugin.initializer = Mock()
    plugin.initializer.memory_engine = Mock()
    plugin.initializer.memory_processor = Mock()
    plugin._llm_tools_registered = False

    plugin._register_agent_tools_if_needed()

    plugin.context.add_llm_tools.assert_called_once()
    tools = plugin.context.add_llm_tools.call_args.args
    assert [tool.name for tool in tools] == ["memorize_long_term_memory"]
    assert isinstance(tools[0], MemoryMemorizeTool)
    assert plugin._llm_tools_registered is True


def test_register_llm_tools_respects_memorize_tool_disabled():
    plugin = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    plugin.context = Mock()
    plugin.config_manager = ConfigManager(
        {"agent_tools": {"enable_recall_tool": True, "enable_memorize_tool": False}}
    )
    plugin.initializer = Mock()
    plugin.initializer.memory_engine = Mock()
    plugin.initializer.memory_processor = Mock()
    plugin._llm_tools_registered = False

    plugin._register_agent_tools_if_needed()

    plugin.context.add_llm_tools.assert_called_once()
    tools = plugin.context.add_llm_tools.call_args.args
    assert [tool.name for tool in tools] == ["recall_long_term_memory"]
    assert isinstance(tools[0], MemorySearchTool)
    assert plugin._llm_tools_registered is True


def test_register_llm_tools_respects_all_tools_disabled():
    plugin = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    plugin.context = Mock()
    plugin.config_manager = ConfigManager(
        {"agent_tools": {"enable_recall_tool": False, "enable_memorize_tool": False}}
    )
    plugin.initializer = Mock()
    plugin.initializer.memory_engine = Mock()
    plugin.initializer.memory_processor = Mock()
    plugin._llm_tools_registered = False

    plugin._register_agent_tools_if_needed()

    plugin.context.add_llm_tools.assert_not_called()
    assert plugin._llm_tools_registered is True
