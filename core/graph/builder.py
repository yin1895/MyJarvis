"""
Graph Builder for Jarvis V7.0

This module constructs the LangGraph workflow with nodes and edges.
Phase 3 implements persistence and safety interceptor.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Sequence, Union, cast, Any
from functools import lru_cache

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import BaseTool

from core.graph.state import AgentState, NodeOutput
from core.llm_provider import LLMFactory, RoleType
from config import Config

# Import tools from centralized registry (single source of truth)
from tools import get_native_tools, get_tool_risk_level
from tools.role import ROLE_SWITCH_MARKER

logger = logging.getLogger(__name__)


# ============== Tool Registry ==============

@lru_cache(maxsize=1)
def _build_tool_registry() -> dict[str, BaseTool]:
    """Build tool registry once and cache it."""
    return {tool.name: tool for tool in get_all_tools()}


def get_tool_by_name(name: str) -> Optional[BaseTool]:
    """
    Get a tool instance by its name.
    
    Args:
        name: The tool name
        
    Returns:
        The tool instance or None if not found
    """
    return _build_tool_registry().get(name)


def get_all_tools() -> List[BaseTool]:
    """
    Get all available native tools from centralized registry.
    
    Returns:
        List of LangChain tool instances
    """
    return get_native_tools()


def get_safe_tools() -> List[BaseTool]:
    """
    Get only safe tools (risk_level == "safe").
    
    Returns:
        List of safe tool instances
    """
    return [t for t in get_all_tools() if get_tool_risk_level(t) == "safe"]


def get_dangerous_tools() -> List[BaseTool]:
    """
    Get dangerous tools (risk_level == "dangerous").
    
    Returns:
        List of dangerous tool instances
    """
    return [t for t in get_all_tools() if get_tool_risk_level(t) == "dangerous"]


def check_tool_calls_safety(tool_calls: Sequence[Union[dict, Any]]) -> tuple[bool, List[str]]:
    """
    Check if all tool calls are safe.
    
    Args:
        tool_calls: Sequence of tool call objects (dict or ToolCall) from AIMessage
        
    Returns:
        Tuple of (all_safe: bool, dangerous_tool_names: List[str])
    """
    dangerous_tools = []
    
    for call in tool_calls:
        # Support both dict and ToolCall objects
        tool_name = call.get("name", "") if isinstance(call, dict) else getattr(call, "name", "")
        tool = get_tool_by_name(tool_name)
        
        if tool is None:
            # Unknown tool - treat as dangerous
            dangerous_tools.append(tool_name)
        elif get_tool_risk_level(tool) == "dangerous":
            dangerous_tools.append(tool_name)
    
    return len(dangerous_tools) == 0, dangerous_tools


# ============== System Prompt ==============

def get_system_prompt(mode: str = "text", role: str = "default") -> str:
    """
    Generate dynamic system prompt based on interaction mode and role.
    
    根据交互模式（语音/文字）和当前角色动态生成 system prompt，
    确保语音模式下输出简洁、文字模式下可以详细。
    
    Args:
        mode: "voice" 或 "text"，影响输出风格约束
        role: LLM 角色，如 "default", "smart", "coder", "vision", "fast"
        
    Returns:
        完整的 system prompt 字符串
    """
    personality = Config.PERSONALITY
    base = personality.get("base", {})
    voice_cfg = personality.get("voice_mode", {})
    text_cfg = personality.get("text_mode", {})
    role_traits = personality.get("roles", {})
    
    # 基础人格
    name = base.get("name", "Jarvis")
    trait = base.get("trait", "简洁、专业、友好")
    language = base.get("language", "中文")
    
    prompt_parts = [
        f"你是 {name}，一个智能 AI 助手。",
        f"你的特点：{trait}",
        f"使用{language}与用户交流。",
    ]
    
    # 角色特定人格
    if role in role_traits:
        prompt_parts.append(f"\n【当前角色模式】{role}: {role_traits[role]}")
    
    # 交互模式约束（核心差异点）
    if mode == "voice":
        style = voice_cfg.get("style", "极度简洁，1-2句话")
        rules = voice_cfg.get("rules", [])
        prompt_parts.append(f"\n【语音模式 - 极其重要】\n风格要求：{style}")
        if rules:
            prompt_parts.append("必须遵守的规则：")
            for rule in rules:
                prompt_parts.append(f"- {rule}")
        # 正反例对比
        bad = voice_cfg.get("example_bad")
        good = voice_cfg.get("example_good")
        if bad and good:
            prompt_parts.append(f"\n❌ 不要这样回答：\"{bad}\"")
            prompt_parts.append(f"✅ 要这样回答：\"{good}\"")
    else:
        # 文字模式
        style = text_cfg.get("style", "清晰准确，可以适当详细")
        rules = text_cfg.get("rules", [])
        prompt_parts.append(f"\n【文字模式】\n风格：{style}")
        if rules:
            for rule in rules:
                prompt_parts.append(f"- {rule}")
    
    # 工具说明
    tool_instructions = """
【可用工具】
安全工具（自动执行）：system_control, memory_operation, knowledge_query, vision_analyze, switch_role
危险工具（需确认）：file_operation, shell_execute, python_interpreter, browser_navigate, knowledge_ingest

根据用户需求选择合适的工具，不需要工具时直接回答。"""
    
    prompt_parts.append(tool_instructions)
    
    return "\n".join(prompt_parts)


# Legacy constant for backward compatibility
DEFAULT_SYSTEM_PROMPT = get_system_prompt()


# ============== Graph Nodes ==============

async def state_updater_node(state: AgentState) -> NodeOutput:
    """
    Post-tool node that checks for role switch markers in tool results.
    
    This node runs after tools execute and before returning to chatbot.
    It detects ROLE_SWITCH_MARKER in ToolMessage content and updates
    the current_role in state accordingly.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with new current_role if switch detected
    """
    messages = state.get("messages", [])
    current_role = state.get("current_role", "default")
    
    # Check recent messages for role switch marker
    # Only look at last few messages to avoid old matches
    for msg in reversed(messages[-5:]):
        # Check if this is a ToolMessage from switch_role
        msg_name = getattr(msg, 'name', '')
        if msg_name == 'switch_role':
            content = str(getattr(msg, 'content', ''))
            if ROLE_SWITCH_MARKER in content:
                # Extract new role from marker
                for line in content.split('\n'):
                    if line.startswith(ROLE_SWITCH_MARKER):
                        new_role = line.split(':')[1].strip()
                        if new_role and new_role != current_role:
                            logger.info(f"Role switch detected: {current_role} -> {new_role}")
                            return {"current_role": new_role}
                break
    
    # No role switch detected
    return {}


def _sanitize_messages_for_gemini(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    清理消息历史以符合 Gemini API 的严格要求。
    
    Gemini API 对消息序列有特殊要求：
    1. function call (AIMessage with tool_calls) 后必须紧跟 function response (ToolMessage)
    2. 不能有连续的 AI 消息（除非是 tool response 后）
    3. 消息序列必须以 user message 或 function response 开始（相对于上一个 AI turn）
    
    此函数通过以下策略确保兼容性：
    - 移除孤立的 tool_calls（没有对应 ToolMessage 的 AIMessage.tool_calls）
    - 确保 tool_calls 和 ToolMessage 配对完整
    
    Args:
        messages: 原始消息列表
        
    Returns:
        清理后的消息列表，符合 Gemini API 要求
    """
    if not messages:
        return messages
    
    sanitized = []
    i = 0
    
    while i < len(messages):
        msg = messages[i]
        
        # 检查是否是带有 tool_calls 的 AIMessage
        if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
            tool_calls = msg.tool_calls
            tool_call_ids = {tc.get('id') or tc.id for tc in tool_calls if hasattr(tc, 'id') or isinstance(tc, dict)}
            
            # 查找后续的 ToolMessage，确保所有 tool_calls 都有对应的响应
            j = i + 1
            found_tool_messages = []
            found_ids = set()
            
            while j < len(messages):
                next_msg = messages[j]
                if isinstance(next_msg, ToolMessage):
                    tool_call_id = getattr(next_msg, 'tool_call_id', None)
                    if tool_call_id in tool_call_ids:
                        found_tool_messages.append(next_msg)
                        found_ids.add(tool_call_id)
                        j += 1
                        continue
                # 遇到非 ToolMessage 或不匹配的 ToolMessage，停止搜索
                break
            
            # 检查是否所有 tool_calls 都有对应的 ToolMessage
            if found_ids == tool_call_ids and len(found_ids) > 0:
                # 完整的 tool call 序列，保留
                sanitized.append(msg)
                sanitized.extend(found_tool_messages)
                i = j
            else:
                # 不完整的 tool call 序列
                # 创建一个没有 tool_calls 的新 AIMessage，只保留 content
                if msg.content:
                    # 保留文本内容，移除 tool_calls
                    clean_msg = AIMessage(content=msg.content)
                    sanitized.append(clean_msg)
                    logger.debug(f"Sanitized incomplete tool_calls from AIMessage")
                # 跳过孤立的 ToolMessage
                i = j if j > i + 1 else i + 1
        else:
            # 普通消息，直接保留
            sanitized.append(msg)
            i += 1
    
    # 最后检查：确保不以 ToolMessage 结尾（除非后面紧跟 AI 响应）
    # Gemini 要求最后一条消息必须是 user 或 AI（非 tool_calls）
    while sanitized and isinstance(sanitized[-1], ToolMessage):
        logger.debug("Removing trailing ToolMessage for Gemini compatibility")
        sanitized.pop()
    
    return sanitized


async def chatbot_node(state: AgentState) -> NodeOutput:
    """
    Main chatbot node that processes user messages and generates responses.
    
    This node:
    1. Gets the current LLM based on the role in state
    2. Binds all available tools to the LLM
    3. Prepends a system message based on interaction_mode and role
    4. Truncates message history to avoid context overflow
    5. Invokes the LLM with the message history
    6. Returns the response (may contain tool calls)
    
    Args:
        state: The current agent state containing messages and metadata
        
    Returns:
        A dict with the new message(s) to append to state
    """
    messages = state.get("messages", [])
    role_str = state.get("current_role") or "default"
    mode = state.get("interaction_mode") or "text"  # 默认文字模式
    role = cast(RoleType, role_str)
    
    # Get the LLM for the current role
    llm = LLMFactory.create(role)
    
    # Bind tools to LLM
    tools = get_all_tools()
    llm_with_tools = llm.bind_tools(tools)
    
    # Generate dynamic system prompt based on mode and role
    system_prompt = get_system_prompt(mode=mode, role=role_str)
    
    # Filter out old system messages to avoid confusion
    filtered_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    
    # 🔧 消息截断：避免 context 超出限制
    # 保留最近的 N 条消息（可通过 Config 配置）
    MAX_HISTORY_MESSAGES = getattr(Config, 'MAX_HISTORY_MESSAGES', 30)
    if len(filtered_messages) > MAX_HISTORY_MESSAGES:
        # 保留最近的消息，确保最后一条是用户消息
        filtered_messages = filtered_messages[-MAX_HISTORY_MESSAGES:]
        logger.info(f"Truncated message history to {MAX_HISTORY_MESSAGES} messages")
    
    # 🔧 Gemini 兼容性处理：清理不完整的 tool_calls 序列
    # Gemini API 要求：function call 后必须紧跟 function response
    # 如果历史消息中有孤立的 tool_calls（没有对应的 ToolMessage），会导致错误
    if role_str in ("vision", "smart") or "gemini" in str(getattr(llm, 'model', '')).lower():
        filtered_messages = _sanitize_messages_for_gemini(filtered_messages)
    
    messages_to_send = [SystemMessage(content=system_prompt)] + filtered_messages
    
    logger.debug(f"Invoking LLM with {len(messages_to_send)} messages, mode={mode}, role={role_str}")
    
    # Invoke the LLM asynchronously with error handling
    try:
        response: AIMessage = await llm_with_tools.ainvoke(messages_to_send)
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        # 返回错误消息而不是崩溃
        error_msg = f"抱歉，AI 模型调用失败：{str(e)[:100]}"
        return {"messages": [AIMessage(content=error_msg)]}
    
    # 检查空响应
    if not response.content and not response.tool_calls:
        logger.warning("LLM returned empty response")
        return {"messages": [AIMessage(content="抱歉，我没有收到有效的响应。请重试或检查网络连接。")]}
    
    logger.debug(f"LLM response: {str(response.content)[:100]}...")
    
    # Return the response to be added to messages via the reducer
    return {"messages": [response]}


def create_graph(
    role: RoleType = "default",
    system_prompt: Optional[str] = None,
    checkpointer: Optional[Any] = None,
    interrupt_before_tools: bool = True
) -> CompiledStateGraph:
    """
    Create and compile a new LangGraph workflow with tool support.
    
    This creates a graph with:
    - chatbot node: LLM with bound tools
    - tools node: ToolNode for executing tool calls
    - Routing: chatbot -> tools (if tool_calls) -> chatbot, or chatbot -> END
    - Optional persistence via checkpointer
    - Optional interrupt before tools for safety
    
    Graph flow:
        START -> chatbot -> tools_condition -> tools -> chatbot
                                           -> END
    
    Args:
        role: Default LLM role for the graph
        system_prompt: Optional custom system prompt
        checkpointer: Optional SQLite or memory checkpointer for persistence
        interrupt_before_tools: If True, interrupt before tool execution for safety check
        
    Returns:
        A compiled LangGraph StateGraph ready for execution
        
    Example:
        >>> from langgraph.checkpoint.sqlite import SqliteSaver
        >>> checkpointer = SqliteSaver.from_conn_string("data/state.db")
        >>> graph = create_graph(checkpointer=checkpointer, interrupt_before_tools=True)
    """
    # Get all tools
    tools = get_all_tools()
    
    # Create the state graph with our AgentState schema
    workflow = StateGraph(AgentState)
    
    # Add the chatbot node
    workflow.add_node("chatbot", chatbot_node)
    
    # Add the tools node using prebuilt ToolNode
    tool_node = ToolNode(tools=tools)
    workflow.add_node("tools", tool_node)
    
    # Add state updater node (runs after tools, updates role if needed)
    workflow.add_node("state_updater", state_updater_node)
    
    # Define the graph edges
    # START -> chatbot
    workflow.add_edge(START, "chatbot")
    
    # chatbot -> tools (if tool_calls) or END
    # tools_condition routes to "tools" if there are tool calls, otherwise to END
    workflow.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    
    # tools -> state_updater -> chatbot (loop back after tool execution)
    workflow.add_edge("tools", "state_updater")
    workflow.add_edge("state_updater", "chatbot")
    
    # Compile options
    compile_kwargs: dict[str, Any] = {}
    
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    
    if interrupt_before_tools:
        # Interrupt before tools node for safety inspection
        compile_kwargs["interrupt_before"] = ["tools"]
    
    # Compile the graph
    compiled = workflow.compile(**compile_kwargs)
    
    logger.info(f"Graph compiled with {len(tools)} tools, "
                f"checkpointer={'enabled' if checkpointer else 'disabled'}, "
                f"interrupt={'enabled' if interrupt_before_tools else 'disabled'}")
    
    return compiled


# Pre-compiled default graph instance for convenience
# This is lazily evaluated when first accessed
_default_graph: Optional[CompiledStateGraph] = None


def get_graph() -> CompiledStateGraph:
    """
    Get the default compiled graph instance (singleton).
    
    Returns:
        The compiled default graph
    """
    global _default_graph
    if _default_graph is None:
        _default_graph = create_graph()
    return _default_graph


# Export a default graph instance
# Note: This is created at import time for convenience
# For custom configurations, use create_graph() instead
graph = None  # Will be lazily initialized


def init_graph() -> CompiledStateGraph:
    """
    Initialize and return the global graph instance.
    
    Call this at application startup to ensure the graph is ready.
    
    Returns:
        The initialized compiled graph
    """
    global graph
    if graph is None:
        graph = create_graph()
    return graph
