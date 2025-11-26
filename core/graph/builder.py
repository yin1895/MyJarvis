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
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool

from core.graph.state import AgentState, NodeOutput
from core.llm_provider import LLMFactory, RoleType
from config import Config

# Import tools from centralized registry (single source of truth)
from tools import get_native_tools, get_tool_risk_level

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

def get_system_prompt() -> str:
    """
    Get the system prompt from config with tool instructions.
    
    Combines user's personality prompt with tool usage instructions.
    """
    personality = getattr(Config, 'PERSONALITY_PROMPT', '')
    
    tool_instructions = """
你可以使用以下工具来帮助用户：

【安全工具 - 自动执行】
- system_control: 系统控制（音量、亮度、媒体、打开应用）
- memory_operation: 记忆管理（记住用户偏好、添加备忘、查询档案）
- knowledge_query: 知识库查询（RAG 检索已学习的文档）
- vision_analyze: 视觉分析（截取屏幕并分析内容）
- switch_role: 切换 AI 模式（default/smart/coder/fast/vision）

【危险工具 - 需要用户确认】
- file_operation: 文件操作（读取、写入、列目录、删除）
- shell_execute: 执行 Shell 命令
- python_interpreter: 执行 Python 代码
- browser_navigate: 浏览器自动化
- knowledge_ingest: 学习文件到知识库

请根据用户的需求选择合适的工具。如果不需要工具，直接回答即可。
"""
    
    return personality + "\n" + tool_instructions


# Legacy constant for backward compatibility
DEFAULT_SYSTEM_PROMPT = get_system_prompt()


# ============== Graph Nodes ==============

async def chatbot_node(state: AgentState) -> NodeOutput:
    """
    Main chatbot node that processes user messages and generates responses.
    
    This node:
    1. Gets the current LLM based on the role in state
    2. Binds all available tools to the LLM
    3. Prepends a system message if not present
    4. Invokes the LLM with the message history
    5. Returns the response (may contain tool calls)
    
    Args:
        state: The current agent state containing messages and metadata
        
    Returns:
        A dict with the new message(s) to append to state
    """
    messages = state.get("messages", [])
    role_str = state.get("current_role") or "default"
    role = cast(RoleType, role_str)
    
    # Get the LLM for the current role
    llm = LLMFactory.create(role)
    
    # Bind tools to LLM
    tools = get_all_tools()
    llm_with_tools = llm.bind_tools(tools)
    
    # Prepare messages with system prompt if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        system_prompt = get_system_prompt()
        messages_to_send = [SystemMessage(content=system_prompt)] + list(messages)
    else:
        messages_to_send = list(messages)
    
    logger.debug(f"Invoking LLM with {len(messages_to_send)} messages and {len(tools)} tools")
    
    # Invoke the LLM asynchronously
    response: AIMessage = await llm_with_tools.ainvoke(messages_to_send)
    
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
    
    # Define the graph edges
    # START -> chatbot
    workflow.add_edge(START, "chatbot")
    
    # chatbot -> tools (if tool_calls) or END
    # tools_condition routes to "tools" if there are tool calls, otherwise to END
    workflow.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    
    # tools -> chatbot (loop back after tool execution)
    workflow.add_edge("tools", "chatbot")
    
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
