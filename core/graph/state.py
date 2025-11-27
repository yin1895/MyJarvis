"""
Agent State Definition for Jarvis V7.0

This module defines the core state structure used throughout the LangGraph
agent system. The state is immutable and passed between nodes in the graph.
"""

from __future__ import annotations

from typing import Annotated, Optional, Any, Literal
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


# 交互模式类型
InteractionMode = Literal["voice", "text"]


class AgentState(TypedDict):
    """
    Core state for the Jarvis agent graph.
    
    This state is passed between all nodes in the graph and represents
    the current context of the conversation and agent execution.
    
    Attributes:
        messages: The conversation history. Uses `add_messages` reducer
                  which automatically handles message appending and deduplication.
        current_role: The active LLM role being used (default, smart, coder, fast, vision)
        interaction_mode: Current interaction mode ("voice" or "text")
        metadata: Optional metadata for the current turn (tool results, context, etc.)
    
    The `add_messages` annotation is a LangGraph reducer that:
    - Appends new messages to the existing list
    - Handles message ID deduplication
    - Supports both single messages and lists of messages
    
    Example:
        state = {
            "messages": [HumanMessage(content="Hello")],
            "current_role": "default",
            "interaction_mode": "voice",
            "metadata": {}
        }
    """
    
    # Core message history with LangGraph's add_messages reducer
    # This automatically handles appending and deduplication
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Current LLM role being used (for multi-model support)
    current_role: Optional[str]
    
    # 交互模式：voice(语音) 或 text(文字)，影响 system prompt 生成
    interaction_mode: Optional[InteractionMode]
    
    # Flexible metadata storage for tool results, intermediate data, etc.
    metadata: Optional[dict[str, Any]]


# Type alias for return values from nodes
# Nodes can return partial state updates
NodeOutput = dict[str, Any]
