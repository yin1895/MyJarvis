# Jarvis Cortex Protocol - Tools Subsystem

"""
Tool ecosystem components for the Jarvis Cortex Protocol.

Exports:
- BaseTool: Abstract base class for all tools
- ToolExecutor: Middleware for safe tool execution
- ToolRegistry: Dynamic tool discovery and registration
"""

from core.tools.base import BaseTool, RiskLevel, ToolResult, EmptyInput
from core.tools.executor import ToolExecutor, ConfirmationResult, get_default_executor
from core.tools.registry import ToolRegistry, get_registry, register_tool, ToolNotFoundError

__all__ = [
    # Layer 1: Base
    "BaseTool",
    "RiskLevel", 
    "ToolResult",
    "EmptyInput",
    # Layer 2: Executor
    "ToolExecutor",
    "ConfirmationResult",
    "get_default_executor",
    # Layer 3: Registry
    "ToolRegistry",
    "get_registry",
    "register_tool",
    "ToolNotFoundError",
]
