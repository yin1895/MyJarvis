# Jarvis V7.0 - Tools Package
# tools/__init__.py

"""
Tool implementations for the Jarvis ecosystem.

V7.0 Native Tools:
- Native LangChain tools using @tool decorator
- Direct integration with LangGraph
- Risk level attributes for safety routing
"""

# ============== V7.0 Native Tools ==============
# These are the only tools used in the current system

from tools.system import system_control, SystemControlInput
from tools.file import file_operation, FileOperationInput
from tools.shell import shell_execute, ShellExecuteInput
from tools.python import python_interpreter, PythonInterpreterInput
from tools.browser import browser_navigate, BrowserNavigateInput
from tools.memory import memory_operation, MemoryOperationInput
from tools.knowledge import knowledge_query, knowledge_ingest, KnowledgeQueryInput, KnowledgeIngestInput
from tools.role import switch_role, SwitchRoleInput, ROLE_SWITCH_MARKER
from tools.vision import vision_analyze, VisionAnalyzeInput

# Native tool collection
NATIVE_TOOLS = [
    # Safe tools
    switch_role,
    system_control,
    memory_operation,
    knowledge_query,
    vision_analyze,
    # Dangerous tools
    file_operation,
    shell_execute,
    python_interpreter,
    browser_navigate,
    knowledge_ingest,
]


def get_tool_risk_level(tool) -> str:
    """Get risk level from tool metadata."""
    if hasattr(tool, 'metadata') and isinstance(tool.metadata, dict):
        return tool.metadata.get('risk_level', 'safe')
    return 'safe'


def get_native_tools():
    """Get all native LangChain tools."""
    return NATIVE_TOOLS.copy()


def get_safe_native_tools():
    """Get native tools with risk_level == 'safe'."""
    return [t for t in NATIVE_TOOLS if get_tool_risk_level(t) == "safe"]


def get_dangerous_native_tools():
    """Get native tools with risk_level == 'dangerous'."""
    return [t for t in NATIVE_TOOLS if get_tool_risk_level(t) == "dangerous"]


# ============== Exports ==============

__all__ = [
    # V7.0 Native Tools
    "switch_role",
    "system_control",
    "file_operation",
    "shell_execute",
    "python_interpreter",
    "browser_navigate",
    "memory_operation",
    "knowledge_query",
    "knowledge_ingest",
    "vision_analyze",
    # Input Schemas
    "SwitchRoleInput",
    "SystemControlInput",
    "FileOperationInput",
    "ShellExecuteInput",
    "PythonInterpreterInput",
    "BrowserNavigateInput",
    "MemoryOperationInput",
    "KnowledgeQueryInput",
    "KnowledgeIngestInput",
    "VisionAnalyzeInput",
    # Role switch marker
    "ROLE_SWITCH_MARKER",
    # Helper functions
    "get_native_tools",
    "get_safe_native_tools",
    "get_dangerous_native_tools",
    "get_tool_risk_level",
    "NATIVE_TOOLS",
]
