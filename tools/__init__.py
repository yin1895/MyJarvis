# Jarvis Cortex Protocol - Tools Package
# tools/__init__.py

"""
Tool implementations for the Jarvis ecosystem.

V7.0 Native Tools:
- Native LangChain tools using @tool decorator
- Direct integration with LangGraph
- Risk level attributes for safety routing

Legacy V6 Tools:
- Tools inheriting from BaseTool (deprecated)
- Will be phased out in future versions

To add a new native tool:
1. Create a new file (e.g., native_xxx.py)
2. Define a Pydantic InputSchema
3. Use @tool(args_schema=...) decorator
4. Set tool.risk_level attribute ("safe" or "dangerous")
"""

# ============== V7.0 Native Tools ==============
# These are the recommended tools for LangGraph integration

from tools.native_system import system_control, SystemControlInput
from tools.native_file import file_operation, FileOperationInput
from tools.native_shell import shell_execute, ShellExecuteInput
from tools.native_python import python_interpreter, PythonInterpreterInput
from tools.native_browser import browser_navigate, BrowserNavigateInput
from tools.native_memory import memory_operation, MemoryOperationInput
from tools.native_knowledge import knowledge_query, knowledge_ingest, KnowledgeQueryInput, KnowledgeIngestInput
from tools.native_role import switch_role, SwitchRoleInput, ROLE_SWITCH_MARKER
from tools.native_vision import vision_analyze, VisionAnalyzeInput

# Native tool collection
NATIVE_TOOLS = [
    # Safe tools
    switch_role,
    system_control,
    memory_operation,
    knowledge_query,
    vision_analyze,  # V7.0: Native vision tool
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


# ============== Legacy V6 Tools (Deprecated) ==============
# These tools are kept for backward compatibility but will be removed
# Using lazy imports to avoid dependency issues

def _import_legacy_tools():
    """Lazy import legacy tools to avoid dependency issues."""
    global MemoryTool, KnowledgeQueryTool, KnowledgeIngestTool, VisionTool
    global BrowserTool, SystemTool, SchedulerTool, PythonExecutorTool
    global ShellTool, WebSearchTool, GetTimeTool, GetWeatherTool
    
    from tools.memory_tool import MemoryTool
    from tools.knowledge_tool import KnowledgeQueryTool, KnowledgeIngestTool
    from tools.vision_tool import VisionTool
    from tools.browser_tool import BrowserTool
    from tools.system_tool import SystemTool
    from tools.scheduler_tool import SchedulerTool
    from tools.python_tool import PythonExecutorTool
    from tools.shell_tool import ShellTool
    from tools.search_tool import WebSearchTool, GetTimeTool, GetWeatherTool


# Legacy tools will be None until explicitly imported
MemoryTool = None
KnowledgeQueryTool = None
KnowledgeIngestTool = None
VisionTool = None
BrowserTool = None
SystemTool = None
SchedulerTool = None
PythonExecutorTool = None
ShellTool = None
WebSearchTool = None
GetTimeTool = None
GetWeatherTool = None


# ============== Exports ==============

__all__ = [
    # V7.0 Native Tools (recommended)
    "switch_role",
    "system_control",
    "file_operation",
    "shell_execute",
    "python_interpreter",
    "browser_navigate",
    "memory_operation",
    "knowledge_query",
    "knowledge_ingest",
    # Native tool schemas
    "SwitchRoleInput",
    "SystemControlInput",
    "FileOperationInput",
    "ShellExecuteInput",
    "PythonInterpreterInput",
    "BrowserNavigateInput",
    "MemoryOperationInput",
    "KnowledgeQueryInput",
    "KnowledgeIngestInput",
    # Vision tool
    "vision_analyze",
    "VisionAnalyzeInput",
    # Role switch marker
    "ROLE_SWITCH_MARKER",
    # Helper functions
    "get_native_tools",
    "get_safe_native_tools",
    "get_dangerous_native_tools",
    "get_tool_risk_level",
    "NATIVE_TOOLS",
    # Legacy V6 tools (deprecated)
    "MemoryTool",
    "KnowledgeQueryTool", 
    "KnowledgeIngestTool",
    "VisionTool",
    "BrowserTool",
    "SystemTool",
    "SchedulerTool",
    "PythonExecutorTool",
    "ShellTool",
    "WebSearchTool",
    "GetTimeTool",
    "GetWeatherTool",
]
