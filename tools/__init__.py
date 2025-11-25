# Jarvis Cortex Protocol - Tools Package
# tools/__init__.py

"""
Tool implementations for the Jarvis ecosystem.

All tools in this package inherit from BaseTool and will be
auto-discovered by the ToolRegistry.

To add a new tool:
1. Create a new file in this directory (e.g., my_tool.py)
2. Define a Pydantic InputSchema
3. Create a class inheriting from BaseTool
4. Implement the execute() method

The tool will be automatically registered when registry.scan("tools/") is called.
"""

# Note: Tools are also loaded dynamically by ToolRegistry.scan()

# Explicit exports for IDE support and direct imports
from tools.memory_tool import MemoryTool
from tools.knowledge_tool import KnowledgeQueryTool, KnowledgeIngestTool
from tools.vision_tool import VisionTool
from tools.browser_tool import BrowserTool
from tools.system_tool import SystemTool
from tools.scheduler_tool import SchedulerTool
from tools.python_tool import PythonExecutorTool
from tools.shell_tool import ShellTool
from tools.search_tool import WebSearchTool, GetTimeTool, GetWeatherTool

__all__ = [
    # Core tools
    "MemoryTool",
    "KnowledgeQueryTool", 
    "KnowledgeIngestTool",
    "VisionTool",
    "BrowserTool",
    "SystemTool",
    "SchedulerTool",
    # Smart tools (V6.1)
    "PythonExecutorTool",
    "ShellTool",
    # Search & utility tools
    "WebSearchTool",
    "GetTimeTool",
    "GetWeatherTool",
]
