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

# Note: Imports are intentionally minimal here.
# Tools are loaded dynamically by ToolRegistry.scan()
