# Jarvis V7.0 - Agents Package
# agents/__init__.py

"""
Agent implementations for the Jarvis ecosystem.

V7.0 Note:
- ManagerAgent has been deprecated in favor of LangGraph workflow.
- Specialized agents (VisionAgent, etc.) are kept as tool backends.
"""

from agents.base import BaseAgent

__all__ = [
    "BaseAgent",
]