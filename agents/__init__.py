# Jarvis Cortex Protocol - Agents Package
# agents/__init__.py

"""
Agent implementations for the Jarvis ecosystem.

This package contains specialized agents for specific task domains.
The Manager agent orchestrates routing between these agents and tools.
"""

from agents.base import BaseAgent
from agents.manager import ManagerAgent

__all__ = [
    "BaseAgent",
    "ManagerAgent",
]