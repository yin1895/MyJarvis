# Jarvis Cortex Protocol - Memory Tool
# tools/memory_tool.py

"""
MemoryTool: Manage user profile and notes.

Migrated from: ManagerAgent._handle_memory_update + direct service calls
Risk Level: SAFE (no external side effects, local file storage only)
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field

from core.tools.base import BaseTool, RiskLevel, ToolResult
from services.memory_service import MemoryService


class MemoryInput(BaseModel):
    """Input schema for memory operations."""
    action: Literal["add_note", "update_profile"] = Field(
        ...,
        description="要执行的操作类型: add_note (添加备忘) 或 update_profile (更新档案)"
    )
    key: Optional[str] = Field(
        default=None,
        description="档案字段名 (仅 update_profile 时需要)，如 'name' 或自定义偏好键"
    )
    value: str = Field(
        ...,
        description="要存储的内容: 备忘内容 (add_note) 或 字段值 (update_profile)"
    )


class MemoryTool(BaseTool[MemoryInput]):
    """
    Manage user profile and notes.
    
    Features:
    - Add notes/reminders to user's memory
    - Update user profile fields (name, preferences)
    - Persistent storage to data/user_profile.json
    
    This tool wraps MemoryService and provides a standardized interface
    for the Cortex Protocol.
    """
    
    name = "memory_tool"
    description = "管理用户档案和备忘录。可添加备忘 (add_note) 或更新个人信息 (update_profile)。"
    risk_level = RiskLevel.SAFE
    InputSchema = MemoryInput
    tags = ["memory", "profile", "notes", "user"]
    
    def __init__(self):
        super().__init__()
        self.memory_service = MemoryService()
    
    def execute(self, params: MemoryInput) -> ToolResult:
        """Execute memory operation based on action type."""
        try:
            if params.action == "add_note":
                return self._add_note(params.value)
            elif params.action == "update_profile":
                return self._update_profile(params.key, params.value)
            else:
                return ToolResult(
                    success=False,
                    error=f"未知操作类型: {params.action}"
                )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"记忆操作失败: {str(e)}",
                metadata={"action": params.action}
            )
    
    def _add_note(self, content: str) -> ToolResult:
        """Add a note to user's memory."""
        if not content or not content.strip():
            return ToolResult(
                success=False,
                error="备忘内容不能为空"
            )
        
        self.memory_service.add_note(content.strip())
        
        return ToolResult(
            success=True,
            data={
                "action": "add_note",
                "content": content.strip(),
                "message": "已添加到备忘录"
            },
            metadata={"notes_count": len(self.memory_service.profile.get("notes", []))}
        )
    
    def _update_profile(self, key: Optional[str], value: str) -> ToolResult:
        """Update user profile field."""
        if not key:
            return ToolResult(
                success=False,
                error="更新档案需要指定字段名 (key)"
            )
        
        if not value or not value.strip():
            return ToolResult(
                success=False,
                error="字段值不能为空"
            )
        
        self.memory_service.update_profile(key, value.strip())
        
        # Determine where the value was stored
        location = "根字段" if key == "name" else "偏好设置"
        
        return ToolResult(
            success=True,
            data={
                "action": "update_profile",
                "key": key,
                "value": value.strip(),
                "location": location,
                "message": f"已更新{location}: {key} = {value.strip()}"
            }
        )
