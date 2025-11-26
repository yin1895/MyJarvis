"""
Native Memory Tool for Jarvis V7.0

Wraps MemoryService as a LangChain native tool using @tool decorator.
Provides user profile and notes management capabilities.

Risk Level: SAFE (only modifies local user data)
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from services.memory_service import MemoryService


class MemoryOperationInput(BaseModel):
    """Input schema for memory operations."""
    
    action: Literal["add_note", "update_profile", "get_profile"] = Field(
        ...,
        description="操作类型: add_note(添加备忘), update_profile(更新用户信息), get_profile(获取用户档案)"
    )
    key: Optional[str] = Field(
        None,
        description="update_profile 时必填: 要更新的字段名 (如 'name', 或存入 preferences 的键)"
    )
    value: Optional[str] = Field(
        None,
        description="add_note 时为备忘内容; update_profile 时为字段值"
    )


@tool(args_schema=MemoryOperationInput, return_direct=False)
def memory_operation(
    action: Literal["add_note", "update_profile", "get_profile"],
    key: Optional[str] = None,
    value: Optional[str] = None,
) -> str:
    """
    管理用户档案和备忘录。
    
    使用场景:
    - 当用户说 "记住我喜欢..." → add_note
    - 当用户说 "叫我..." 或 "我的名字是..." → update_profile (key='name')
    - 当用户说 "你知道我是谁吗" → get_profile
    
    Examples:
    - add_note: {"action": "add_note", "value": "主人喜欢喝拿铁咖啡"}
    - update_profile: {"action": "update_profile", "key": "name", "value": "Alice"}
    - get_profile: {"action": "get_profile"}
    """
    try:
        service = MemoryService()  # Singleton
        
        if action == "add_note":
            if not value:
                return "错误: add_note 需要提供 value (备忘内容)"
            service.add_note(value)
            return f"已记住: {value}"
        
        elif action == "update_profile":
            if not key or not value:
                return "错误: update_profile 需要提供 key 和 value"
            service.update_profile(key, value)
            if key == "name":
                return f"好的，我以后会称呼您为 {value}"
            return f"已更新 {key} = {value}"
        
        elif action == "get_profile":
            profile = service.profile
            name = profile.get("name", "主人")
            prefs = profile.get("preferences", {})
            notes = profile.get("notes", [])
            
            result = f"用户档案:\n- 称呼: {name}\n"
            if prefs:
                result += f"- 偏好: {prefs}\n"
            if notes:
                result += f"- 备忘 ({len(notes)}条): {notes[-5:]}"  # 最近5条
            else:
                result += "- 备忘: 无"
            return result
        
        else:
            return f"未知操作: {action}"
            
    except Exception as e:
        return f"Memory 操作失败: {str(e)}"


# Set risk level in metadata
memory_operation.metadata = {"risk_level": "safe"}
