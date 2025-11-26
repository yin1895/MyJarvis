# Jarvis V7.0 - Native Role Switching Tool
# tools/native_role.py

"""
Native LangChain Tool for switching LLM roles.

This tool allows users to dynamically switch between different AI models
during a conversation. The actual model switch happens in the main loop
by detecting the special return marker.

Available Roles:
- default: 默认模型 (balanced)
- smart: 高智能模型 (GPT-4o for complex tasks)
- coder: 编程模型 (optimized for code)
- fast: 快速模型 (for quick responses)
- vision: 视觉模型 (Gemini, supports image analysis)

Risk Level: safe
"""

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from core.llm_provider import LLMFactory, RoleType


# ============== Constants ==============

# Special marker for role switch detection in main loop
ROLE_SWITCH_MARKER = "__JARVIS_SWITCH_ROLE__"

# Role aliases for natural language matching
ROLE_ALIASES = {
    # Default
    "default": "default",
    "默认": "default",
    "普通": "default",
    "normal": "default",
    
    # Smart
    "smart": "smart",
    "高智商": "smart",
    "聪明": "smart",
    "gpt4": "smart",
    "gpt-4": "smart",
    "gpt-4o": "smart",
    
    # Coder
    "coder": "coder",
    "编程": "coder",
    "代码": "coder",
    "deepseek": "coder",
    "程序员": "coder",
    
    # Fast
    "fast": "fast",
    "快速": "fast",
    "llama": "fast",
    "ollama": "fast",
    
    # Vision
    "vision": "vision",
    "视觉": "vision",
    "gemini": "vision",
    "图像": "vision",
    "看图": "vision",
    "图片": "vision",
}


# ============== Input Schema ==============

class SwitchRoleInput(BaseModel):
    """Input schema for switch_role tool."""
    role: str = Field(
        description=(
            "目标角色名称。可选值:\n"
            "- default/默认: 平衡模式\n"
            "- smart/高智商/gpt4: 高智能模式 (GPT-4o)\n"
            "- coder/编程/代码: 编程模式 (DeepSeek-Coder)\n"
            "- fast/快速/llama: 快速响应模式\n"
            "- vision/视觉/gemini: 视觉分析模式 (支持图片)"
        )
    )


# ============== Helper Functions ==============

def resolve_role_alias(role_input: str) -> str:
    """
    Resolve role alias to canonical role name.
    
    Args:
        role_input: User input (may be alias like "gemini", "高智商", etc.)
        
    Returns:
        Canonical role name (default, smart, coder, fast, vision)
    """
    normalized = role_input.strip().lower()
    return ROLE_ALIASES.get(normalized, normalized)


def get_role_description(role: str) -> str:
    """Get human-readable description of a role."""
    try:
        from typing import cast
        role_info = LLMFactory.get_role_info(cast(RoleType, role))
        descriptions = {
            "default": "默认模式 - 平衡的通用对话能力",
            "smart": "高智能模式 - GPT-4o，适合复杂推理和创意任务",
            "coder": "编程模式 - DeepSeek-Coder，优化的代码生成能力",
            "fast": "快速模式 - Llama3，本地运行，响应迅速",
            "vision": "视觉模式 - Gemini，支持图像分析和多模态理解",
        }
        desc = descriptions.get(role, "未知模式")
        return f"{desc}\n[Provider: {role_info['provider']}, Model: {role_info['model']}]"
    except Exception:
        return f"角色: {role}"


# ============== Native Tool ==============

@tool(args_schema=SwitchRoleInput)
def switch_role(role: str) -> str:
    """
    切换 AI 的模型角色。
    
    这个工具允许用户在对话中动态切换不同的 AI 模型，以获得不同的能力：
    
    - **default** (默认): 平衡的通用对话模型
    - **smart** (高智商/GPT-4): 复杂推理和创意任务
    - **coder** (编程/DeepSeek): 代码生成和技术问题
    - **fast** (快速/Llama): 本地快速响应
    - **vision** (视觉/Gemini): 图像分析和多模态理解
    
    使用示例:
    - "切换到Gemini" → 切换到视觉模式
    - "使用高智商模式" → 切换到GPT-4o
    - "换成编程模式" → 切换到DeepSeek-Coder
    
    Args:
        role: 目标角色名称（支持别名如 gemini, 高智商, 编程 等）
        
    Returns:
        切换结果说明，包含特殊标记供主循环处理
    """
    # Resolve alias to canonical role
    canonical_role = resolve_role_alias(role)
    
    # Validate role exists
    available_roles = LLMFactory.get_available_roles()
    if canonical_role not in available_roles:
        return (
            f"无法识别的角色: {role}\n"
            f"可用角色: {', '.join(available_roles)}\n"
            "请使用正确的角色名称。"
        )
    
    # Get role info
    role_desc = get_role_description(canonical_role)
    
    # Return special marker for main loop to detect
    # Format: __JARVIS_SWITCH_ROLE__:role_name
    return f"{ROLE_SWITCH_MARKER}:{canonical_role}\n已切换到 {canonical_role} 模式。\n{role_desc}"


# Set risk level (safe - no side effects)
switch_role.metadata = {"risk_level": "safe"}


# ============== Export ==============
__all__ = [
    "switch_role",
    "SwitchRoleInput",
    "ROLE_SWITCH_MARKER",
    "resolve_role_alias",
    "ROLE_ALIASES",
]
