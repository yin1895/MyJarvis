# Jarvis Cortex Protocol - Vision Tool
# tools/vision_tool.py

"""
VisionTool: Screen capture and visual analysis.

Migrated from: agents/vision_agent.py (wrapped as Cortex Tool)
Risk Level: SAFE (passive observation, no side effects)
"""

from pydantic import BaseModel, Field

from core.tools.base import BaseTool, RiskLevel, ToolResult


class VisionInput(BaseModel):
    """Input schema for vision analysis."""
    query: str = Field(
        default="描述当前屏幕内容",
        description="视觉分析指令，如 '屏幕上显示什么' 或 '帮我看看这张图片'"
    )


class VisionTool(BaseTool[VisionInput]):
    """
    Screen capture and visual analysis using Vision LLM.
    
    Features:
    - Automatic screenshot capture
    - Multi-modal LLM analysis (Gemini/GPT-4V)
    - Image resizing for optimal processing
    
    This tool is read-only and marked as SAFE.
    The underlying VisionAgent is lazy-loaded on first use.
    """
    
    name = "vision_tool"
    description = "查看屏幕、分析图片、视觉问答。用于需要'看'的任务。"
    risk_level = RiskLevel.SAFE
    InputSchema = VisionInput
    tags = ["vision", "screenshot", "image", "multimodal"]
    
    def __init__(self):
        super().__init__()
        self._agent = None  # Lazy load
    
    @property
    def agent(self):
        """Lazy load VisionAgent to avoid import overhead."""
        if self._agent is None:
            from agents.vision_agent import VisionAgent
            self._agent = VisionAgent()
        return self._agent
    
    def execute(self, params: VisionInput) -> ToolResult:
        """Execute vision analysis."""
        try:
            # Call the underlying VisionAgent
            result = self.agent.run(params.query)
            
            if not result:
                return ToolResult(
                    success=False,
                    error="视觉分析未返回结果"
                )
            
            # Check for error indicators
            if ("抱歉" in result and "无法" in result) or "出错" in result or "Error" in result:
                return ToolResult(
                    success=False,
                    error=result
                )
            
            return ToolResult(
                success=True,
                data={
                    "query": params.query,
                    "analysis": result
                },
                metadata={"model": getattr(self.agent, 'model_name', 'unknown')}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"视觉分析失败: {str(e)}",
                metadata={"query": params.query}
            )
