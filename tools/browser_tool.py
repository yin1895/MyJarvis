# Jarvis Cortex Protocol - Browser Automation Tool
# tools/browser_tool.py

"""
BrowserTool: Web browser automation for complex tasks.

Migrated from: agents/web_surfer_agent.py (wrapped as Cortex Tool)
Risk Level: DANGEROUS (web actions can have real-world consequences)
"""

from pydantic import BaseModel, Field

from core.tools.base import BaseTool, RiskLevel, ToolResult


class BrowserInput(BaseModel):
    """Input schema for browser automation."""
    instruction: str = Field(
        ...,
        description="浏览器自动化指令，如 '打开百度搜索Python教程' 或 '登录GitHub并创建仓库'",
        examples=[
            "打开 google.com 搜索最新 AI 新闻",
            "访问 github.com 并查看 trending 项目",
            "在淘宝搜索机械键盘并找到销量最高的"
        ]
    )


class BrowserTool(BaseTool[BrowserInput]):
    """
    Browser automation using AI-powered web agent.
    
    Features:
    - Natural language to browser actions
    - Autonomous web navigation
    - Form filling and data extraction
    - Uses browser-use library with LangChain
    
    This tool can perform real web actions and is marked as DANGEROUS.
    The Manager will intercept and request user confirmation before execution.
    The underlying WebSurferAgent is lazy-loaded on first use.
    """
    
    name = "browser_tool"
    description = "浏览器自动化，执行复杂网页操作、表单填写、数据抓取。"
    risk_level = RiskLevel.DANGEROUS  # Web actions require confirmation
    InputSchema = BrowserInput
    tags = ["browser", "web", "automation", "scraping"]
    
    def __init__(self):
        super().__init__()
        self._agent = None  # Lazy load
    
    @property
    def agent(self):
        """Lazy load WebSurferAgent to avoid import overhead."""
        if self._agent is None:
            from agents.web_surfer_agent import WebSurferAgent
            self._agent = WebSurferAgent()
        return self._agent
    
    def execute(self, params: BrowserInput) -> ToolResult:
        """Execute browser automation task."""
        try:
            # Call the underlying WebSurferAgent
            result = self.agent.run(params.instruction)
            
            if not result:
                return ToolResult(
                    success=True,  # Empty result might still be success
                    data={
                        "instruction": params.instruction,
                        "result": "任务已完成，但没有返回具体内容。"
                    }
                )
            
            # Check for error indicators
            if "Failed" in result or "Error" in result:
                return ToolResult(
                    success=False,
                    error=result,
                    metadata={"instruction": params.instruction}
                )
            
            return ToolResult(
                success=True,
                data={
                    "instruction": params.instruction,
                    "result": result
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"浏览器自动化失败: {str(e)}",
                metadata={"instruction": params.instruction}
            )
