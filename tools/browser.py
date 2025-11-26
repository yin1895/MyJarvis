# Jarvis V7.0 - Native Browser Automation Tool
# tools/browser.py

"""
Native LangChain Tool for browser automation.

Features:
- Web page navigation and interaction
- Form filling and data extraction
- Uses browser-use library with AI agent
- Graceful handling if browser-use is not installed
- Timeout protection for long-running tasks

Risk Level: DANGEROUS (web actions can have real-world consequences)
"""

import asyncio
import logging
from typing import Optional, Any, cast
from pydantic import BaseModel, Field
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ============== Constants ==============

BROWSER_TASK_TIMEOUT = 120  # seconds


# ============== Input Schema ==============

class BrowserNavigateInput(BaseModel):
    """Input schema for browser automation."""
    instruction: str = Field(
        ...,
        description="浏览器自动化指令",
        examples=[
            "打开 google.com 搜索最新 AI 新闻",
            "访问 github.com 并查看 trending 项目",
            "在淘宝搜索机械键盘并找到销量最高的"
        ]
    )


# ============== LLM Wrapper ==============

class LLMWrapper:
    """
    LLM 包装类，为 browser-use 库添加 provider 属性兼容性。
    browser-use 内部会访问 llm.provider 来区分不同模型类型，
    但标准 LangChain ChatOpenAI 对象没有此属性。
    """
    def __init__(self, llm: Any, provider: str = "openai"):
        object.__setattr__(self, '_llm', llm)
        object.__setattr__(self, 'provider', provider)
    
    def __getattr__(self, name: str) -> Any:
        _llm = object.__getattribute__(self, '_llm')
        return getattr(_llm, name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'provider':
            object.__setattr__(self, name, value)
        else:
            _llm = object.__getattribute__(self, '_llm')
            setattr(_llm, name, value)


# ============== Helper Functions ==============

def _get_provider_name(model_name: str) -> str:
    """Infer provider type from model name."""
    model_lower = model_name.lower()
    if "gemini" in model_lower:
        return "google"
    elif "claude" in model_lower:
        return "anthropic"
    else:
        return "openai"


def _get_browser_llm():
    """
    Get LLM instance for browser-use.
    Uses LLMFactory to get the default LLM.
    
    Note: Import is done inside the function to avoid top-level
    circular imports (tools should not depend on core at import time).
    """
    try:
        # Lazy import to avoid circular dependency
        from core.llm_provider import LLMFactory
        llm = LLMFactory.create("default")
        model_name = getattr(llm, "model_name", "") or getattr(llm, "model", "unknown")
        provider = _get_provider_name(str(model_name))
        return LLMWrapper(llm, provider=provider)
    except Exception as e:
        raise RuntimeError(f"无法初始化 LLM: {e}")


async def _run_browser_task(task: str, timeout: int = BROWSER_TASK_TIMEOUT) -> str:
    """
    Execute browser automation task asynchronously with timeout protection.
    
    Args:
        task: The browser task instruction
        timeout: Maximum execution time in seconds
        
    Returns:
        Task result or error message
    """
    try:
        # Try to import browser-use
        from browser_use import Agent
    except ImportError:
        return "错误：browser-use 库未安装。请运行 `pip install browser-use` 安装。"
    
    try:
        # Get LLM
        wrapped_llm = _get_browser_llm()
        
        # Create Browser Use Agent
        agent = Agent(
            task=task,
            llm=cast(Any, wrapped_llm),
        )
        
        # Execute task with timeout protection
        logger.info(f"Starting browser task with {timeout}s timeout: {task[:50]}...")
        result = await asyncio.wait_for(agent.run(), timeout=timeout)
        final = result.final_result()
        
        return final if final is not None else "任务已完成，但没有返回具体内容。"
    
    except asyncio.TimeoutError:
        logger.warning(f"Browser task timed out after {timeout}s")
        return f"浏览器任务超时（{timeout}秒）。请尝试简化任务或稍后重试。"
        
    except Exception as e:
        logger.exception("Browser automation failed")
        return f"浏览器自动化失败: {e}"


def _run_sync(task: str) -> str:
    """
    Run browser task synchronously (wraps async).
    """
    try:
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in a running loop, create a new task
            # This shouldn't normally happen for our use case
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _run_browser_task(task))
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(_run_browser_task(task))
    except Exception as e:
        return f"执行失败: {e}"


# ============== Native Tool ==============

@tool(args_schema=BrowserNavigateInput)
def browser_navigate(instruction: str) -> str:
    """
    浏览器自动化工具：执行网页操作、表单填写、数据抓取。
    
    使用场景:
    - 打开网页: instruction="打开 google.com"
    - 搜索内容: instruction="在百度搜索 Python 教程"
    - 数据抓取: instruction="访问 github.com/trending 获取热门项目"
    - 表单操作: instruction="登录 xxx 网站"
    
    注意：此工具会控制真实浏览器，请谨慎使用。
    
    Args:
        instruction: 浏览器自动化指令（自然语言）
        
    Returns:
        任务执行结果
    """
    if not instruction or not instruction.strip():
        return "错误：请提供浏览器操作指令"
    
    instruction = instruction.strip()
    
    # Execute browser task
    result = _run_sync(instruction)
    
    # Check for error indicators
    if result.startswith("错误：") or result.startswith("浏览器自动化失败"):
        return result
    
    return f"浏览器任务结果:\n{result}"


# ============== Risk Level Metadata ==============
browser_navigate.metadata = {"risk_level": "dangerous"}


# ============== Export ==============
__all__ = ["browser_navigate", "BrowserNavigateInput"]
