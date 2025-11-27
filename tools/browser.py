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

from config import Config

logger = logging.getLogger(__name__)

# ============== Constants ==============

# 使用 Config 中的统一配置
BROWSER_TASK_TIMEOUT = Config.BROWSER_TASK_TIMEOUT


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
    LLM 包装类，为 browser-use 库添加兼容性。
    
    browser-use 库会：
    1. 访问 llm.provider 来区分不同模型类型
    2. 访问 llm.model 来获取模型名称（用于 token 统计）
    3. 动态设置 ainvoke 等属性来追踪 token 使用量
    
    这个包装器拦截这些操作，避免 Pydantic 模型报错。
    
    注意：LangChain 不同 Provider 的模型名称属性不一致：
    - ChatOpenAI: model_name
    - ChatGoogleGenerativeAI: model
    - ChatAnthropic: model
    本包装器统一提供 model 和 model_name 两个属性。
    """
    def __init__(self, llm: Any, provider: str = "openai", model_name: Optional[str] = None):
        # 使用 object.__setattr__ 避免触发自定义 __setattr__
        object.__setattr__(self, '_llm', llm)
        
        # 从底层 LLM 提取模型名称（兼容不同 Provider）
        if model_name is None:
            model_name = (
                getattr(llm, 'model_name', None) or 
                getattr(llm, 'model', None) or 
                'unknown'
            )
        
        # 预设 browser-use 库可能访问的所有属性
        object.__setattr__(self, '_extra_attrs', {
            'provider': provider,
            'model': model_name,       # browser-use 期望的属性
            'model_name': model_name,  # 保持两个属性一致
        })
    
    def __getattr__(self, name: str) -> Any:
        # 优先从额外属性中获取
        extra = object.__getattribute__(self, '_extra_attrs')
        if name in extra:
            return extra[name]
        
        # 否则从底层 LLM 获取
        _llm = object.__getattribute__(self, '_llm')
        
        # 防御性处理：model 属性特殊处理（避免 AttributeError）
        if name == 'model':
            return (
                getattr(_llm, 'model_name', None) or 
                getattr(_llm, 'model', None) or 
                'unknown'
            )
        
        return getattr(_llm, name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        # 所有动态设置的属性都存储在 _extra_attrs 中
        # 不尝试设置到底层 Pydantic LLM 对象上
        extra = object.__getattribute__(self, '_extra_attrs')
        extra[name] = value


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
        from core.llm_provider import LLMFactory, get_model_name
        llm = LLMFactory.create("default")
        model_name = get_model_name(llm)
        provider = _get_provider_name(model_name)
        return LLMWrapper(llm, provider=provider, model_name=model_name)
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


async def _run_async(task: str) -> str:
    """
    Run browser task asynchronously.
    This is the preferred entry point when called from an async context.
    """
    return await _run_browser_task(task)


def _run_sync(task: str) -> str:
    """
    Run browser task synchronously (wraps async).
    Handles both cases: running inside an existing event loop or standalone.
    """
    try:
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop is not None and loop.is_running():
            # We're inside a running event loop (e.g., called from LangGraph async node)
            # Use nest_asyncio if available, otherwise run in a separate thread
            try:
                import nest_asyncio
                nest_asyncio.apply()
                return asyncio.run(_run_browser_task(task))
            except ImportError:
                # Fallback: run in a separate thread to avoid blocking
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    # Create a new event loop in the thread
                    def run_in_new_loop():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(_run_browser_task(task))
                        finally:
                            new_loop.close()
                    
                    future = pool.submit(run_in_new_loop)
                    return future.result(timeout=BROWSER_TASK_TIMEOUT + 10)
        else:
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
