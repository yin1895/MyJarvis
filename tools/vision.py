# Jarvis V7.0 - Native Vision Tool
# tools/vision.py

"""
Native LangChain Tool for visual analysis and screen capture.

This tool provides screen capture and multi-modal AI analysis capabilities:
- Automatic screenshot capture using pyautogui
- Image analysis via LLMFactory "vision" role (Gemini/GPT-4o)
- LangChain-compatible multi-modal message format

Risk Level: safe (passive observation, no side effects)

Usage by LLM:
- "看看我屏幕上有什么"
- "帮我截图分析一下"
- "屏幕上显示什么内容"
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


# ============== Input Schema ==============

class VisionAnalyzeInput(BaseModel):
    """Input schema for vision_analyze tool."""
    query: str = Field(
        default="描述当前屏幕内容",
        description="视觉分析指令。例如：'屏幕上显示什么'、'帮我看看这个界面'、'分析这张图片'"
    )


# ============== Helper Functions ==============

def capture_screen() -> Optional[str]:
    """
    Capture the screen and return as base64-encoded JPEG string.
    
    Features:
    - Automatic resize to max 1024px (preserves aspect ratio)
    - JPEG compression at 80% quality for optimal API performance
    - Error handling with graceful fallback
    
    Returns:
        Base64-encoded image string, or None if capture fails
    """
    try:
        import pyautogui
        from PIL import Image
        
        # Capture screenshot
        screenshot = pyautogui.screenshot()
        
        # Resize for optimal API processing (max 1024px on longest side)
        max_size = 1024
        width, height = screenshot.size
        if width > max_size or height > max_size:
            ratio = min(max_size / width, max_size / height)
            new_size = (int(width * ratio), int(height * ratio))
            screenshot = screenshot.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to JPEG bytes
        buffer = io.BytesIO()
        screenshot.save(buffer, format="JPEG", quality=80)
        image_bytes = buffer.getvalue()
        
        # Encode to base64
        base64_str = base64.b64encode(image_bytes).decode('utf-8')
        
        logger.debug(f"Screenshot captured: {len(base64_str)} bytes (base64)")
        return base64_str
        
    except ImportError as e:
        logger.error(f"Missing dependency for screenshot: {e}")
        return None
    except Exception as e:
        logger.error(f"Screenshot capture failed: {e}")
        return None


def analyze_image_with_llm(base64_image: str, query: str) -> str:
    """
    Analyze an image using the vision LLM.
    
    Uses LangChain's ChatGoogleGenerativeAI or ChatOpenAI with multi-modal messages.
    Respects voice mode constraints for concise responses.
    
    Args:
        base64_image: Base64-encoded image string
        query: User's analysis query/instruction
        
    Returns:
        LLM's analysis response text
    """
    from core.llm_provider import LLMFactory
    from langchain_core.messages import SystemMessage
    
    try:
        # Get vision-capable LLM
        llm = LLMFactory.create("vision")
        
        # System prompt: 强制简洁（因为这个结果会被朗读或转述）
        system_prompt = (
            "你是一个视觉分析助手。直接描述你看到的内容，不要废话。"
            "规则："
            "1-2句话概括重点，不要长篇大论；"
            "不要说'我看到'这种开头，直接说内容；"
            "不要过度分析或推测用户意图；"
            "不要反问用户。"
            "示例回答：'VS Code 打开了 main.py，正在调试 Python 程序。'"
        )
        
        # Construct multi-modal message (compatible with both OpenAI and Gemini)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            )
        ]
        
        # Invoke LLM
        response = llm.invoke(messages)
        
        # Extract text content
        if hasattr(response, 'content'):
            content = response.content
            if isinstance(content, list):
                return " ".join(str(c) for c in content)
            return str(content)
        
        return str(response)
        
    except Exception as e:
        logger.error(f"Vision LLM analysis failed: {e}")
        return f"视觉分析出错：{e}"


# ============== Native Tool ==============

@tool(args_schema=VisionAnalyzeInput)
def vision_analyze(query: str = "描述当前屏幕内容") -> str:
    """
    截取屏幕并使用 AI 视觉模型进行分析。
    
    这个工具可以：
    - 自动截取当前屏幕画面
    - 使用 Gemini/GPT-4o 等视觉模型分析图像内容
    - 回答关于屏幕内容的问题
    
    使用场景：
    - 用户说 "看看我屏幕上有什么"
    - 用户说 "帮我看看这个界面"
    - 用户说 "屏幕上显示什么"
    - 用户说 "这是什么软件"
    
    Args:
        query: 视觉分析指令，描述你想了解屏幕上的什么内容
        
    Returns:
        AI 对屏幕内容的分析结果
    """
    logger.info(f"Vision analyze requested: {query}")
    
    # Step 1: Capture screen
    base64_image = capture_screen()
    
    if not base64_image:
        return (
            "抱歉，我无法截取屏幕画面。\n"
            "可能的原因：\n"
            "- 缺少 pyautogui 或 Pillow 库\n"
            "- 系统权限不足\n"
            "- 在无头环境中运行\n"
            "请检查依赖并重试。"
        )
    
    # Step 2: Analyze with vision LLM
    result = analyze_image_with_llm(base64_image, query)
    
    return result


# Set risk level metadata (safe - passive observation only)
vision_analyze.metadata = {"risk_level": "safe"}


# ============== Export ==============

__all__ = [
    "vision_analyze",
    "VisionAnalyzeInput",
    "capture_screen",
    "analyze_image_with_llm",
]
