import base64
import io
import logging

from agents.base import BaseAgent
from typing import Any, cast

logger = logging.getLogger(__name__)

# 尝试导入新版 LLMFactory
try:
    from core.llm_provider import LLMFactory as NewLLMFactory, RoleType
    HAS_NEW_FACTORY = True
except ImportError:
    HAS_NEW_FACTORY = False
    NewLLMFactory = None


class VisionAgent(BaseAgent):
    """
    Vision Agent for screen analysis.
    
    V7.0 Refactor:
    - 使用 core.llm_provider.LLMFactory 创建视觉模型
    - 支持 Gemini 多模态 API
    - 向后兼容 OpenAI 兼容 API
    """
    
    def __init__(self):
        super().__init__(name="VisionAgent")
        self._use_langchain = HAS_NEW_FACTORY
        
        if self._use_langchain:
            logger.info(f"[{self.name}] Using LangChain LLMFactory for vision")
        else:
            logger.info(f"[{self.name}] Using legacy OpenAI client")

    def _take_screenshot(self) -> str:
        """截取屏幕并转换为 Base64 字符串"""
        try:
            import pyautogui
            from PIL import Image
            
            # 截图
            screenshot = pyautogui.screenshot()
            
            # 调整大小 (最大边长 1024，加快处理)
            max_size = 1024
            width, height = screenshot.size
            if width > max_size or height > max_size:
                ratio = min(max_size / width, max_size / height)
                new_size = (int(width * ratio), int(height * ratio))
                screenshot = screenshot.resize(new_size, Image.Resampling.LANCZOS)
            
            # 转为 JPEG 字节流
            buffer = io.BytesIO()
            screenshot.save(buffer, format="JPEG", quality=80)
            image_bytes = buffer.getvalue()
            
            # 转为 Base64
            base64_str = base64.b64encode(image_bytes).decode('utf-8')
            return base64_str
            
        except Exception as e:
            logger.error(f"[VisionAgent] 截图失败: {e}")
            return ""

    def run(self, user_input: str) -> str:
        """执行视觉分析任务"""
        logger.info("[VisionAgent]: 正在截取屏幕...")
        base64_image = self._take_screenshot()
        
        if not base64_image:
            return "抱歉，我无法截取屏幕画面，请检查权限。"

        logger.info(f"[VisionAgent]: 正在请求 Vision Model 分析...")
        
        # 构建多模态消息 (OpenAI/Gemini 兼容格式)
        user_content = [
            {"type": "text", "text": f"这是我当前屏幕的截图。用户指令: {user_input}。请分析图片并直接回答用户指令，不要啰嗦。"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]

        try:
            # V7.0: 使用 LangChain LLMFactory
            if self._use_langchain and NewLLMFactory:
                from langchain_core.messages import HumanMessage
                
                llm = NewLLMFactory.create("vision")
                
                message = HumanMessage(content=user_content)
                response = llm.invoke([message])
                
                if hasattr(response, 'content'):
                    content = response.content
                    if isinstance(content, list):
                        return " ".join(str(c) for c in content)
                    return str(content)
                return str(response)
            
            # 回退: 使用 BaseAgent 的 OpenAI client
            elif self.client:
                messages = [{"role": "user", "content": user_content}]
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=cast(Any, messages),
                    max_tokens=1000
                )
                return response.choices[0].message.content or "我看完了，但好像没法描述它。"
            
            else:
                return "视觉模型未配置。请在 .env 中设置 VISION_LLM_API_KEY。"
            
        except Exception as e:
            logger.error(f"[VisionAgent Error]: {e}")
            return f"视觉分析出错了: {e}"

