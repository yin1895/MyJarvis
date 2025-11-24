import base64
import io
import pyautogui
import httpx
from PIL import Image
from openai import OpenAI
from agents.base import BaseAgent
from config import Config
from typing import Any, cast

class VisionAgent(BaseAgent):
    def __init__(self):
        # 不调用 super().__init__，因为我们需要连接到 Google 而不是默认 LLM
        self.name = "VisionAgent"
        
        # 专门为 VisionAgent 创建连接到 Google 的 Client
        self._http_client = httpx.Client() # 代理已由 Config.setup_env_proxy() 全局设置
        
        self.client = OpenAI(
            api_key=Config.GOOGLE_API_KEY,
            base_url=Config.GOOGLE_BASE_URL,
            http_client=self._http_client
        )
        self.model = Config.VISION_MODEL

    def _take_screenshot(self) -> str:
        """截取屏幕并转换为 Base64 字符串"""
        try:
            # 1. 截图
            screenshot = pyautogui.screenshot()
            
            # 2. 调整大小 (最大边长 1024，保持比例，加快 Gemini 处理)
            max_size = 1024
            width, height = screenshot.size
            if width > max_size or height > max_size:
                ratio = min(max_size / width, max_size / height)
                new_size = (int(width * ratio), int(height * ratio))
                screenshot = screenshot.resize(new_size, Image.Resampling.LANCZOS)
            
            # 3. 转为 JPEG 字节流
            buffer = io.BytesIO()
            screenshot.save(buffer, format="JPEG", quality=80)
            image_bytes = buffer.getvalue()
            
            # 4. 转为 Base64
            base64_str = base64.b64encode(image_bytes).decode('utf-8')
            return base64_str
            
        except Exception as e:
            print(f"[VisionAgent] 截图失败: {e}")
            return ""

    def run(self, user_input: str) -> str:
        """执行视觉分析任务"""
        print("[VisionAgent]: 正在截取屏幕...", flush=True)
        base64_image = self._take_screenshot()
        
        if not base64_image:
            return "抱歉，我无法截取屏幕画面，请检查权限。"

        print(f"[VisionAgent]: 正在请求 Gemini ({self.model}) 分析...", flush=True)
        
        # 构建多模态消息 (OpenAI 兼容格式)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"这是我当前屏幕的截图。用户指令: {user_input}。请分析图片并直接回答用户指令，不要啰嗦。"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=cast(Any, messages),
                max_tokens=1000
            )
            return response.choices[0].message.content or "我看完了，但好像没法描述它。"
            
        except Exception as e:
            print(f"[VisionAgent Error]: {e}")
            return f"视觉分析出错了: {e}"

    def close(self):
        if hasattr(self, '_http_client') and self._http_client:
            self._http_client.close()
