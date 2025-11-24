# agents/base.py
import httpx
from openai import OpenAI
from config import Config
from typing import Any, cast

class BaseAgent:
    def __init__(self, name="BaseAgent"):
        self.name = name
        
        # 【关键修改】不要传 proxies 参数，直接初始化
        # Config.setup_env_proxy() 会在 main.py 里把代理设置进环境变量
        self._http_client = httpx.Client()
        
        self.client = OpenAI(
            api_key=Config.LLM_API_KEY, 
            base_url=Config.LLM_BASE_URL,
            http_client=self._http_client
        )
    
    def _call_llm(self, messages, temperature=0.7):
        """统一的 LLM 调用接口"""
        try:
            response = self.client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=cast(Any, messages),
                temperature=temperature
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"[{self.name} Error]: {e}"

    def run(self, input_text: str) -> str:
        """子类必须实现此方法"""
        raise NotImplementedError

    def close(self):
        if hasattr(self, '_http_client') and self._http_client:
            self._http_client.close()