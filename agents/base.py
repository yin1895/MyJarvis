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
        
        # 初始化默认模型配置
        self.update_model_config("default")
    
    def update_model_config(self, preset_name: str) -> bool:
        """
        更新当前 Agent 的 LLM 配置
        """
        if preset_name not in Config.MODEL_PRESETS:
            print(f"[{self.name}] Warning: Model preset '{preset_name}' not found.")
            return False
            
        config = Config.MODEL_PRESETS[preset_name]
        
        # 检查 API Key 是否存在
        if not config.get("api_key"):
            print(f"[{self.name}] 切换失败：预设 {preset_name} 缺少 API Key")
            return False

        # 重新实例化 OpenAI Client
        self.client = OpenAI(
            api_key=config["api_key"], 
            base_url=config["base_url"],
            http_client=self._http_client
        )
        self.model_name = config["model"]
        # 兼容部分子类可能使用 self.model 的情况
        self.model = config["model"]
        print(f"[{self.name}] 模型已切换至: {preset_name}")
        return True

    def _call_llm(self, messages, temperature=0.7):
        """统一的 LLM 调用接口"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
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