# agents/base.py
import httpx
from openai import OpenAI
from config import Config
from typing import Any, cast

class BaseAgent:
    def __init__(self, name="BaseAgent"):
        self.name = name
        
        # 初始化 HTTP 客户端
        self._http_client = httpx.Client()
        
        # --- 自动路由逻辑 ---
        # 1. 获取当前 Agent 类名对应的预设角色
        target_role = Config.AGENT_MODEL_MAP.get(self.__class__.__name__, "default")
        
        # 2. 检查该角色的配置是否有效 (是否有 API Key)
        preset_config = Config.MODEL_PRESETS.get(target_role, {})
        if not preset_config.get("api_key"):
            print(f"[{self.name}] Role '{target_role}' not configured (missing API Key). Falling back to 'default'.")
            target_role = "default"
            
        # 3. 应用配置 (调用 update_model_config 以复用逻辑)
        success = self.update_model_config(target_role)
        
        # 如果连 default 都失败了 (例如用户完全没配 .env)，这是一个严重错误
        if not success and target_role == "default":
             print(f"[{self.name}] CRITICAL: Default model configuration is missing or invalid!")

    def update_model_config(self, preset_name: str) -> bool:
        """
        更新当前 Agent 的 LLM 配置。
        支持运行时热切换 (Hot-Swap)。
        """
        if preset_name not in Config.MODEL_PRESETS:
            print(f"[{self.name}] Warning: Model preset '{preset_name}' not found.")
            return False
            
        config = Config.MODEL_PRESETS[preset_name]
        api_key = config.get("api_key")
        
        # 再次检查 API Key，防止运行时切换到空配置
        if not api_key:
            print(f"[{self.name}] Switch failed: Preset '{preset_name}' has no API Key.")
            return False

        # 重新实例化 OpenAI Client
        # 注意：所有兼容 OpenAI 协议的供应商 (DeepSeek, Google, Groq 等) 都走这里
        self.client = OpenAI(
            api_key=api_key, 
            base_url=config.get("base_url"),
            http_client=self._http_client
        )
        self.model_name = config.get("model")
        self.model = self.model_name # 兼容性保留
        
        print(f"[{self.name}] Loaded Model: {self.model_name} (Role: {preset_name})")
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