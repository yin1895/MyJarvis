import asyncio
import os
from typing import Any, cast
from langchain_openai import ChatOpenAI
from browser_use import Agent
from agents.base import BaseAgent
from config import Config
from pydantic import SecretStr


class LLMWrapper:
    """
    LLM 包装类，为 browser-use 库添加 provider 属性兼容性。
    browser-use 内部会访问 llm.provider 来区分不同模型类型，
    但标准 LangChain ChatOpenAI 对象没有此属性。
    
    V6.1 Fix:
    - 使用 object.__setattr__ 避免 __getattr__ 递归
    - 显式定义 provider 属性
    - 正确代理所有属性访问
    """
    def __init__(self, llm: Any, provider: str = "openai"):
        # 使用 object.__setattr__ 避免触发 __getattr__
        object.__setattr__(self, '_llm', llm)
        object.__setattr__(self, 'provider', provider)
    
    def __getattr__(self, name: str) -> Any:
        # 代理所有其他属性/方法调用到原始 llm
        _llm = object.__getattribute__(self, '_llm')
        return getattr(_llm, name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        # provider 存储在 wrapper 本身
        if name == 'provider':
            object.__setattr__(self, name, value)
        else:
            # 其他属性设置到底层 llm
            _llm = object.__getattribute__(self, '_llm')
            setattr(_llm, name, value)


class WebSurferAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="WebSurferAgent")
        # 初始化 LangChain LLM (browser-use 需要)
        self._init_langchain_llm()

    def _init_langchain_llm(self):
        """初始化 LangChain ChatOpenAI 对象供 browser-use 使用"""
        # 使用 BaseAgent 已加载的配置
        presets = getattr(Config, 'MODEL_PRESETS', {})
        if callable(presets):
            presets = Config.get_model_presets()
        
        agent_map = getattr(Config, 'AGENT_MODEL_MAP', {})
        preset_key = agent_map.get(self.__class__.__name__, "default")
        preset = presets.get(preset_key, presets.get("default", {}))
        
        try:
            self.llm = ChatOpenAI(
                api_key=preset.get("api_key", ""),
                base_url=preset.get("base_url"),
                model=preset.get("model", "gpt-4o"),
                temperature=0.0  # 保持低温以确保操作精准
            )
        except Exception as e:
            print(f"[{self.name}] LangChain LLM 初始化失败: {e}")
            self.llm = None

    def _get_provider_name(self) -> str:
        """根据模型名推断 provider 类型"""
        # 优先使用 self.model_name (从 BaseAgent 继承)
        model_name = getattr(self, "model_name", "") or ""
        if not model_name and self.llm:
            model_name = getattr(self.llm, "model_name", "") or getattr(self.llm, "model", "")
        model_lower = str(model_name).lower()
        
        if "gemini" in model_lower:
            return "google"
        elif "claude" in model_lower:
            return "anthropic"
        else:
            return "openai"

    async def _run_browser_task(self, task: str) -> str:
        try:
            # 检查 LLM 是否已初始化
            if self.llm is None:
                return "Browser Task Failed: LangChain LLM 未初始化，请检查 Vision 模型配置。"
            
            # 包装 LLM 以添加 provider 属性兼容性
            wrapped_llm = LLMWrapper(self.llm, provider=self._get_provider_name())
            
            # 创建 Browser Use Agent
            # 注意：默认情况下 browser-use 可能会尝试打开浏览器窗口 (headless=False)
            agent = Agent(
                task=task,
                llm=cast(Any, wrapped_llm),
            )
            
            
            # 执行任务
            result = await agent.run()
            final = result.final_result()
            # 确保始终返回字符串，避免返回 None 导致类型不匹配
            return final if final is not None else ""
        except Exception as e:
            return f"Browser Task Failed: {e}"

    def run(self, task: str) -> str:
        """
        执行浏览器自动化任务
        """
        try:
            # browser-use 是异步库，需要运行在 asyncio 事件循环中
            result = asyncio.run(self._run_browser_task(task))
            return result if result is not None else ""
        except Exception as e:
            return f"[{self.name} Error]: {e}"
        
    def update_model_config(self, preset_name: str) -> bool:
        # 1. 先调用父类方法更新基础 client
        if not super().update_model_config(preset_name):
            return False
            
        # 2. 获取新配置
        presets = getattr(Config, 'MODEL_PRESETS', {})
        if callable(presets):
            presets = Config.get_model_presets()
        preset = presets.get(preset_name)
        if not preset:
            return False
        
        # 3. 检查 API Key
        api_key = preset.get("api_key")
        if not api_key:
            print(f"[{self.name}] Switch failed: Preset '{preset_name}' has no API Key.")
            return False
        
        # 4. 重新初始化 LangChain 的 ChatOpenAI 对象
        # 注意：Browser-use 强依赖高智商模型，如果切换到 local 模型可能会导致任务失败，但我们仍需允许切换以便测试
        try:
            self.llm = ChatOpenAI(
                api_key=api_key,
                base_url=preset.get("base_url"),
                model=preset.get("model", "gpt-4o"),
                temperature=0.0  # 保持低温以确保操作精准
            )
            print(f"[{self.name}] LangChain LLM 已同步切换至: {preset_name}")
            return True
        except Exception as e:
            print(f"[{self.name}] LangChain LLM 切换失败: {e}")
            return False
