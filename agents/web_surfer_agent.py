import asyncio
import os
from typing import Any, cast
from langchain_openai import ChatOpenAI
from browser_use import Agent
from agents.base import BaseAgent
from config import Config
from pydantic import SecretStr

class WebSurferAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="WebSurferAgent")
        # 初始化 LangChain 兼容的 LLM
        # browser-use 推荐使用 GPT-4o 或同等能力的模型以获得最佳效果
        self.llm = ChatOpenAI(
            api_key=SecretStr(Config.LLM_API_KEY) if Config.LLM_API_KEY is not None else None,
            base_url=Config.LLM_BASE_URL,
            model=Config.LLM_MODEL,
            temperature=0.0, # 动作执行需要精确
        )
        

    async def _run_browser_task(self, task: str) -> str:
        try:
            # 创建 Browser Use Agent
            # 注意：默认情况下 browser-use 可能会尝试打开浏览器窗口 (headless=False)
            agent = Agent(
                task=task,
                llm=cast(Any, self.llm),
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
        preset = Config.MODEL_PRESETS.get(preset_name)
        if not preset:
            return False
        
        # 3. 重新初始化 LangChain 的 ChatOpenAI 对象
        # 注意：Browser-use 强依赖高智商模型，如果切换到 local 模型可能会导致任务失败，但我们仍需允许切换以便测试
        try:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                api_key=preset["api_key"],
                base_url=preset["base_url"],
                model=preset["model"],
                temperature=0.0 # 保持低温以确保操作精准
            )
            print(f"[{self.name}] LangChain LLM 已同步切换至: {preset_name}")
            return True
        except Exception as e:
            print(f"[{self.name}] LangChain LLM 切换失败: {e}")
            return False
