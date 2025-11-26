# agents/base.py
"""
BaseAgent - V7.0 Compatible

提供与旧版 Agent 的兼容层，同时支持新的 LLMFactory。
新代码应直接使用 core.llm_provider.LLMFactory。
"""

import httpx
from openai import OpenAI
from config import Config
from typing import Any, cast, Optional

# 尝试导入新版 LLMFactory
try:
    from core.llm_provider import LLMFactory as NewLLMFactory
    HAS_NEW_FACTORY = True
except ImportError:
    HAS_NEW_FACTORY = False


class BaseAgent:
    """
    基础 Agent 类，为旧版代码提供兼容层。
    
    V7.0 更新：
    - 支持 Config.LLM_ROLES 配置（优先）
    - 支持多 Provider (openai, ollama, gemini)
    - 向后兼容 Config.MODEL_PRESETS
    """
    
    def __init__(self, name="BaseAgent"):
        self.name = name
        self._http_client: Optional[httpx.Client] = None
        self.client: Optional[OpenAI] = None
        self.model_name: str = ""
        self.model: str = ""  # 兼容性别名
        self._current_role: str = "default"
        
        # --- 自动路由逻辑 ---
        # 1. 获取当前 Agent 类名对应的预设角色
        target_role = Config.AGENT_MODEL_MAP.get(self.__class__.__name__, "default")
        
        # 2. 检查该角色的配置是否有效
        if not self._is_role_valid(target_role):
            print(f"[{self.name}] Role '{target_role}' not configured. Falling back to 'default'.")
            target_role = "default"
            
        # 3. 应用配置
        success = self.update_model_config(target_role)
        
        if not success and target_role == "default":
            print(f"[{self.name}] CRITICAL: Default model configuration is missing!")
    
    def _is_role_valid(self, role: str) -> bool:
        """检查角色配置是否有效"""
        # 优先检查 LLM_ROLES
        roles = getattr(Config, 'LLM_ROLES', {})
        if role in roles:
            config = roles[role]
            provider = config.get('provider', 'openai')
            if provider == 'gemini':
                return bool(config.get('api_key'))
            elif provider == 'ollama':
                return bool(config.get('host'))
            else:  # openai
                return bool(config.get('api_key'))
        
        # 回退到 MODEL_PRESETS
        presets = getattr(Config, 'MODEL_PRESETS', {})
        if callable(presets):
            presets = Config.get_model_presets()
        if role in presets:
            return bool(presets[role].get('api_key'))
        
        return False

    def update_model_config(self, preset_name: str) -> bool:
        """
        更新当前 Agent 的 LLM 配置。
        
        V7.0: 支持多 Provider，优先使用 LLM_ROLES 配置。
        """
        # 优先使用 LLM_ROLES
        roles = getattr(Config, 'LLM_ROLES', {})
        config = roles.get(preset_name)
        
        if config:
            provider = config.get('provider', 'openai')
            
            # Gemini 和 Ollama 不使用 OpenAI 客户端
            if provider in ('gemini', 'ollama'):
                # 这些 provider 应该使用新的 LLMFactory
                if HAS_NEW_FACTORY:
                    print(f"[{self.name}] Using {provider} provider via LLMFactory")
                    self._current_role = preset_name
                    self.model_name = config.get('model', '')
                    self.model = self.model_name
                    # 不创建 OpenAI client，后续调用会使用 LLMFactory
                    return True
                else:
                    print(f"[{self.name}] Warning: {provider} requires LLMFactory, falling back to OpenAI")
            
            # OpenAI 兼容模式
            api_key = config.get('api_key')
            if api_key:
                self._ensure_http_client()
                self.client = OpenAI(
                    api_key=api_key, 
                    base_url=config.get('base_url'),
                    http_client=self._http_client
                )
                self.model_name = config.get('model', 'gpt-3.5-turbo')
                self.model = self.model_name
                self._current_role = preset_name
                print(f"[{self.name}] Loaded Model: {self.model_name} (Role: {preset_name})")
                return True
        
        # 回退到 MODEL_PRESETS (兼容旧配置)
        presets = getattr(Config, 'MODEL_PRESETS', {})
        if callable(presets):
            presets = Config.get_model_presets()
        
        if preset_name not in presets:
            print(f"[{self.name}] Warning: Model preset '{preset_name}' not found.")
            return False
            
        preset_config = presets[preset_name]
        api_key = preset_config.get("api_key")
        
        if not api_key:
            print(f"[{self.name}] Switch failed: Preset '{preset_name}' has no API Key.")
            return False

        self._ensure_http_client()
        self.client = OpenAI(
            api_key=api_key, 
            base_url=preset_config.get("base_url"),
            http_client=self._http_client
        )
        self.model_name = preset_config.get("model", "gpt-3.5-turbo")
        self.model = self.model_name
        self._current_role = preset_name
        
        print(f"[{self.name}] Loaded Model: {self.model_name} (Role: {preset_name})")
        return True
    
    def _ensure_http_client(self):
        """确保 HTTP 客户端已初始化"""
        if self._http_client is None:
            self._http_client = httpx.Client()

    def _call_llm(self, messages, temperature=0.7):
        """统一的 LLM 调用接口"""
        # 如果使用非 OpenAI provider，使用 LLMFactory
        roles = getattr(Config, 'LLM_ROLES', {})
        config = roles.get(self._current_role, {})
        provider = config.get('provider', 'openai')
        
        if provider in ('gemini', 'ollama') and HAS_NEW_FACTORY:
            try:
                from core.llm_provider import LLMFactory, RoleType
                llm = LLMFactory.create(cast(RoleType, self._current_role), temperature=temperature)
                from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
                
                # 转换消息格式
                lc_messages = []
                for msg in messages:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    if role == 'system':
                        lc_messages.append(SystemMessage(content=content))
                    elif role == 'assistant':
                        lc_messages.append(AIMessage(content=content))
                    else:
                        lc_messages.append(HumanMessage(content=content))
                
                response = llm.invoke(lc_messages)
                return response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                return f"[{self.name} Error]: {e}"
        
        # OpenAI 兼容模式
        if self.client is None:
            return f"[{self.name} Error]: No LLM client configured"
        
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
        """关闭资源"""
        if self._http_client:
            self._http_client.close()
            self._http_client = None