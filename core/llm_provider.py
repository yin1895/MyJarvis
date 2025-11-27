"""
LLM Provider Factory for Jarvis V7.0

Provides a unified factory to create LangChain BaseChatModel instances
based on the configuration in config.py.

Supported Providers:
- OpenAI (ChatOpenAI) - including compatible APIs (DeepSeek, xAI, OpenRouter)
- Ollama (ChatOllama) - local models
- Gemini (ChatGoogleGenerativeAI) - Google's Gemini models
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from config import Config

logger = logging.getLogger(__name__)

# Type alias for supported roles
RoleType = Literal["default", "smart", "coder", "fast", "vision"]

# Type alias for supported providers
ProviderType = Literal["openai", "ollama", "gemini"]


class LLMConfig(BaseModel):
    """Validated LLM configuration."""
    provider: ProviderType
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    host: Optional[str] = None  # For Ollama
    timeout: int = Field(default=60)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class LLMFactory:
    """
    Factory class for creating LangChain Chat Model instances.
    
    Usage:
        # Get model for a specific role
        model = LLMFactory.create("smart")
        
        # Get default model
        model = LLMFactory.create()
        
        # Get model with custom temperature
        model = LLMFactory.create("coder", temperature=0.2)
    """
    
    _DEFAULT_ROLE: RoleType = "default"
    
    @classmethod
    def _get_role_config(cls, role: RoleType) -> dict:
        """
        Get configuration for a specific role from Config.LLM_ROLES.
        Falls back to 'default' if role not found.
        """
        roles = getattr(Config, 'LLM_ROLES', {})
        
        if role not in roles:
            logger.warning(f"Role '{role}' not found in LLM_ROLES, falling back to 'default'")
            role = cls._DEFAULT_ROLE
        
        config = roles.get(role, {})
        
        # If the role config is empty or missing critical fields, fallback to default
        if not config or not config.get('provider'):
            logger.warning(f"Role '{role}' has invalid config, falling back to 'default'")
            config = roles.get(cls._DEFAULT_ROLE, {})
        
        return config
    
    @classmethod
    def _resolve_provider(cls, config: dict) -> ProviderType:
        """
        Resolve the provider type from config.
        Implements configuration-level fallback:
        - If provider is 'ollama' but no host configured, fallback to 'openai'
        - If provider is 'gemini' but no api_key configured, fallback to 'openai'
        """
        provider = config.get('provider', 'openai')
        
        if provider == 'ollama':
            # Check if Ollama is properly configured
            host = config.get('host')
            if not host:
                logger.warning("Ollama provider selected but no host configured, falling back to OpenAI")
                return 'openai'
            return 'ollama'
        
        elif provider == 'gemini':
            # Check if Gemini API key is available
            api_key = config.get('api_key')
            if not api_key:
                logger.warning("Gemini provider selected but no API key configured, falling back to OpenAI")
                return 'openai'
            return 'gemini'
        
        return 'openai'
    
    @classmethod
    def _create_openai(
        cls,
        config: dict,
        temperature: float,
        **kwargs
    ) -> ChatOpenAI:
        """Create a ChatOpenAI instance."""
        return ChatOpenAI(
            model=config.get('model', 'gpt-3.5-turbo'),
            api_key=config.get('api_key'),
            base_url=config.get('base_url'),
            temperature=temperature,
            timeout=config.get('timeout', 60),
            **kwargs
        )
    
    @classmethod
    def _create_ollama(
        cls,
        config: dict,
        temperature: float,
        **kwargs
    ) -> ChatOllama:
        """Create a ChatOllama instance."""
        # Note: ChatOllama uses num_predict for timeout-like behavior
        # timeout is not a direct parameter, we use request_timeout via kwargs if needed
        ollama_kwargs = {
            "model": config.get('model', 'llama3:8b'),
            "base_url": config.get('host', 'http://localhost:11434'),
            "temperature": temperature,
            **kwargs
        }
        
        # Set request timeout if specified in config
        if config.get('timeout'):
            ollama_kwargs["num_ctx"] = 4096  # Reasonable context for longer operations
        
        return ChatOllama(**ollama_kwargs)
    
    @classmethod
    def _create_gemini(
        cls,
        config: dict,
        temperature: float,
        **kwargs
    ) -> ChatGoogleGenerativeAI:
        """Create a ChatGoogleGenerativeAI instance."""
        return ChatGoogleGenerativeAI(
            model=config.get('model', 'gemini-1.5-flash'),
            google_api_key=config.get('api_key'),
            temperature=temperature,
            timeout=config.get('timeout', 60),
            **kwargs
        )
    
    @classmethod
    def create(
        cls,
        role: RoleType = "default",
        temperature: Optional[float] = None,
        **kwargs
    ) -> BaseChatModel:
        """
        Create a LangChain Chat Model instance for the specified role.
        
        Args:
            role: The LLM role preset ('default', 'smart', 'coder', 'fast', 'vision')
            temperature: Override the default temperature (0.0-2.0)
            **kwargs: Additional arguments passed to the model constructor
            
        Returns:
            A LangChain BaseChatModel instance (ChatOpenAI, ChatOllama, or ChatGoogleGenerativeAI)
            
        Raises:
            ValueError: If the configuration is invalid
            
        Example:
            >>> model = LLMFactory.create("smart")
            >>> response = await model.ainvoke([HumanMessage(content="Hello")])
        """
        config = cls._get_role_config(role)
        provider = cls._resolve_provider(config)
        
        # Use provided temperature or default to 0.7
        temp = temperature if temperature is not None else 0.7
        
        logger.info(f"Creating LLM: role={role}, provider={provider}, model={config.get('model')}")
        
        if provider == 'openai':
            return cls._create_openai(config, temp, **kwargs)
        elif provider == 'ollama':
            return cls._create_ollama(config, temp, **kwargs)
        elif provider == 'gemini':
            return cls._create_gemini(config, temp, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @classmethod
    def get_available_roles(cls) -> list[RoleType]:
        """Get list of configured roles."""
        roles = getattr(Config, 'LLM_ROLES', {})
        return list(roles.keys())
    
    @classmethod
    def get_role_info(cls, role: RoleType) -> dict:
        """Get provider and model info for a role (for debugging)."""
        config = cls._get_role_config(role)
        provider = cls._resolve_provider(config)
        return {
            "role": role,
            "provider": provider,
            "model": config.get('model'),
            "base_url": config.get('base_url') or config.get('host'),
        }


# Convenience function for quick access
def get_llm(role: RoleType = "default", **kwargs) -> BaseChatModel:
    """
    Shortcut function to get a LangChain Chat Model.
    
    Args:
        role: The LLM role preset
        **kwargs: Additional arguments (temperature, etc.)
        
    Returns:
        A configured BaseChatModel instance
    """
    return LLMFactory.create(role, **kwargs)


def get_model_name(llm: BaseChatModel) -> str:
    """
    统一获取 LangChain LLM 实例的模型名称。
    
    不同 Provider 的模型名称属性不一致：
    - ChatOpenAI: model_name
    - ChatOllama: model
    - ChatGoogleGenerativeAI: model
    - ChatAnthropic: model
    
    此函数统一处理这些差异，供全项目复用。
    
    Args:
        llm: LangChain BaseChatModel 实例
        
    Returns:
        模型名称字符串，如果无法获取则返回 'unknown'
    """
    # 优先尝试 model_name（ChatOpenAI 使用）
    model_name = getattr(llm, 'model_name', None)
    if model_name:
        return str(model_name)
    
    # 其次尝试 model（Gemini, Ollama, Anthropic 使用）
    model = getattr(llm, 'model', None)
    if model:
        return str(model)
    
    # 最后尝试从 _identifying_params 获取
    try:
        params = llm._identifying_params
        if isinstance(params, dict):
            return str(params.get('model_name', params.get('model', 'unknown')))
    except Exception:
        pass
    
    return 'unknown'
