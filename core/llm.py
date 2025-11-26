# Jarvis Cortex Protocol - LLM Factory
# core/llm.py

"""
Multi-Provider LLM Abstraction Layer (Jarvis V6.1)

This module provides a unified interface for different LLM providers:
- OpenAI-compatible APIs (OpenAI, DeepSeek, xAI, OpenRouter)
- Ollama (local models)
- Google Gemini

The LLMFactory maps roles to specific providers/models based on config.
Roles: 'default', 'smart', 'coder', 'fast', 'vision'

Design:
- Abstract LLMProvider base class
- Concrete implementations for each provider
- LLMFactory singleton with role-based model selection
- Auto-fallback to 'default' if local provider unavailable
"""

import os
import re
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache

logger = logging.getLogger("jarvis.llm")


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Send messages to LLM and get response.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 - 1.0)
            **kwargs: Provider-specific options
            
        Returns:
            LLM response text
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is currently available."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass


class OpenAIProvider(LLMProvider):
    """
    OpenAI-compatible provider.
    
    Works with:
    - OpenAI (api.openai.com)
    - DeepSeek (api.deepseek.com)
    - xAI/Grok (api.x.ai)
    - OpenRouter (openrouter.ai)
    - Any OpenAI-compatible endpoint
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self._model_name = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self._client = None
        self._http_client = None
    
    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            import httpx
            from openai import OpenAI
            
            self._http_client = httpx.Client()
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=self._http_client,
            )
        return self._client
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=messages,  # type: ignore
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"[LLM Error]: {e}"
    
    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)
    
    def close(self):
        """Close HTTP client."""
        if self._http_client:
            self._http_client.close()


class OllamaProvider(LLMProvider):
    """
    Ollama provider for local LLM inference.
    
    Connects to Ollama server (default: localhost:11434).
    Falls back gracefully if server is unavailable.
    """
    
    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self._model_name = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._available: Optional[bool] = None
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        import requests
        
        url = f"{self.host}/api/chat"
        payload = {
            "model": self._model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        try:
            response = requests.post(
                url, 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Ollama server not reachable at {self.host}")
            self._available = False
            return "[Ollama Error]: Server not reachable"
        except requests.exceptions.Timeout:
            logger.warning(f"Ollama request timeout ({self.timeout}s)")
            return "[Ollama Error]: Request timeout"
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"[Ollama Error]: {e}"
    
    def is_available(self) -> bool:
        """Check if Ollama server is running and model exists."""
        if self._available is not None:
            return self._available
        
        import requests
        
        try:
            # Check if server is reachable
            response = requests.get(
                f"{self.host}/api/tags",
                timeout=5
            )
            if response.status_code != 200:
                self._available = False
                return False
            
            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            
            # Also check exact match with tags
            full_names = [m.get("name", "") for m in models]
            
            self._available = (
                self._model_name in model_names or 
                self._model_name in full_names or
                any(self._model_name in n for n in full_names)
            )
            
            if not self._available:
                logger.warning(
                    f"Ollama model '{self._model_name}' not found. "
                    f"Available: {model_names}"
                )
            
            return self._available
            
        except requests.exceptions.ConnectionError:
            logger.info(f"Ollama server not running at {self.host}")
            self._available = False
            return False
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {e}")
            self._available = False
            return False


class GeminiProvider(LLMProvider):
    """
    Google Gemini provider.
    
    Uses google-generativeai SDK for API access.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-flash",
    ):
        self.api_key = api_key
        self._model_name = model
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai  # type: ignore
                genai.configure(api_key=self.api_key)  # type: ignore
                self._client = genai.GenerativeModel(self._model_name)  # type: ignore
            except ImportError:
                raise ImportError(
                    "google-generativeai package not found. "
                    "Install with: pip install google-generativeai"
                )
        return self._client
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Convert OpenAI message format to Gemini format and call API.
        """
        try:
            # Convert messages to Gemini format
            # Gemini uses 'user' and 'model' roles
            gemini_messages = []
            system_prompt = ""
            
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    # Gemini handles system prompts differently
                    system_prompt += content + "\n"
                elif role == "assistant":
                    gemini_messages.append({
                        "role": "model",
                        "parts": [content]
                    })
                else:  # user
                    gemini_messages.append({
                        "role": "user",
                        "parts": [content]
                    })
            
            # Prepend system prompt to first user message if exists
            if system_prompt and gemini_messages:
                first_user = gemini_messages[0]
                if first_user["role"] == "user":
                    first_user["parts"][0] = system_prompt + first_user["parts"][0]
            
            # Create chat and get response
            chat = self.client.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
            
            last_message = gemini_messages[-1]["parts"][0] if gemini_messages else ""
            response = chat.send_message(
                last_message,
                generation_config={"temperature": temperature}
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"[Gemini Error]: {e}"
    
    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)


class LLMFactory:
    """
    Factory for creating LLM providers based on roles.
    
    Roles:
    - 'default': General purpose (fallback)
    - 'smart': High-capability model for complex reasoning
    - 'coder': Code generation specialist
    - 'fast': Quick responses for simple tasks
    - 'vision': Multimodal with image understanding
    
    Usage:
        llm = LLMFactory.get_model("coder")
        response = llm.chat([{"role": "user", "content": "Write a Python sort"}])
    """
    
    _instance: Optional["LLMFactory"] = None
    _providers: Dict[str, LLMProvider] = {}
    
    def __new__(cls) -> "LLMFactory":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._default_provider: Optional[LLMProvider] = None
    
    @classmethod
    def get_model(cls, role: str = "default") -> LLMProvider:
        """
        Get LLM provider for a specific role.
        
        Args:
            role: Role name ('default', 'smart', 'coder', 'fast', 'vision')
            
        Returns:
            LLMProvider instance, with auto-fallback if needed
        """
        factory = cls()
        
        # Check cache first
        if role in cls._providers:
            provider = cls._providers[role]
            if provider.is_available():
                return provider
            else:
                logger.warning(f"Cached provider for '{role}' no longer available")
                del cls._providers[role]
        
        # Build provider from config
        provider = factory._build_provider(role)
        
        # Check availability and fallback if needed
        if not provider.is_available():
            logger.warning(
                f"Provider for role '{role}' is not available. "
                f"Falling back to 'default'."
            )
            if role != "default":
                provider = factory._build_provider("default")
                if not provider.is_available():
                    raise RuntimeError(
                        "No LLM provider available. "
                        "Please configure DEFAULT_LLM_API_KEY in .env"
                    )
        
        # Cache and return
        cls._providers[role] = provider
        return provider
    
    def _build_provider(self, role: str) -> LLMProvider:
        """Build a provider instance for the given role."""
        from config import Config
        
        # Get role configuration
        role_config = getattr(Config, "LLM_ROLES", {}).get(role)
        
        if not role_config:
            # Fallback to legacy MODEL_PRESETS (now a classmethod)
            presets = Config.get_model_presets() if hasattr(Config, 'get_model_presets') else {}
            role_config = presets.get(role, presets.get("default", {}))
        
        provider_type = role_config.get("provider", "openai")
        
        if provider_type == "ollama":
            return OllamaProvider(
                model=role_config.get("model", "llama3"),
                host=role_config.get("host", "http://localhost:11434"),
                timeout=role_config.get("timeout", 120),
            )
        
        elif provider_type == "gemini":
            api_key = role_config.get("api_key", "")
            if not api_key:
                raise ValueError(
                    f"LLM role '{role}' (Gemini) requires an API key. "
                    f"Please set GEMINI_API_KEY or VISION_LLM_API_KEY in your .env file."
                )
            return GeminiProvider(
                api_key=api_key,
                model=role_config.get("model", "gemini-1.5-flash"),
            )
        
        else:  # openai (default)
            api_key = role_config.get("api_key", "")
            if not api_key:
                raise ValueError(
                    f"LLM role '{role}' (OpenAI-compatible) requires an API key. "
                    f"Please set DEFAULT_LLM_API_KEY in your .env file."
                )
            return OpenAIProvider(
                api_key=api_key,
                model=role_config.get("model", "gpt-3.5-turbo"),
                base_url=role_config.get("base_url"),
            )
    
    @classmethod
    def clear_cache(cls):
        """Clear cached providers (useful for testing or hot reload)."""
        cls._providers.clear()
    
    @classmethod
    def list_available_roles(cls) -> List[str]:
        """List all configured roles."""
        from config import Config
        
        roles = set()
        
        # From new LLM_ROLES
        if hasattr(Config, "LLM_ROLES"):
            roles.update(Config.LLM_ROLES.keys())
        
        # From legacy MODEL_PRESETS (now a classmethod)
        if hasattr(Config, "get_model_presets"):
            roles.update(Config.get_model_presets().keys())
        
        return list(roles)


# === Utility Functions ===

def extract_code_block(text: str, language: str = "python") -> Optional[str]:
    """
    Extract code block from LLM response.
    
    Args:
        text: LLM response text
        language: Expected language (for ```language blocks)
        
    Returns:
        Extracted code, or None if not found
    """
    # Try language-specific block first
    pattern = rf"```{language}(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Try generic code block
    match = re.search(r"```(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return None


def extract_json(text: str) -> Optional[str]:
    """
    Extract JSON from LLM response.
    
    Handles common LLM output formats:
    - Raw JSON
    - ```json ... ``` blocks
    - JSON with surrounding text
    """
    # Remove markdown code blocks
    text = text.replace("```json", "").replace("```", "").strip()
    
    # Try to find JSON object or array
    import json
    
    # Look for { ... } or [ ... ]
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start != -1:
            # Find matching end
            depth = 0
            for i, char in enumerate(text[start:], start):
                if char == start_char:
                    depth += 1
                elif char == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i+1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except json.JSONDecodeError:
                            break
    
    return None


# === Convenience Functions ===

def get_llm(role: str = "default") -> LLMProvider:
    """Shortcut for LLMFactory.get_model()."""
    return LLMFactory.get_model(role)


def quick_chat(
    prompt: str, 
    role: str = "default",
    system: Optional[str] = None,
    temperature: float = 0.7,
) -> str:
    """
    Quick one-shot chat with LLM.
    
    Args:
        prompt: User message
        role: LLM role to use
        system: Optional system prompt
        temperature: Sampling temperature
        
    Returns:
        LLM response text
    """
    llm = get_llm(role)
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    return llm.chat(messages, temperature=temperature)


# === Module-level exports ===

__all__ = [
    "LLMProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "GeminiProvider",
    "LLMFactory",
    "get_llm",
    "quick_chat",
    "extract_code_block",
    "extract_json",
]
