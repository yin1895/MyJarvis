import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

class Config:
    # 基础配置
    PROXY_ENABLED = os.getenv("PROXY_ENABLED", "false").lower() == "true"
    PROXY_URL = os.getenv("PROXY_URL", "http://127.0.0.1:7897")
    
    # Picovoice
    PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
    USE_BUILTIN_KEYWORD = os.getenv("USE_BUILTIN_KEYWORD", "true").lower() == "true"
    WAKE_WORD_FILE = os.getenv("WAKE_WORD_FILE", "jarvis.ppn")
    
    # --- TTS ---
    TTS_VOICE = os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural")
    
    # --- 唤醒词灵敏度 ---
    try:
        WAKE_SENSITIVITY = float(os.getenv("WAKE_SENSITIVITY", "0.7"))
        if WAKE_SENSITIVITY < 0.0: WAKE_SENSITIVITY = 0.0
        if WAKE_SENSITIVITY > 1.0: WAKE_SENSITIVITY = 1.0
    except Exception:
        WAKE_SENSITIVITY = 0.8

    # --- LLM 角色预设 (Role Presets) ---
    MODEL_PRESETS = {
        "default": {
            "api_key": os.getenv("DEFAULT_LLM_API_KEY"),
            "base_url": os.getenv("DEFAULT_LLM_BASE_URL", "https://api.openai.com/v1"),
            "model": os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo")
        },
        "smart": {
            "api_key": os.getenv("SMART_LLM_API_KEY"),
            "base_url": os.getenv("SMART_LLM_BASE_URL"),
            "model": os.getenv("SMART_LLM_MODEL")
        },
        "vision": {
            "api_key": os.getenv("VISION_LLM_API_KEY"),
            "base_url": os.getenv("VISION_LLM_BASE_URL"),
            "model": os.getenv("VISION_LLM_MODEL")
        }
    }

    # --- Agent 路由映射 (Legacy - 兼容旧代码) ---
    # 未在此列出的 Agent 默认使用 "default"
    AGENT_MODEL_MAP = {
        "PythonAgent": "smart",
        "VisionAgent": "vision",
        "WebSurferAgent": "vision"
    }

    # === LLM 角色配置 (Jarvis V6.1 - 多 LLM 架构) ===
    # 支持的 provider: "openai", "ollama", "gemini"
    # 
    # Ollama 配置示例 (本地模型):
    #   "coder": {"provider": "ollama", "model": "deepseek-coder:6.7b"}
    #   "fast": {"provider": "ollama", "model": "llama3:8b"}
    #
    # 如果 Ollama 不可用，自动回退到 'default'
    LLM_ROLES = {
        "default": {
            "provider": "openai",
            "api_key": os.getenv("DEFAULT_LLM_API_KEY"),
            "base_url": os.getenv("DEFAULT_LLM_BASE_URL", "https://api.openai.com/v1"),
            "model": os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo"),
        },
        "smart": {
            "provider": "openai",
            "api_key": os.getenv("SMART_LLM_API_KEY") or os.getenv("DEFAULT_LLM_API_KEY"),
            "base_url": os.getenv("SMART_LLM_BASE_URL") or os.getenv("DEFAULT_LLM_BASE_URL"),
            "model": os.getenv("SMART_LLM_MODEL", "gpt-4o"),
        },
        "coder": {
            # 优先使用 Ollama 本地 DeepSeek-Coder，不可用时回退到 default
            "provider": os.getenv("CODER_LLM_PROVIDER", "ollama"),
            "model": os.getenv("CODER_LLM_MODEL", "deepseek-coder:6.7b"),
            "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            "timeout": 180,
            # OpenAI fallback config (used if provider=openai or ollama unavailable)
            "api_key": os.getenv("CODER_LLM_API_KEY") or os.getenv("DEFAULT_LLM_API_KEY"),
            "base_url": os.getenv("CODER_LLM_BASE_URL") or os.getenv("DEFAULT_LLM_BASE_URL"),
        },
        "fast": {
            # 优先使用 Ollama 本地 Llama3，不可用时回退到 default
            "provider": os.getenv("FAST_LLM_PROVIDER", "ollama"),
            "model": os.getenv("FAST_LLM_MODEL", "llama3:8b"),
            "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            "timeout": 60,
            # OpenAI fallback config
            "api_key": os.getenv("FAST_LLM_API_KEY") or os.getenv("DEFAULT_LLM_API_KEY"),
            "base_url": os.getenv("FAST_LLM_BASE_URL") or os.getenv("DEFAULT_LLM_BASE_URL"),
        },
        "vision": {
            "provider": os.getenv("VISION_LLM_PROVIDER", "gemini"),
            "api_key": os.getenv("VISION_LLM_API_KEY") or os.getenv("GEMINI_API_KEY"),
            "model": os.getenv("VISION_LLM_MODEL", "gemini-1.5-flash"),
            # OpenAI fallback
            "base_url": os.getenv("VISION_LLM_BASE_URL"),
        },
    }

    @staticmethod
    def get_proxy_config():
        """获取 httpx 兼容的代理配置字典"""
        if Config.PROXY_ENABLED and Config.PROXY_URL:
            return {
                "http://": Config.PROXY_URL,
                "https://": Config.PROXY_URL
            }
        return None

    @staticmethod
    def setup_env_proxy():
        """设置环境变量代理 (供 requests 等库使用)"""
        if Config.PROXY_ENABLED and Config.PROXY_URL:
            os.environ["http_proxy"] = Config.PROXY_URL
            os.environ["https_proxy"] = Config.PROXY_URL