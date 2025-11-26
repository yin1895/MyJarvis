"""
Jarvis V7.0 Configuration

统一配置中心，支持：
- 多 LLM Provider (OpenAI, Ollama, Gemini)
- 角色切换 (default, smart, coder, fast, vision)
- 人格 Prompt 定制
- 代理设置
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


class Config:
    """Jarvis 统一配置类"""
    
    # =========================================
    # 🎭 人格 Prompt 配置 (Personality Prompt)
    # =========================================
    # 自定义 Jarvis 的人格特征，影响对话风格和行为
    PERSONALITY_PROMPT = os.getenv("JARVIS_PERSONALITY", """你是 Jarvis，一个智能 AI 助手。你友好、有帮助，并且能够协助用户完成各种任务。

你的特点：
- 简洁明了地回答问题
- 在需要时提供详细的解释
- 保持友好和专业的态度
- 使用中文与用户交流（除非用户使用其他语言）
""")
    
    # 用户自定义名称（用于个性化称呼）
    USER_NAME = os.getenv("JARVIS_USER_NAME", "主人")
    
    # 助手名称
    ASSISTANT_NAME = os.getenv("JARVIS_ASSISTANT_NAME", "Jarvis")
    
    # =========================================
    # 🌐 网络与代理配置
    # =========================================
    PROXY_ENABLED = os.getenv("PROXY_ENABLED", "false").lower() == "true"
    PROXY_URL = os.getenv("PROXY_URL", "http://127.0.0.1:7897")
    
    # =========================================
    # 🎤 语音与唤醒词配置
    # =========================================
    PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
    USE_BUILTIN_KEYWORD = os.getenv("USE_BUILTIN_KEYWORD", "true").lower() == "true"
    WAKE_WORD_FILE = os.getenv("WAKE_WORD_FILE", "jarvis.ppn")
    TTS_VOICE = os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural")
    
    # 唤醒词灵敏度 (0.0 - 1.0)
    try:
        WAKE_SENSITIVITY = float(os.getenv("WAKE_SENSITIVITY", "0.7"))
        WAKE_SENSITIVITY = max(0.0, min(1.0, WAKE_SENSITIVITY))
    except Exception:
        WAKE_SENSITIVITY = 0.7

    # =========================================
    # 🤖 LLM 角色配置 (V7.0 统一架构)
    # =========================================
    # 支持的 provider: "openai", "ollama", "gemini"
    # 
    # 角色说明:
    # - default: 平衡模式，日常对话和任务
    # - smart: 高智能模式，复杂推理和创意任务
    # - coder: 编程模式，代码生成和技术问题
    # - fast: 快速模式，本地 Ollama 低延迟响应
    # - vision: 视觉模式，图像分析和多模态理解
    #
    # 如果本地模型不可用，自动回退到 default
    
    LLM_ROLES = {
        "default": {
            "provider": "openai",
            "api_key": os.getenv("DEFAULT_LLM_API_KEY"),
            "base_url": os.getenv("DEFAULT_LLM_BASE_URL", "https://api.openai.com/v1"),
            "model": os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo"),
            "timeout": 60,
        },
        "smart": {
            "provider": "openai",
            "api_key": os.getenv("SMART_LLM_API_KEY") or os.getenv("DEFAULT_LLM_API_KEY"),
            "base_url": os.getenv("SMART_LLM_BASE_URL") or os.getenv("DEFAULT_LLM_BASE_URL"),
            "model": os.getenv("SMART_LLM_MODEL", "gpt-4o"),
            "timeout": 120,
        },
        "coder": {
            "provider": os.getenv("CODER_LLM_PROVIDER", "ollama"),
            "model": os.getenv("CODER_LLM_MODEL", "deepseek-coder:6.7b"),
            "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            "timeout": 180,
            # OpenAI fallback (当 Ollama 不可用时使用)
            "api_key": os.getenv("CODER_LLM_API_KEY") or os.getenv("DEFAULT_LLM_API_KEY"),
            "base_url": os.getenv("CODER_LLM_BASE_URL") or os.getenv("DEFAULT_LLM_BASE_URL"),
        },
        "fast": {
            "provider": os.getenv("FAST_LLM_PROVIDER", "ollama"),
            "model": os.getenv("FAST_LLM_MODEL", "llama3:8b"),
            "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            "timeout": 60,
            # OpenAI fallback
            "api_key": os.getenv("FAST_LLM_API_KEY") or os.getenv("DEFAULT_LLM_API_KEY"),
            "base_url": os.getenv("FAST_LLM_BASE_URL") or os.getenv("DEFAULT_LLM_BASE_URL"),
        },
        "vision": {
            "provider": os.getenv("VISION_LLM_PROVIDER", "gemini"),
            "api_key": os.getenv("VISION_LLM_API_KEY") or os.getenv("GEMINI_API_KEY"),
            "model": os.getenv("VISION_LLM_MODEL", "gemini-1.5-flash"),
            "timeout": 60,
            # OpenAI fallback (如 GPT-4o)
            "base_url": os.getenv("VISION_LLM_BASE_URL"),
        },
    }
    
    # =========================================
    # 🔧 兼容性配置 (Legacy - 将在未来版本移除)
    # =========================================
    # MODEL_PRESETS: 为旧版 BaseAgent 提供兼容
    # 新代码请使用 LLM_ROLES
    
    @classmethod
    def get_model_presets(cls) -> dict:
        """生成兼容旧版的 MODEL_PRESETS（从 LLM_ROLES 派生）"""
        return {
            role: {
                "api_key": config.get("api_key"),
                "base_url": config.get("base_url") or config.get("host"),
                "model": config.get("model"),
            }
            for role, config in cls.LLM_ROLES.items()
        }
    
    # 动态属性：兼容旧代码
    MODEL_PRESETS = property(lambda self: Config.get_model_presets())
    
    # Agent 路由映射 (兼容旧版 BaseAgent)
    AGENT_MODEL_MAP = {
        "PythonAgent": "smart",
        "VisionAgent": "vision",
        "WebSurferAgent": "vision"
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