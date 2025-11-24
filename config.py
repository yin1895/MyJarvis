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
    
    # LLM
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL")
    LLM_MODEL = os.getenv("LLM_MODEL", "x-ai/grok-4.1-fast:free")
    
    # Google Gemini (Vision)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_BASE_URL = os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    VISION_MODEL = os.getenv("VISION_MODEL", "gemini-1.5-flash")
    
    # Groq
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # TTS
    TTS_VOICE = os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural")
    # 唤醒词灵敏度 (0.0 - 1.0)
    try:
        WAKE_SENSITIVITY = float(os.getenv("WAKE_SENSITIVITY", "0.7"))
        # clamp
        if WAKE_SENSITIVITY < 0.0:
            WAKE_SENSITIVITY = 0.0
        if WAKE_SENSITIVITY > 1.0:
            WAKE_SENSITIVITY = 1.0
    except Exception:
        WAKE_SENSITIVITY = 0.8

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