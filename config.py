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
    # 🎭 人格配置系统 (Personality System)
    # =========================================
    # 支持语音/文字模式的差异化人格，以及角色特定的行为约束
    
    PERSONALITY = {
        # 基础人格（所有模式共享）
        "base": {
            "name": os.getenv("JARVIS_ASSISTANT_NAME", "Jarvis"),
            "trait": "简洁、专业、友好",
            "language": "中文",
        },
        
        # 语音模式约束（被朗读出来，必须简洁口语化）
        "voice_mode": {
            "style": "极度简洁，1-2句话解决问题，像朋友聊天",
            "rules": [
                "不要长篇大论，用户在听不是在看",
                "不要使用 markdown、列表、代码块",
                "不要分析过程，直接给结果",
                "不要反问，除非真的需要澄清",
            ],
            "example_bad": "我看到您的屏幕上显示的是一个代码编辑器，可能是 VS Code，并且您刚刚执行了一个切换模型的操作...",
            "example_good": "屏幕上是 VS Code，打开了 main.py。",
        },
        
        # 文字模式约束（可以适当详细）
        "text_mode": {
            "style": "清晰准确，可以适当详细，支持 markdown",
            "rules": [
                "可以使用格式化提高可读性",
                "复杂问题可以分步骤解释",
            ],
        },
        
        # 角色特定人格补充
        "roles": {
            "default": "平衡通用，日常对话和任务执行",
            "smart": "深度思考，但仍保持简洁，适合复杂推理",
            "coder": "技术精准，代码优先，少废话",
            "vision": "描述所见即可，不要过度分析和推测",
            "fast": "极速响应，一句话搞定",
        },
    }
    
    # 兼容旧版：保留 PERSONALITY_PROMPT（从新配置生成）
    @classmethod
    def get_personality_prompt(cls) -> str:
        """获取基础人格 Prompt（兼容旧代码）"""
        base = cls.PERSONALITY.get("base", {})
        return f"""你是 {base.get('name', 'Jarvis')}，一个智能 AI 助手。
你的特点：{base.get('trait', '简洁、专业、友好')}
使用{base.get('language', '中文')}与用户交流。"""
    
    # 保持向后兼容
    PERSONALITY_PROMPT = property(lambda self: Config.get_personality_prompt())
    
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
    # ⚙️ 运行时参数 (Runtime Settings)
    # =========================================
    # 集中管理各模块的阈值和超时，避免硬编码
    
    # 浏览器自动化
    BROWSER_TASK_TIMEOUT = int(os.getenv("BROWSER_TASK_TIMEOUT", "120"))  # 秒
    
    # 知识库 RAG
    KNOWLEDGE_CHUNK_SIZE = int(os.getenv("KNOWLEDGE_CHUNK_SIZE", "500"))  # 字符
    KNOWLEDGE_CHUNK_OVERLAP = int(os.getenv("KNOWLEDGE_CHUNK_OVERLAP", "50"))  # 字符
    KNOWLEDGE_MAX_RESULTS = int(os.getenv("KNOWLEDGE_MAX_RESULTS", "5"))  # 条
    
    # 语音识别 VAD
    VAD_PAUSE_THRESHOLD = float(os.getenv("VAD_PAUSE_THRESHOLD", "0.8"))  # 秒
    VAD_MAX_RECORD_SECONDS = int(os.getenv("VAD_MAX_RECORD_SECONDS", "30"))  # 秒
    
    # 对话历史
    MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "30"))  # 条，防止 context 溢出
    
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