# Jarvis AI Assistant (V7.0)

Jarvis 是一个基于 **LangGraph** 的多模态本地桌面语音助手。它集成了语音交互、视觉分析、系统控制、RAG 知识库和深度搜索能力，支持多 LLM 提供商动态切换，旨在成为你的全能 AI 伙伴。

## ✨ 核心功能

*   **👂 双模交互**: 支持 **语音模式** (唤醒词 "Jarvis") 和 **文字模式** (命令行聊天)。
*   **🔄 多模型切换**: 运行时动态切换 LLM 角色 (default/smart/coder/fast/vision)，自动回退到可用 Provider。
*   **🧠 RAG 知识库**: 支持读取本地文件 (代码、文档) 并基于此回答问题。
*   **👁 视觉能力**: 基于 Gemini/OpenAI 的屏幕分析能力，可以"看"到你的屏幕并回答相关问题。
*   **🖐 系统控制**: 语音控制音量、屏幕亮度、媒体播放和打开应用。
*   **🌍 深度搜索**: 集成 Playwright 进行抗反爬虫的联网深度搜索。
*   **⏰ 时间管理**: 智能自然语言定时提醒 (例如："10分钟后提醒我喝水")。
*   **💾 状态持久化**: 基于 SQLite 的对话状态持久化，支持断点续聊。

## 🛠️ 架构概览

项目采用 **LangGraph 响应式** 架构：

```
┌─────────────────────────────────────────────────────────┐
│                      main.py                            │
│              (语音/文字模式入口)                          │
├─────────────────────────────────────────────────────────┤
│                  core/graph/builder.py                  │
│          (LangGraph StateGraph 构建器)                   │
├───────────────┬─────────────────────────────────────────┤
│ LLMFactory    │           Native Tools                  │
│ (多Provider)   │  shell, python, system, vision, etc.   │
├───────────────┴─────────────────────────────────────────┤
│                     Services                            │
│     (IO, Memory, Knowledge, Scheduler)                  │
└─────────────────────────────────────────────────────────┘
```

**核心模块**:
*   **core/graph/**: LangGraph 工作流构建器和状态管理
*   **core/llm_provider.py**: 统一的 LLM 工厂，支持 OpenAI/Ollama/Gemini
*   **tools/native_*.py**: 原生 LangChain 工具（Shell、Python、系统、视觉等）
*   **services/**: 底层服务支持 (IO, Memory, Knowledge, Scheduler)
*   **agents/**: 旧版 Agent 架构（保留兼容）

## 🚀 安装指南

### 前置要求
*   **Python 3.10+** (必须)
*   **PyTorch (GPU 版)**: 强烈建议安装 GPU 版本以加速 Embedding。
    *   访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取安装命令。

### 步骤

1.  **克隆仓库**
    ```bash
    git clone https://github.com/your-repo/jarvis-project.git
    cd jarvis-project
    ```

2.  **创建虚拟环境**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

4.  **安装 Playwright 浏览器**
    ```bash
    playwright install
    ```

5.  **配置环境变量**
    *   复制 `.env.example` 为 `.env`。
    *   填入你的 API Keys 和个性化配置。

## 📖 使用说明

### 运行主程序
```bash
# 默认语音模式 (需要麦克风和 Picovoice Key)
python main.py

# 文字模式 (推荐，适合调试或无麦克风环境)
python main.py -t

# 静音模式 (只显示文字不说话)
python main.py --mute

# 组合模式
python main.py -t --mute
```

### 运行时命令
在对话中，你可以使用以下命令：
*   `/exit` 或 `/quit` - 退出程序
*   `切换到 <角色>` - 动态切换模型角色 (default/smart/coder/fast/vision)

### 角色说明
| 角色 | 说明 | 推荐场景 |
|------|------|----------|
| `default` | 默认通用模型 | 日常对话 |
| `smart` | 高智能模型 | 复杂推理 |
| `coder` | 代码专家模型 | 编程任务 |
| `fast` | 快速响应模型 | 简单查询 |
| `vision` | 视觉能力模型 | 屏幕分析 |

### 训练知识库
让 Jarvis 学习当前项目的代码或文档：
```bash
python train_jarvis.py
```
训练完成后，你可以问它："Jarvis 的架构是怎样的？" 或 "GraphBuilder 是怎么实现的？"

## ⚙️ 配置说明

### 环境变量 (.env)

```bash
# === LLM Provider API Keys ===
OPENAI_API_KEY=sk-xxx              # OpenAI API Key
OPENAI_BASE_URL=                   # 可选: 自定义 API 端点
GEMINI_API_KEY=                    # Google Gemini API Key
OLLAMA_BASE_URL=http://localhost:11434  # Ollama 本地服务地址

# === 语音相关 ===
PICOVOICE_ACCESS_KEY=              # Picovoice 唤醒词 Key
TTS_VOICE_NAME=zh-CN-XiaoxiaoNeural  # TTS 语音名称

# === 个性化配置 ===
USER_NAME=用户                      # 用户称呼
ASSISTANT_NAME=Jarvis              # 助手名称
PERSONALITY_PROMPT=你是一个有帮助的AI助手  # 人格提示词

# === 网络配置 ===
PROXY_URL=                         # 代理地址（可选）
```

### 模型预设 (config.py)

```python
MODEL_PRESETS = {
    "default": ("openai", "gpt-4o-mini"),
    "smart":   ("openai", "gpt-4o"),
    "coder":   ("openai", "gpt-4o"),
    "fast":    ("ollama", "qwen3:4b"),
    "vision":  ("gemini", "gemini-2.0-flash"),
}
```

## 📁 项目结构

```
Jarvis_Project/
├── main.py              # 主入口 (语音/文字模式)
├── config.py            # 配置中心
├── train_jarvis.py      # 知识库训练脚本
├── requirements.txt     # Python 依赖
├── .env.example         # 环境变量模板
├── core/
│   ├── llm_provider.py  # LLM 工厂 (OpenAI/Ollama/Gemini)
│   ├── llm.py           # 旧版 LLM 封装
│   └── graph/
│       ├── builder.py   # LangGraph 构建器
│       └── state.py     # 状态定义
├── tools/
│   ├── native_shell.py  # Shell 命令工具
│   ├── native_python.py # Python 执行工具
│   ├── native_system.py # 系统控制工具
│   ├── native_vision.py # 屏幕视觉工具
│   └── ...              # 其他工具
├── services/
│   ├── io_service.py    # 语音 I/O 服务
│   ├── memory_service.py # 记忆服务
│   └── knowledge_service.py # RAG 知识服务
├── agents/              # 旧版 Agent (保留兼容)
├── data/
│   ├── state.db         # 对话状态持久化
│   ├── user_profile.json # 用户画像
│   └── vector_db/       # RAG 向量数据库
└── workspace/           # 工作目录
```

## ⚠️ 注意事项

*   **隐私**: `data/` 目录包含你的个人画像和向量数据库，请勿提交到 Git。
*   **代理**: 如果你在国内访问 OpenAI/Gemini，请在 `.env` 中配置 `PROXY_URL`。
*   **Ollama**: 使用 Ollama 前请确保服务已启动 (`ollama serve`)。
*   **视觉功能**: 需要配置 Gemini API Key 或支持视觉的 OpenAI 模型。

## 🔄 版本历史

*   **V7.0** - LangGraph 架构重构，多 Provider 支持，状态持久化
*   **V6.0** - Agent 架构，意图分发
*   **V4.0** - 多模态集成，RAG 知识库

---
*Version 7.0 - LangGraph Edition*
