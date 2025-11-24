# Jarvis AI Assistant (V4.0)

Jarvis 是一个基于 Python 的多模态本地桌面语音助手。它集成了语音交互、视觉分析、系统控制、RAG 知识库和深度搜索能力，旨在成为你的全能 AI 伙伴。

## ✨ 核心功能

*   **👂 双模交互**: 支持 **语音模式** (唤醒词 "Jarvis") 和 **文字模式** (命令行聊天)。
*   **🧠 RAG 知识库**: 支持读取本地文件 (代码、文档) 并基于此回答问题。
    *   使用 `train_jarvis.py` 训练知识库。
*   **👁 视觉能力**: 基于 Google Gemini 的屏幕分析能力，可以"看"到你的屏幕并回答相关问题。
*   **🖐 系统控制**: 语音控制音量、屏幕亮度、媒体播放 (上一首/下一首/暂停) 和打开应用。
*   **🌍 深度搜索**: 集成 Playwright 进行抗反爬虫的联网深度搜索。
*   **⏰ 时间管理**: 智能自然语言定时提醒 (例如："10分钟后提醒我喝水")。

## 🛠️ 架构概览

项目采用模块化 Agent 架构：
*   **ManagerAgent**: 中央大脑，负责意图识别和任务分发。
*   **SearchAgent**: 联网搜索专家。
*   **SystemAgent**: 操作系统控制专家。
*   **VisionAgent**: 视觉分析专家。
*   **FileAgent**: 文件系统操作专家。
*   **Services**: 提供底层支持 (IO, Memory, Knowledge, Scheduler)。

## 🚀 安装指南

### 前置要求
*   **Python 3.10+** (必须)
*   **PyTorch (GPU 版)**: 强烈建议安装 GPU 版本以加速 Embedding 和本地模型推理。
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
    *   填入你的 API Keys (OpenAI/Groq/Google/Picovoice)。

## 📖 使用说明

### 运行主程序
```bash
# 默认语音模式 (需要麦克风)
python main.py

# 文字模式 (适合调试或无麦克风环境)
python main.py -t

# 静音模式 (Alice 只显示文字不说话)
python main.py --mute
```

### 训练知识库
如果你想让 Jarvis 学习当前项目的代码或文档：
```bash
python train_jarvis.py
```
训练完成后，你就可以问它："Jarvis 的架构是怎样的？" 或 "SystemAgent 是怎么实现的？"。

## ⚠️ 注意事项
*   **隐私**: `data/` 目录包含你的个人画像和向量数据库，请勿提交到 Git。
*   **代理**: 如果你在国内，请在 `.env` 中配置 `PROXY_URL`。

---
*Version 4.0 - Release 1.0*
