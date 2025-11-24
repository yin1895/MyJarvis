import json
import os
from typing import List, Dict, Any
import dateparser
from datetime import datetime
from agents.base import BaseAgent
from agents.search_agent import SearchAgent
from agents.system_agent import SystemAgent
from agents.vision_agent import VisionAgent
from agents.file_agent import FileAgent
from agents.shell_agent import ShellAgent
from agents.python_agent import PythonAgent
from agents.web_surfer_agent import WebSurferAgent
from services.memory_service import MemoryService
from services.knowledge_service import KnowledgeService
import tools

class ManagerAgent(BaseAgent):
    def __init__(self, scheduler=None):
        super().__init__()
        self.scheduler = scheduler
        # 初始化所有特工
        self.search_agent = SearchAgent()
        self.system_agent = SystemAgent()
        self.vision_agent = VisionAgent()
        self.file_agent = FileAgent()
        self.shell_agent = ShellAgent()
        self.python_agent = PythonAgent()
        self.web_surfer = WebSurferAgent()
        # 初始化服务
        self.memory = MemoryService()
        self.knowledge_service = KnowledgeService()
        
        self.history: List[Dict[str, str]] = []
        self.max_history = 10
        
        # 加载用户画像
        self.profile = self.memory.load_profile()
        
        self.base_persona = """
你现在的名字是“爱丽丝”（Alice），是主人的贴身全能女仆。
请遵守以下规则：
1. **绝对禁止使用Markdown**：不要用加粗、标题、列表符号。
2. **口语化**：像正常说话一样，不要列点。
3. **语气**：极度温柔、体贴，偶尔带一点点俏皮或日式翻译腔（如“呐，主人...”）。
4. **记忆**：请根据【关于主人的记忆】来调整你的回答，比如使用正确的称呼。
"""

    def _prune_history(self):
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

    def _identify_intent(self, user_input: str) -> dict:
        # 1. 获取上下文 (最近一条 Assistant 回复)
        context_msg = "无"
        if self.history:
            # 倒序查找最近的一条 assistant 消息
            for msg in reversed(self.history):
                if msg["role"] == "assistant":
                    context_msg = msg["content"]
                    break

        # 2. 构建 Prompt
        prompt = [
            {"role": "system", "content": """
你是一个智能意图决策中枢。请先进行【思考】，分析用户需求最适合哪个工具，然后输出 JSON。

### 核心工具定义 (边界严格区分)
1. **python_task**: 【最高优先级】复杂逻辑、数据处理、批量文件操作、画图、计算、未知任务的自动化。
   - *例子*: "把所有jpg转png", "分析excel", "画个正弦图", "清空workspace目录".
2. **shell**: 【系统级操作】Git操作、安装依赖、系统状态、Ping、运行脚本。
   - *例子*: "git status", "pip install pandas", "运行 python script.py".
3. **file_io**: 【仅限单文件读取/查看】绝对不做修改/删除操作。
   - *例子*: "读一下 main.py", "看看当前目录下有什么文件".
4. **search**: 需要联网获取实时信息。
5. **query_knowledge**: 询问关于项目代码库的问题、报错解决方案、文档内容 (RAG)。
   - *例子*: "ManagerAgent是怎么实现的?", "根据文档解释架构".
6. **schedule**: 包含具体时间的提醒。
7. **switch_model**: 切换底层 LLM 模型。
   - *例子*: "切换到 gemini", "用本地模型", "换回默认模型".
8. **chat**: 纯闲聊，不涉及操作。

### 输出格式 (JSON)
{
    "thought": "用户的意图是... 涉及到... 应该使用...",
    "intent": "python_task",
    "param": "..."
}
"""},
            {"role": "user", "content": "用户: \"帮我把 data 目录清空\""},
            {"role": "assistant", "content": """{
    "thought": "这是批量删除操作，file_io 做不到，shell 有风险，用 python 沙箱最安全。",
    "intent": "python_task",
    "param": "清空 data 目录"
}"""},
            {"role": "user", "content": "用户: \"提交代码\""},
            {"role": "assistant", "content": """{
    "thought": "这是 Git 操作，属于系统级命令。",
    "intent": "shell",
    "param": "git add . && git commit -m 'update'"
}"""},
            {"role": "user", "content": "用户: \"读一下 README.md\""},
            {"role": "assistant", "content": """{
    "thought": "这是读取单个文件内容。",
    "intent": "file_io",
    "param": "read README.md"
}"""},
            {"role": "user", "content": f"Context: {context_msg}\nUser Input: {user_input}"}
        ]
        
        try:
            response = self._call_llm(prompt, temperature=0.1)
            # 清理可能存在的 markdown 标记
            clean_json = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            
            if "thought" in data:
                print(f"[Manager 思考]: {data['thought']}")
                
            return data
        except:
            return {"intent": "chat", "param": ""}

    def _handle_memory_update(self, user_input: str) -> str:
        """分析用户输入并更新记忆"""
        prompt = [
            {"role": "system", "content": """
请分析用户的话，提取记忆信息。返回 JSON。
格式：
{"type": "name", "value": "新名字"} 
{"type": "preference", "key": "偏好项", "value": "偏好内容"}
{"type": "note", "value": "备忘内容"}
"""},
            {"role": "user", "content": user_input}
        ]
        
        try:
            response = self._call_llm(prompt, temperature=0.1)
            clean_json = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            
            if data["type"] == "name":
                self.memory.update_profile("name", data["value"])
                return f"好的，我已经记住了，以后就叫您 {data['value']}。"
            elif data["type"] == "preference":
                self.memory.update_profile(data["key"], data["value"])
                return f"好的，记住了您的偏好：{data['value']}。"
            elif data["type"] == "note":
                self.memory.add_note(data["value"])
                return "好的，已经添加到备忘录了。"
        except Exception as e:
            print(f"[Memory Update Error]: {e}")
            
        return "抱歉，我没太听清您想让我记住什么。"

    def run(self, user_input: str) -> str:
        if not user_input: return ""
        
        # 1. 历史管理
        self._prune_history()
        self.history.append({"role": "user", "content": user_input})
        
        # 2. 意图识别
        print(f"[Manager]: 分析意图 - {user_input}")
        intent_data = self._identify_intent(user_input)
        intent = intent_data.get("intent", "chat")
        param = intent_data.get("param", "")
        print(f"[Manager]: 识别结果 - {intent} ({param})")

        # 3. 执行任务获取上下文
        tool_output = ""
        
        if intent == "search":
            tool_output = self.search_agent.run(param)
        
        elif intent == "open_app":
            success, msg = tools.open_app(param)
            tool_output = msg
        
        elif intent == "system_control":
            tool_output = self.system_agent.run(param)
        
        elif intent == "vision":
            tool_output = self.vision_agent.run(param)
        
        elif intent == "file_io":
            tool_output = self.file_agent.run(param)
        
        elif intent == "shell":
            tool_output = self.shell_agent.run(param)
        
        elif intent == "python_task":
            tool_output = self.python_agent.run(param)
        
        elif intent == "browse_task":
            tool_output = self.web_surfer.run(param)
        
        elif intent == "schedule":
            if self.scheduler:
                # 简单的时间提取逻辑
                prompt = [
                    {"role": "system", "content": '提取时间与内容。格式：{"time_str": "...", "content": "..."}'},
                    {"role": "user", "content": param or user_input}
                ]
                try:
                    resp = self._call_llm(prompt, temperature=0.1)
                    clean = resp.replace("```json", "").replace("```", "").strip()
                    data = json.loads(clean)
                    time_str = data.get("time_str", "")
                    content = data.get("content", "提醒")
                    
                    dt_obj = dateparser.parse(time_str)
                    if dt_obj:
                        if dt_obj < datetime.now():
                            # 简单的过时修正逻辑，实际项目可优化
                            pass 
                        self.scheduler.add_reminder(content, dt_obj)
                        tool_output = f"好的，已设定在 {dt_obj.strftime('%H:%M')} 提醒您：{content}"
                    else:
                        tool_output = "抱歉，我没听懂具体的时间。"
                except Exception as e:
                    tool_output = f"设定提醒失败：{str(e)}"
            else:
                tool_output = "抱歉，调度服务未启动。"
        
        elif intent == "time":
            tool_output = f"现在的北京时间是 {tools.get_current_time()}"
        
        elif intent == "remember":
            tool_output = self._handle_memory_update(param or user_input)
            
        elif intent == "learn":
            # 【RAG 学习流程】
            target_path = param.strip()
            # 简单处理路径：如果不存在，尝试在当前目录找
            if not os.path.exists(target_path):
                potential_path = os.path.join(os.getcwd(), target_path)
                if os.path.exists(potential_path):
                    target_path = potential_path
            
            tool_output = self.knowledge_service.ingest_file(target_path)
            
        elif intent == "query_knowledge":
            # 【RAG 检索流程】
            docs = self.knowledge_service.query_knowledge(param or user_input)
            if docs:
                # 将检索到的文档作为上下文注入
                tool_output = "检索到的参考资料（请基于此回答）：\n" + "\n---\n".join(docs)
            else:
                tool_output = "知识库中没有找到相关内容，请尝试联网搜索。"

        elif intent == "switch_model":
            # 提取目标模式名称
            target_model = param.lower()
            if "gemini" in target_model: target_model = "gemini"
            elif "local" in target_model or "本地" in target_model: target_model = "local"
            elif "default" in target_model or "默认" in target_model: target_model = "default"
            
            # 自身切换
            success = self.update_model_config(target_model)
            
            if success:
                # 级联切换所有子 Agent
                sub_agents = [
                    self.search_agent, self.system_agent, self.vision_agent,
                    self.file_agent, self.shell_agent, self.python_agent,
                    self.web_surfer
                ]
                for agent in sub_agents:
                    if hasattr(agent, 'update_model_config'):
                        agent.update_model_config(target_model)
                tool_output = f"已成功切换至 {target_model} 模式。"
            else:
                tool_output = f"切换失败：未找到模式 {target_model}。"
        
        # 4. 构建最终 Prompt
        # 实时获取最新的 System Prompt (包含动态记忆)
        system_prompt = self.base_persona + self.memory.get_system_prompt_suffix()
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history) # 包含当前 user_input
        
        # 如果有工具输出，注入 System Message 告诉 LLM 发生了什么
        if tool_output:
            messages.append({"role": "system", "content": f"【系统执行结果】: {tool_output}\n请根据执行结果回复主人。"})
            
        # 5. 生成回复
        final_reply = self._call_llm(messages)
        
        # 6. 记录助手回复
        self.history.append({"role": "assistant", "content": final_reply})
        
        return final_reply

    def close(self):
        super().close()
        # 关闭所有子 Agent (如果有需要关闭的资源)
        if hasattr(self.search_agent, 'close'): self.search_agent.close()
        if hasattr(self.system_agent, 'close'): self.system_agent.close()
        if hasattr(self.vision_agent, 'close'): self.vision_agent.close()
        if hasattr(self.file_agent, 'close'): self.file_agent.close()
        if hasattr(self.shell_agent, 'close'): self.shell_agent.close()
        if hasattr(self.python_agent, 'close'): self.python_agent.close()
        if hasattr(self.web_surfer, 'close'): self.web_surfer.close()