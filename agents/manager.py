import json
import os
from typing import List, Dict, Any, Optional
import dateparser
from datetime import datetime
from agents.base import BaseAgent
# 保留遗留代理 (Legacy Agents)
from agents.system_agent import SystemAgent
from agents.vision_agent import VisionAgent
from agents.web_surfer_agent import WebSurferAgent
# 新的工具系统 (Cortex Protocol)
from core.tools import ToolRegistry, ToolExecutor, RiskLevel
from core.tools.base import BaseTool
from services.memory_service import MemoryService
from services.knowledge_service import KnowledgeService
import legacy_tools


class ManagerAgent(BaseAgent):
    """
    混合架构 Manager Agent
    
    新系统 (Cortex Protocol): ToolRegistry + ToolExecutor
    遗留系统 (Legacy): VisionAgent, WebSurferAgent
    """
    
    # 意图到工具名的映射 (兼容旧 intent 字符串)
    INTENT_TO_TOOL_MAP = {
        "shell": "shell_execute",
        "python_task": "python_execute",
        "search": "web_search",
        "file_io": "file_read",
        "time": "get_time",
    }
    
    # ========== 确认关键词定义 ==========
    CONFIRM_POSITIVE = frozenset({
        "yes", "y", "ok", "okay", "confirm", "do it", "proceed", "go ahead",
        "确认", "是", "是的", "好", "好的", "可以", "执行", "没问题", "行", "对"
    })
    CONFIRM_NEGATIVE = frozenset({
        "no", "n", "cancel", "stop", "abort", "don't", "nope",
        "不", "不要", "别", "取消", "算了", "停", "不行", "拒绝"
    })
    
    def __init__(self, scheduler=None):
        super().__init__()
        self.scheduler = scheduler
        
        # ========== 会话状态: 待确认操作 ==========
        self.pending_action: Optional[Dict[str, Any]] = None
        # 结构: {"tool": BaseTool, "params": Dict, "intent": str, "description": str}
        
        # ========== 新系统: Cortex Protocol ==========
        self.registry = ToolRegistry()
        registered_tools = self.registry.scan("tools/")
        print(f"[Manager] Cortex Protocol: 已注册 {len(registered_tools)} 个工具: {registered_tools}")
        
        # Executor 不再负责确认，Manager 是唯一的确认守门人
        self.executor = ToolExecutor(
            require_confirmation_for=[]  # 空列表: 禁用 Executor 内置确认
        )
        
        # ========== 遗留系统: Legacy Agents ==========
        self.system_agent = SystemAgent()
        self.vision_agent = VisionAgent()
        self.web_surfer = WebSurferAgent()
        
        # ========== 服务层 ==========
        self.memory = MemoryService()
        self.knowledge_service = KnowledgeService()
        
        self.history: List[Dict[str, str]] = []
        self.max_history = 10
        self.profile = self.memory.load_profile()
        
        self.base_persona = """
你现在的名字是"爱丽丝"（Alice），是主人的贴身全能女仆。
请遵守以下规则：
1. **绝对禁止使用Markdown**：不要用加粗、标题、列表符号。
2. **口语化**：像正常说话一样，不要列点。
3. **语气**：极度温柔、体贴，偶尔带一点点俏皮或日式翻译腔（如"呐，主人..."）。
4. **记忆**：请根据【关于主人的记忆】来调整你的回答，比如使用正确的称呼。
"""

    # ========== 确认状态检测方法 ==========
    def _is_awaiting_confirmation(self) -> bool:
        """检查是否正在等待用户确认"""
        return self.pending_action is not None
    
    def _check_confirmation_response(self, user_input: str) -> Optional[str]:
        """
        检查用户输入是否为确认响应
        
        Returns:
            "confirmed" - 用户确认执行
            "rejected" - 用户拒绝执行  
            None - 不是确认响应（新的指令）
        """
        normalized = user_input.strip().lower()
        
        # 检查是否匹配肯定关键词
        if normalized in self.CONFIRM_POSITIVE:
            return "confirmed"
        # 检查是否包含肯定关键词（处理 "好的确认" 等变体）
        for kw in self.CONFIRM_POSITIVE:
            if len(kw) >= 2 and kw in normalized and len(normalized) <= len(kw) + 4:
                return "confirmed"
        
        # 检查是否匹配否定关键词
        if normalized in self.CONFIRM_NEGATIVE:
            return "rejected"
        for kw in self.CONFIRM_NEGATIVE:
            if len(kw) >= 2 and kw in normalized and len(normalized) <= len(kw) + 4:
                return "rejected"
        
        # 不是确认响应，是新指令
        return None
    
    def _execute_pending_action(self) -> str:
        """
        执行待确认的操作
        
        Returns:
            工具执行结果的自然语言描述
        """
        if not self.pending_action:
            return "没有待执行的操作。"
        
        tool = self.pending_action["tool"]
        params = self.pending_action["params"]
        
        print(f"[Manager] 用户已确认，执行危险操作: {tool.name}")
        
        # 执行工具（skip_confirmation=True 跳过 Executor 内部确认）
        result = self.executor.run(tool, params, skip_confirmation=True)
        
        # 清除待确认状态
        self.pending_action = None
        
        return result.to_natural_language()
    
    def _cancel_pending_action(self) -> str:
        """取消待确认的操作"""
        if not self.pending_action:
            return "没有待取消的操作。"
        
        tool_name = self.pending_action["tool"].name
        self.pending_action = None
        
        print(f"[Manager] 用户已拒绝，取消操作: {tool_name}")
        return f"好的，已取消 {tool_name} 操作。"

    def _prune_history(self):
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

    def _build_tools_prompt(self) -> str:
        """动态生成工具描述，从 Registry 获取"""
        # 从注册表获取工具描述
        tools_desc = self.registry.get_tools_description()
        
        return f"""
你是一个智能意图决策中枢。请先进行【思考】，分析用户需求最适合哪个工具，然后输出 JSON。

### 已注册工具 (来自 Cortex Protocol)
{tools_desc}

### 特殊处理 (遗留系统 / 无需工具)
- **vision**: 【视觉能力】查看屏幕、分析图片、看图说话。
- **browse_task**: 【浏览器自动化】复杂网页操作、表单填写、数据抓取。
- **schedule**: 包含具体时间的提醒。
- **switch_model**: 切换底层 LLM 模型。
- **query_knowledge**: 询问关于项目代码库的问题 (RAG)。
- **remember**: 让我记住某些信息。
- **learn**: 学习某个文件/目录到知识库。
- **chat**: 纯闲聊，不涉及操作。

### 意图选择指南
1. **python_task/python_execute**: 复杂逻辑、数据处理、批量文件操作、画图、计算。
2. **shell/shell_execute**: Git操作、安装依赖、系统命令、运行脚本。
3. **search/web_search**: 需要联网获取实时信息。
4. **file_io/file_read**: 仅限单文件读取/查看。
5. **vision**: 看屏幕、分析图片。
6. **browse_task**: 复杂浏览器自动化任务。

### 输出格式 (JSON)
{{
    "thought": "用户的意图是... 涉及到... 应该使用...",
    "intent": "工具名或特殊类别",
    "param": "传递给工具的参数"
}}
"""

    def _identify_intent(self, user_input: str) -> dict:
        """意图识别 - 动态注入工具描述"""
        # 获取上下文 (最近一条 Assistant 回复)
        context_msg = "无"
        if self.history:
            for msg in reversed(self.history):
                if msg["role"] == "assistant":
                    context_msg = msg["content"]
                    break

        # 动态构建 System Prompt
        system_prompt = self._build_tools_prompt()

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "用户: \"帮我把 data 目录清空\""},
            {"role": "assistant", "content": """{
    "thought": "这是批量删除操作，需要用 python 沙箱执行。",
    "intent": "python_task",
    "param": "清空 data 目录"
}"""},
            {"role": "user", "content": "用户: \"提交代码\""},
            {"role": "assistant", "content": """{
    "thought": "这是 Git 操作，属于系统级命令。",
    "intent": "shell",
    "param": "git add . && git commit -m 'update'"
}"""},
            {"role": "user", "content": "用户: \"看看我屏幕上是什么\""},
            {"role": "assistant", "content": """{
    "thought": "这需要截图并分析，使用 vision。",
    "intent": "vision",
    "param": "分析屏幕内容"
}"""},
            {"role": "user", "content": f"Context: {context_msg}\nUser Input: {user_input}"}
        ]
        
        try:
            response = self._call_llm(prompt, temperature=0.1)
            clean_json = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            
            if "thought" in data:
                print(f"[Manager 思考]: {data['thought']}")
                
            return data
        except Exception as e:
            print(f"[Manager] 意图识别失败: {e}")
            return {"intent": "chat", "param": ""}

    def _adapt_params_for_tool(self, tool_name: str, param: str) -> Dict[str, Any]:
        """
        参数适配器：将字符串参数转换为工具所需的 Dict 格式
        """
        # 根据工具名适配参数
        if tool_name == "python_execute":
            # Python 工具需要 LLM 先生成代码
            return {"code": param, "timeout": 60}
        
        elif tool_name == "shell_execute":
            return {"command": param, "timeout": 30}
        
        elif tool_name == "web_search":
            return {"query": param, "max_results": 4}
        
        elif tool_name == "file_read":
            return {"path": param}
        
        elif tool_name == "get_time":
            return {"timezone": "Asia/Shanghai"}
        
        # 默认：尝试作为单一参数传递
        return {"input": param}

    def _generate_python_code(self, task_description: str) -> str:
        """使用 LLM 生成 Python 代码"""
        prompt = [
            {"role": "system", "content": """
你是一个 Python 代码生成专家。根据用户任务生成可执行的 Python 代码。
规则：
1. 只输出代码块，不要解释
2. 代码在 workspace/ 目录下执行
3. 使用 print() 输出结果
4. 如需生成文件，保存在当前目录
"""},
            {"role": "user", "content": task_description}
        ]
        
        response = self._call_llm(prompt, temperature=0.2)
        
        # 提取代码块
        import re
        match = re.search(r"```python(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"```(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()

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

    def _execute_with_registry(self, intent: str, param: str) -> Optional[str]:
        """
        尝试通过 Registry 执行工具
        
        对于 DANGEROUS 级别的工具，会拦截并存入 pending_action，
        返回确认询问文本而非直接执行。
        
        Returns:
            工具输出字符串，或确认询问文本，或 None 表示未找到工具
        """
        # 先尝试意图映射
        tool_name = self.INTENT_TO_TOOL_MAP.get(intent, intent)
        
        # 从注册表获取工具
        tool = self.registry.get_by_intent(tool_name)
        if tool is None:
            # 尝试直接用 intent 作为工具名
            tool = self.registry.get_by_intent(intent)
        
        if tool is None:
            return None  # 未找到，回退到遗留系统
        
        print(f"[Manager] 使用 Cortex Protocol: {tool.name} (Risk: {tool.risk_level.value})")
        
        # 特殊处理: python_task 需要先生成代码
        if tool.name == "python_execute" and not param.strip().startswith(("import ", "def ", "class ", "from ", "#")):
            # param 是自然语言描述，需要先转换为代码
            print(f"[Manager] 生成 Python 代码...")
            code = self._generate_python_code(param)
            params = {"code": code, "timeout": 60}
        else:
            # 适配参数格式
            params = self._adapt_params_for_tool(tool.name, param)
        
        # ========== Step A: 拦截危险操作 ==========
        if tool.risk_level == RiskLevel.DANGEROUS:
            # 存储待确认操作
            self.pending_action = {
                "tool": tool,
                "params": params,
                "intent": intent,
                "description": tool.description
            }
            
            print(f"[Manager] 检测到危险操作，等待用户确认: {tool.name}")
            
            # 返回 None，让主流程通过 pending_action 状态构建 System Report
            # 不再硬中断返回，改用 Soft Context Injection 模式
            return None
        
        # ========== SAFE / MODERATE: 直接执行 ==========
        result = self.executor.run(tool, params)
        
        # 格式化输出
        return result.to_natural_language()

    def run(self, user_input: str) -> str:
        if not user_input:
            return ""
        
        # ========== Step B: 处理待确认状态 ==========
        if self._is_awaiting_confirmation():
            confirmation_response = self._check_confirmation_response(user_input)
            
            if confirmation_response == "confirmed":
                # 用户确认执行
                self.history.append({"role": "user", "content": user_input})
                tool_output = self._execute_pending_action()
                
                # 生成回复
                system_prompt = self.base_persona + self.memory.get_system_prompt_suffix()
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(self.history)
                messages.append({"role": "system", "content": f"【系统执行结果】: {tool_output}\n请根据执行结果回复主人。"})
                
                final_reply = self._call_llm(messages)
                self.history.append({"role": "assistant", "content": final_reply})
                return final_reply
            
            elif confirmation_response == "rejected":
                # 用户拒绝执行
                self.history.append({"role": "user", "content": user_input})
                cancel_msg = self._cancel_pending_action()
                
                # 生成友好的取消回复
                system_prompt = self.base_persona + self.memory.get_system_prompt_suffix()
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(self.history)
                messages.append({"role": "system", "content": f"【系统消息】: {cancel_msg}\n请温柔地告诉主人操作已取消。"})
                
                final_reply = self._call_llm(messages)
                self.history.append({"role": "assistant", "content": final_reply})
                return final_reply
            
            else:
                # 不是确认响应，是新指令 → 隐式拒绝，清除状态
                print(f"[Manager] 收到新指令，隐式取消待确认操作")
                self.pending_action = None
                # 继续正常流程处理新指令
        
        # ========== 正常流程 ==========
        # 1. 历史管理
        self._prune_history()
        self.history.append({"role": "user", "content": user_input})
        
        # 2. 意图识别
        print(f"[Manager]: 分析意图 - {user_input}")
        intent_data = self._identify_intent(user_input)
        intent = intent_data.get("intent", "chat")
        param = intent_data.get("param", "")
        print(f"[Manager]: 识别结果 - {intent} ({param})")

        # 3. 执行任务 (混合架构)
        tool_output = ""
        
        # ========== Priority 1: 尝试新系统 (Cortex Protocol) ==========
        registry_result = self._execute_with_registry(intent, param)
        if registry_result is not None:
            tool_output = registry_result
        
        # ========== Step C: 检测待确认状态 (Soft Context Injection) ==========
        # 如果 _execute_with_registry 设置了 pending_action，构建 System Report
        if self._is_awaiting_confirmation() and not tool_output:
            pending = self.pending_action
            assert pending is not None  # Type guard: _is_awaiting_confirmation 已确保非空
            param_preview = str(pending["params"])[:100] + "..." if len(str(pending["params"])) > 100 else str(pending["params"])
            
            # 构建结构化的 System Report，注入 LLM 上下文
            tool_output = (
                f"⛔ 【系统报告 - 操作已拦截】\n"
                f"状态: BLOCKED - 待用户确认 (操作尚未执行)\n"
                f"⚠️ 风险等级: 危险\n"
                f"📋 工具名称: {pending['tool'].name}\n"
                f"📝 操作描述: {pending['description']}\n"
                f"🔧 操作参数: {param_preview}\n\n"
                f"【指令】此操作因高风险被系统拦截，尚未执行。"
                f"请以你的人格向主人解释这个操作的潜在风险，并温柔地询问主人是否确认执行（说「是」或「确认」来执行，说「不」或「取消」来放弃）。"
                f"切勿声称操作已完成。"
            )
        
        # ========== Priority 2: 遗留系统回退 ==========
        elif intent == "vision":
            tool_output = self.vision_agent.run(param)
        
        elif intent == "browse_task":
            tool_output = self.web_surfer.run(param)
        
        elif intent == "system_control":
            tool_output = self.system_agent.run(param)
        
        elif intent == "open_app":
            success, msg = legacy_tools.open_app(param)
            tool_output = msg
        
        elif intent == "schedule":
            if self.scheduler:
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
                        self.scheduler.add_reminder(content, dt_obj)
                        tool_output = f"好的，已设定在 {dt_obj.strftime('%H:%M')} 提醒您：{content}"
                    else:
                        tool_output = "抱歉，我没听懂具体的时间。"
                except Exception as e:
                    tool_output = f"设定提醒失败：{str(e)}"
            else:
                tool_output = "抱歉，调度服务未启动。"
        
        elif intent == "remember":
            tool_output = self._handle_memory_update(param or user_input)
            
        elif intent == "learn":
            target_path = param.strip()
            if not os.path.exists(target_path):
                potential_path = os.path.join(os.getcwd(), target_path)
                if os.path.exists(potential_path):
                    target_path = potential_path
            tool_output = self.knowledge_service.ingest_file(target_path)
            
        elif intent == "query_knowledge":
            docs = self.knowledge_service.query_knowledge(param or user_input)
            if docs:
                tool_output = "检索到的参考资料（请基于此回答）：\n" + "\n---\n".join(docs)
            else:
                tool_output = "知识库中没有找到相关内容，请尝试联网搜索。"

        elif intent == "switch_model":
            target_model = param.lower()
            if "gemini" in target_model or "vision" in target_model:
                target_model = "vision"
            elif "smart" in target_model or "高智商" in target_model:
                target_model = "smart"
            elif "default" in target_model or "默认" in target_model:
                target_model = "default"
            
            success = self.update_model_config(target_model)
            if success:
                # 级联切换遗留代理
                for agent in [self.system_agent, self.vision_agent, self.web_surfer]:
                    if hasattr(agent, 'update_model_config'):
                        agent.update_model_config(target_model)
                tool_output = f"已成功切换至 {target_model} 模式。"
            else:
                tool_output = f"切换失败：未找到模式 {target_model}。"
        
        # 4. 构建最终 Prompt
        system_prompt = self.base_persona + self.memory.get_system_prompt_suffix()
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history)
        
        if tool_output:
            messages.append({"role": "system", "content": f"【系统执行结果】: {tool_output}\n请根据执行结果回复主人。"})
            
        # 5. 生成回复
        final_reply = self._call_llm(messages)
        
        # 6. 记录助手回复
        self.history.append({"role": "assistant", "content": final_reply})
        
        return final_reply

    def close(self):
        super().close()
        # 关闭遗留代理
        if hasattr(self.system_agent, 'close'):
            self.system_agent.close()
        if hasattr(self.vision_agent, 'close'):
            self.vision_agent.close()
        if hasattr(self.web_surfer, 'close'):
            self.web_surfer.close()
