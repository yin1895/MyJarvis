import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from agents.base import BaseAgent
# 所有代理已迁移到 Cortex Protocol (Phase 1-3)
# 新的工具系统 (Cortex Protocol)
from core.tools import ToolRegistry, ToolExecutor, RiskLevel
from core.tools.base import BaseTool
from services.memory_service import MemoryService
from services.knowledge_service import KnowledgeService
from services.scheduler_service import SchedulerService


class ManagerAgent(BaseAgent):
    """
    统一架构 Manager Agent (Cortex Protocol)
    
    所有工具通过 ToolRegistry + ToolExecutor 统一管理
    唯一保留的特殊处理: switch_model (切换 LLM 模型)
    """
    
    # 意图到工具名的映射 (兼容旧 intent 字符串)
    INTENT_TO_TOOL_MAP = {
        "shell": "shell_execute",
        "python_task": "python_execute",
        "search": "web_search",
        "file_io": "file_read",
        "time": "get_time",
        # Memory & Knowledge (Cortex Protocol Migration Phase 1)
        "remember": "memory_tool",
        "memory_op": "memory_tool",
        "learn": "knowledge_ingest",
        "query_knowledge": "knowledge_query",
        # Vision & Browser (Cortex Protocol Migration Phase 2)
        "vision": "vision_tool",
        "browse_task": "browser_tool",
        # System & Schedule (Cortex Protocol Migration Phase 3)
        "system_control": "system_tool",
        "open_app": "system_tool",
        "schedule": "scheduler_tool",
        # Utility Tools
        "weather": "get_weather",
        "get_weather": "get_weather",
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
        
        # ========== SchedulerService 单例初始化 ==========
        # 使用单例模式，如果传入了 scheduler，设置 speak_callback
        if scheduler:
            # scheduler 是外部传入的 SchedulerService 实例，用于设置 speak_callback
            self._scheduler_service = SchedulerService()
            if hasattr(scheduler, 'speak_callback'):
                self._scheduler_service.set_speak_callback(scheduler.speak_callback)
        else:
            # 确保 SchedulerService 单例被初始化
            self._scheduler_service = SchedulerService()
        
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

### 特殊处理 (仅保留模型切换)
- **switch_model**: 切换底层 LLM 模型。
- **chat**: 纯闲聊，不涉及操作。

### 意图选择指南
1. **python_task/python_execute**: 复杂逻辑、数据处理、批量文件操作、画图、计算。
2. **shell/shell_execute**: Git操作、安装依赖、系统命令、运行脚本。
3. **search/web_search**: 需要联网获取实时信息。
4. **file_io/file_read**: 仅限单文件读取/查看。
5. **vision/vision_tool**: 看屏幕、分析图片、视觉问答。
6. **browse_task/browser_tool**: 复杂浏览器自动化任务。

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
            # Ensure response is a string
            if isinstance(response, list):
                response = str(response[0]) if response else ""
            response_str = str(response) if response else ""
            clean_json = response_str.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            
            if "thought" in data:
                print(f"[Manager 思考]: {data['thought']}")
                
            return data
        except Exception as e:
            print(f"[Manager] 意图识别失败: {e}")
            return {"intent": "chat", "param": ""}

    def _adapt_params_for_tool(self, tool_name: str, param: str, user_input: str = "") -> Dict[str, Any]:
        """
        参数适配器：将字符串参数转换为工具所需的 Dict 格式
        
        Args:
            tool_name: 工具名称
            param: 意图识别提取的参数
            user_input: 原始用户输入 (用于需要 LLM 解析的场景)
        """
        # 根据工具名适配参数
        # ========== Smart Tools (V6.1 - 自带 LLM 代码/命令生成) ==========
        if tool_name == "python_execute":
            # Smart Python Tool: 传递 instruction，工具内部生成代码
            return {"instruction": param, "timeout": 60, "max_retries": 2}
        
        elif tool_name == "shell_execute":
            # Smart Shell Tool: 传递 instruction，工具内部生成命令
            return {"instruction": param, "timeout": 30}
        
        elif tool_name == "web_search":
            return {"query": param, "max_results": 4}
        
        elif tool_name == "file_read":
            return {"path": param}
        
        elif tool_name == "get_time":
            return {"timezone": "Asia/Shanghai"}
        
        # ========== Memory & Knowledge (Cortex Protocol Phase 1) ==========
        elif tool_name == "memory_tool":
            # 使用 LLM 解析用户输入，提取记忆结构
            params = self._handle_memory_update(user_input or param)
            # 检查是否返回了错误标记
            if params.get("_error"):
                # 返回特殊错误格式，让 _execute_with_registry 处理
                return {"_error": True, "message": params.get("message", "记忆解析失败")}
            return params
        
        elif tool_name == "knowledge_query":
            return {"query": param, "n_results": 3}
        
        elif tool_name == "knowledge_ingest":
            return {"file_path": param}
        
        # ========== Vision & Browser (Cortex Protocol Phase 2) ==========
        elif tool_name == "vision_tool":
            return {"query": param or "描述当前屏幕内容"}
        
        elif tool_name == "browser_tool":
            return {"instruction": param}
        
        # ========== System & Schedule (Cortex Protocol Phase 3) ==========
        elif tool_name == "system_tool":
            # 使用 LLM 解析系统控制意图
            return self._parse_system_intent(user_input or param)
        
        elif tool_name == "scheduler_tool":
            # 使用 LLM 解析时间和内容
            return self._parse_schedule_intent(user_input or param)
        
        # ========== Utility Tools ==========
        elif tool_name == "get_weather":
            return {"city": param or "Beijing"}
        
        # 默认：尝试作为单一参数传递
        return {"input": param}

    # NOTE: _generate_python_code 已移除 (Jarvis V6.1)
    # 代码生成逻辑已下沉到 PythonExecutorTool (Smart Tool)
    # Manager 不再负责代码生成，只负责意图识别和工具调度

    def _handle_memory_update(self, user_input: str) -> Dict[str, Any]:
        """
        使用 LLM 分析用户输入，提取记忆信息并转换为 MemoryTool 参数格式。
        
        Returns:
            Dict: MemoryTool 所需的参数 {"action": ..., "key": ..., "value": ...}
                  或错误标记 {"_error": True, "message": ...}
        """
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
            # Ensure response is a string
            if isinstance(response, list):
                response = str(response[0]) if response else ""
            response_str = str(response) if response else ""
            clean_json = response_str.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            
            # 验证必要字段
            if "type" not in data:
                raise ValueError("Missing 'type' field in LLM response")
            
            if data["type"] == "name":
                if "value" not in data or not str(data["value"]).strip():
                    raise ValueError("Name value is empty")
                return {"action": "update_profile", "key": "name", "value": str(data["value"]).strip()}
            elif data["type"] == "preference":
                if "value" not in data or not str(data["value"]).strip():
                    raise ValueError("Preference value is empty")
                key = data.get("key", "偏好")
                if not key or not str(key).strip():
                    key = "偏好"
                return {"action": "update_profile", "key": str(key).strip(), "value": str(data["value"]).strip()}
            elif data["type"] == "note":
                if "value" not in data or not str(data["value"]).strip():
                    raise ValueError("Note value is empty")
                return {"action": "add_note", "value": str(data["value"]).strip()}
            else:
                raise ValueError(f"Unknown memory type: {data['type']}")
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[Memory Update Error]: {e}")
            # 返回错误标记，而非静默回退到 add_note
            return {
                "_error": True,
                "message": f"无法理解您想让我记住什么。请更清楚地说明，例如：'记住我叫小明' 或 '记住我喜欢咖啡'。"
            }

    def _parse_system_intent(self, user_input: str) -> Dict[str, Any]:
        """
        使用 LLM 解析系统控制意图，提取 action/value。
        
        Returns:
            Dict: SystemTool 所需的参数 {"action": ..., "value": ...}
        """
        prompt = [
            {"role": "system", "content": """
分析用户输入，识别系统控制操作。返回 JSON。

可用操作 (action 必须是以下之一):
- volume: 音量控制 (value: 0-100 或 "+10"/"-10")
- brightness: 亮度控制 (value: 0-100)
- media_control: 媒体控制 (value: "play"/"pause"/"next"/"prev")
- open_app: 打开应用 (value: 应用名称)

格式示例:
{"action": "volume", "value": "50"}
{"action": "open_app", "value": "微信"}
{"action": "media_control", "value": "pause"}
{"action": "brightness", "value": "70"}
"""},
            {"role": "user", "content": user_input}
        ]
        
        try:
            response = self._call_llm(prompt, temperature=0.1)
            # Ensure response is a string
            if isinstance(response, list):
                response = str(response[0]) if response else ""
            response_str = str(response) if response else ""
            clean_json = response_str.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            
            action = data.get("action", "open_app")
            result = {"action": action}
            
            # 统一使用 value 字段
            if "value" in data:
                result["value"] = str(data["value"])
            elif "target" in data:
                # 兼容旧格式: target -> value
                result["value"] = str(data["target"])
            
            return result
            
        except Exception as e:
            print(f"[System Intent Parse Error]: {e}")
            # 默认当作打开应用处理
            return {"action": "open_app", "value": user_input}

    def _parse_schedule_intent(self, user_input: str) -> Dict[str, Any]:
        """
        使用 LLM 解析日程/提醒意图，提取时间和内容。
        
        Returns:
            Dict: SchedulerTool 所需的参数 {"action": ..., "content": ..., "time_str": ...}
        """
        prompt = [
            {"role": "system", "content": """
分析用户输入，提取提醒信息。返回 JSON。

可用操作 (action 必须是以下之一):
- add_reminder: 添加新提醒 (需要 time_str 和 content)
- list_reminders: 列出所有提醒

格式示例:
{"action": "add_reminder", "time_str": "明天上午9点", "content": "开会"}
{"action": "add_reminder", "time_str": "5分钟后", "content": "喝水"}
{"action": "list_reminders"}

注意:
- time_str 保留用户原始时间表达，如 "明天下午3点", "10分钟后"
- content 是提醒内容
- 如果用户问有什么提醒/任务，用 action: "list_reminders"
"""},
            {"role": "user", "content": user_input}
        ]
        
        try:
            response = self._call_llm(prompt, temperature=0.1)
            # Ensure response is a string
            if isinstance(response, list):
                response = str(response[0]) if response else ""
            response_str = str(response) if response else ""
            clean_json = response_str.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            
            result = {"action": data.get("action", "add_reminder")}
            if "content" in data:
                result["content"] = str(data["content"])
            if "time_str" in data:
                result["time_str"] = str(data["time_str"])
            
            return result
            
        except Exception as e:
            print(f"[Schedule Intent Parse Error]: {e}")
            # 无法解析时返回错误标记
            return {
                "_error": True,
                "message": "无法理解您的提醒时间。请说明具体时间，例如：'10分钟后提醒我喝水'"
            }

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
        
        # 适配参数格式 (传入 user_input 用于需要 LLM 解析的工具)
        # NOTE: V6.1 - Smart Tools (python_execute, shell_execute) 自带 LLM 代码生成
        #       Manager 只传递 instruction，工具内部处理代码生成
        user_input = self.history[-1]["content"] if self.history else param
        params = self._adapt_params_for_tool(tool.name, param, user_input)
        
        # ========== 检查参数适配是否返回错误 ==========
        if isinstance(params, dict) and params.get("_error"):
            # 参数解析失败，直接返回错误消息
            return params.get("message", "参数解析失败")
        
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
                # Ensure final_reply is a string
                if isinstance(final_reply, list):
                    final_reply = str(final_reply[0]) if final_reply else ""
                final_reply = str(final_reply) if final_reply else ""
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
                # Ensure final_reply is a string
                if isinstance(final_reply, list):
                    final_reply = str(final_reply[0]) if final_reply else ""
                final_reply = str(final_reply) if final_reply else ""
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
            # 构建强权威性的 System Report，防止 LLM "拒绝幻觉"
            tool_output = (
                f"【⚡ 系统权限验证】检测到高权限操作请求：{intent}。\n"
                f"执行动作: {param}\n"
                f"当前状态: ⛔ 已挂起 (PENDING)，等待授权。\n\n"
                "👉 对 LLM 的强制指令：\n"
                "1. **绝对不要道歉**，也不要说你无法执行（系统其实可以执行，只是在等信号）。\n"
                "2. 请用温柔自信的语气告诉主人：'这个操作涉及系统/网络权限，为了安全，我已经把它暂停了。'\n"
                "3. **必须**询问主人：'您确认要执行吗？'。"
            )
        
        # ========== Priority 2: 仅保留模型切换 (switch_model) ==========
        # NOTE: 所有工具已迁移到 Cortex Protocol (Phase 1-3)
        # system_control, open_app, schedule 已迁移到 system_tool, scheduler_tool
        
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
        
        # Ensure final_reply is a string
        if isinstance(final_reply, list):
            final_reply = str(final_reply[0]) if final_reply else ""
        final_reply = str(final_reply) if final_reply else ""
        
        # 6. 记录助手回复
        self.history.append({"role": "assistant", "content": final_reply})
        
        return final_reply

    def close(self):
        super().close()
        # 所有代理已迁移到 Cortex Protocol 工具层，由工具自行管理生命周期
        pass
