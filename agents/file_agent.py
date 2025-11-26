import os
import json
from pathlib import Path
from agents.base import BaseAgent

class FileAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="FileAgent")
        self.work_dir = Path(os.getcwd()).resolve()

    def _is_safe_path(self, path_str: str) -> bool:
        """安全检查：防止路径遍历攻击"""
        try:
            target_path = (self.work_dir / path_str).resolve()
            # 检查 target_path 是否以 work_dir 开头
            return str(target_path).startswith(str(self.work_dir))
        except Exception:
            return False

    def _list_dir(self, rel_path: str = ".") -> str:
        """列出目录内容"""
        if not self._is_safe_path(rel_path):
            return "❌ 访问被拒绝：只能访问工作区内的文件。"
        
        target_path = (self.work_dir / rel_path).resolve()
        if not target_path.exists():
            return f"❌ 路径不存在: {rel_path}"
        if not target_path.is_dir():
            return f"❌ 这不是一个文件夹: {rel_path}"

        try:
            items = []
            for item in target_path.iterdir():
                type_icon = "📁" if item.is_dir() else "📄"
                items.append(f"{type_icon} {item.name}")
            
            if not items:
                return "📂 空文件夹"
            return "\n".join(items)
        except Exception as e:
            return f"❌ 列出目录失败: {e}"

    def _read_file(self, rel_path: str) -> str:
        """读取文件内容"""
        if not self._is_safe_path(rel_path):
            return "❌ 访问被拒绝：只能访问工作区内的文件。"
            
        target_path = (self.work_dir / rel_path).resolve()
        if not target_path.exists():
            return f"❌ 文件不存在: {rel_path}"
        if not target_path.is_file():
            return f"❌ 这不是一个文件: {rel_path}"

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                content = f.read()
                if len(content) > 2000:
                    return content[:2000] + "\n\n... (文件太长，已截断) ..."
                return content
        except UnicodeDecodeError:
            return "❌ 无法读取二进制文件或编码格式不支持。"
        except Exception as e:
            return f"❌ 读取失败: {e}"

    def _write_file(self, rel_path: str, content: str) -> str:
        """写入文件"""
        if not self._is_safe_path(rel_path):
            return "❌ 访问被拒绝：只能访问工作区内的文件。"
            
        target_path = (self.work_dir / rel_path).resolve()
        try:
            # 自动创建父目录
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"✅ 文件已保存: {rel_path}"
        except Exception as e:
            return f"❌ 写入失败: {e}"

    def run(self, user_input: str) -> str:
        """
        解析用户指令并执行文件操作。
        使用 LLM 将自然语言转换为 JSON 指令。
        """
        prompt = [
            {"role": "system", "content": """
你是一个文件系统助手。请将用户的自然语言指令转换为 JSON 操作。
工作目录: . (当前项目根目录)

支持的操作 (op) 和参数:
1. list: 列出目录 (param: 目录路径，默认为 ".")
2. read: 读取文件 (param: 文件路径)
3. write: 写入文件 (param: 文件路径, content: 内容)

示例:
"看看当前目录下有什么" -> {"op": "list", "param": "."}
"列出 agents 文件夹" -> {"op": "list", "param": "agents"}
"读取 main.py" -> {"op": "read", "param": "main.py"}
"创建一个 test.txt 内容是 hello" -> {"op": "write", "param": "test.txt", "content": "hello"}
"""},
            {"role": "user", "content": user_input}
        ]

        try:
            response = self._call_llm(prompt, temperature=0.1)
            # 确保 response 是字符串
            if isinstance(response, list):
                response = str(response[0]) if response else ""
            response_str = str(response)
            clean_json = response_str.replace("```json", "").replace("```", "").strip()
            cmd = json.loads(clean_json)
            
            op = cmd.get("op")
            param = cmd.get("param", ".")
            
            if op == "list":
                return self._list_dir(param)
            elif op == "read":
                return self._read_file(param)
            elif op == "write":
                content = cmd.get("content", "")
                return self._write_file(param, content)
            else:
                return "无法理解该文件指令。"

        except Exception as e:
            print(f"[FileAgent Error]: {e}")
            return "文件操作解析失败。"
