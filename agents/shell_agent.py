import subprocess
import os
from agents.base import BaseAgent

class ShellAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ShellAgent")
        self.forbidden_commands = ["rm", "del", "format", "rd", "rmdir"]

    def _execute_command(self, command: str) -> str:
        try:
            # Security check (simple keyword matching)
            cmd_parts = command.lower().split()
            for forbidden in self.forbidden_commands:
                # Check if forbidden command is the start of the command or after a pipe/semicolon
                # This is a basic check.
                if forbidden in command.lower():
                     return f"❌ 安全警告: 禁止执行高危命令 '{forbidden}'"

            result = subprocess.run(
                command,
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                shell=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return result.stdout.strip() or "✅ 执行成功 (无输出)"
            else:
                return f"❌ 执行出错:\n{result.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return "❌ 执行超时 (30s)"
        except Exception as e:
            return f"❌ 系统错误: {str(e)}"

    def run(self, user_input: str) -> str:
        # 1. LLM 解析命令
        prompt = [
            {"role": "system", "content": """
你是一个 Shell 命令生成器。请将用户的自然语言转换为 Windows PowerShell 或 CMD 命令。
规则：
1. 只返回命令本身，不要包含 Markdown (```), 不要解释。
2. 如果用户想执行 Git 操作、安装依赖、文件管理等，生成对应命令。
3. 确保命令在 Windows 环境下有效。
"""},
            {"role": "user", "content": user_input}
        ]
        
        command = self._call_llm(prompt, temperature=0.1).strip()
        
        # 清理可能的 markdown
        command = command.replace("```powershell", "").replace("```bash", "").replace("```cmd", "").replace("```", "").strip()
        
        print(f"[ShellAgent]: 拟执行命令 -> {command}")
        
        # 2. 执行
        output = self._execute_command(command)
        
        # 3. 总结 (如果输出太长，截断)
        if len(output) > 1000:
            summary = output[:1000] + "\n...(输出过长已截断)"
        else:
            summary = output
            
        return f"执行命令: `{command}`\n\n结果:\n{summary}"
