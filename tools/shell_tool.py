# Jarvis Cortex Protocol - Shell Tool (Smart Tool V6.1)
# tools/shell_tool.py

"""
ShellTool: Smart shell command generation and execution.

Jarvis V6.1 Upgrade:
- Accepts 'instruction' (natural language) OR 'command' (raw shell command)
- Uses LLMFactory.get_model("fast") for command generation
- Self-contained: no dependency on ManagerAgent for command translation
- Safety checks for dangerous commands

Risk Level: DANGEROUS (requires user confirmation)
"""

import os
import subprocess
from typing import Optional, List
from pydantic import BaseModel, Field

from core.tools.base import BaseTool, RiskLevel, ToolResult


class ShellInput(BaseModel):
    """
    Input schema for Shell tool.
    
    Accepts either:
    - instruction: Natural language description (LLM generates command)
    - command: Raw shell command to execute directly
    
    If both provided, 'command' takes precedence.
    """
    instruction: Optional[str] = Field(
        default=None,
        description="自然语言指令 (如 '查看 git 状态', '安装 pandas')。工具会自动生成对应命令。"
    )
    command: Optional[str] = Field(
        default=None,
        description="要直接执行的 Shell 命令 (PowerShell/CMD)。如果提供，将跳过命令生成。",
        examples=["git status", "pip list", "dir"]
    )
    cwd: Optional[str] = Field(
        default=None,
        description="命令执行的工作目录 (默认为当前目录)"
    )
    timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="执行超时时间 (秒)"
    )


class ShellTool(BaseTool[ShellInput]):
    """
    Smart Shell Tool: Generate and execute shell commands.
    
    Features:
    - Natural language to command via LLM (using 'fast' role)
    - Direct command execution
    - Forbidden command blacklist (rm, del, format, etc.)
    - Configurable timeout
    - Structured output with exit code
    
    This is a "Smart Tool" that encapsulates its own LLM logic,
    freeing ManagerAgent from command translation responsibilities.
    """
    
    name = "shell_execute"
    description = "执行 Shell 命令。可接受自然语言指令（自动生成命令）或直接执行命令。适用于 Git 操作、包管理、系统命令等。"
    risk_level = RiskLevel.DANGEROUS
    InputSchema = ShellInput
    tags = ["shell", "system", "command", "smart"]
    
    # Command safety configuration
    FORBIDDEN_PATTERNS: List[str] = [
        "rm -rf",
        "rm -r",
        "del /s",
        "del /f",
        "format",
        "rd /s",
        "rmdir /s",
        ":(){:|:&};:",  # Fork bomb
        "> /dev/sda",
        "mkfs",
        "dd if=",
    ]
    
    FORBIDDEN_COMMANDS: List[str] = [
        "format",
        "shutdown",
        "reboot",
    ]
    
    # Command generation system prompt
    CMD_GEN_PROMPT = """你是一个 Shell 命令专家。根据用户的自然语言指令生成 Windows PowerShell 命令。

规则：
1. 只输出命令本身，不要解释
2. 命令应该在 Windows PowerShell 或 CMD 中可执行
3. 如果任务需要多条命令，用 ; 或 && 连接
4. 避免危险命令 (rm -rf, format, shutdown 等)
5. Git 操作、包管理、文件操作都可以

示例:
- "查看 git 状态" -> git status
- "安装 pandas" -> pip install pandas
- "列出当前目录文件" -> dir 或 Get-ChildItem
- "创建名为 test 的文件夹" -> mkdir test
- "提交代码" -> git add . && git commit -m "update"
"""
    
    def __init__(self):
        super().__init__()
        self.default_cwd = os.getcwd()
        self._llm = None  # Lazy init
    
    @property
    def llm(self):
        """Lazy initialization of LLM (fast role)."""
        if self._llm is None:
            from core.llm import LLMFactory
            self._llm = LLMFactory.get_model("fast")
            print(f"[ShellTool] Using LLM: {self._llm.model_name}")
        return self._llm
    
    def _generate_command(self, instruction: str) -> str:
        """Generate shell command from natural language instruction."""
        messages = [
            {"role": "system", "content": self.CMD_GEN_PROMPT},
            {"role": "user", "content": instruction}
        ]
        
        response = self.llm.chat(messages, temperature=0.1)
        
        # Clean up response - remove code blocks if present
        command = response.strip()
        if command.startswith("```"):
            # Extract from code block
            lines = command.split("\n")
            # Filter out ``` lines
            command_lines = [l for l in lines if not l.strip().startswith("```")]
            command = "\n".join(command_lines).strip()
        
        # Remove leading $ or > if present (common in examples)
        if command.startswith("$ ") or command.startswith("> "):
            command = command[2:]
        
        return command
    
    def _check_command_safety(self, command: str) -> Optional[str]:
        """
        Check if command contains dangerous patterns.
        
        Returns:
            None if safe, or error message if dangerous
        """
        cmd_lower = command.lower()
        
        # Check forbidden patterns
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern.lower() in cmd_lower:
                return f"安全拦截：检测到危险模式 '{pattern}'"
        
        # Check if command starts with forbidden command
        cmd_parts = cmd_lower.strip().split()
        if cmd_parts:
            first_cmd = cmd_parts[0]
            for forbidden in self.FORBIDDEN_COMMANDS:
                if first_cmd == forbidden or first_cmd.endswith(f"\\{forbidden}"):
                    return f"安全拦截：禁止直接执行 '{forbidden}' 命令"
        
        return None
    
    def execute(self, params: ShellInput) -> ToolResult:
        """Execute shell command with optional command generation."""
        
        # Determine command source
        if params.command:
            command = params.command.strip()
            generated = False
        elif params.instruction:
            print(f"[ShellTool] Generating command for: {params.instruction[:50]}...")
            command = self._generate_command(params.instruction)
            generated = True
            print(f"[ShellTool] Generated command: {command}")
        else:
            return ToolResult(
                success=False,
                error="请提供 'instruction' (自然语言指令) 或 'command' (Shell 命令)"
            )
        
        if not command:
            return ToolResult(
                success=False,
                error="命令生成失败，请尝试更清晰的指令描述"
            )
        
        # Safety check
        safety_error = self._check_command_safety(command)
        if safety_error:
            return ToolResult(
                success=False,
                error=safety_error,
                metadata={"command": command, "blocked": True, "generated": generated}
            )
        
        cwd = params.cwd or self.default_cwd
        timeout = params.timeout
        
        # Validate working directory
        if not os.path.isdir(cwd):
            return ToolResult(
                success=False,
                error=f"工作目录不存在: {cwd}",
                metadata={"command": command}
            )
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                shell=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace'
            )
            
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            
            # Truncate long output
            max_output_length = 4000
            if len(stdout) > max_output_length:
                stdout = stdout[:max_output_length] + "\n...[输出过长已截断]"
            if len(stderr) > max_output_length:
                stderr = stderr[:max_output_length] + "\n...[错误信息过长已截断]"
            
            output_data = {
                "command": command,
                "output": stdout or "(无输出)",
                "exit_code": result.returncode,
            }
            
            if generated:
                output_data["instruction"] = params.instruction
                output_data["generated"] = True
            
            if result.returncode == 0:
                return ToolResult(
                    success=True,
                    data=output_data,
                    metadata={"cwd": cwd, "generated": generated}
                )
            else:
                output_data["stderr"] = stderr
                return ToolResult(
                    success=False,
                    data=output_data,
                    error=f"命令执行失败 (退出码: {result.returncode}): {stderr[:200]}"
                )
                
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"命令执行超时 ({timeout}秒)",
                metadata={"command": command, "timeout": timeout, "generated": generated}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"执行异常: {str(e)}",
                metadata={"command": command, "exception_type": type(e).__name__}
            )


# === Backward Compatibility Aliases ===

# Old input class name
ShellInput_Legacy = ShellInput

# NaturalLanguageShellTool is now merged into ShellTool
# Keep alias for backward compatibility
NaturalLanguageShellTool = ShellTool
