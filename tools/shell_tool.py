# Jarvis Cortex Protocol - Shell Tool
# tools/shell_tool.py

"""
ShellTool: Execute shell commands with safety checks.

Migrated from: agents/shell_agent.py
Risk Level: DANGEROUS (requires user confirmation)
"""

import os
import subprocess
from typing import Optional, List
from pydantic import BaseModel, Field

from core.tools.base import BaseTool, RiskLevel, ToolResult


class ShellInput(BaseModel):
    """Input schema for shell command execution."""
    command: str = Field(
        ..., 
        description="要执行的 Shell 命令 (PowerShell/CMD)",
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
    Execute shell commands with safety checks.
    
    Features:
    - Forbidden command blacklist (rm, del, format, etc.)
    - Configurable timeout
    - Structured output with exit code
    
    This tool is marked as DANGEROUS and requires user confirmation
    before execution through the ToolExecutor middleware.
    """
    
    name = "shell_execute"
    description = "执行 Shell 命令 (PowerShell/CMD)。适用于 Git 操作、包管理、系统命令等。"
    risk_level = RiskLevel.DANGEROUS
    InputSchema = ShellInput
    tags = ["shell", "system", "command"]
    
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
    
    def __init__(self):
        super().__init__()
        self.default_cwd = os.getcwd()
    
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
        """Execute the shell command."""
        command = params.command.strip()
        cwd = params.cwd or self.default_cwd
        timeout = params.timeout
        
        # Safety check
        safety_error = self._check_command_safety(command)
        if safety_error:
            return ToolResult(
                success=False,
                error=safety_error,
                metadata={"command": command, "blocked": True}
            )
        
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
            
            if result.returncode == 0:
                return ToolResult(
                    success=True,
                    data={
                        "command": command,
                        "output": stdout or "(无输出)",
                        "exit_code": 0
                    },
                    metadata={"cwd": cwd}
                )
            else:
                return ToolResult(
                    success=False,
                    data={
                        "command": command,
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": result.returncode
                    },
                    error=f"命令执行失败 (退出码: {result.returncode}): {stderr[:200]}"
                )
                
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"命令执行超时 ({timeout}秒)",
                metadata={"command": command, "timeout": timeout}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"执行异常: {str(e)}",
                metadata={"command": command, "exception_type": type(e).__name__}
            )


# === LLM-Powered Natural Language to Command Tool ===

class NaturalLanguageShellInput(BaseModel):
    """Input for natural language shell command generation."""
    instruction: str = Field(
        ...,
        description="用户的自然语言指令 (如 '查看 git 状态', '安装 pandas')"
    )


class NaturalLanguageShellTool(BaseTool[NaturalLanguageShellInput]):
    """
    Convert natural language to shell commands using LLM.
    
    This tool wraps ShellTool and adds LLM-based command generation.
    Note: Actual LLM call is delegated to the Agent layer (ManagerAgent).
    This tool just defines the interface.
    """
    
    name = "shell_nl"
    description = "将自然语言转换为 Shell 命令并执行。如 '提交代码' -> 'git commit'"
    risk_level = RiskLevel.DANGEROUS
    InputSchema = NaturalLanguageShellInput
    tags = ["shell", "nlp", "command"]
    
    def execute(self, params: NaturalLanguageShellInput) -> ToolResult:
        """
        This method is a placeholder.
        
        In the actual implementation, ManagerAgent will:
        1. Call LLM to convert instruction to command
        2. Execute the command via ShellTool
        
        This structure exists for schema generation purposes.
        """
        return ToolResult(
            success=False,
            error="此工具需要 LLM 支持，请通过 ManagerAgent 调用",
            metadata={"instruction": params.instruction}
        )
