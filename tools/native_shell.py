# Jarvis V7.0 - Native Shell Execution Tool
# tools/native_shell.py

"""
Native LangChain Tool for shell command execution.

Features:
- Execute PowerShell/CMD commands
- Safety checks for dangerous commands
- Configurable timeout
- Output truncation

Risk Level: DANGEROUS (shell commands can have system-wide effects)
"""

import os
import subprocess
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ============== Input Schema ==============

class ShellExecuteInput(BaseModel):
    """Input schema for shell command execution."""
    command: str = Field(
        ...,
        description="要执行的 Shell 命令 (PowerShell/CMD)",
        examples=["git status", "pip list", "dir", "Get-ChildItem"]
    )
    cwd: Optional[str] = Field(
        default=None,
        description="命令执行的工作目录（默认为当前目录）"
    )
    timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="执行超时时间（秒）"
    )


# ============== Safety Configuration ==============

# Dangerous command patterns - will be blocked
FORBIDDEN_PATTERNS: List[str] = [
    "rm -rf",
    "rm -r /",
    "del /s /q",
    "del /f",
    "format c:",
    "format d:",
    "rd /s /q",
    "rmdir /s /q",
    ":(){:|:&};:",  # Fork bomb
    "> /dev/sda",
    "mkfs",
    "dd if=/dev/zero",
    "shutdown",
    "reboot",
    "Remove-Item -Recurse -Force /",
    "Remove-Item -Recurse -Force C:\\",
]

# Commands that start with these are blocked
FORBIDDEN_COMMANDS: List[str] = [
    "format",
    "shutdown",
    "reboot",
    "halt",
]


# ============== Helper Functions ==============

def _check_command_safety(command: str) -> Optional[str]:
    """
    Check if command contains dangerous patterns.
    
    Returns:
        None if safe, or error message if dangerous
    """
    cmd_lower = command.lower()
    
    # Check forbidden patterns
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.lower() in cmd_lower:
            return f"安全拦截：检测到危险模式 '{pattern}'"
    
    # Check if command starts with forbidden command
    cmd_parts = cmd_lower.strip().split()
    if cmd_parts:
        first_cmd = cmd_parts[0]
        for forbidden in FORBIDDEN_COMMANDS:
            if first_cmd == forbidden or first_cmd.endswith(f"\\{forbidden}"):
                return f"安全拦截：禁止直接执行 '{forbidden}' 命令"
    
    return None


def _execute_command(command: str, cwd: str, timeout: int) -> dict:
    """
    Execute shell command and return results.
    
    Returns:
        dict with keys: success, stdout, stderr, exit_code
    """
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
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "exit_code": result.returncode,
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"命令执行超时 ({timeout}秒)",
            "exit_code": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "exit_code": -1,
        }


# ============== Native Tool ==============

@tool(args_schema=ShellExecuteInput)
def shell_execute(command: str, cwd: Optional[str] = None, timeout: int = 30) -> str:
    """
    执行 Shell 命令（PowerShell/CMD）。
    
    使用场景:
    - Git 操作: command="git status"
    - 包管理: command="pip install pandas"
    - 文件操作: command="dir" 或 command="Get-ChildItem"
    - 系统命令: command="ipconfig"
    
    注意：危险命令（如 rm -rf, format）会被自动拦截。
    
    Args:
        command: 要执行的 Shell 命令
        cwd: 工作目录（默认为当前目录）
        timeout: 超时时间（秒）
        
    Returns:
        命令执行结果
    """
    if not command or not command.strip():
        return "错误：请提供要执行的命令"
    
    command = command.strip()
    
    # Safety check
    safety_error = _check_command_safety(command)
    if safety_error:
        return safety_error
    
    # Resolve working directory
    working_dir = cwd or os.getcwd()
    if not os.path.isdir(working_dir):
        return f"错误：工作目录不存在 - {working_dir}"
    
    # Execute command
    result = _execute_command(command, working_dir, timeout)
    
    # Format output
    stdout = result["stdout"]
    stderr = result["stderr"]
    exit_code = result["exit_code"]
    
    # Truncate long output
    max_output_length = 4000
    if len(stdout) > max_output_length:
        stdout = stdout[:max_output_length] + "\n...[输出过长已截断]"
    if len(stderr) > max_output_length:
        stderr = stderr[:max_output_length] + "\n...[错误信息过长已截断]"
    
    # Build response
    response_parts = [f"命令: {command}"]
    
    if result["success"]:
        response_parts.append(f"状态: 成功 (退出码: {exit_code})")
        if stdout:
            response_parts.append(f"输出:\n{stdout}")
        else:
            response_parts.append("输出: (无)")
    else:
        response_parts.append(f"状态: 失败 (退出码: {exit_code})")
        if stdout:
            response_parts.append(f"输出:\n{stdout}")
        if stderr:
            response_parts.append(f"错误:\n{stderr}")
    
    return "\n".join(response_parts)


# ============== Risk Level Metadata ==============
shell_execute.metadata = {"risk_level": "dangerous"}


# ============== Export ==============
__all__ = ["shell_execute", "ShellExecuteInput"]
