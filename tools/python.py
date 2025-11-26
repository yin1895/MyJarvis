# Jarvis V7.0 - Native Python Interpreter Tool
# tools/python.py

"""
Native LangChain Tool for Python code execution.

Features:
- Execute Python code in sandboxed workspace/
- Capture stdout/stderr
- Track generated files
- Configurable timeout

Risk Level: DANGEROUS (code execution)
"""

import os
import subprocess
import sys
from typing import Optional, Set
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ============== Input Schema ==============

class PythonInterpreterInput(BaseModel):
    """Input schema for Python code execution."""
    code: str = Field(
        ...,
        description="要执行的 Python 代码"
    )
    timeout: int = Field(
        default=60,
        ge=5,
        le=600,
        description="执行超时时间（秒）"
    )


# ============== Configuration ==============

# Default workspace directory for code execution
WORKSPACE_DIR = os.path.join(os.getcwd(), "workspace")

# Ensure workspace exists
os.makedirs(WORKSPACE_DIR, exist_ok=True)


# ============== Helper Functions ==============

def _get_existing_files(workspace: str) -> Set[str]:
    """Get set of files currently in workspace."""
    try:
        return set(os.listdir(workspace))
    except Exception:
        return set()


def _detect_new_files(before: Set[str], after: Set[str]) -> Set[str]:
    """Detect newly created files, excluding temp files."""
    new_files = after - before
    excluded = {"script.py", "__pycache__", ".ipynb_checkpoints"}
    return {f for f in new_files if f not in excluded and not f.startswith(".")}


def _execute_code(code: str, workspace: str, timeout: int) -> dict:
    """
    Execute Python code and return results.
    
    Returns:
        dict with keys: success, stdout, stderr, exit_code, new_files
    """
    script_path = os.path.join(workspace, "script.py")
    
    # Track files before execution
    files_before = _get_existing_files(workspace)
    
    # Write code to file
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"无法写入脚本文件: {e}",
            "exit_code": -1,
            "new_files": [],
        }
    
    # Execute
    try:
        result = subprocess.run(
            [sys.executable, "script.py"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace"
        )
        
        files_after = _get_existing_files(workspace)
        new_files = _detect_new_files(files_before, files_after)
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "exit_code": result.returncode,
            "new_files": list(new_files),
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"代码执行超时 ({timeout}秒)",
            "exit_code": -1,
            "new_files": [],
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "exit_code": -1,
            "new_files": [],
        }


# ============== Native Tool ==============

@tool(args_schema=PythonInterpreterInput)
def python_interpreter(code: str, timeout: int = 60) -> str:
    """
    执行 Python 代码。代码在 workspace/ 目录下运行。
    
    使用场景:
    - 数据处理和分析
    - 数学计算
    - 文件处理
    - 生成图表（matplotlib）
    - 自动化脚本
    
    注意：
    - 代码在 workspace/ 目录执行
    - 生成的文件保存在 workspace/
    - 使用 print() 输出结果
    
    Args:
        code: 要执行的 Python 代码
        timeout: 超时时间（秒，默认60）
        
    Returns:
        执行结果（stdout 和生成的文件列表）
    """
    if not code or not code.strip():
        return "错误：请提供要执行的代码"
    
    code = code.strip()
    
    # Execute code
    result = _execute_code(code, WORKSPACE_DIR, timeout)
    
    # Format output
    stdout = result["stdout"]
    stderr = result["stderr"]
    new_files = result["new_files"]
    
    # Truncate long output
    max_output_length = 3000
    if len(stdout) > max_output_length:
        stdout = stdout[:max_output_length] + "\n...[输出过长已截断]"
    if len(stderr) > max_output_length:
        stderr = stderr[:max_output_length] + "\n...[错误信息过长已截断]"
    
    # Build response
    response_parts = []
    
    if result["success"]:
        response_parts.append("状态: 执行成功")
        
        if stdout:
            response_parts.append(f"输出:\n{stdout}")
        else:
            response_parts.append("输出: (无)")
        
        if new_files:
            response_parts.append(f"生成的文件: {', '.join(new_files)}")
    else:
        response_parts.append(f"状态: 执行失败 (退出码: {result['exit_code']})")
        
        if stdout:
            response_parts.append(f"输出:\n{stdout}")
        
        if stderr:
            response_parts.append(f"错误:\n{stderr}")
    
    return "\n".join(response_parts)


# ============== Risk Level Metadata ==============
python_interpreter.metadata = {"risk_level": "dangerous"}


# ============== Export ==============
__all__ = ["python_interpreter", "PythonInterpreterInput", "WORKSPACE_DIR"]
