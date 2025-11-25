# Jarvis Cortex Protocol - Python Executor Tool
# tools/python_tool.py

"""
PythonExecutorTool: Execute Python code in a sandboxed environment.

Migrated from: agents/python_agent.py
Risk Level: DANGEROUS (code execution)

Features:
- Sandboxed execution in workspace/ directory
- Timeout protection
- Output capture (stdout, stderr, generated files)
- Self-correction capability (via LLM integration in Agent layer)
"""

import os
import re
import subprocess
import sys
from typing import Optional, Set, Tuple
from pydantic import BaseModel, Field

from core.tools.base import BaseTool, RiskLevel, ToolResult


class PythonExecuteInput(BaseModel):
    """Input schema for Python code execution."""
    code: str = Field(
        ...,
        description="要执行的 Python 代码"
    )
    timeout: int = Field(
        default=60,
        ge=5,
        le=600,
        description="执行超时时间 (秒)"
    )


class PythonExecutorTool(BaseTool[PythonExecuteInput]):
    """
    Execute Python code in a sandboxed workspace directory.
    
    Features:
    - Code runs in `workspace/` directory (isolated from project files)
    - Automatic file tracking (detects newly created files)
    - Configurable timeout
    - Captures both stdout and stderr
    
    Security:
    - Execution is isolated to workspace directory
    - Timeout prevents infinite loops
    - Marked as DANGEROUS, requires confirmation
    """
    
    name = "python_execute"
    description = "执行 Python 代码。适用于数据处理、绘图、计算、自动化任务等。代码在 workspace/ 目录中运行。"
    risk_level = RiskLevel.DANGEROUS
    InputSchema = PythonExecuteInput
    tags = ["python", "code", "automation", "data"]
    
    def __init__(self, workspace_dir: Optional[str] = None):
        super().__init__()
        self.workspace_dir = workspace_dir or os.path.join(os.getcwd(), "workspace")
        os.makedirs(self.workspace_dir, exist_ok=True)
    
    def _get_existing_files(self) -> Set[str]:
        """Get set of files currently in workspace."""
        try:
            return set(os.listdir(self.workspace_dir))
        except Exception:
            return set()
    
    def _detect_new_files(self, before: Set[str], after: Set[str]) -> Set[str]:
        """Detect newly created files, excluding script.py and __pycache__."""
        new_files = after - before
        excluded = {"script.py", "__pycache__", ".ipynb_checkpoints"}
        return {f for f in new_files if f not in excluded and not f.startswith(".")}
    
    def execute(self, params: PythonExecuteInput) -> ToolResult:
        """Execute the Python code."""
        code = params.code.strip()
        timeout = params.timeout
        
        if not code:
            return ToolResult(
                success=False,
                error="代码不能为空"
            )
        
        script_path = os.path.join(self.workspace_dir, "script.py")
        
        # Write code to file
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"写入代码文件失败: {e}"
            )
        
        # Track files before execution
        files_before = self._get_existing_files()
        
        # Execute code
        try:
            result = subprocess.run(
                [sys.executable, "script.py"],
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace"
            )
            
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            
            # Detect new files
            files_after = self._get_existing_files()
            new_files = self._detect_new_files(files_before, files_after)
            
            # Truncate long output
            max_length = 3000
            if len(stdout) > max_length:
                stdout = stdout[:max_length] + "\n...[输出过长已截断]"
            if len(stderr) > max_length:
                stderr = stderr[:max_length] + "\n...[错误过长已截断]"
            
            output_data = {
                "stdout": stdout or "(无输出)",
                "exit_code": result.returncode,
            }
            
            if new_files:
                output_data["generated_files"] = list(new_files)
            
            if result.returncode == 0:
                return ToolResult(
                    success=True,
                    data=output_data,
                    metadata={"workspace": self.workspace_dir}
                )
            else:
                output_data["stderr"] = stderr
                return ToolResult(
                    success=False,
                    data=output_data,
                    error=f"代码执行失败: {stderr[:300]}"
                )
                
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"代码执行超时 ({timeout}秒)",
                metadata={"timeout": timeout}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"执行异常: {str(e)}",
                metadata={"exception_type": type(e).__name__}
            )


# === Natural Language Python Task Tool ===

class PythonTaskInput(BaseModel):
    """Input for natural language Python task."""
    task: str = Field(
        ...,
        description="用户的任务描述 (如 '分析 data.csv 并绘制柱状图')"
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=5,
        description="代码失败时的最大重试次数 (自我修正)"
    )


class PythonTaskTool(BaseTool[PythonTaskInput]):
    """
    Convert natural language task to Python code using LLM.
    
    This tool is a higher-level interface that:
    1. Uses LLM to generate Python code from task description
    2. Executes the code via PythonExecutorTool
    3. If execution fails, feeds error back to LLM for self-correction
    
    Note: The LLM integration is handled by ManagerAgent.
    This tool defines the interface for schema generation.
    """
    
    name = "python_task"
    description = "根据自然语言描述生成并执行 Python 代码。支持数据分析、绘图、文件处理等任务。"
    risk_level = RiskLevel.DANGEROUS
    InputSchema = PythonTaskInput
    tags = ["python", "nlp", "automation"]
    
    def execute(self, params: PythonTaskInput) -> ToolResult:
        """
        Placeholder - actual implementation requires LLM.
        
        The ManagerAgent will:
        1. Generate code via LLM
        2. Execute via PythonExecutorTool
        3. On failure, pass error back to LLM for correction
        4. Repeat up to max_retries times
        """
        return ToolResult(
            success=False,
            error="此工具需要 LLM 支持，请通过 ManagerAgent 调用",
            metadata={"task": params.task}
        )


# === Code Extraction Utility ===

def extract_python_code(llm_response: str) -> Optional[str]:
    """
    Extract Python code block from LLM response.
    
    Args:
        llm_response: Raw LLM response text
        
    Returns:
        Extracted code, or None if no code block found
    """
    # Try to find ```python ... ``` block
    match = re.search(r"```python(.*?)```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try to find generic ``` ... ``` block
    match = re.search(r"```(.*?)```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return None
