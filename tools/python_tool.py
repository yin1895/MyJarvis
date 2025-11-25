# Jarvis Cortex Protocol - Python Executor Tool (Smart Tool V6.1)
# tools/python_tool.py

"""
PythonExecutorTool: Smart Python code generation and execution.

Jarvis V6.1 Upgrade:
- Accepts 'instruction' (natural language) OR 'code' (raw Python)
- Uses LLMFactory.get_model("coder") for code generation
- Self-contained: no dependency on ManagerAgent for code generation
- Supports self-correction on execution failure

Risk Level: DANGEROUS (code execution, requires confirmation)
"""

import os
import re
import subprocess
import sys
from typing import Optional, Set, List
from pydantic import BaseModel, Field

from core.tools.base import BaseTool, RiskLevel, ToolResult


class PythonInput(BaseModel):
    """
    Input schema for Python tool.
    
    Accepts either:
    - instruction: Natural language task description (LLM generates code)
    - code: Raw Python code to execute directly
    
    If both provided, 'code' takes precedence.
    """
    instruction: Optional[str] = Field(
        default=None,
        description="任务描述 (如 '分析 data.csv 并绘制柱状图')。工具会自动生成 Python 代码。"
    )
    code: Optional[str] = Field(
        default=None,
        description="要直接执行的 Python 代码。如果提供，将跳过代码生成。"
    )
    timeout: int = Field(
        default=60,
        ge=5,
        le=600,
        description="执行超时时间 (秒)"
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="代码执行失败时的自动修复重试次数"
    )


class PythonExecutorTool(BaseTool[PythonInput]):
    """
    Smart Python Tool: Generate and execute Python code.
    
    Features:
    - Natural language to code via LLM (using 'coder' role)
    - Direct code execution in sandboxed workspace/
    - Automatic error correction with LLM feedback loop
    - File generation tracking
    - Configurable timeout
    
    This is a "Smart Tool" that encapsulates its own LLM logic,
    freeing ManagerAgent from code generation responsibilities.
    """
    
    name = "python_execute"
    description = "执行 Python 任务。可接受任务描述（自动生成代码）或直接执行代码。适用于数据处理、绘图、计算、自动化等。"
    risk_level = RiskLevel.DANGEROUS
    InputSchema = PythonInput
    tags = ["python", "code", "automation", "data", "smart"]
    
    # Code generation system prompt
    CODE_GEN_PROMPT = """你是一个 Python 代码生成专家。根据用户任务生成可执行的 Python 代码。

规则：
1. 只输出代码块，用 ```python 包裹
2. 代码将在 workspace/ 目录下执行
3. 使用 print() 输出结果供用户查看
4. 如需生成文件，保存在当前目录 (workspace/)
5. 代码必须完整可执行，不要省略导入语句
6. 优先使用标准库，如需第三方库请注明

常用场景示例：
- 数据分析: 使用 pandas, matplotlib
- 文件操作: 使用 os, shutil, pathlib
- 网络请求: 使用 requests
- 图像处理: 使用 PIL/Pillow
"""

    CODE_FIX_PROMPT = """上次执行的代码出错了，请修复。

原代码:
```python
{code}
```

错误信息:
{error}

请生成修复后的完整代码（用 ```python 包裹）。只输出代码，不要解释。
"""
    
    def __init__(self, workspace_dir: Optional[str] = None):
        super().__init__()
        self.workspace_dir = workspace_dir or os.path.join(os.getcwd(), "workspace")
        os.makedirs(self.workspace_dir, exist_ok=True)
        self._llm = None  # Lazy init
    
    @property
    def llm(self):
        """Lazy initialization of LLM (coder role)."""
        if self._llm is None:
            from core.llm import LLMFactory
            self._llm = LLMFactory.get_model("coder")
            print(f"[PythonTool] Using LLM: {self._llm.model_name}")
        return self._llm
    
    def _generate_code(self, instruction: str) -> str:
        """Generate Python code from natural language instruction."""
        messages = [
            {"role": "system", "content": self.CODE_GEN_PROMPT},
            {"role": "user", "content": instruction}
        ]
        
        response = self.llm.chat(messages, temperature=0.2)
        
        # Extract code from response
        code = self._extract_code(response)
        if code:
            return code
        
        # If extraction failed, return raw response (might be code without blocks)
        return response.strip()
    
    def _fix_code(self, code: str, error: str) -> str:
        """Attempt to fix code based on error message."""
        prompt = self.CODE_FIX_PROMPT.format(code=code, error=error[:1000])
        
        messages = [
            {"role": "system", "content": self.CODE_GEN_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.chat(messages, temperature=0.1)
        
        fixed_code = self._extract_code(response)
        return fixed_code or code  # Return original if extraction fails
    
    def _extract_code(self, text: str) -> Optional[str]:
        """Extract Python code block from LLM response."""
        # Try ```python ... ``` block
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Try generic ``` ... ``` block
        match = re.search(r"```(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return None
    
    def _get_existing_files(self) -> Set[str]:
        """Get set of files currently in workspace."""
        try:
            return set(os.listdir(self.workspace_dir))
        except Exception:
            return set()
    
    def _detect_new_files(self, before: Set[str], after: Set[str]) -> Set[str]:
        """Detect newly created files, excluding temp files."""
        new_files = after - before
        excluded = {"script.py", "__pycache__", ".ipynb_checkpoints"}
        return {f for f in new_files if f not in excluded and not f.startswith(".")}
    
    def _execute_code(self, code: str, timeout: int) -> dict:
        """
        Execute Python code and return results.
        
        Returns:
            dict with keys: success, stdout, stderr, exit_code, new_files
        """
        script_path = os.path.join(self.workspace_dir, "script.py")
        
        # Write code to file
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
        
        # Track files before execution
        files_before = self._get_existing_files()
        
        # Execute
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
            
            files_after = self._get_existing_files()
            new_files = self._detect_new_files(files_before, files_after)
            
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
    
    def execute(self, params: PythonInput) -> ToolResult:
        """Execute Python task with optional code generation and self-correction."""
        
        # Determine code source
        if params.code:
            code = params.code.strip()
            generated = False
        elif params.instruction:
            print(f"[PythonTool] Generating code for: {params.instruction[:50]}...")
            code = self._generate_code(params.instruction)
            generated = True
            print(f"[PythonTool] Generated {len(code)} chars of code")
        else:
            return ToolResult(
                success=False,
                error="请提供 'instruction' (任务描述) 或 'code' (Python 代码)"
            )
        
        if not code:
            return ToolResult(
                success=False,
                error="代码生成失败，请尝试更详细的任务描述"
            )
        
        # Execute with retry loop
        attempt = 0
        max_attempts = params.max_retries + 1
        last_error = ""
        result = {}
        
        while attempt < max_attempts:
            attempt += 1
            print(f"[PythonTool] Execution attempt {attempt}/{max_attempts}")
            
            result = self._execute_code(code, params.timeout)
            
            if result["success"]:
                # Success!
                output_data = {
                    "stdout": result["stdout"][:3000] or "(无输出)",
                    "exit_code": 0,
                }
                
                if result["new_files"]:
                    output_data["generated_files"] = result["new_files"]
                
                if generated:
                    output_data["generated_code"] = code[:500] + ("..." if len(code) > 500 else "")
                
                return ToolResult(
                    success=True,
                    data=output_data,
                    metadata={
                        "workspace": self.workspace_dir,
                        "attempts": attempt,
                        "code_generated": generated,
                    }
                )
            
            # Execution failed
            last_error = result["stderr"] or result["stdout"]
            
            if attempt < max_attempts:
                print(f"[PythonTool] Execution failed, attempting fix...")
                code = self._fix_code(code, last_error)
        
        # All attempts failed
        return ToolResult(
            success=False,
            data={
                "stdout": result.get("stdout", ""),
                "stderr": last_error[:1000],
                "exit_code": result.get("exit_code", -1),
                "code": code[:500] + ("..." if len(code) > 500 else ""),
            },
            error=f"代码执行失败 (尝试 {attempt} 次): {last_error[:200]}",
            metadata={
                "workspace": self.workspace_dir,
                "attempts": attempt,
            }
        )


# === Backward Compatibility Aliases ===

# Old schema name (for existing code references)
PythonExecuteInput = PythonInput

# PythonTaskTool is now merged into PythonExecutorTool
# Keep alias for backward compatibility
PythonTaskTool = PythonExecutorTool


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
