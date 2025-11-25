import os
import subprocess
import sys
import re
from typing import Tuple
from agents.base import BaseAgent

class PythonAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="PythonAgent")
        # 设定工作目录为项目根目录下的 workspace
        self.work_dir = os.path.join(os.getcwd(), "workspace")
        os.makedirs(self.work_dir, exist_ok=True)

    def _is_safe_path(self, filepath: str) -> bool:
        """确保路径在 workspace 内部，防止路径遍历"""
        # 解析绝对路径
        abs_path = os.path.abspath(os.path.join(self.work_dir, filepath))
        # 检查是否以 work_dir 开头
        return abs_path.startswith(os.path.abspath(self.work_dir))

    def run_code(self, code: str) -> Tuple[str, bool]:
        """
        执行 Python 代码
        返回: (output_message, is_success)
        """
        script_path = os.path.join(self.work_dir, 'script.py')
        
        # 1. 写入文件
        try:
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(code)
        except Exception as e:
            return f"写入代码文件失败: {e}", False

        # 记录执行前的文件列表
        files_before = set(os.listdir(self.work_dir))

        # 2. 执行代码
        try:
            # 使用当前 Python 解释器执行
            # cwd=self.work_dir 确保生成的文件的默认路径在 workspace 内
            result = subprocess.run(
                [sys.executable, 'script.py'],
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=60,
                encoding='utf-8',
                errors='replace' # 防止编码错误导致崩溃
            )
            
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            
            # 3. 检查新生成的文件
            files_after = set(os.listdir(self.work_dir))
            new_files = files_after - files_before
            # 过滤掉 script.py 自身和 __pycache__
            new_files = {f for f in new_files if f != 'script.py' and f != '__pycache__'}
            
            output_msg = ""
            if stdout:
                output_msg += f"【标准输出】:\n{stdout}\n"
            
            if new_files:
                output_msg += f"\n【生成文件】: {', '.join(new_files)} (位于 workspace 目录)"

            if result.returncode != 0:
                # 执行失败
                error_msg = f"{output_msg}\n【错误信息】:\n{stderr}"
                return error_msg, False
            
            return output_msg if output_msg else "代码执行成功，无输出。", True

        except subprocess.TimeoutExpired:
            return "代码执行超时 (60s)。", False
        except Exception as e:
            return f"执行过程发生异常: {e}", False

    def run(self, user_requirement: str) -> str:
        """
        根据用户需求生成并执行代码，包含自我修正机制
        """
        max_retries = 3
        history = []
        
        # 1. 获取当前工作区文件列表 (Context Injection)
        try:
            files = os.listdir(self.work_dir)
            file_list_str = ", ".join(files) if files else "无"
            context_info = f"当前工作区文件列表: [{file_list_str}]"
        except Exception as e:
            context_info = f"无法获取文件列表: {e}"

        system_prompt = f"""
你是一个 Python 数据专家和代码解释器。
你的任务是编写 Python 代码来解决用户的问题。

【环境约束】
1. 工作目录: `{self.work_dir}`
2. 已安装库: pandas, numpy, matplotlib, seaborn, sklearn, requests 等。
3. 文件操作: 
   - 读取文件前，必须使用 `os.path.exists()` 检查文件是否存在。
   - 假设文件就在当前目录下，直接使用文件名。
   - 处理 CSV/Excel 前，建议先 `print(df.head())` 查看列名，防止猜测错误。
4. 绘图规范:
   - 必须设置中文字体防止乱码: `plt.rcParams['font.sans-serif'] = ['SimHei']` (Windows) 或 `['Arial Unicode MS']` (Mac)。
   - 必须保存为图片文件 (e.g. `plt.savefig('result.png')`)，禁止使用 `plt.show()`。
5. 输出规范:
   - 任何结果必须使用 `print()` 输出到标准输出，否则用户看不到。
6. **网络请求**: 使用 `requests` 库时，**严禁**直接调用 `.json()`。必须先检查 `if response.status_code == 200:`。如果状态码不是 200，必须执行 `print(f"Error: {response.status_code}, Content: {response.text}")` 以便调试。

【回复格式】
请直接返回可执行的 Python 代码块，包裹在 ```python ... ``` 中。
不要包含多余的解释文字，除非是代码注释。
"""

        # 将环境上下文注入到用户请求中
        current_prompt = f"{context_info}\n用户需求: {user_requirement}"
        
        for attempt in range(max_retries):
            # 2. 调用 LLM 生成代码
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": current_prompt})
            
            response = self._call_llm(messages, temperature=0.1)
            
            # 3. 提取代码块
            code_match = re.search(r"```python(.*?)```", response, re.DOTALL)
            if not code_match:
                # 如果没有代码块，可能 LLM 拒绝或者是纯文本回复，直接返回
                return response
            
            code = code_match.group(1).strip()
            
            # 4. 执行代码
            output, success = self.run_code(code)
            
            if success:
                return f"代码执行成功。\n{output}"
            else:
                # 5. 失败处理
                print(f"[PythonAgent] Attempt {attempt+1} failed: {output}")
                
                # 检查是否是缺少依赖
                if "ModuleNotFoundError" in output:
                    return f"执行失败，缺少依赖库。\n错误信息：{output}\n请提示用户安装相关库。"
                
                # 将错误信息回传给 LLM 进行修正
                history.append({"role": "assistant", "content": response})
                current_prompt = f"代码执行报错，请修复代码。\n错误信息：\n{output}"
        
        return "抱歉，尝试了多次修复代码，但依然执行失败。"
