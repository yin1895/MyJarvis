# Jarvis Cortex Protocol - Layer 2: The Middleware Interceptor
# core/tools/executor.py

"""
ToolExecutor: Middleware for safe and observable tool execution.

This layer provides:
1. Safety: Pre-execution risk checks with user confirmation callbacks
2. Observability: Structured logging of inputs, outputs, and timing
3. Error Handling: Exception capture and natural language formatting

Design Principles:
- Stateless execution (tool state is managed by tools themselves)
- Configurable confirmation callback for dangerous operations
- Unified error formatting for LLM consumption
"""

import logging
import time
import traceback
from typing import Any, Callable, Dict, Optional, Protocol, Union
from functools import wraps
from datetime import datetime
from enum import Enum

from core.tools.base import BaseTool, RiskLevel, ToolResult

# Configure module logger
logger = logging.getLogger("jarvis.executor")


class ConfirmationResult(Enum):
    """Result of user confirmation prompt."""
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


class ConfirmationCallback(Protocol):
    """
    Protocol for user confirmation callbacks.
    
    Implementations should:
    1. Present the confirmation message to the user
    2. Wait for user response (with optional timeout)
    3. Return the appropriate ConfirmationResult
    
    Example implementations:
    - CLI: input() prompt
    - TTS: Voice prompt with STT response
    - GUI: Modal dialog
    """
    def __call__(
        self, 
        tool_name: str, 
        description: str, 
        params: Dict[str, Any]
    ) -> ConfirmationResult:
        """
        Request user confirmation for dangerous operation.
        
        Args:
            tool_name: Name of the tool requesting execution
            description: Human-readable description of the operation
            params: Input parameters being passed to the tool
            
        Returns:
            ConfirmationResult indicating user's decision
        """
        ...


def default_confirmation_callback(
    tool_name: str, 
    description: str, 
    params: Dict[str, Any]
) -> ConfirmationResult:
    """
    Default CLI-based confirmation callback.
    
    Prompts user via console input. For production use, replace with
    TTS/STT or GUI-based confirmation.
    """
    print("\n" + "=" * 60)
    print("⚠️  危险操作确认 (Dangerous Operation Confirmation)")
    print("=" * 60)
    print(f"工具: {tool_name}")
    print(f"描述: {description}")
    print(f"参数: {params}")
    print("-" * 60)
    
    try:
        response = input("是否允许执行？(y/n): ").strip().lower()
        if response in ("y", "yes", "是", "确认"):
            return ConfirmationResult.APPROVED
        else:
            return ConfirmationResult.REJECTED
    except (EOFError, KeyboardInterrupt):
        return ConfirmationResult.REJECTED


class ExecutionLog:
    """
    Structured log entry for tool execution.
    
    Captures all relevant information for debugging and analytics.
    """
    def __init__(
        self,
        tool_name: str,
        risk_level: RiskLevel,
        params: Dict[str, Any],
    ):
        self.tool_name = tool_name
        self.risk_level = risk_level
        self.params = params
        self.timestamp_start: datetime = datetime.now()
        self.timestamp_end: Optional[datetime] = None
        self.execution_time_ms: Optional[float] = None
        self.success: Optional[bool] = None
        self.result_summary: Optional[str] = None
        self.error: Optional[str] = None
        self.error_traceback: Optional[str] = None
        self.confirmation_required: bool = False
        self.confirmation_result: Optional[ConfirmationResult] = None
    
    def complete(
        self, 
        success: bool, 
        result_summary: Optional[str] = None,
        error: Optional[str] = None,
        error_traceback: Optional[str] = None
    ) -> None:
        """Mark execution as complete and calculate duration."""
        self.timestamp_end = datetime.now()
        self.execution_time_ms = (
            self.timestamp_end - self.timestamp_start
        ).total_seconds() * 1000
        self.success = success
        self.result_summary = result_summary
        self.error = error
        self.error_traceback = error_traceback
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tool_name": self.tool_name,
            "risk_level": self.risk_level.value,
            "params": self.params,
            "timestamp_start": self.timestamp_start.isoformat(),
            "timestamp_end": self.timestamp_end.isoformat() if self.timestamp_end else None,
            "execution_time_ms": self.execution_time_ms,
            "success": self.success,
            "result_summary": self.result_summary,
            "error": self.error,
            "confirmation_required": self.confirmation_required,
            "confirmation_result": self.confirmation_result.value if self.confirmation_result else None,
        }
    
    def __str__(self) -> str:
        status = "✅" if self.success else "❌"
        time_str = f"{self.execution_time_ms:.2f}ms" if self.execution_time_ms else "N/A"
        return f"[{status}] {self.tool_name} ({time_str})"


class ToolExecutor:
    """
    Middleware executor for safe and observable tool execution.
    
    Features:
    - Pre-execution safety checks for dangerous tools
    - Configurable user confirmation callback
    - Structured logging with timing metrics
    - Exception handling with natural language formatting
    - Execution history for debugging
    
    Usage:
        executor = ToolExecutor(
            confirmation_callback=my_tts_confirmation,
            require_confirmation_for=[RiskLevel.DANGEROUS]
        )
        
        result = executor.run(shell_tool, {"command": "rm -rf temp/"})
    """
    
    def __init__(
        self,
        confirmation_callback: Optional[ConfirmationCallback] = None,
        require_confirmation_for: Optional[list[RiskLevel]] = None,
        max_history: int = 100,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the ToolExecutor.
        
        Args:
            confirmation_callback: Function to call for user confirmation.
                                   If None, uses default CLI prompt.
            require_confirmation_for: Risk levels that require confirmation.
                                      Default: [RiskLevel.DANGEROUS]
            max_history: Maximum execution logs to retain in memory.
            log_level: Logging level for execution events.
        """
        self.confirmation_callback = confirmation_callback or default_confirmation_callback
        self.require_confirmation_for = require_confirmation_for or [RiskLevel.DANGEROUS]
        self.max_history = max_history
        self.log_level = log_level
        
        # Execution history (circular buffer behavior)
        self._history: list[ExecutionLog] = []
        
        # Statistics
        self._stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "rejected_confirmations": 0,
            "total_execution_time_ms": 0.0,
        }
        
        logger.setLevel(log_level)
        logger.info("ToolExecutor initialized")
    
    def run(
        self, 
        tool: BaseTool, 
        params: Dict[str, Any],
        skip_confirmation: bool = False,
    ) -> ToolResult:
        """
        Execute a tool with full middleware pipeline.
        
        Pipeline:
        1. Create execution log
        2. Check risk level and request confirmation if needed
        3. Validate input parameters
        4. Execute tool
        5. Log results and update statistics
        6. Return formatted result
        
        Args:
            tool: The tool instance to execute
            params: Input parameters (will be validated against tool's InputSchema)
            skip_confirmation: If True, bypass confirmation for this execution
                               (use with caution, e.g., for whitelisted operations)
        
        Returns:
            ToolResult containing execution outcome
        """
        # Step 1: Initialize execution log
        log_entry = ExecutionLog(
            tool_name=tool.name,
            risk_level=tool.risk_level,
            params=self._sanitize_params_for_log(params),
        )
        
        self._stats["total_executions"] += 1
        
        logger.log(
            self.log_level,
            f"[EXEC START] {tool.name} | Risk: {tool.risk_level.value} | Params: {params}"
        )
        
        try:
            # Step 2: Safety check - require confirmation for dangerous operations
            if not skip_confirmation and tool.risk_level in self.require_confirmation_for:
                log_entry.confirmation_required = True
                
                confirmation = self.confirmation_callback(
                    tool_name=tool.name,
                    description=tool.description,
                    params=params
                )
                log_entry.confirmation_result = confirmation
                
                if confirmation != ConfirmationResult.APPROVED:
                    self._stats["rejected_confirmations"] += 1
                    log_entry.complete(
                        success=False,
                        error=f"用户拒绝执行 ({confirmation.value})"
                    )
                    self._add_to_history(log_entry)
                    
                    logger.warning(f"[EXEC REJECTED] {tool.name} - User declined confirmation")
                    
                    return ToolResult(
                        success=False,
                        error="操作已被取消：用户未确认执行此危险操作。",
                        metadata={"confirmation_result": confirmation.value}
                    )
            
            # Step 3: Validate input
            try:
                validated_params = tool.validate_input(params)
            except Exception as validation_error:
                log_entry.complete(
                    success=False,
                    error=f"参数验证失败: {str(validation_error)}"
                )
                self._add_to_history(log_entry)
                self._stats["failed_executions"] += 1
                
                logger.error(f"[EXEC VALIDATION ERROR] {tool.name} - {validation_error}")
                
                return ToolResult(
                    success=False,
                    error=self._format_validation_error(validation_error),
                    metadata={"error_type": "validation"}
                )
            
            # Step 4: Execute tool
            start_time = time.perf_counter()
            result = tool.execute(validated_params)
            execution_time = (time.perf_counter() - start_time) * 1000
            
            # Inject execution time if not set
            if result.execution_time_ms is None:
                result.execution_time_ms = execution_time
            
            # Step 5: Log results
            log_entry.complete(
                success=result.success,
                result_summary=self._summarize_result(result),
                error=result.error if not result.success else None
            )
            self._add_to_history(log_entry)
            
            # Update statistics
            if result.success:
                self._stats["successful_executions"] += 1
            else:
                self._stats["failed_executions"] += 1
            self._stats["total_execution_time_ms"] += execution_time
            
            logger.log(
                self.log_level,
                f"[EXEC {'SUCCESS' if result.success else 'FAILED'}] {tool.name} | "
                f"Time: {execution_time:.2f}ms | "
                f"Result: {self._summarize_result(result)[:100]}..."
            )
            
            return result
            
        except Exception as e:
            # Step 6: Handle unexpected exceptions
            error_tb = traceback.format_exc()
            
            log_entry.complete(
                success=False,
                error=str(e),
                error_traceback=error_tb
            )
            self._add_to_history(log_entry)
            self._stats["failed_executions"] += 1
            
            logger.error(
                f"[EXEC EXCEPTION] {tool.name} - {type(e).__name__}: {e}\n{error_tb}"
            )
            
            return ToolResult(
                success=False,
                error=self._format_exception_for_llm(e, tool.name),
                metadata={
                    "error_type": type(e).__name__,
                    "traceback": error_tb if logger.level <= logging.DEBUG else None
                }
            )
    
    def run_batch(
        self, 
        tool: BaseTool, 
        params_list: list[Dict[str, Any]],
        stop_on_failure: bool = False,
    ) -> list[ToolResult]:
        """
        Execute a tool multiple times with different parameters.
        
        Args:
            tool: The tool instance to execute
            params_list: List of parameter dictionaries
            stop_on_failure: If True, stop execution on first failure
            
        Returns:
            List of ToolResult objects
        """
        results = []
        for params in params_list:
            result = self.run(tool, params)
            results.append(result)
            if stop_on_failure and not result.success:
                break
        return results
    
    # === Helper Methods ===
    
    def _add_to_history(self, log_entry: ExecutionLog) -> None:
        """Add log entry to history with circular buffer behavior."""
        self._history.append(log_entry)
        if len(self._history) > self.max_history:
            self._history.pop(0)
    
    def _sanitize_params_for_log(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize parameters for logging (hide sensitive data).
        
        Redacts values for keys containing: password, secret, token, key, api
        """
        sensitive_keys = {"password", "secret", "token", "key", "api", "credential"}
        sanitized = {}
        for k, v in params.items():
            if any(s in k.lower() for s in sensitive_keys):
                sanitized[k] = "[REDACTED]"
            elif isinstance(v, str) and len(v) > 500:
                sanitized[k] = v[:500] + "...[TRUNCATED]"
            else:
                sanitized[k] = v
        return sanitized
    
    def _summarize_result(self, result: ToolResult) -> str:
        """Create a brief summary of the result for logging."""
        if result.success:
            data_str = str(result.data)
            if len(data_str) > 200:
                return data_str[:200] + "..."
            return data_str
        return f"Error: {result.error}"
    
    def _format_validation_error(self, error: Exception) -> str:
        """Format validation error for LLM consumption."""
        error_str = str(error)
        # Pydantic validation errors are verbose, simplify them
        if "validation error" in error_str.lower():
            # Extract key information
            lines = error_str.split("\n")
            if len(lines) > 1:
                return f"参数验证失败：{lines[1].strip()}"
        return f"参数格式不正确：{error_str}"
    
    def _format_exception_for_llm(self, exception: Exception, tool_name: str) -> str:
        """
        Format an exception as natural language for LLM feedback.
        
        This helps the LLM understand what went wrong and potentially
        suggest corrections to the user.
        """
        error_type = type(exception).__name__
        error_msg = str(exception)
        
        # Common error patterns and their natural language equivalents
        error_patterns = {
            "FileNotFoundError": "找不到指定的文件或目录",
            "PermissionError": "没有权限执行此操作",
            "TimeoutError": "操作超时",
            "ConnectionError": "网络连接失败",
            "JSONDecodeError": "返回数据格式解析失败",
            "ValueError": "参数值无效",
            "TypeError": "参数类型不匹配",
            "KeyError": "缺少必要的参数",
            "OSError": "系统操作失败",
        }
        
        friendly_type = error_patterns.get(error_type, f"执行错误 ({error_type})")
        
        return f"{tool_name} 执行失败：{friendly_type}。详情：{error_msg}"
    
    # === Statistics & History Access ===
    
    @property
    def history(self) -> list[ExecutionLog]:
        """Get execution history (read-only copy)."""
        return self._history.copy()
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        total = self._stats["total_executions"]
        return {
            **self._stats,
            "success_rate": (
                self._stats["successful_executions"] / total * 100 
                if total > 0 else 0
            ),
            "average_execution_time_ms": (
                self._stats["total_execution_time_ms"] / total 
                if total > 0 else 0
            ),
        }
    
    def get_recent_errors(self, limit: int = 5) -> list[ExecutionLog]:
        """Get recent failed executions for debugging."""
        failed = [log for log in self._history if not log.success]
        return failed[-limit:]
    
    def clear_history(self) -> None:
        """Clear execution history (statistics are preserved)."""
        self._history.clear()
        logger.info("Execution history cleared")


# === Convenience: Create a global executor instance ===

_default_executor: Optional[ToolExecutor] = None


def get_default_executor() -> ToolExecutor:
    """Get or create the default global ToolExecutor instance."""
    global _default_executor
    if _default_executor is None:
        _default_executor = ToolExecutor()
    return _default_executor


def set_default_executor(executor: ToolExecutor) -> None:
    """Set the default global ToolExecutor instance."""
    global _default_executor
    _default_executor = executor


# === Decorator for quick tool execution ===

def execute_tool(
    tool: BaseTool,
    skip_confirmation: bool = False,
) -> Callable:
    """
    Decorator to wrap a function's return value through tool execution.
    
    Example:
        @execute_tool(my_tool)
        def process_data(data):
            return {"input": data}  # This becomes the tool params
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> ToolResult:
            params = func(*args, **kwargs)
            return get_default_executor().run(
                tool, 
                params, 
                skip_confirmation=skip_confirmation
            )
        return wrapper
    return decorator


# === Example Usage ===

if __name__ == "__main__":
    from pydantic import BaseModel, Field
    
    # Configure logging for demo
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    
    # Define a sample dangerous tool
    class DeleteFileInput(BaseModel):
        path: str = Field(..., description="要删除的文件路径")
        force: bool = Field(default=False, description="是否强制删除")
    
    class DeleteFileTool(BaseTool[DeleteFileInput]):
        name = "delete_file"
        description = "删除指定的文件（危险操作）"
        risk_level = RiskLevel.DANGEROUS
        InputSchema = DeleteFileInput
        
        def execute(self, params: DeleteFileInput) -> ToolResult:
            # Simulate file deletion (not actually deleting)
            return ToolResult(
                success=True,
                data=f"已删除文件: {params.path}",
                metadata={"force": params.force}
            )
    
    # Create executor and run tool
    executor = ToolExecutor(
        require_confirmation_for=[RiskLevel.DANGEROUS, RiskLevel.MODERATE]
    )
    
    tool = DeleteFileTool()
    
    # This will trigger confirmation prompt
    result = executor.run(tool, {"path": "/tmp/test.txt", "force": True})
    print(f"\nResult: {result}")
    print(f"\nStats: {executor.stats}")
    print(f"\nHistory: {[str(log) for log in executor.history]}")
