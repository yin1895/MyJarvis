# Jarvis Cortex Protocol - Layer 1: The Atom Standard
# core/tools/base.py

"""
BaseTool: Abstract base class for all Jarvis tools.

All tools in the Jarvis ecosystem must inherit from BaseTool and implement:
1. InputSchema: A Pydantic model defining the tool's input parameters
2. execute(): The core logic of the tool
3. Metadata: name, description, risk_level

This design enables:
- Native LLM Function Calling via to_openai_schema()
- Type-safe input validation
- Uniform safety classification
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Type, TypeVar, Generic
from pydantic import BaseModel, Field
import time
import logging

# Configure module logger
logger = logging.getLogger("jarvis.tools")


class RiskLevel(str, Enum):
    """
    Tool risk classification for safety middleware.
    
    - SAFE: No side effects, read-only operations (e.g., get_time, search)
    - MODERATE: Limited side effects, reversible (e.g., file read, web browse)
    - DANGEROUS: System modifications, irreversible (e.g., shell exec, file delete)
    """
    SAFE = "safe"
    MODERATE = "moderate"
    DANGEROUS = "dangerous"


class ToolResult(BaseModel):
    """
    Standardized tool execution result.
    
    Attributes:
        success: Whether the tool executed successfully
        data: The output data from the tool (if successful)
        error: Error message (if failed)
        execution_time_ms: Execution duration in milliseconds
        metadata: Additional context (e.g., warnings, suggestions)
    """
    success: bool = True
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_natural_language(self) -> str:
        """Format result for LLM consumption with intelligent formatting."""
        if self.success:
            if self.data is None:
                return "操作已成功完成。"
            
            if isinstance(self.data, dict):
                # 智能格式化结构化数据
                if "stdout" in self.data:
                    output = self.data.get("stdout", "(无输出)")
                    if self.data.get("generated_files"):
                        output += f"\n生成文件: {', '.join(self.data['generated_files'])}"
                    return output
                if "output" in self.data:
                    return self.data.get("output", "(无输出)")
                if "results" in self.data:  # 搜索结果
                    results = self.data["results"]
                    if isinstance(results, list) and results:
                        formatted = []
                        for i, r in enumerate(results[:5], 1):
                            title = r.get("title", "无标题") if isinstance(r, dict) else str(r)
                            snippet = r.get("snippet", "")[:200] if isinstance(r, dict) else ""
                            formatted.append(f"【{i}】{title}\n{snippet}")
                        return "\n\n".join(formatted)
                if "time" in self.data or "datetime" in self.data:
                    return f"当前时间: {self.data.get('datetime', self.data.get('time'))}"
                if "weather" in self.data:
                    return self.data.get("weather", str(self.data))
                if "query" in self.data and "raw_content" in self.data:
                    return self.data.get("raw_content", "(无内容)")
                # 默认: 格式化为可读文本
                import json
                return json.dumps(self.data, ensure_ascii=False, indent=2)
            
            return str(self.data)
        else:
            return f"执行失败：{self.error}"


# Type variable for InputSchema
TInput = TypeVar("TInput", bound=BaseModel)


class BaseTool(ABC, Generic[TInput]):
    """
    Abstract base class for all Jarvis tools.
    
    Subclasses must define:
    - name: Unique tool identifier (snake_case recommended)
    - description: Human-readable description for LLM tool selection
    - risk_level: Safety classification (RiskLevel enum)
    - InputSchema: Pydantic model for input validation
    - execute(): Core tool logic
    
    Example:
        class GetTimeTool(BaseTool[GetTimeInput]):
            name = "get_current_time"
            description = "获取当前系统时间"
            risk_level = RiskLevel.SAFE
            InputSchema = GetTimeInput
            
            def execute(self, params: GetTimeInput) -> ToolResult:
                return ToolResult(success=True, data=datetime.now().isoformat())
    """
    
    # === Required Metadata (must be overridden) ===
    name: str = ""
    description: str = ""
    risk_level: RiskLevel = RiskLevel.SAFE
    InputSchema: Type[TInput] = None  # type: ignore
    
    # === Optional Metadata ===
    version: str = "1.0.0"
    author: str = "Jarvis Team"
    tags: list[str] = []
    
    def __init__(self):
        """Initialize the tool and validate metadata."""
        self._validate_metadata()
        logger.debug(f"Tool initialized: {self.name} (risk={self.risk_level.value})")
    
    def _validate_metadata(self) -> None:
        """Ensure required metadata is properly defined."""
        if not self.name:
            raise ValueError(f"{self.__class__.__name__} must define 'name' attribute")
        if not self.description:
            raise ValueError(f"{self.__class__.__name__} must define 'description' attribute")
        if self.InputSchema is None:
            raise ValueError(f"{self.__class__.__name__} must define 'InputSchema' class attribute")
    
    @abstractmethod
    def execute(self, params: TInput) -> ToolResult:
        """
        Execute the tool with validated parameters.
        
        Args:
            params: Validated input parameters (instance of InputSchema)
            
        Returns:
            ToolResult containing success status, data, or error message
            
        Note:
            This method should NOT handle exceptions internally (except for
            expected business logic errors). Let the ToolExecutor middleware
            handle unexpected exceptions for consistent error formatting.
        """
        pass
    
    def validate_input(self, raw_input: Dict[str, Any]) -> TInput:
        """
        Validate and parse raw input dictionary into InputSchema.
        
        Args:
            raw_input: Dictionary of input parameters
            
        Returns:
            Validated InputSchema instance
            
        Raises:
            pydantic.ValidationError: If validation fails
        """
        return self.InputSchema.model_validate(raw_input)
    
    def run(self, raw_input: Dict[str, Any]) -> ToolResult:
        """
        Convenience method: validate input and execute.
        
        For production use, prefer using ToolExecutor.run() which adds
        safety checks, logging, and error handling.
        
        Args:
            raw_input: Dictionary of input parameters
            
        Returns:
            ToolResult from execution
        """
        start_time = time.perf_counter()
        try:
            params = self.validate_input(raw_input)
            result = self.execute(params)
            result.execution_time_ms = (time.perf_counter() - start_time) * 1000
            return result
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.perf_counter() - start_time) * 1000
            )
    
    # === OpenAI Function Calling Schema Generation ===
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """
        Generate OpenAI-compatible function calling schema.
        
        Returns:
            Dictionary conforming to OpenAI's function calling format:
            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": { ... JSON Schema ... }
                }
            }
        """
        # Get JSON schema from Pydantic model
        json_schema = self.InputSchema.model_json_schema()
        
        # Clean up schema for OpenAI compatibility
        # Remove 'title' from root (OpenAI doesn't need it)
        json_schema.pop("title", None)
        
        # Ensure 'type' is present
        if "type" not in json_schema:
            json_schema["type"] = "object"
        
        # Handle $defs (Pydantic v2 nested models) - inline them
        if "$defs" in json_schema:
            json_schema = self._inline_refs(json_schema)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": json_schema
            }
        }
    
    def _inline_refs(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively inline $ref definitions for OpenAI compatibility.
        
        OpenAI function calling doesn't support JSON Schema $ref,
        so we need to inline all referenced definitions.
        """
        defs = schema.pop("$defs", {})
        
        def resolve_refs(obj: Any) -> Any:
            if isinstance(obj, dict):
                if "$ref" in obj:
                    ref_path = obj["$ref"]  # e.g., "#/$defs/MyModel"
                    ref_name = ref_path.split("/")[-1]
                    if ref_name in defs:
                        # Replace $ref with actual definition
                        resolved = defs[ref_name].copy()
                        resolved.pop("title", None)
                        return resolve_refs(resolved)
                    return obj
                return {k: resolve_refs(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_refs(item) for item in obj]
            return obj
        
        return resolve_refs(schema)
    
    @classmethod
    def to_openai_tools(cls, tools: list["BaseTool"]) -> list[Dict[str, Any]]:
        """
        Convert a list of tools to OpenAI tools parameter format.
        
        Args:
            tools: List of BaseTool instances
            
        Returns:
            List of OpenAI function schemas
        """
        return [tool.to_openai_schema() for tool in tools]
    
    # === Utility Methods ===
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, risk={self.risk_level.value})>"
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"


# === Example: Empty Input Schema for tools with no parameters ===

class EmptyInput(BaseModel):
    """Input schema for tools that require no parameters."""
    pass


# === Example Implementation (for reference, can be removed in production) ===

if __name__ == "__main__":
    # Demonstration of how to create a tool
    from datetime import datetime
    
    class GetTimeInput(BaseModel):
        """Input for GetTimeTool."""
        timezone: str = Field(
            default="Asia/Shanghai",
            description="IANA timezone name (e.g., 'Asia/Shanghai', 'UTC')"
        )
    
    class GetTimeTool(BaseTool[GetTimeInput]):
        name = "get_current_time"
        description = "获取当前时间，支持指定时区"
        risk_level = RiskLevel.SAFE
        InputSchema = GetTimeInput
        tags = ["utility", "time"]
        
        def execute(self, params: GetTimeInput) -> ToolResult:
            try:
                from zoneinfo import ZoneInfo
                tz = ZoneInfo(params.timezone)
                current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
                return ToolResult(success=True, data=current_time)
            except Exception as e:
                return ToolResult(success=False, error=f"时区无效: {params.timezone}")
    
    # Test the tool
    tool = GetTimeTool()
    print(f"Tool: {tool}")
    print(f"\nOpenAI Schema:\n{tool.to_openai_schema()}")
    
    # Execute
    result = tool.run({"timezone": "UTC"})
    print(f"\nExecution Result: {result}")
