# Jarvis Cortex Protocol - Layer 3: Dynamic Registry
# core/tools/registry.py

"""
ToolRegistry: Dynamic tool discovery, registration, and retrieval.

This layer provides:
1. Auto-discovery: Scan directories and auto-register BaseTool subclasses
2. Lookup: Get tools by name or intent mapping
3. Schema Export: Generate OpenAI tools array for function calling
4. (Future) Semantic Retrieval: Top-K tool selection via embeddings

Design Principles:
- Singleton pattern for global registry access
- Lazy loading support for performance
- Intent-to-tool mapping for ManagerAgent integration
"""

import importlib
import importlib.util
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Callable
from functools import lru_cache

from core.tools.base import BaseTool, RiskLevel

# Configure module logger
logger = logging.getLogger("jarvis.registry")


class ToolNotFoundError(Exception):
    """Raised when a requested tool is not found in the registry."""
    pass


class ToolRegistrationError(Exception):
    """Raised when tool registration fails."""
    pass


class ToolRegistry:
    """
    Dynamic registry for Jarvis tools.
    
    Features:
    - Auto-scan directories for BaseTool subclasses
    - Register tools by name with intent mappings
    - Export OpenAI-compatible function schemas
    - Query tools by name, intent, or tags
    - (Future) Semantic similarity search for tool selection
    
    Usage:
        registry = ToolRegistry()
        registry.scan("tools/")
        
        # Get tool by name
        shell_tool = registry.get("shell_execute")
        
        # Get tool by intent (for ManagerAgent compatibility)
        tool = registry.get_by_intent("shell")
        
        # Export for OpenAI function calling
        tools_schema = registry.to_openai_tools()
    """
    
    # Singleton instance
    _instance: Optional["ToolRegistry"] = None
    
    def __new__(cls) -> "ToolRegistry":
        """Singleton pattern - ensure only one registry exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the registry (only runs once due to singleton)."""
        if self._initialized:
            return
            
        # Tool storage: name -> tool instance
        self._tools: Dict[str, BaseTool] = {}
        
        # Intent mapping: intent -> tool name (for backward compatibility)
        self._intent_map: Dict[str, str] = {}
        
        # Tag index: tag -> set of tool names
        self._tag_index: Dict[str, set] = {}
        
        # Risk level index
        self._risk_index: Dict[RiskLevel, set] = {
            RiskLevel.SAFE: set(),
            RiskLevel.MODERATE: set(),
            RiskLevel.DANGEROUS: set(),
        }
        
        # Scan history (to prevent duplicate scans)
        self._scanned_paths: set = set()
        
        # Embedding cache for semantic search (future)
        self._embeddings: Dict[str, Any] = {}
        self._embedding_model: Optional[Any] = None
        
        self._initialized = True
        logger.info("ToolRegistry initialized (singleton)")
    
    # === Core Registration Methods ===
    
    def register(
        self, 
        tool: BaseTool, 
        intents: Optional[List[str]] = None,
        override: bool = False,
    ) -> None:
        """
        Register a tool instance.
        
        Args:
            tool: BaseTool instance to register
            intents: Optional list of intent strings that map to this tool
                     (for backward compatibility with ManagerAgent)
            override: If True, allow overwriting existing tool with same name
            
        Raises:
            ToolRegistrationError: If tool with same name exists and override=False
        """
        name = tool.name
        
        if name in self._tools and not override:
            raise ToolRegistrationError(
                f"Tool '{name}' is already registered. Use override=True to replace."
            )
        
        # Register tool
        self._tools[name] = tool
        
        # Update risk index
        self._risk_index[tool.risk_level].add(name)
        
        # Update tag index
        for tag in getattr(tool, 'tags', []):
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(name)
        
        # Register intent mappings
        if intents:
            for intent in intents:
                self._intent_map[intent] = name
                logger.debug(f"Intent '{intent}' -> tool '{name}'")
        
        logger.info(f"Registered tool: {name} (risk={tool.risk_level.value})")
    
    def register_class(
        self, 
        tool_class: Type[BaseTool],
        intents: Optional[List[str]] = None,
        override: bool = False,
    ) -> BaseTool:
        """
        Register a tool class (instantiates it automatically).
        
        Args:
            tool_class: BaseTool subclass to instantiate and register
            intents: Optional intent mappings
            override: Allow overwriting existing tools
            
        Returns:
            The instantiated tool
        """
        tool = tool_class()
        self.register(tool, intents=intents, override=override)
        return tool
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a tool by name.
        
        Args:
            name: Tool name to unregister
            
        Returns:
            True if tool was removed, False if not found
        """
        if name not in self._tools:
            return False
        
        tool = self._tools.pop(name)
        
        # Clean up indexes
        self._risk_index[tool.risk_level].discard(name)
        for tag_set in self._tag_index.values():
            tag_set.discard(name)
        
        # Clean up intent mappings
        intents_to_remove = [k for k, v in self._intent_map.items() if v == name]
        for intent in intents_to_remove:
            del self._intent_map[intent]
        
        logger.info(f"Unregistered tool: {name}")
        return True
    
    # === Discovery & Auto-Registration ===
    
    def scan(
        self, 
        directory: str,
        recursive: bool = True,
        intent_extractor: Optional[Callable[[BaseTool], List[str]]] = None,
    ) -> List[str]:
        """
        Scan a directory for BaseTool subclasses and register them.
        
        Args:
            directory: Path to directory containing tool modules
            recursive: If True, scan subdirectories
            intent_extractor: Optional function to extract intents from tool
                              Default: Uses tool.name as the intent
        
        Returns:
            List of registered tool names
            
        Example:
            # Scan tools/ directory
            registered = registry.scan("tools/")
            print(f"Registered {len(registered)} tools")
        """
        dir_path = Path(directory).resolve()
        
        if not dir_path.exists():
            logger.warning(f"Scan directory does not exist: {dir_path}")
            return []
        
        if str(dir_path) in self._scanned_paths:
            logger.debug(f"Directory already scanned: {dir_path}")
            return []
        
        self._scanned_paths.add(str(dir_path))
        registered_tools = []
        
        # Find all Python files
        pattern = "**/*.py" if recursive else "*.py"
        for py_file in dir_path.glob(pattern):
            if py_file.name.startswith("_"):
                continue  # Skip __init__.py, __pycache__, etc.
            
            try:
                tools = self._load_tools_from_file(py_file)
                for tool in tools:
                    # Extract intents
                    if intent_extractor:
                        intents = intent_extractor(tool)
                    else:
                        # Default: use tool name as intent
                        intents = [tool.name]
                    
                    self.register(tool, intents=intents)
                    registered_tools.append(tool.name)
                    
            except Exception as e:
                logger.error(f"Failed to load tools from {py_file}: {e}")
        
        logger.info(f"Scanned {dir_path}: registered {len(registered_tools)} tools")
        return registered_tools
    
    def _load_tools_from_file(self, file_path: Path) -> List[BaseTool]:
        """
        Load all BaseTool subclasses from a Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of instantiated BaseTool objects
        """
        tools = []
        
        # Create module name from file path
        module_name = f"tools.{file_path.stem}"
        
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module spec from {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise ImportError(f"Failed to execute module {file_path}: {e}")
        
        # Find all BaseTool subclasses
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseTool) 
                and obj is not BaseTool
                and not inspect.isabstract(obj)
                and obj.__module__ == module.__name__
            ):
                try:
                    tool = obj()
                    tools.append(tool)
                    logger.debug(f"Loaded tool class: {name} -> {tool.name}")
                except Exception as e:
                    logger.warning(f"Failed to instantiate {name}: {e}")
        
        return tools
    
    # === Lookup Methods ===
    
    def get(self, name: str) -> BaseTool:
        """
        Get a tool by its name.
        
        Args:
            name: Tool name
            
        Returns:
            BaseTool instance
            
        Raises:
            ToolNotFoundError: If tool not found
        """
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' not found in registry")
        return self._tools[name]
    
    def get_by_intent(self, intent: str) -> Optional[BaseTool]:
        """
        Get a tool by intent mapping (for ManagerAgent compatibility).
        
        Args:
            intent: Intent string from _identify_intent()
            
        Returns:
            BaseTool instance, or None if no mapping exists
        """
        tool_name = self._intent_map.get(intent)
        if tool_name:
            return self._tools.get(tool_name)
        
        # Fallback: try direct name match
        if intent in self._tools:
            return self._tools[intent]
        
        return None
    
    def get_by_tag(self, tag: str) -> List[BaseTool]:
        """Get all tools with a specific tag."""
        names = self._tag_index.get(tag, set())
        return [self._tools[name] for name in names if name in self._tools]
    
    def get_by_risk(self, risk_level: RiskLevel) -> List[BaseTool]:
        """Get all tools with a specific risk level."""
        names = self._risk_index.get(risk_level, set())
        return [self._tools[name] for name in names if name in self._tools]
    
    def get_safe_tools(self) -> List[BaseTool]:
        """Get all tools that don't require confirmation."""
        return self.get_by_risk(RiskLevel.SAFE)
    
    def exists(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def list_tools(self) -> List[str]:
        """Get list of all registered tool names."""
        return list(self._tools.keys())
    
    def list_intents(self) -> Dict[str, str]:
        """Get intent to tool name mapping."""
        return self._intent_map.copy()
    
    # === Schema Export ===
    
    def to_openai_tools(
        self, 
        filter_tags: Optional[List[str]] = None,
        exclude_dangerous: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Export all tools as OpenAI function calling schemas.
        
        Args:
            filter_tags: If provided, only include tools with these tags
            exclude_dangerous: If True, exclude DANGEROUS risk level tools
            
        Returns:
            List of OpenAI function schemas
        """
        schemas = []
        
        for tool in self._tools.values():
            # Apply filters
            if exclude_dangerous and tool.risk_level == RiskLevel.DANGEROUS:
                continue
            
            if filter_tags:
                tool_tags = getattr(tool, 'tags', [])
                if not any(tag in tool_tags for tag in filter_tags):
                    continue
            
            schemas.append(tool.to_openai_schema())
        
        return schemas
    
    def get_tools_description(self) -> str:
        """
        Generate a human-readable description of all tools.
        
        Useful for including in system prompts.
        """
        lines = ["可用工具列表："]
        for name, tool in self._tools.items():
            risk_emoji = {
                RiskLevel.SAFE: "✅",
                RiskLevel.MODERATE: "⚠️",
                RiskLevel.DANGEROUS: "🔴"
            }.get(tool.risk_level, "")
            lines.append(f"  {risk_emoji} {name}: {tool.description}")
        return "\n".join(lines)
    
    # === Semantic Retrieval (Future-Proofing) ===
    
    def init_semantic_search(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """
        Initialize semantic search capability.
        
        Args:
            model_name: SentenceTransformer model name
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            self._embedding_model = SentenceTransformer(model_name)
            
            # Pre-compute embeddings for all tools
            for name, tool in self._tools.items():
                text = f"{tool.name} {tool.description}"
                embedding = self._embedding_model.encode(text)
                self._embeddings[name] = embedding
            
            logger.info(f"Semantic search initialized with {model_name}")
            
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Semantic search disabled. Run: pip install sentence-transformers"
            )
    
    def get_tools_by_query(
        self, 
        query: str, 
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> List[BaseTool]:
        """
        Retrieve top-K tools by semantic similarity to query.
        
        Args:
            query: Natural language query
            top_k: Maximum number of tools to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of most relevant tools
        """
        if self._embedding_model is None:
            logger.warning("Semantic search not initialized, returning all tools")
            return list(self._tools.values())[:top_k]
        
        import numpy as np
        
        # Encode query
        query_embedding = self._embedding_model.encode(query)
        
        # Calculate similarities
        similarities = []
        for name, embedding in self._embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            if similarity >= threshold:
                similarities.append((name, similarity))
        
        # Sort by similarity and get top-K
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_names = [name for name, _ in similarities[:top_k]]
        
        return [self._tools[name] for name in top_names]
    
    # === Statistics ===
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_tools": len(self._tools),
            "by_risk": {
                level.value: len(names) 
                for level, names in self._risk_index.items()
            },
            "intents_mapped": len(self._intent_map),
            "tags": list(self._tag_index.keys()),
            "scanned_paths": list(self._scanned_paths),
            "semantic_search_enabled": self._embedding_model is not None,
        }
    
    # === Utility ===
    
    def clear(self) -> None:
        """Clear all registered tools (useful for testing)."""
        self._tools.clear()
        self._intent_map.clear()
        self._tag_index.clear()
        for level in RiskLevel:
            self._risk_index[level].clear()
        self._scanned_paths.clear()
        self._embeddings.clear()
        logger.info("Registry cleared")
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
    
    def __iter__(self):
        return iter(self._tools.values())


# === Global Registry Access ===

def get_registry() -> ToolRegistry:
    """Get the global ToolRegistry singleton."""
    return ToolRegistry()


# === Decorator for Easy Registration ===

def register_tool(
    intents: Optional[List[str]] = None,
    override: bool = False,
) -> Callable[[Type[BaseTool]], Type[BaseTool]]:
    """
    Decorator to register a tool class on definition.
    
    Example:
        @register_tool(intents=["shell", "execute_command"])
        class ShellTool(BaseTool[ShellInput]):
            name = "shell_execute"
            ...
    """
    def decorator(cls: Type[BaseTool]) -> Type[BaseTool]:
        registry = get_registry()
        registry.register_class(cls, intents=intents, override=override)
        return cls
    return decorator


# === Example Usage ===

if __name__ == "__main__":
    import json
    from pydantic import BaseModel, Field
    from core.tools.base import ToolResult
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    
    # Define sample tools
    class PingInput(BaseModel):
        host: str = Field(..., description="要 ping 的主机地址")
        count: int = Field(default=4, description="ping 次数")
    
    class PingTool(BaseTool[PingInput]):
        name = "ping"
        description = "Ping 一个主机地址检查网络连通性"
        risk_level = RiskLevel.SAFE
        InputSchema = PingInput
        tags = ["network", "utility"]
        
        def execute(self, params: PingInput) -> ToolResult:
            return ToolResult(success=True, data=f"Pinging {params.host}...")
    
    class RebootInput(BaseModel):
        delay: int = Field(default=0, description="延迟秒数")
    
    class RebootTool(BaseTool[RebootInput]):
        name = "system_reboot"
        description = "重启系统（危险操作）"
        risk_level = RiskLevel.DANGEROUS
        InputSchema = RebootInput
        tags = ["system", "power"]
        
        def execute(self, params: RebootInput) -> ToolResult:
            return ToolResult(success=True, data="Rebooting...")
    
    # Test registry
    registry = get_registry()
    registry.clear()  # Clear for clean test
    
    # Register tools
    registry.register_class(PingTool, intents=["ping", "network_check"])
    registry.register_class(RebootTool, intents=["reboot", "restart"])
    
    print("\n=== Registry Stats ===")
    print(json.dumps(registry.stats, indent=2))
    
    print("\n=== Get by Intent ===")
    tool = registry.get_by_intent("ping")
    print(f"Intent 'ping' -> {tool}")
    
    print("\n=== Get by Tag ===")
    network_tools = registry.get_by_tag("network")
    print(f"Tag 'network' -> {[t.name for t in network_tools]}")
    
    print("\n=== OpenAI Tools Schema ===")
    schemas = registry.to_openai_tools()
    print(json.dumps(schemas, indent=2, ensure_ascii=False))
    
    print("\n=== Safe Tools Only ===")
    safe = registry.get_safe_tools()
    print(f"Safe tools: {[t.name for t in safe]}")
    
    print("\n=== Tools Description ===")
    print(registry.get_tools_description())
