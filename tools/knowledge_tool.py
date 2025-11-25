# Jarvis Cortex Protocol - Knowledge Tools
# tools/knowledge_tool.py

"""
KnowledgeQueryTool & KnowledgeIngestTool: RAG-based knowledge management.

Migrated from: ManagerAgent direct service calls (query_knowledge, learn intents)
Risk Levels:
- KnowledgeQueryTool: SAFE (read-only RAG query)
- KnowledgeIngestTool: DANGEROUS (modifies vector database, requires confirmation)
"""

from typing import Optional, List
from pydantic import BaseModel, Field

from core.tools.base import BaseTool, RiskLevel, ToolResult
from services.knowledge_service import KnowledgeService


# ============================================================================
# Knowledge Query Tool (SAFE)
# ============================================================================

class KnowledgeQueryInput(BaseModel):
    """Input schema for knowledge query."""
    query: str = Field(
        ...,
        description="要查询的问题或关键词",
        examples=["这个项目的架构是什么？", "ToolRegistry 怎么用？"]
    )
    n_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="返回的最大结果数量"
    )


class KnowledgeQueryTool(BaseTool[KnowledgeQueryInput]):
    """
    Query the local knowledge base using semantic search.
    
    Features:
    - RAG (Retrieval-Augmented Generation) query
    - Returns relevant document chunks
    - Uses sentence-transformers for embedding
    
    This tool is read-only and marked as SAFE.
    """
    
    name = "knowledge_query"
    description = "查询本地知识库 (RAG)。用于询问关于已学习文档的问题。"
    risk_level = RiskLevel.SAFE
    InputSchema = KnowledgeQueryInput
    tags = ["knowledge", "rag", "search", "query"]
    
    def __init__(self):
        super().__init__()
        self.knowledge_service = KnowledgeService()
    
    def execute(self, params: KnowledgeQueryInput) -> ToolResult:
        """Execute knowledge query."""
        try:
            # Check if knowledge base has content
            stats = self.knowledge_service.get_stats()
            if stats == 0:
                return ToolResult(
                    success=True,
                    data={
                        "query": params.query,
                        "results": [],
                        "message": "知识库为空，请先使用 knowledge_ingest 学习文件。"
                    },
                    metadata={"kb_size": 0}
                )
            
            # Perform query
            docs = self.knowledge_service.query_knowledge(
                params.query, 
                n_results=params.n_results
            )
            
            if not docs:
                return ToolResult(
                    success=True,
                    data={
                        "query": params.query,
                        "results": [],
                        "message": "未找到相关内容"
                    },
                    metadata={"kb_size": stats}
                )
            
            # Format results for LLM consumption
            formatted_results = "\n\n---\n\n".join(docs)
            
            return ToolResult(
                success=True,
                data={
                    "query": params.query,
                    "results": docs,
                    "raw_content": f"检索到的参考资料（请基于此回答）：\n{formatted_results}",
                    "count": len(docs)
                },
                metadata={"kb_size": stats}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"知识库查询失败: {str(e)}",
                metadata={"query": params.query}
            )


# ============================================================================
# Knowledge Ingest Tool (MODERATE)
# ============================================================================

class KnowledgeIngestInput(BaseModel):
    """Input schema for knowledge ingestion."""
    file_path: str = Field(
        ...,
        description="要学习的文件路径 (支持 .txt, .md, .py, .pdf 等)",
        examples=["./docs/README.md", "D:/projects/code.py"]
    )


class KnowledgeIngestTool(BaseTool[KnowledgeIngestInput]):
    """
    Ingest files into the local knowledge base.
    
    Features:
    - Supports multiple file types (text, PDF, code)
    - Automatic chunking and embedding
    - Deduplication via file hash
    
    This tool modifies the vector database and is marked as DANGEROUS
    to require user confirmation before execution.
    """
    
    name = "knowledge_ingest"
    description = "学习文件到知识库。将文档切片、向量化并存储，供后续 RAG 查询使用。"
    risk_level = RiskLevel.DANGEROUS  # Changed from MODERATE: requires user confirmation
    InputSchema = KnowledgeIngestInput
    tags = ["knowledge", "rag", "ingest", "learn"]
    
    def __init__(self):
        super().__init__()
        self.knowledge_service = KnowledgeService()
    
    def execute(self, params: KnowledgeIngestInput) -> ToolResult:
        """Execute file ingestion."""
        import os
        
        file_path = params.file_path.strip()
        
        # Path resolution: try relative to cwd if not absolute
        if not os.path.isabs(file_path):
            potential_path = os.path.join(os.getcwd(), file_path)
            if os.path.exists(potential_path):
                file_path = potential_path
        
        # Validate file exists
        if not os.path.exists(file_path):
            return ToolResult(
                success=False,
                error=f"文件不存在: {file_path}",
                metadata={"original_path": params.file_path}
            )
        
        # Validate it's a file (not directory)
        if os.path.isdir(file_path):
            return ToolResult(
                success=False,
                error=f"路径是目录而非文件: {file_path}。暂不支持批量学习目录。",
                metadata={"path": file_path}
            )
        
        try:
            # Perform ingestion
            result_message = self.knowledge_service.ingest_file(file_path)
            
            # Determine success based on result message
            is_success = "成功" in result_message or "跳过" in result_message
            
            return ToolResult(
                success=is_success,
                data={
                    "file_path": file_path,
                    "message": result_message,
                    "kb_size": self.knowledge_service.get_stats()
                } if is_success else None,
                error=result_message if not is_success else None,
                metadata={"file_name": os.path.basename(file_path)}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"文件学习失败: {str(e)}",
                metadata={"file_path": file_path}
            )
