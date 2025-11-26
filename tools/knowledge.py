"""
Native Knowledge Tools for Jarvis V7.0

Wraps KnowledgeService as LangChain native tools using @tool decorator.
Provides RAG query and file ingestion capabilities.

Lazy Loading: Heavy dependencies (torch, transformers) are only loaded
when the tool is first invoked, ensuring fast startup.

Risk Levels:
- knowledge_query: SAFE (read-only)
- knowledge_ingest: DANGEROUS (modifies vector database)
"""

from __future__ import annotations

import logging
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Lazy-loaded singleton reference
_knowledge_service = None


def _get_knowledge_service():
    """
    Lazy load KnowledgeService on first use.
    
    This delays loading of heavy dependencies (torch, sentence-transformers, chromadb)
    until the tool is actually invoked, ensuring Jarvis starts in seconds.
    """
    global _knowledge_service
    if _knowledge_service is None:
        logger.info("Lazy loading KnowledgeService (first use)...")
        from services.knowledge_service import KnowledgeService
        _knowledge_service = KnowledgeService()
        logger.info("KnowledgeService loaded successfully.")
    return _knowledge_service


# ==================== Knowledge Query Tool ====================

class KnowledgeQueryInput(BaseModel):
    """Input schema for knowledge query."""
    
    query: str = Field(
        ...,
        description="要查询的问题或关键词"
    )
    n_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="返回结果数量 (1-10)"
    )


@tool(args_schema=KnowledgeQueryInput, return_direct=False)
def knowledge_query(query: str, n_results: int = 3) -> str:
    """
    查询本地知识库 (RAG)。
    
    使用场景:
    - 当用户询问之前学习过的文档内容时
    - 当用户问 "根据我的资料..." 或 "我之前告诉过你..."
    - 当用户想要检索保存的知识时
    
    注意: 如果知识库为空，会返回空结果。建议先用 knowledge_ingest 学习文件。
    
    Examples:
    - {"query": "项目的技术架构是什么", "n_results": 3}
    - {"query": "合同中的付款条款"}
    """
    try:
        service = _get_knowledge_service()
        
        # Check if knowledge base is empty
        doc_count = service.get_stats()
        if doc_count == 0:
            return "知识库为空。请先使用 knowledge_ingest 工具学习文件。"
        
        # Query
        results = service.query_knowledge(query, n_results)
        
        if not results:
            return f"未找到与 '{query}' 相关的内容。"
        
        # Format results
        formatted = f"找到 {len(results)} 条相关内容:\n\n"
        for i, doc in enumerate(results, 1):
            # Truncate long documents for display
            preview = doc[:500] + "..." if len(doc) > 500 else doc
            formatted += f"【{i}】{preview}\n\n"
        
        return formatted.strip()
        
    except Exception as e:
        logger.exception("Knowledge query failed")
        return f"知识库查询失败: {str(e)}"


# Set risk level
knowledge_query.metadata = {"risk_level": "safe"}


# ==================== Knowledge Ingest Tool ====================

class KnowledgeIngestInput(BaseModel):
    """Input schema for knowledge ingestion."""
    
    file_path: str = Field(
        ...,
        description="要学习的文件路径 (支持 .txt, .md, .pdf)"
    )


@tool(args_schema=KnowledgeIngestInput, return_direct=False)
def knowledge_ingest(file_path: str) -> str:
    """
    学习文件到知识库。
    
    使用场景:
    - 当用户说 "学习这个文件" 或 "记住这份文档"
    - 当用户提供文件路径并希望 Jarvis 记住内容
    
    支持格式: .txt, .md, .pdf
    
    注意: 这是一个危险操作，会修改本地向量数据库。
    相同文件只会被索引一次（基于 MD5 去重）。
    
    Examples:
    - {"file_path": "D:/Documents/project_spec.pdf"}
    - {"file_path": "./notes/meeting.txt"}
    """
    try:
        service = _get_knowledge_service()
        result = service.ingest_file(file_path)
        return result
        
    except Exception as e:
        logger.exception("Knowledge ingest failed")
        return f"文件学习失败: {str(e)}"


# Set risk level - modifies database
knowledge_ingest.metadata = {"risk_level": "dangerous"}
