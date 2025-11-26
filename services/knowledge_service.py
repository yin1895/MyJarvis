import os
import hashlib
import logging
import threading
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import chromadb
    from chromadb.api.models.Collection import Collection

# 配置日志
logger = logging.getLogger(__name__)


class KnowledgeService:
    """
    知识库管理服务 (Singleton with Lazy Loading)
    
    特性:
    1. 单例模式：整个应用只有一个实例
    2. 懒加载：ChromaDB 和 Embedding 模型在首次使用时才加载
    3. 线程安全：所有初始化操作都有锁保护
    
    好处: Jarvis 秒级启动，1GB+ 模型只在需要时加载
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # 线程安全的初始化检查
        with self._lock:
            if self._initialized:
                return
            
            self.persist_directory = os.path.join("data", "vector_db")
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # 懒加载占位符 - 不在初始化时加载重型依赖
            self._client: Optional[Any] = None
            self._collection: Optional[Any] = None
            self._model: Optional[Any] = None
            
            self._initialized = True
            logger.debug("KnowledgeService initialized (lazy mode)")
    
    @property
    def client(self):
        """Lazy load ChromaDB client on first access."""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    import chromadb
                    logger.info("Loading ChromaDB client...")
                    self._client = chromadb.PersistentClient(path=self.persist_directory)
                    logger.info("ChromaDB client loaded.")
        return self._client
    
    @property
    def collection(self) -> "Collection":
        """Lazy load ChromaDB collection on first access."""
        if self._collection is None:
            with self._lock:
                if self._collection is None:
                    self._collection = self.client.get_or_create_collection(name="jarvis_knowledge")
        return self._collection  # type: ignore[return-value]
    
    @property
    def model(self):
        """Lazy load SentenceTransformer model on first access."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info("Loading embedding model (all-MiniLM-L6-v2)... This may take a moment.")
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer("all-MiniLM-L6-v2")
                    logger.info("Embedding model loaded.")
        return self._model

    def _calculate_hash(self, file_path):
        """计算文件的 MD5"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except FileNotFoundError:
            return None

    def _read_file_content(self, file_path):
        """读取文件内容 (支持 PDF 和 文本)"""
        ext = os.path.splitext(file_path)[1].lower()
        content = ""
        
        try:
            if ext == ".pdf":
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                for page in reader.pages:
                    extract = page.extract_text()
                    if extract:
                        content += extract + "\n"
            else:
                # 优先尝试 UTF-8，失败则回退到 GBK (适配 Windows 中文文档)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    with open(file_path, "r", encoding="gbk") as f:
                        content = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
            
        return content

    def ingest_file(self, file_path: str):
        """处理并存入文件 (带去重机制)"""
        if not os.path.exists(file_path):
            return f"文件不存在: {file_path}"

        # 1. 计算 Hash
        file_hash = self._calculate_hash(file_path)
        if not file_hash:
            return "Hash计算失败"
        
        # 2. 【性能优化】检查是否已存在
        # 只查询 metadata，不返回 documents 和 embeddings，极大提高速度
        existing = self.collection.get(
            where={"source": file_path},
            limit=1,
            include=["metadatas"] 
        )
        
        if existing and existing['metadatas']:
            stored_hash = existing['metadatas'][0].get('hash')
            if stored_hash == file_hash:
                logger.info(f"File skipped (unchanged): {file_path}")
                return "文件未变更，已跳过。"
            else:
                logger.info(f"File changed, re-indexing: {file_path}")
                # 只有内容变了才删除旧的
                self.collection.delete(where={"source": file_path})

        # 3. 读取内容
        text = self._read_file_content(file_path)
        if not text or len(text.strip()) == 0:
            return "无法读取文件内容或内容为空"

        # 4. 切片 (Chunking) - 简单滑动窗口
        chunk_size = 500
        overlap = 50
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 10: # 忽略太短的碎片
                chunks.append(chunk)
        
        if not chunks:
            return "文件有效内容过少，未生成切片。"

        # 5. Embedding
        # 这一步最耗时，打印日志提示用户
        logger.info(f"Embedding {len(chunks)} chunks for {file_path}...")
        embeddings = self.model.encode(chunks).tolist()

        # 6. 存储
        # ID 格式: hash_索引
        ids = [f"{file_hash}_{i}" for i in range(len(chunks))]

        metadatas = [{"source": file_path, "hash": file_hash} for _ in chunks]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas  # type: ignore[arg-type]
        )
        
        return f"成功学习文件：{os.path.basename(file_path)}，共 {len(chunks)} 个知识片段。"

    def query_knowledge(self, query_text, n_results=3):
        """查询知识库"""
        if self.collection.count() == 0:
            return []

        query_embedding = self.model.encode([query_text]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        # results['documents'] 是 [[doc1, doc2...]]
        if results and results['documents']:
            return results['documents'][0]
        return []

    def get_stats(self):
        """获取当前知识库状态"""
        return self.collection.count()