# Jarvis V7.0 - Native File Operation Tool
# tools/file.py

"""
Native LangChain Tool for file operations.

Features:
- Read file content
- Write/create files
- List directory contents
- Delete files

Risk Level: 
- "safe" for read/list operations
- "dangerous" for write/delete operations
"""

import os
import shutil
from pathlib import Path
from typing import Literal, Optional, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ============== Input Schema ==============

class FileOperationInput(BaseModel):
    """Input schema for file operations."""
    action: Literal["read", "write", "list", "delete", "exists", "info"] = Field(
        ...,
        description="操作类型: read(读取), write(写入), list(列目录), delete(删除), exists(检查存在), info(文件信息)"
    )
    path: str = Field(
        ...,
        description="文件或目录的路径（绝对路径或相对于workspace的路径）"
    )
    content: Optional[str] = Field(
        default=None,
        description="写入的内容（仅用于 write 操作）"
    )
    encoding: str = Field(
        default="utf-8",
        description="文件编码（默认 utf-8）"
    )


# ============== Constants ==============

# Default workspace directory
WORKSPACE_DIR = os.path.join(os.getcwd(), "workspace")

# Forbidden paths (security) - patterns are case-insensitive
FORBIDDEN_PATTERNS = [
    # Windows system paths
    "\\windows\\",
    "\\program files",
    "\\programdata",
    "\\users\\administrator",
    "\\system32",
    "\\syswow64",
    # Linux system paths
    "/etc/",
    "/usr/",
    "/bin/",
    "/sbin/",
    "/var/",
    "/root/",
    "/proc/",
    "/sys/",
    # Sensitive files
    ".ssh",
    ".git/config",
    ".gitconfig",
    ".env",
    ".aws",
    ".azure",
    "id_rsa",
    "id_ed25519",
    "credentials",
]


# ============== Helper Functions ==============

def _resolve_path(path: str) -> Path:
    """
    Resolve path - if relative, resolve relative to workspace.
    """
    p = Path(path)
    if not p.is_absolute():
        p = Path(WORKSPACE_DIR) / path
    return p.resolve()


def _is_safe_path(path: Path, require_workspace: bool = False) -> tuple[bool, str]:
    """
    Check if path is safe to operate on.
    
    Uses path.resolve() to normalize and prevent path traversal attacks.
    
    Args:
        path: The path to check (should already be resolved)
        require_workspace: If True, path must be within WORKSPACE_DIR
        
    Returns:
        Tuple of (is_safe: bool, reason: str)
    """
    try:
        # Resolve to absolute path (handles ../ and symlinks)
        resolved = path.resolve()
        path_str = str(resolved).lower()
        
        # Check workspace constraint first
        workspace = Path(WORKSPACE_DIR).resolve()
        is_in_workspace = False
        try:
            resolved.relative_to(workspace)
            is_in_workspace = True
        except ValueError:
            pass
        
        if require_workspace and not is_in_workspace:
            return False, f"安全拦截：只能操作 workspace 目录内的文件 - {resolved}"
        
        # Allow anything in workspace (safest zone)
        if is_in_workspace:
            return True, ""
        
        # Check forbidden patterns for paths outside workspace
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.lower() in path_str:
                return False, f"安全拦截：禁止访问此路径 - {resolved}"
        
        return True, ""
        
    except (OSError, ValueError) as e:
        return False, f"路径验证失败: {e}"


def _read_file(path: Path, encoding: str) -> str:
    """Read file content."""
    if not path.exists():
        return f"错误：文件不存在 - {path}"
    
    if not path.is_file():
        return f"错误：路径不是文件 - {path}"
    
    is_safe, reason = _is_safe_path(path, require_workspace=False)
    if not is_safe:
        return reason
    
    try:
        # Check file size (limit to 100KB for safety)
        size = path.stat().st_size
        if size > 100 * 1024:
            return f"错误：文件过大 ({size / 1024:.1f} KB)，最大支持 100KB"
        
        content = path.read_text(encoding=encoding)
        
        # Truncate if too long
        max_length = 10000
        if len(content) > max_length:
            content = content[:max_length] + f"\n\n...[内容已截断，总长度 {len(content)} 字符]"
        
        return f"文件内容 ({path.name}):\n```\n{content}\n```"
    except UnicodeDecodeError:
        return f"错误：无法以 {encoding} 编码读取文件（可能是二进制文件）"
    except Exception as e:
        return f"读取文件失败: {e}"


def _write_file(path: Path, content: str, encoding: str) -> str:
    """Write content to file."""
    # Write operations require path to be in workspace for extra safety
    is_safe, reason = _is_safe_path(path, require_workspace=True)
    if not is_safe:
        return reason
    
    try:
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content
        path.write_text(content, encoding=encoding)
        
        return f"文件已保存: {path} ({len(content)} 字符)"
    except Exception as e:
        return f"写入文件失败: {e}"


def _list_directory(path: Path) -> str:
    """List directory contents."""
    if not path.exists():
        return f"错误：目录不存在 - {path}"
    
    if not path.is_dir():
        return f"错误：路径不是目录 - {path}"
    
    is_safe, reason = _is_safe_path(path, require_workspace=False)
    if not is_safe:
        return reason
    
    try:
        items: List[str] = []
        for item in sorted(path.iterdir()):
            if item.is_dir():
                items.append(f"📁 {item.name}/")
            else:
                size = item.stat().st_size
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f} MB"
                items.append(f"📄 {item.name} ({size_str})")
        
        if not items:
            return f"目录为空: {path}"
        
        return f"目录内容 ({path}):\n" + "\n".join(items)
    except Exception as e:
        return f"列出目录失败: {e}"


def _delete_file(path: Path) -> str:
    """Delete file or directory."""
    if not path.exists():
        return f"错误：路径不存在 - {path}"
    
    # Delete operations require path to be in workspace for maximum safety
    is_safe, reason = _is_safe_path(path, require_workspace=True)
    if not is_safe:
        return reason
    
    try:
        if path.is_file():
            path.unlink()
            return f"文件已删除: {path}"
        elif path.is_dir():
            shutil.rmtree(path)
            return f"目录已删除: {path}"
        else:
            return f"未知的路径类型: {path}"
    except Exception as e:
        return f"删除失败: {e}"


def _check_exists(path: Path) -> str:
    """Check if path exists."""
    if path.exists():
        if path.is_file():
            return f"存在：文件 - {path}"
        elif path.is_dir():
            return f"存在：目录 - {path}"
        else:
            return f"存在：其他类型 - {path}"
    else:
        return f"不存在: {path}"


def _get_file_info(path: Path) -> str:
    """Get file/directory information."""
    if not path.exists():
        return f"错误：路径不存在 - {path}"
    
    try:
        stat = path.stat()
        info_lines = [
            f"路径: {path}",
            f"类型: {'文件' if path.is_file() else '目录'}",
            f"大小: {stat.st_size} 字节",
        ]
        
        # Format timestamps
        from datetime import datetime
        mtime = datetime.fromtimestamp(stat.st_mtime)
        ctime = datetime.fromtimestamp(stat.st_ctime)
        info_lines.append(f"修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        info_lines.append(f"创建时间: {ctime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if path.is_file():
            info_lines.append(f"后缀: {path.suffix or '(无)'}")
        
        return "\n".join(info_lines)
    except Exception as e:
        return f"获取文件信息失败: {e}"


# ============== Native Tool ==============

@tool(args_schema=FileOperationInput)
def file_operation(
    action: str, 
    path: str, 
    content: Optional[str] = None, 
    encoding: str = "utf-8"
) -> str:
    """
    文件操作工具：读取、写入、列目录、删除、检查文件。
    
    使用场景:
    - 读取文件: action="read", path="file.txt"
    - 写入文件: action="write", path="file.txt", content="内容"
    - 列目录: action="list", path="./folder"
    - 删除文件: action="delete", path="file.txt"
    - 检查存在: action="exists", path="file.txt"
    - 文件信息: action="info", path="file.txt"
    
    注意：相对路径将相对于 workspace/ 目录解析。
    
    Args:
        action: 操作类型 (read/write/list/delete/exists/info)
        path: 文件或目录路径
        content: 写入内容（仅 write 操作需要）
        encoding: 文件编码（默认 utf-8）
        
    Returns:
        操作结果描述
    """
    resolved_path = _resolve_path(path)
    
    if action == "read":
        return _read_file(resolved_path, encoding)
    
    elif action == "write":
        if content is None:
            return "错误：write 操作需要提供 content 参数"
        return _write_file(resolved_path, content, encoding)
    
    elif action == "list":
        return _list_directory(resolved_path)
    
    elif action == "delete":
        return _delete_file(resolved_path)
    
    elif action == "exists":
        return _check_exists(resolved_path)
    
    elif action == "info":
        return _get_file_info(resolved_path)
    
    else:
        return f"未知操作类型: {action}"


# ============== Risk Level Metadata ==============
# Note: This tool handles both safe (read/list) and dangerous (write/delete) operations
# The graph should check the 'action' parameter to determine actual risk
file_operation.metadata = {"risk_level": "dangerous"}


# ============== Export ==============
__all__ = ["file_operation", "FileOperationInput", "WORKSPACE_DIR"]
