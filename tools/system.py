# Jarvis V7.0 - Native System Control Tool
# tools/system.py

"""
Native LangChain Tool for system control operations.

Features:
- Volume control (master & per-app)
- Screen brightness control
- Media playback control
- Application launching
- System info

Risk Level: SAFE (local system operations, user-initiated)
"""

import re
import subprocess
import webbrowser
import platform
from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ============== Input Schema ==============

class SystemControlInput(BaseModel):
    """Input schema for system control operations."""
    action: Literal[
        "volume",
        "brightness", 
        "media_control",
        "open_app",
        "system_info"
    ] = Field(
        ...,
        description="操作类型: volume(音量), brightness(亮度), media_control(媒体控制), open_app(打开应用), system_info(系统信息)"
    )
    value: Optional[str] = Field(
        default=None,
        description="操作参数: 音量/亮度数值(0-100), 媒体指令(play/pause/next/prev), 应用名称等"
    )
    target: Optional[str] = Field(
        default=None,
        description="目标应用名称(用于应用音量控制，如'网易云'、'抖音')"
    )


# ============== Helper Functions ==============

# App name mapping (Chinese -> process keyword)
APP_MAP = {
    "抖音": "douyin",
    "网易云": "cloudmusic",
    "音乐": "cloudmusic",
    "浏览器": "msedge",
    "谷歌": "chrome",
    "edge": "msedge",
    "微信": "wechat",
    "qq": "qq"
}

_com_initialized = False


def _init_com():
    """Initialize COM library for audio control."""
    global _com_initialized
    if _com_initialized:
        return
    try:
        import comtypes
        comtypes.CoInitialize()
        _com_initialized = True
    except Exception:
        pass


def _get_master_volume_ctrl():
    """Get master volume controller."""
    _init_com()
    import comtypes
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from ctypes import POINTER, cast as ctypes_cast
    from typing import Any, cast as typing_cast
    
    devices = AudioUtilities.GetSpeakers()
    if not devices:
        raise RuntimeError("未找到音频输出设备")
    
    activate = getattr(devices, "Activate", None) or getattr(devices, "activate", None)
    if not callable(activate):
        raise RuntimeError("音频设备不支持 Activate 方法")
    
    interface = activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None)
    # Use comtypes.cast for COM interface - type checking disabled for dynamic COM interface
    volume_interface: Any = comtypes.cast(interface, POINTER(IAudioEndpointVolume))  # type: ignore[arg-type]
    return volume_interface


def _set_master_volume(val: int) -> str:
    """Set master volume (0-100)."""
    try:
        volume = _get_master_volume_ctrl()
        scalar = max(0.0, min(1.0, val / 100.0))
        volume.SetMasterVolumeLevelScalar(scalar, None)
        return f"主音量已调整为 {val}%"
    except Exception as e:
        return f"调整主音量失败: {e}"


def _set_app_volume(app_name: str, val: int) -> str:
    """Set volume for a specific application."""
    _init_com()
    from pycaw.pycaw import AudioUtilities
    
    target_process = APP_MAP.get(app_name.lower(), app_name).lower()
    found = False
    
    try:
        sessions = AudioUtilities.GetAllSessions()
        for session in sessions:
            if not session.Process:
                continue
            
            proc_name = session.Process.name().lower()
            if target_process in proc_name.replace(".exe", ""):
                interface = session.SimpleAudioVolume
                scalar = max(0.0, min(1.0, val / 100.0))
                interface.SetMasterVolume(scalar, None)
                found = True
        
        if found:
            return f"已将 {app_name} 的音量设为 {val}%"
        else:
            return f"没找到正在运行的 {app_name} 应用（它必须在运行且发出过声音）"
    except Exception as e:
        return f"控制应用音量出错: {e}"


def _adjust_volume_step(step: int) -> str:
    """Adjust volume by step (+/- value)."""
    try:
        volume = _get_master_volume_ctrl()
        current = volume.GetMasterVolumeLevelScalar() * 100
        new_val = int(max(0, min(100, current + step)))
        return _set_master_volume(new_val)
    except Exception as e:
        return f"音量微调失败: {e}"


def _set_brightness(val: int) -> str:
    """Set screen brightness (0-100)."""
    try:
        import screen_brightness_control as sbc
        sbc.set_brightness(val)
        return f"屏幕亮度已设为 {val}%"
    except Exception as e:
        return f"亮度调整失败: {e}"


def _media_control(command: str) -> str:
    """Control media playback."""
    try:
        import keyboard
        
        cmd_map = {
            "play": "play/pause",
            "pause": "play/pause",
            "stop": "play/pause",
            "next": "next track",
            "prev": "previous track",
            "previous": "previous track",
        }
        
        key = cmd_map.get(command.lower())
        if not key:
            return f"未知的媒体命令: {command}"
        
        keyboard.send(key)
        
        action_desc = {
            "play": "继续播放",
            "pause": "已暂停",
            "stop": "已暂停",
            "next": "切换到下一首",
            "prev": "切换到上一首",
            "previous": "切换到上一首",
        }
        
        return action_desc.get(command.lower(), "媒体控制已执行")
    except Exception as e:
        return f"媒体控制失败: {e}"


def _open_app(app_name: str) -> str:
    """Open an application."""
    app_name_lower = app_name.lower()
    
    try:
        if "记事本" in app_name_lower or "notepad" in app_name_lower:
            subprocess.Popen("notepad.exe")
            return "记事本已打开"
        
        elif "计算器" in app_name_lower or "calc" in app_name_lower:
            subprocess.Popen("calc.exe")
            return "计算器已打开"
        
        elif "浏览器" in app_name_lower or "browser" in app_name_lower:
            webbrowser.open("https://www.bilibili.com")
            return "浏览器已打开"
        
        elif "cmd" in app_name_lower or "命令" in app_name_lower or "终端" in app_name_lower:
            subprocess.Popen("cmd.exe")
            return "命令提示符已打开"
        
        elif "powershell" in app_name_lower:
            subprocess.Popen("powershell.exe")
            return "PowerShell已打开"
        
        elif "资源管理器" in app_name_lower or "文件" in app_name_lower or "explorer" in app_name_lower:
            subprocess.Popen("explorer.exe")
            return "资源管理器已打开"
        
        else:
            # Try to open as URL if it looks like one
            if "." in app_name and ("http" in app_name or "www" in app_name or ".com" in app_name):
                url = app_name if app_name.startswith("http") else f"https://{app_name}"
                webbrowser.open(url)
                return f"已打开 {url}"
            
            return f"抱歉，我还没学会怎么打开 {app_name}"
    except Exception as e:
        return f"打开应用失败: {e}"


def _get_system_info() -> str:
    """Get system information."""
    info = {
        "os": platform.system(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor()
    }
    return f"系统: {info['os']} {info['version']}, 架构: {info['machine']}"


# ============== Native Tool ==============

@tool(args_schema=SystemControlInput, return_direct=False)
def system_control(action: str, value: Optional[str] = None, target: Optional[str] = None) -> str:
    """
    系统控制工具：音量调节、屏幕亮度、媒体播放控制、打开应用程序。
    
    使用场景:
    - 调整音量: action="volume", value="50" 或 value="+10"
    - 调整应用音量: action="volume", value="30", target="网易云"
    - 调整亮度: action="brightness", value="70"
    - 媒体控制: action="media_control", value="play/pause/next/prev"
    - 打开应用: action="open_app", value="记事本/计算器/浏览器"
    - 系统信息: action="system_info"
    
    Args:
        action: 操作类型
        value: 操作参数值
        target: 目标应用(仅用于应用音量控制)
        
    Returns:
        操作结果描述
    """
    value = value or ""
    
    try:
        if action == "volume":
            if not value:
                return "请指定音量值，如 '50' 或 '+10'"
            
            # Check if it's a step adjustment
            if value.startswith("+") or value.startswith("-"):
                step = int(value)
                return _adjust_volume_step(step)
            
            # Check for target app
            if target:
                nums = re.findall(r"\d+", value)
                if nums:
                    return _set_app_volume(target, int(nums[0]))
                return "无法解析音量值"
            
            # Master volume
            nums = re.findall(r"\d+", value)
            if nums:
                return _set_master_volume(int(nums[0]))
            
            return "无法解析音量值"
        
        elif action == "brightness":
            if not value:
                return "请指定亮度值 (0-100)"
            
            nums = re.findall(r"\d+", value)
            if nums:
                return _set_brightness(int(nums[0]))
            
            return "无法解析亮度值"
        
        elif action == "media_control":
            if not value:
                return "请指定媒体命令 (play/pause/next/prev)"
            return _media_control(value)
        
        elif action == "open_app":
            if not value:
                return "请指定要打开的应用名称"
            return _open_app(value)
        
        elif action == "system_info":
            return _get_system_info()
        
        else:
            return f"未知操作类型: {action}"
        
    except Exception as e:
        return f"系统操作失败: {str(e)}"


# ============== Risk Level Metadata ==============
# Store risk level in tool's metadata dict
system_control.metadata = {"risk_level": "safe"}

# Helper to get risk level
def get_tool_risk_level(tool) -> str:
    """Get risk level from tool metadata."""
    if hasattr(tool, 'metadata') and isinstance(tool.metadata, dict):
        return tool.metadata.get('risk_level', 'safe')
    return 'safe'


# ============== Export ==============
__all__ = ["system_control", "SystemControlInput", "get_tool_risk_level"]
