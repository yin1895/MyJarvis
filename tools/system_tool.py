# Jarvis Cortex Protocol - System Control Tool
# tools/system_tool.py

"""
SystemTool: System-level operations including volume, brightness, media control, and app launching.

Migrated from: agents/system_agent.py + legacy_tools.open_app
Risk Level: SAFE (local system operations, user-initiated)
"""

import re
import subprocess
import webbrowser
from typing import Optional, Literal
from pydantic import BaseModel, Field

from core.tools.base import BaseTool, RiskLevel, ToolResult


class SystemInput(BaseModel):
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
        description="目标应用名称(用于应用音量控制)"
    )


class SystemTool(BaseTool[SystemInput]):
    """
    System control tool for volume, brightness, media, and app operations.
    
    Features:
    - Master and per-app volume control (via pycaw)
    - Screen brightness control (via screen-brightness-control)
    - Media playback control (via keyboard simulation)
    - App launching (common Windows applications)
    
    This tool performs local system operations and is marked as SAFE.
    Dependencies are lazy-loaded to avoid import overhead.
    """
    
    name = "system_tool"
    description = "系统控制：音量调节、屏幕亮度、媒体播放控制、打开应用程序。"
    risk_level = RiskLevel.SAFE
    InputSchema = SystemInput
    tags = ["system", "volume", "brightness", "media", "app"]
    
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
    
    def __init__(self):
        super().__init__()
        self._com_initialized = False
    
    def _init_com(self):
        """Initialize COM library for audio control."""
        if self._com_initialized:
            return
        try:
            import comtypes
            comtypes.CoInitialize()
            self._com_initialized = True
        except:
            pass
    
    def _get_master_volume_ctrl(self):
        """Get master volume controller."""
        self._init_com()
        import comtypes
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        from ctypes import POINTER
        
        devices = AudioUtilities.GetSpeakers()
        if not devices:
            raise RuntimeError("未找到音频输出设备")
        
        activate = getattr(devices, "Activate", None) or getattr(devices, "activate", None)
        if not callable(activate):
            raise RuntimeError("音频设备不支持 Activate 方法")
        
        interface = activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None)
        return comtypes.cast(interface, POINTER(IAudioEndpointVolume))  # type: ignore[arg-type]
    
    def _set_master_volume(self, val: int) -> ToolResult:
        """Set master volume (0-100)."""
        try:
            volume = self._get_master_volume_ctrl()
            scalar = max(0.0, min(1.0, val / 100.0))
            volume.SetMasterVolumeLevelScalar(scalar, None)  # type: ignore[attr-defined]
            return ToolResult(
                success=True,
                data={"action": "volume", "value": val, "message": f"主音量已调整为 {val}%"}
            )
        except Exception as e:
            return ToolResult(success=False, error=f"调整主音量失败: {e}")
    
    def _set_app_volume(self, app_name: str, val: int) -> ToolResult:
        """Set volume for a specific application."""
        self._init_com()
        from pycaw.pycaw import AudioUtilities
        
        target_process = self.APP_MAP.get(app_name.lower(), app_name).lower()
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
                return ToolResult(
                    success=True,
                    data={"action": "app_volume", "app": app_name, "value": val,
                          "message": f"已将 {app_name} 的音量设为 {val}%"}
                )
            else:
                return ToolResult(
                    success=False,
                    error=f"没找到正在运行的 {app_name} 应用（它必须在运行且发出过声音）"
                )
        except Exception as e:
            return ToolResult(success=False, error=f"控制应用音量出错: {e}")
    
    def _adjust_volume_step(self, step: int) -> ToolResult:
        """Adjust volume by step (+/- value)."""
        try:
            volume = self._get_master_volume_ctrl()
            current = volume.GetMasterVolumeLevelScalar() * 100  # type: ignore[attr-defined]
            new_val = int(max(0, min(100, current + step)))
            return self._set_master_volume(new_val)
        except Exception as e:
            return ToolResult(success=False, error=f"音量微调失败: {e}")
    
    def _set_brightness(self, val: int) -> ToolResult:
        """Set screen brightness (0-100)."""
        try:
            import screen_brightness_control as sbc
            sbc.set_brightness(val)
            return ToolResult(
                success=True,
                data={"action": "brightness", "value": val, "message": f"屏幕亮度已设为 {val}%"}
            )
        except Exception as e:
            return ToolResult(success=False, error=f"亮度调整失败: {e}")
    
    def _media_control(self, command: str) -> ToolResult:
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
                return ToolResult(success=False, error=f"未知的媒体命令: {command}")
            
            keyboard.send(key)
            
            action_desc = {
                "play": "继续播放",
                "pause": "已暂停",
                "stop": "已暂停",
                "next": "切换到下一首",
                "prev": "切换到上一首",
                "previous": "切换到上一首",
            }
            
            return ToolResult(
                success=True,
                data={"action": "media", "command": command, "message": action_desc.get(command.lower(), "已执行")}
            )
        except Exception as e:
            return ToolResult(success=False, error=f"媒体控制失败: {e}")
    
    def _open_app(self, app_name: str) -> ToolResult:
        """Open an application."""
        app_name_lower = app_name.lower()
        
        try:
            if "记事本" in app_name_lower or "notepad" in app_name_lower:
                subprocess.Popen("notepad.exe")
                return ToolResult(success=True, data={"app": "记事本", "message": "记事本打开啦"})
            
            elif "计算器" in app_name_lower or "calc" in app_name_lower:
                subprocess.Popen("calc.exe")
                return ToolResult(success=True, data={"app": "计算器", "message": "计算器来喽"})
            
            elif "浏览器" in app_name_lower or "browser" in app_name_lower:
                webbrowser.open("https://www.bilibili.com")
                return ToolResult(success=True, data={"app": "浏览器", "message": "浏览器打开了"})
            
            elif "cmd" in app_name_lower or "命令" in app_name_lower or "终端" in app_name_lower:
                subprocess.Popen("cmd.exe")
                return ToolResult(success=True, data={"app": "命令提示符", "message": "命令提示符已打开"})
            
            elif "powershell" in app_name_lower:
                subprocess.Popen("powershell.exe")
                return ToolResult(success=True, data={"app": "PowerShell", "message": "PowerShell已打开"})
            
            elif "资源管理器" in app_name_lower or "文件" in app_name_lower or "explorer" in app_name_lower:
                subprocess.Popen("explorer.exe")
                return ToolResult(success=True, data={"app": "资源管理器", "message": "资源管理器已打开"})
            
            else:
                # Try to open as URL if it looks like one
                if "." in app_name and ("http" in app_name or "www" in app_name or ".com" in app_name):
                    url = app_name if app_name.startswith("http") else f"https://{app_name}"
                    webbrowser.open(url)
                    return ToolResult(success=True, data={"app": "浏览器", "url": url, "message": f"已打开 {url}"})
                
                return ToolResult(
                    success=False,
                    error=f"抱歉，我还没学会怎么打开 {app_name}"
                )
        except Exception as e:
            return ToolResult(success=False, error=f"打开应用失败: {e}")
    
    def execute(self, params: SystemInput) -> ToolResult:
        """Execute system control operation."""
        action = params.action
        value = params.value or ""
        
        try:
            if action == "volume":
                # Parse volume value or step
                if not value:
                    return ToolResult(success=False, error="请指定音量值，如 '50' 或 '+10'")
                
                # Check if it's a step adjustment
                if value.startswith("+") or value.startswith("-"):
                    step = int(value)
                    return self._adjust_volume_step(step)
                
                # Check for target app
                if params.target:
                    return self._set_app_volume(params.target, int(value))
                
                # Master volume
                nums = re.findall(r"\d+", value)
                if nums:
                    return self._set_master_volume(int(nums[0]))
                
                return ToolResult(success=False, error="无法解析音量值")
            
            elif action == "brightness":
                if not value:
                    return ToolResult(success=False, error="请指定亮度值 (0-100)")
                
                nums = re.findall(r"\d+", value)
                if nums:
                    return self._set_brightness(int(nums[0]))
                
                return ToolResult(success=False, error="无法解析亮度值")
            
            elif action == "media_control":
                if not value:
                    return ToolResult(success=False, error="请指定媒体命令 (play/pause/next/prev)")
                return self._media_control(value)
            
            elif action == "open_app":
                if not value:
                    return ToolResult(success=False, error="请指定要打开的应用名称")
                return self._open_app(value)
            
            elif action == "system_info":
                import platform
                info = {
                    "os": platform.system(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor()
                }
                return ToolResult(
                    success=True,
                    data={"system_info": info, "message": f"系统: {info['os']} {info['version']}"}
                )
            
            else:
                return ToolResult(success=False, error=f"未知操作类型: {action}")
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"系统操作失败: {str(e)}",
                metadata={"action": action, "value": value}
            )
