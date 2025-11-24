from .base import BaseAgent
import re

# 引入必要的库
try:
    import comtypes
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume, ISimpleAudioVolume
    from ctypes import POINTER
    import screen_brightness_control as sbc
    import keyboard
except ImportError:
    print("[SystemAgent]: 缺少依赖库，请运行 pip install pycaw screen-brightness-control keyboard comtypes")

class SystemAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="SystemAgent")
        # 常用软件名称映射表 (中文 -> 进程关键词)
        self.app_map = {
            "抖音": "douyin",
            "网易云": "cloudmusic",
            "音乐": "cloudmusic",
            "浏览器": "msedge", # 或者 chrome
            "谷歌": "chrome",
            "edge": "msedge",
            "微信": "wechat",
            "qq": "qq"
        }

    def _init_com(self):
        """关键修复：初始化 COM 库，防止多线程下报错"""
        try:
            comtypes.CoInitialize()
        except:
            pass # 已经初始化过则忽略

    def _get_master_volume_ctrl(self):
        """获取主音量控制器"""
        self._init_com()
        devices = AudioUtilities.GetSpeakers()
        if not devices:
            raise RuntimeError("未找到音频输出设备")
        # 使用 getattr 安全获取 Activate 方法，避免静态分析或平台差异导致的属性访问错误
        activate = getattr(devices, "Activate", None)
        if not callable(activate):
            # 有些实现可能使用小写 'activate'
            activate = getattr(devices, "activate", None)
        if not callable(activate):
            raise RuntimeError("音频设备不支持 Activate 方法")
        interface = activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None)
        return comtypes.cast(interface, POINTER(IAudioEndpointVolume)) # type: ignore

    def _set_master_volume(self, val: int):
        """设置主音量 (0-100)"""
        try:
            volume = self._get_master_volume_ctrl()
            # Pycaw 需要分贝值或标量，这里用标量 (Scalar) 0.0 - 1.0
            scalar = max(0.0, min(1.0, val / 100.0))
            volume.SetMasterVolumeLevelScalar(scalar, None) # type: ignore
            return True, f"主音量已调整为 {val}%"
        except Exception as e:
            return False, f"调整主音量失败: {e}"

    def _set_app_volume(self, app_name_key: str, val: int):
        """设置特定 App 音量"""
        self._init_com()
        target_process = self.app_map.get(app_name_key, app_name_key).lower()
        found = False
        
        try:
            sessions = AudioUtilities.GetAllSessions()
            for session in sessions:
                if not session.Process: continue
                
                # 匹配进程名 (例如 "Douyin.exe")
                proc_name = session.Process.name().lower()
                if target_process in proc_name.replace(".exe", ""):
                    interface = session.SimpleAudioVolume
                    scalar = max(0.0, min(1.0, val / 100.0))
                    interface.SetMasterVolume(scalar, None)
                    found = True
            
            if found:
                return True, f"已将 {app_name_key} ({target_process}) 的音量设为 {val}%"
            else:
                return False, f"没找到正在播放声音的“{app_name_key}”应用哦（它必须在运行且发出过声音）"
        except Exception as e:
            return False, f"控制应用音量出错: {e}"

    def _adjust_volume_step(self, target: str, step: int):
        """增量调节 (+10 / -10)"""
        # 这里为了简化，先只做主音量增量。
        try:
            volume = self._get_master_volume_ctrl()
            current = volume.GetMasterVolumeLevelScalar() * 100 # type: ignore
            new_val = int(current + step)
            return self._set_master_volume(new_val)
        except Exception as e:
            return False, f"微调失败: {e}"

    def run(self, user_input: str) -> str:
        """解析指令并执行"""
        user_input = user_input.lower()
        
        # --- 1. 媒体控制 ---
        if "暂停" in user_input or "停止" in user_input:
            keyboard.send("play/pause")
            return "已暂停播放"
        if "播放" in user_input or "继续" in user_input:
            keyboard.send("play/pause")
            return "继续播放"
        if "下一首" in user_input:
            keyboard.send("next track")
            return "切歌啦"
        if "上一首" in user_input:
            keyboard.send("previous track")
            return "切回上一首"

        # --- 2. 亮度控制 ---
        if "亮度" in user_input or "屏幕" in user_input:
            try:
                # 简单提取数字
                nums = re.findall(r"\d+", user_input)
                if nums:
                    val = int(nums[0])
                    sbc.set_brightness(val)
                    return f"屏幕亮度已设为 {val}%"
            except Exception as e:
                return f"亮度调整失败: {e}"

        # --- 3. 音量控制 ---
        # 提取数值，默认 20 (如果用户只说"调低点")
        nums = re.findall(r"\d+", user_input)
        val = int(nums[0]) if nums else None
        
        # 判断是 "调低/小" 还是 "调高/大" 还是 "设为"
        step = 0
        if "低" in user_input or "小" in user_input:
            step = -20
        elif "高" in user_input or "大" in user_input:
            step = +20
            
        # 确定目标应用
        target_app = "master"
        for name in self.app_map:
            if name in user_input:
                target_app = name
                break
        
        # 执行
        if val is not None:
            # 用户说了具体数字 "设为 30"
            if target_app == "master":
                success, msg = self._set_master_volume(val)
            else:
                success, msg = self._set_app_volume(target_app, val)
            return msg
        elif step != 0:
            # 用户说了模糊指令 "调低点"
            # 如果是 App 模糊调节比较麻烦，这里简单处理：
            # 如果指定了 App 且没说数字，默认给它设为 50
            if target_app != "master":
                return f"调整 {target_app} 音量请直接告诉我具体数值哦，比如'{target_app}音量50'"
            
            success, msg = self._adjust_volume_step("master", step)
            return msg
            
        return "抱歉，我没听懂怎么控制系统，请说'音量50'或'暂停播放'。"
