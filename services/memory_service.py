import json
import os
import logging
import threading

logger = logging.getLogger(__name__)


class MemoryService:
    """
    用户档案与备忘录管理服务 (Singleton)
    
    使用单例模式确保整个应用中只有一个实例，
    避免多个组件（ManagerAgent, MemoryTool）持有不同实例导致数据不一致。
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
            
            self.data_dir = "data"
            self.file_path = os.path.join(self.data_dir, "user_profile.json")
            self._ensure_data_dir()
            self.profile = self.load_profile()
            self._initialized = True

    def _ensure_data_dir(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def load_profile(self):
        if not os.path.exists(self.file_path):
            return {
                "name": "Master",
                "preferences": {},
                "notes": []
            }
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"name": "Master", "preferences": {}, "notes": []}

    def save_profile(self):
        """Thread-safe save profile to disk."""
        with self._lock:
            try:
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump(self.profile, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"[Memory Error] Save failed: {e}")

    def update_profile(self, key, value):
        """更新根字段或 preferences"""
        if key in ["name"]:
            self.profile[key] = value
        else:
            # 默认存入 preferences
            self.profile["preferences"][key] = value
        self.save_profile()

    def add_note(self, content):
        if content not in self.profile["notes"]:
            self.profile["notes"].append(content)
            self.save_profile()

    def get_system_prompt_suffix(self):
        suffix = "\n\n【关于主人的记忆】\n"
        suffix += f"- 称呼: {self.profile.get('name', 'Master')}\n"
        
        prefs = self.profile.get("preferences", {})
        if prefs:
            suffix += f"- 偏好: {json.dumps(prefs, ensure_ascii=False)}\n"
            
        notes = self.profile.get("notes", [])
        if notes:
            suffix += "- 备忘:\n"
            for note in notes[-5:]: # 只显示最近5条
                suffix += f"  * {note}\n"
        
        return suffix
