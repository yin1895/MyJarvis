from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import logging
import threading
from typing import Optional, Callable, Any


class SchedulerService:
    """
    定时任务调度服务 (Singleton)
    
    V6.1 重构:
    - 实现单例模式，确保全局只有一个调度器实例
    - 线程安全的初始化
    - 支持延迟设置 speak_callback
    """
    
    _instance: Optional["SchedulerService"] = None
    _lock = threading.Lock()
    
    def __new__(cls, speak_callback: Optional[Callable[..., Any]] = None):
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, speak_callback: Optional[Callable[..., Any]] = None):
        # 线程安全的初始化检查
        with self._lock:
            if self._initialized:
                # 已初始化，但如果传入了新的 callback，更新它
                if speak_callback is not None:
                    self.speak_callback = speak_callback
                return
            
            self.speak_callback = speak_callback
            self.scheduler = BackgroundScheduler()
            self.scheduler.start()
            logging.info("SchedulerService started (Singleton).")
            self._initialized = True
    
    def set_speak_callback(self, callback: Callable[..., Any]):
        """设置语音回调函数 (支持延迟注入)"""
        self.speak_callback = callback

    def add_reminder(self, task_content: str, trigger_time: datetime):
        logging.info(f"Adding reminder: '{task_content}' at {trigger_time}")
        self.scheduler.add_job(
            self._trigger_reminder,
            'date',
            run_date=trigger_time,
            args=[task_content]
        )

    def _trigger_reminder(self, task_content: str):
        logging.info(f"Triggering reminder: {task_content}")
        if self.speak_callback:
            self.speak_callback(f"主人，提醒时间到了：{task_content}")

    def stop(self):
        if hasattr(self, 'scheduler') and self.scheduler.running:
            self.scheduler.shutdown()
            logging.info("SchedulerService stopped.")
