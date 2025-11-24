from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import logging

class SchedulerService:
    def __init__(self, speak_callback):
        self.speak_callback = speak_callback
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()
        logging.info("SchedulerService started.")

    def add_reminder(self, task_content, trigger_time):
        logging.info(f"Adding reminder: '{task_content}' at {trigger_time}")
        self.scheduler.add_job(
            self._trigger_reminder,
            'date',
            run_date=trigger_time,
            args=[task_content]
        )

    def _trigger_reminder(self, task_content):
        logging.info(f"Triggering reminder: {task_content}")
        if self.speak_callback:
            self.speak_callback(f"主人，提醒时间到了：{task_content}")

    def stop(self):
        if self.scheduler.running:
            self.scheduler.shutdown()
            logging.info("SchedulerService stopped.")
