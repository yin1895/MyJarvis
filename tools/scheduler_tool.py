# Jarvis Cortex Protocol - Scheduler Tool
# tools/scheduler_tool.py

"""
SchedulerTool: Time-based reminders and scheduled tasks.

Migrated from: ManagerAgent inline schedule handling
Risk Level: SAFE (local scheduling, no external effects)

V6.1 Refactor:
- 移除 set_scheduler 全局变量依赖
- 直接使用 SchedulerService 单例模式
"""

from typing import Optional, Literal, Any
from pydantic import BaseModel, Field
import dateparser
from datetime import datetime

from core.tools.base import BaseTool, RiskLevel, ToolResult
from services.scheduler_service import SchedulerService


class SchedulerInput(BaseModel):
    """Input schema for scheduler operations."""
    action: Literal["add_reminder", "list_reminders"] = Field(
        ...,
        description="操作类型: add_reminder(添加提醒) 或 list_reminders(列出提醒)"
    )
    content: Optional[str] = Field(
        default=None,
        description="提醒内容，如 '开会' 或 '吃药'"
    )
    time_str: Optional[str] = Field(
        default=None,
        description="时间描述，如 '下午3点'、'10分钟后'、'明天早上9点'"
    )


class SchedulerTool(BaseTool[SchedulerInput]):
    """
    Schedule reminders and time-based tasks.
    
    Features:
    - Natural language time parsing (via dateparser)
    - Flexible time expressions (relative and absolute)
    - Integration with SchedulerService for execution
    
    This tool schedules local reminders and is marked as SAFE.
    """
    
    name = "scheduler_tool"
    description = "设置定时提醒。支持自然语言时间描述，如 '下午3点提醒我开会' 或 '10分钟后叫我'。"
    risk_level = RiskLevel.SAFE
    InputSchema = SchedulerInput
    tags = ["schedule", "reminder", "timer", "alarm"]
    
    def execute(self, params: SchedulerInput) -> ToolResult:
        """Execute scheduler operation."""
        action = params.action
        
        if action == "add_reminder":
            return self._add_reminder(params.content, params.time_str)
        elif action == "list_reminders":
            return self._list_reminders()
        else:
            return ToolResult(success=False, error=f"未知操作: {action}")
    
    def _add_reminder(self, content: Optional[str], time_str: Optional[str]) -> ToolResult:
        """Add a new reminder."""
        # 使用 SchedulerService 单例
        scheduler = SchedulerService()
        
        if not content:
            return ToolResult(
                success=False,
                error="请指定提醒内容，如 '开会' 或 '吃药'"
            )
        
        if not time_str:
            return ToolResult(
                success=False,
                error="请指定提醒时间，如 '下午3点' 或 '10分钟后'"
            )
        
        # Parse time using dateparser
        try:
            # Configure dateparser for Chinese and relative time support
            dt_obj = dateparser.parse(
                time_str,
                settings={
                    'PREFER_DATES_FROM': 'future',
                    'RELATIVE_BASE': datetime.now()
                }
            )
            
            if dt_obj is None:
                return ToolResult(
                    success=False,
                    error=f"无法理解时间描述: '{time_str}'。请使用如 '下午3点'、'10分钟后' 等格式。"
                )
            
            # Ensure the time is in the future
            if dt_obj <= datetime.now():
                return ToolResult(
                    success=False,
                    error=f"提醒时间 {dt_obj.strftime('%H:%M')} 已经过去了，请指定未来的时间。"
                )
            
            # Add the reminder
            scheduler.add_reminder(content, dt_obj)
            
            return ToolResult(
                success=True,
                data={
                    "action": "add_reminder",
                    "content": content,
                    "time": dt_obj.strftime("%Y-%m-%d %H:%M"),
                    "message": f"好的，已设定在 {dt_obj.strftime('%H:%M')} 提醒您：{content}"
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"设置提醒失败: {str(e)}",
                metadata={"content": content, "time_str": time_str}
            )
    
    def _list_reminders(self) -> ToolResult:
        """List all pending reminders."""
        # 使用 SchedulerService 单例
        scheduler = SchedulerService()
        
        try:
            # Get jobs from APScheduler
            jobs = scheduler.scheduler.get_jobs()
            
            if not jobs:
                return ToolResult(
                    success=True,
                    data={"reminders": [], "message": "目前没有待执行的提醒。"}
                )
            
            reminders = []
            for job in jobs:
                reminders.append({
                    "id": job.id,
                    "next_run": str(job.next_run_time),
                    "args": job.args if hasattr(job, 'args') else []
                })
            
            return ToolResult(
                success=True,
                data={
                    "reminders": reminders,
                    "count": len(reminders),
                    "message": f"当前有 {len(reminders)} 个待执行的提醒。"
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"获取提醒列表失败: {str(e)}"
            )
