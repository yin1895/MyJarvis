"""
Jarvis V7.0 - Production Entry Point

统一入口，支持语音/文字双模式：
- LangGraph 原生工具系统
- Rich UI 美化输出
- 语音 I/O (TTS + STT + 唤醒词)
- SQLite 会话持久化
- 人机安全拦截器

Usage:
    python main.py              # 语音模式 (默认)
    python main.py -t           # 文字模式
    python main.py --mute       # 静音模式 (无TTS)
    python main.py --no-safety  # 禁用安全拦截
    python main.py -r smart     # 使用 smart 角色
"""

from __future__ import annotations

import asyncio
import sys
import os
import io
import argparse
import logging
import warnings
from pathlib import Path
from typing import Optional, Callable, Any

# ==================== 环境修复 ====================
reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(reconfigure):
    try:
        reconfigure(encoding='utf-8', line_buffering=True)
    except:
        pass
else:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    except:
        pass

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module='pygame')

# ==================== 日志配置 ====================
logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ==================== Rich UI ====================
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme
from rich.prompt import Confirm

custom_theme = Theme({
    "user": "bold green",
    "jarvis": "bold cyan",
    "info": "dim cyan",
    "error": "bold red",
    "warning": "bold yellow",
    "danger": "bold red on yellow",
    "safe": "bold green",
})
console = Console(theme=custom_theme)

# ==================== LangChain/LangGraph ====================
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config import Config
from core.graph.builder import create_graph, check_tool_calls_safety, get_tool_by_name
from core.graph.state import AgentState
from core.llm_provider import RoleType, LLMFactory
from services.memory_service import MemoryService
from tools.native_role import ROLE_SWITCH_MARKER

# ==================== 路径配置 ====================
DATA_DIR = Path(__file__).parent / "data"
STATE_DB_PATH = DATA_DIR / "state.db"


def ensure_data_dir() -> None:
    """Ensure the data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ==================== 工具调用辅助函数 ====================

def format_tool_call_display(tool_name: str, tool_args: dict) -> str:
    """Format tool call for display."""
    args_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in tool_args.items())
    return f"{tool_name}({args_str})"


def create_rejection_messages(tool_calls: list) -> list[ToolMessage]:
    """Create rejection ToolMessages for denied tool calls."""
    return [
        ToolMessage(
            content=f"工具调用被用户拒绝。用户不允许执行 '{tc['name']}' 操作。",
            tool_call_id=tc["id"],
            name=tc["name"],
        )
        for tc in tool_calls
    ]


# ==================== 安全确认函数 ====================

def text_confirm_dangerous_tools(tool_calls: list) -> bool:
    """
    Text-mode confirmation for dangerous tools.
    
    Returns:
        True if user approves, False otherwise
    """
    console.print(Panel(
        "[danger]⚠ 检测到危险工具调用，需要您的批准[/danger]",
        title="安全拦截",
        border_style="red",
        expand=False
    ))
    
    for tc in tool_calls:
        tool = get_tool_by_name(tc["name"])
        risk = (tool.metadata or {}).get("risk_level", "unknown") if tool else "unknown"
        risk_style = "danger" if risk == "dangerous" else "safe"
        console.print(f"  [{risk_style}][{risk.upper()}][/{risk_style}] {format_tool_call_display(tc['name'], tc['args'])}")
    
    return Confirm.ask("\n[warning]是否允许执行这些操作？[/warning]", default=False)


def voice_confirm_dangerous_tools(
    tool_calls: list,
    speak_func: Callable[..., Any],
    listen_func: Callable[[], str],
) -> bool:
    """
    Voice-mode confirmation for dangerous tools.
    
    Uses TTS to ask and STT to listen for confirmation.
    
    Returns:
        True if user says "确认/是/执行/好", False otherwise
    """
    # Display in console
    console.print(Panel(
        "[danger]⚠ 检测到危险工具调用，语音确认中...[/danger]",
        title="安全拦截",
        border_style="red",
        expand=False
    ))
    
    for tc in tool_calls:
        console.print(f"  [danger][DANGEROUS][/danger] {format_tool_call_display(tc['name'], tc['args'])}")
    
    # Build confirmation prompt
    tool_names = ", ".join(tc["name"] for tc in tool_calls)
    prompt = f"检测到危险操作：{tool_names}。请说'确认'执行，或'取消'拒绝。"
    
    # Speak the prompt
    console.print(f"[jarvis][Jarvis]: {prompt}[/jarvis]")
    speak_func(prompt)
    
    # Listen for response
    console.print("[info]正在聆听您的回应...[/info]")
    response = listen_func()
    
    if not response:
        console.print("[warning]未检测到回应，默认拒绝[/warning]")
        return False
    
    console.print(f"[user][主人]: {response}[/user]")
    
    # Check for confirmation keywords
    confirm_keywords = ["确认", "是", "执行", "好", "可以", "同意", "yes", "ok"]
    approved = any(kw in response.lower() for kw in confirm_keywords)
    
    if approved:
        console.print("[safe]✓ 语音确认通过[/safe]")
    else:
        console.print("[warning]✗ 语音确认拒绝[/warning]")
    
    return approved


# ==================== Graph 执行引擎 ====================

async def run_graph_with_safety(
    graph,
    messages: list[BaseMessage],
    thread_config: RunnableConfig,
    role: str = "default",
    auto_approve_safe: bool = True,
    confirm_func: Optional[Callable[[list], bool]] = None,
) -> tuple[str, list[BaseMessage]]:
    """
    Run the LangGraph with human-in-the-loop safety checks.
    
    Args:
        graph: Compiled LangGraph instance
        messages: Current message history
        thread_config: Thread configuration for checkpointing
        role: LLM role
        auto_approve_safe: Whether to auto-approve safe tools
        confirm_func: Function to call for dangerous tool confirmation
        
    Returns:
        Tuple of (response_text, updated_messages)
    """
    full_response = ""
    
    # Build initial state
    state: AgentState = {
        "messages": messages.copy(),
        "current_role": role,
        "metadata": {}
    }
    
    # Track if we're resuming from interrupt (use None to resume from checkpoint)
    current_input: Optional[AgentState] = state
    
    while True:
        # Stream until we hit an interrupt or finish
        async for event in graph.astream_events(current_input, config=thread_config, version="v2"):
            kind = event.get("event")
            
            # Handle streaming tokens
            if kind == "on_chat_model_stream":
                content = event.get("data", {}).get("chunk")
                if content and hasattr(content, "content"):
                    token = content.content
                    if token:
                        print(token, end="", flush=True)
                        full_response += token
        
        # Check if we're in an interrupted state
        graph_state = await graph.aget_state(thread_config)
        
        if not graph_state.next:
            # Graph finished normally
            break
        
        # We're interrupted before tools node
        if "tools" in graph_state.next:
            current_messages = graph_state.values.get("messages", [])
            if current_messages:
                last_msg = current_messages[-1]
                if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                    tool_calls = last_msg.tool_calls
                    
                    # Check safety
                    all_safe, dangerous_tools = check_tool_calls_safety(tool_calls)
                    
                    if all_safe and auto_approve_safe:
                        # Auto-approve safe tools
                        console.print("\n[safe]✓ 自动批准安全工具调用[/safe]")
                        for tc in tool_calls:
                            console.print(f"[dim]  → {format_tool_call_display(tc['name'], tc['args'])}[/dim]")
                        
                        # Resume from checkpoint by passing None
                        current_input = None
                        continue
                    
                    else:
                        # Dangerous tools - require confirmation
                        print()  # Newline
                        
                        if confirm_func:
                            approved = confirm_func(tool_calls)
                        else:
                            approved = text_confirm_dangerous_tools(tool_calls)
                        
                        if approved:
                            console.print("[safe]✓ 用户批准，继续执行...[/safe]")
                            # Resume from checkpoint by passing None
                            current_input = None
                            # Print continuation prompt
                            console.print("[jarvis][Jarvis]: [/jarvis]", end="")
                            continue
                        else:
                            console.print("[warning]✗ 用户拒绝，已取消操作[/warning]")
                            # Create rejection messages
                            rejection_msgs = create_rejection_messages(tool_calls)
                            
                            # Update graph state with rejection
                            await graph.aupdate_state(
                                thread_config,
                                {"messages": rejection_msgs},
                                as_node="tools"
                            )
                            
                            # Resume from checkpoint (now with rejection messages added)
                            current_input = None
                            console.print("[jarvis][Jarvis]: [/jarvis]", end="")
                            continue
        
        # Unknown interrupt state
        logger.warning(f"Unknown interrupt state: {graph_state.next}")
        break
    
    # Get final messages
    final_state = await graph.aget_state(thread_config)
    final_messages = final_state.values.get("messages", messages) if final_state.values else messages
    
    return full_response, list(final_messages)


# ==================== 主循环 ====================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Jarvis V7.0 - 智能助手")
    parser.add_argument("-t", "--text", action="store_true", help="文字聊天模式 (无需麦克风)")
    parser.add_argument("--mute", action="store_true", help="静音模式 (不语音合成)")
    parser.add_argument("--no-safety", action="store_true", help="禁用安全拦截器")
    parser.add_argument("-r", "--role", type=str, default="default",
                        choices=["default", "smart", "coder", "fast", "vision"],
                        help="LLM 角色预设")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup environment
    Config.setup_env_proxy()
    ensure_data_dir()
    
    # Display banner
    mode_str = "文字模式" if args.text else "语音模式"
    safety_str = "禁用" if args.no_safety else "启用"
    banner = Panel(
        f"[bold white]Jarvis V7.0 AI Assistant[/bold white]\n"
        f"[dim]模式: {mode_str} | 安全拦截: {safety_str} | Role: {args.role}[/dim]",
        title="System Online",
        border_style="blue",
        expand=False
    )
    console.print(banner)
    
    # Initialize services
    ear_mouth = None
    wake_word = None
    scheduler = None
    
    try:
        with console.status("[info]正在初始化核心模组...[/info]", spinner="dots"):
            # Import heavy services lazily
            from services.io_service import AudioHandler, WakeWord
            from services.scheduler_service import SchedulerService
            
            ear_mouth = AudioHandler()
            scheduler = SchedulerService(speak_callback=ear_mouth.speak)
            
            if not args.text:
                wake_word = WakeWord()
        
        console.print("[info]系统初始化完成。[/info]")
        
        # Load user memory for personalization
        memory = MemoryService()
        user_name = memory.profile.get("name", "主人")
        console.print(f"[info]欢迎回来，{user_name}！[/info]")
        
    except Exception as e:
        console.print(f"[error]初始化失败: {e}[/error]")
        logger.exception("Init failed")
        return
    
    # Run main loop
    try:
        asyncio.run(run_main_loop(args, ear_mouth, wake_word, scheduler))
    
    except KeyboardInterrupt:
        console.print("\n[warning]系统正在退出...[/warning]")
    finally:
        # Cleanup
        with console.status("[info]正在清理资源...[/info]", spinner="dots"):
            try:
                if scheduler:
                    scheduler.stop()
            except:
                pass
            try:
                if ear_mouth:
                    ear_mouth.close()
            except:
                pass
            try:
                if wake_word:
                    wake_word.close()
            except:
                pass
        
        console.print("[info]再见。[/info]")


async def run_main_loop(args, ear_mouth, wake_word, scheduler):
    """
    Main async loop for running Jarvis with AsyncSqliteSaver.
    
    This function is extracted to properly use async context manager
    for the AsyncSqliteSaver checkpointer.
    """
    current_role = args.role  # Track current role for dynamic switching
    
    async with AsyncSqliteSaver.from_conn_string(str(STATE_DB_PATH)) as checkpointer:
        # Create graph
        graph = create_graph(
            role=current_role,
            checkpointer=checkpointer,
            interrupt_before_tools=not args.no_safety,
        )
        
        thread_id = "jarvis-main-thread"
        thread_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        
        # Try to restore previous messages
        try:
            saved_state = await graph.aget_state(thread_config)
            if saved_state.values and "messages" in saved_state.values:
                messages = list(saved_state.values["messages"])
                console.print(f"[info]已恢复上次会话 ({len(messages)} 条消息)[/info]")
            else:
                messages: list[BaseMessage] = []
        except:
            messages: list[BaseMessage] = []
        
        if args.text:
            # ==================== 文字模式循环 ====================
            console.print("[info]提示: 输入指令，'exit' 退出，'clear' 清空会话[/info]")
            
            while True:
                try:
                    console.print("\n[user][主人]: [/user]", end="")
                    user_input = input().strip()
                    
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ["exit", "quit", "退下", "bye"]:
                        console.print("[jarvis][Jarvis]: 好的，再见！[/jarvis]")
                        if not args.mute:
                            ear_mouth.speak("好的，再见！")
                        break
                    
                    if user_input.lower() == "clear":
                        messages.clear()
                        console.print("[info]会话已清空。[/info]")
                        continue
                    
                    if user_input.lower() in ["history", "debug"]:
                        console.print(f"[info]历史消息: {len(messages)} 条[/info]")
                        for i, msg in enumerate(messages[-10:]):
                            console.print(f"[dim]  {i}: [{type(msg).__name__}] {str(msg.content)[:60]}...[/dim]")
                        continue
                    
                    # Add user message
                    messages.append(HumanMessage(content=user_input))
                    
                    # Get response
                    console.print("[jarvis][Jarvis]: [/jarvis]", end="")
                    
                    response, messages = await run_graph_with_safety(
                        graph=graph,
                        messages=messages,
                        thread_config=thread_config,
                        role=current_role,
                        auto_approve_safe=True,
                        confirm_func=text_confirm_dangerous_tools if not args.no_safety else None,
                    )
                    
                    print()  # Newline after response
                    
                    # Check for role switch marker in ToolMessages or response
                    found_marker = False
                    new_role_candidate = None

                    # 1. Check recent ToolMessages for switch_role result (only last 3 to avoid old matches)
                    recent_messages = messages[-3:] if len(messages) > 3 else messages
                    for msg in reversed(recent_messages):
                        # Check ToolMessage with name 'switch_role'
                        is_switch_tool = (
                            (isinstance(msg, ToolMessage) or hasattr(msg, 'tool_call_id')) and
                            getattr(msg, 'name', '') == 'switch_role'
                        )
                        if is_switch_tool:
                            msg_content = str(msg.content)
                            if ROLE_SWITCH_MARKER in msg_content:
                                for line in msg_content.split('\n'):
                                    if line.startswith(ROLE_SWITCH_MARKER):
                                        new_role_candidate = line.split(':')[1]
                                        found_marker = True
                                        break
                        if found_marker:
                            break
                    
                    # 2. Fallback: Check response text (if LLM repeated it)
                    if not found_marker and response and ROLE_SWITCH_MARKER in response:
                        for line in response.split('\n'):
                            if line.startswith(ROLE_SWITCH_MARKER):
                                new_role_candidate = line.split(':')[1]
                                found_marker = True
                                break
                    
                    # Apply switch if found
                    role_switched = False
                    if found_marker and new_role_candidate and new_role_candidate != current_role:
                        current_role = new_role_candidate
                        role_switched = True
                        # Rebuild graph with new role (cast to RoleType)
                        from typing import cast
                        graph = create_graph(
                            role=cast(RoleType, current_role),
                            checkpointer=checkpointer,
                            interrupt_before_tools=not args.no_safety,
                        )
                        role_info = LLMFactory.get_role_info(cast(RoleType, current_role))
                        console.print(f"[info]已切换至 {current_role} 模式 ({role_info['provider']}/{role_info['model']})[/info]")
                    
                    # TTS (filter out the marker from speech, skip if role switched to avoid duplicate)
                    if not args.mute and response and not role_switched:
                        speech_text = response.replace(ROLE_SWITCH_MARKER, "").strip()
                        # Remove the marker line for clean speech
                        speech_lines = [l for l in speech_text.split('\n') if not l.startswith("__JARVIS")]
                        speech_text = '\n'.join(speech_lines)
                        if speech_text:
                            with console.status("[info]语音合成...[/info]", spinner="dots"):
                                ear_mouth.speak(speech_text)
                    
                except KeyboardInterrupt:
                    console.print("\n[warning]已中断。输入 'exit' 退出。[/warning]")
                    continue
                except Exception as e:
                    console.print(f"[error]错误: {e}[/error]")
                    logger.exception("Error in text loop")
                    continue
        
        else:
            # ==================== 语音模式循环 ====================
            console.print("[info]语音监听已启动，请说出唤醒词...[/info]")
            
            # Create voice confirm function
            def voice_confirm(tool_calls):
                return voice_confirm_dangerous_tools(
                    tool_calls,
                    speak_func=ear_mouth.speak,
                    listen_func=ear_mouth.listen,
                )
            
            while True:
                try:
                    # Wait for wake word
                    if wake_word is None or not wake_word.listen():
                        break
                    
                    # Wake feedback
                    ear_mouth.play_ding()
                    console.print("[info]已唤醒，正在聆听...[/info]")
                    
                    # Listen
                    with console.status("[info]正在识别语音...[/info]", spinner="dots"):
                        user_input = ear_mouth.listen()
                    
                    if not user_input:
                        console.print("[dim]未检测到有效语音[/dim]")
                        continue
                    
                    console.print(f"[user][主人]: {user_input}[/user]")
                    
                    # Add user message
                    messages.append(HumanMessage(content=user_input))
                    
                    # Get response
                    console.print("[jarvis][Jarvis]: [/jarvis]", end="")
                    
                    response, messages = await run_graph_with_safety(
                        graph=graph,
                        messages=messages,
                        thread_config=thread_config,
                        role=current_role,
                        auto_approve_safe=True,
                        confirm_func=voice_confirm if not args.no_safety else None,
                    )
                    
                    print()  # Newline
                    
                    # Check for role switch marker in ToolMessages or response
                    found_marker = False
                    new_role_candidate = None

                    # 1. Check recent ToolMessages for switch_role result (only last 3 to avoid old matches)
                    recent_messages = messages[-3:] if len(messages) > 3 else messages
                    for msg in reversed(recent_messages):
                        # Check ToolMessage with name 'switch_role'
                        is_switch_tool = (
                            (isinstance(msg, ToolMessage) or hasattr(msg, 'tool_call_id')) and
                            getattr(msg, 'name', '') == 'switch_role'
                        )
                        if is_switch_tool and ROLE_SWITCH_MARKER in str(msg.content):
                            content = str(msg.content)
                            for line in content.split('\n'):
                                if line.startswith(ROLE_SWITCH_MARKER):
                                    new_role_candidate = line.split(':')[1]
                                    found_marker = True
                                    break
                        if found_marker:
                            break
                    
                    # 2. Fallback: Check response text (if LLM repeated it)
                    if not found_marker and response and ROLE_SWITCH_MARKER in response:
                        for line in response.split('\n'):
                            if line.startswith(ROLE_SWITCH_MARKER):
                                new_role_candidate = line.split(':')[1]
                                found_marker = True
                                break
                    
                    # Apply switch if found
                    role_switched = False
                    if found_marker and new_role_candidate and new_role_candidate != current_role:
                        current_role = new_role_candidate
                        role_switched = True
                        from typing import cast
                        graph = create_graph(
                            role=cast(RoleType, current_role),
                            checkpointer=checkpointer,
                            interrupt_before_tools=not args.no_safety,
                        )
                        role_info = LLMFactory.get_role_info(cast(RoleType, current_role))
                        console.print(f"[info]已切换至 {current_role} 模式 ({role_info['provider']}/{role_info['model']})[/info]")
                        ear_mouth.speak(f"已切换到{current_role}模式")
                    
                    # TTS with interrupt detection (filter out marker, skip if role switched)
                    if response and not role_switched:
                        speech_text = response.replace(ROLE_SWITCH_MARKER, "").strip()
                        speech_lines = [l for l in speech_text.split('\n') if not l.startswith("__JARVIS")]
                        speech_text = '\n'.join(speech_lines)
                        if speech_text:
                            with console.status("[info]语音合成...[/info]", spinner="dots"):
                                wake_word.start()
                                
                                def check_interrupt():
                                    return wake_word.process_frame()
                                
                                ear_mouth.speak(speech_text, interrupt_check_callback=check_interrupt)
                                wake_word.stop()
                    
                except KeyboardInterrupt:
                    console.print("\n[warning]已中断。[/warning]")
                    break
                except Exception as e:
                    console.print(f"[error]错误: {e}[/error]")
                    logger.exception("Error in voice loop")
                    continue


if __name__ == "__main__":
    main()
