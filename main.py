import sys
import io
import warnings
import os
import argparse
import logging
from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.text import Text

# 屏蔽 httpx 和 openai 的 info 日志，只显示 warning 以上
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
# 只保留服务的日志
logging.basicConfig(level=logging.INFO, format='%(message)s')

# 1. 环境修复
reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(reconfigure):
    try: reconfigure(encoding='utf-8', line_buffering=True)
    except: pass
else:
    try: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    except: pass

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module='pygame')

# 2. 导入模块
from config import Config
from services.io_service import AudioHandler, WakeWord
from services.scheduler_service import SchedulerService
from agents.manager import ManagerAgent

# --- UI 配置 ---
custom_theme = Theme({
    "user": "bold green",
    "alice": "bold hot_pink",
    "info": "dim cyan",
    "danger": "bold red"
})
console = Console(theme=custom_theme)

def main():
    # --- 命令行参数解析 ---
    parser = argparse.ArgumentParser(description="Jarvis 智能助手")
    parser.add_argument("-t", "--text", action="store_true", help="进入文字聊天模式 (无需麦克风)")
    parser.add_argument("--mute", action="store_true", help="静音模式 (Alice 不说话，只显示文字)")
    args = parser.parse_args()

    # --- 启动画面 ---
    mode_str = "文字模式" if args.text else "语音模式"
    banner = Panel(
        f"[bold white]Jarvis AI Assistant[/bold white]\n[dim]Version 2.0 | {mode_str}[/dim]",
        title="System Online",
        border_style="blue",
        expand=False
    )
    console.print(banner)

    Config.setup_env_proxy()

    # --- 初始化 ---
    ear_mouth = None
    scheduler = None
    agent_system = None
    wake_word = None

    try:
        with console.status("[info]正在初始化核心模组...[/info]", spinner="dots"):
            # 初始化核心服务
            ear_mouth = AudioHandler()
            scheduler = SchedulerService(speak_callback=ear_mouth.speak)
            agent_system = ManagerAgent(scheduler=scheduler)
            
            # 语音模式才需要唤醒词
            if not args.text:
                wake_word = WakeWord()
        
        console.print("[info]系统初始化完成。[/info]")

    except Exception as e:
        console.print(f"[danger]严重错误: 初始化失败 - {e}[/danger]")
        return

    # --- 主循环逻辑 ---
    try:
        if args.text:
            # === 文字模式循环 ===
            console.print("[info]提示: 您现在可以直接输入指令，输入 'exit' 或 'quit' 退出。[/info]")
            while True:
                try:
                    # 1. 获取键盘输入
                    console.print("\n[user][主人]: [/user]", end="")
                    user_text = input().strip()
                    
                    if not user_text: continue
                    if user_text.lower() in ["exit", "quit", "退下"]:
                        console.print("[alice][Alice]: 好的主人，再见。[/alice]")
                        break

                    # 2. 思考与回复
                    reply = ""
                    with console.status("[info]Alice 正在思考...[/info]", spinner="bouncingBall"):
                        reply = agent_system.run(user_text)
                    
                    # 3. 输出 (文字 + 语音)
                    console.print(f"[alice][Alice]: {reply}[/alice]")
                    
                    if not args.mute:
                        with console.status("[info]正在语音合成...[/info]", spinner="dots"):
                            ear_mouth.speak(reply)
                        
                except KeyboardInterrupt:
                    break

        else:
            # === 语音模式循环 ===
            console.print("[info]语音监听已启动，请说出唤醒词...[/info]")
            while True:
                # 1. 等待唤醒
                if wake_word is None or not wake_word.listen():
                    break 
                
                # 2. 唤醒反馈
                ear_mouth.play_ding()
                console.print("[info]已唤醒，正在聆听...[/info]")
                
                # 3. 聆听
                user_text = ""
                with console.status("[info]正在识别语音...[/info]", spinner="dots"):
                    user_text = ear_mouth.listen()
                
                if not user_text: 
                    console.print("[dim]未检测到有效语音[/dim]")
                    continue
                
                console.print(f"[user][主人]: {user_text}[/user]")
                
                # 4. 思考
                reply = ""
                with console.status("[info]Alice 正在思考...[/info]", spinner="bouncingBall"):
                    reply = agent_system.run(user_text)
                
                # 5. 回复 (支持打断)
                console.print(f"[alice][Alice]: {reply}[/alice]")
                with console.status("[info]正在语音合成...[/info]", spinner="dots"):
                    # 开启唤醒词检测流
                    wake_word.start()
                    # 定义打断检测回调
                    def check_interrupt():
                        return wake_word.process_frame()
                    
                    ear_mouth.speak(reply, interrupt_check_callback=check_interrupt)
                    
                    # 说话结束，关闭流 (避免与下一轮 listen 冲突)
                    wake_word.stop()

    except KeyboardInterrupt:
        console.print("\n[danger]系统正在退出...[/danger]")
    finally:
        # 清理资源
        with console.status("[info]正在清理资源...[/info]", spinner="dots"):
            try: 
                if scheduler: scheduler.stop()
            except: pass
            try: 
                if ear_mouth: ear_mouth.close()
            except: pass
            try: 
                if agent_system: agent_system.close()
            except: pass
            if wake_word:
                try: wake_word.close()
                except: pass
        console.print("[info]再见。[/info]")

if __name__ == "__main__":
    main()
