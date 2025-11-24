import os
import time
from rich.console import Console
from rich.progress import track
from services.knowledge_service import KnowledgeService

console = Console()

def train():
    console.print(Panel.fit("[bold green]Jarvis 知识库训练程序 (Codebase Ingestion)[/bold green]", border_style="green"))
    
    # 初始化服务 (会自动加载 GPU 模型)
    with console.status("[bold cyan]正在初始化向量数据库 (CUDA Accelerated)..."):
        ks = KnowledgeService()
    
    root_dir = os.getcwd()
    # 需要忽略的文件夹
    ignore_dirs = {'.git', '__pycache__', '.venv', 'data', 'dist', 'build', '.idea', '.vscode'}
    # 需要学习的文件后缀
    target_exts = {'.py', '.md', '.txt', '.env.example'}

    files_to_process = []

    # 1. 扫描文件
    for root, dirs, files in os.walk(root_dir):
        # 修改 dirs 列表以跳过忽略的目录
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in target_exts:
                full_path = os.path.join(root, file)
                # 跳过训练脚本本身和旧代码
                if file in ['train_jarvis.py', 'core.old.py']:
                    continue
                files_to_process.append(full_path)

    console.print(f"[info]扫描到 {len(files_to_process)} 个核心文件，准备开始注入知识库...[/info]")

    # 2. 批量处理
    start_time = time.time()
    success_count = 0
    
    # 使用 rich 的进度条
    for file_path in track(files_to_process, description="[bold magenta]正在吞噬代码...[/bold magenta]"):
        try:
            # ingest_file 内部有查重逻辑，重复运行很快
            result = ks.ingest_file(file_path)
            # console.print(f"[dim]{os.path.basename(file_path)}: {result}[/dim]")
            success_count += 1
        except Exception as e:
            console.print(f"[red]处理 {os.path.basename(file_path)} 失败: {e}[/red]")

    end_time = time.time()
    duration = end_time - start_time

    console.print(f"\n[bold green]✨ 训练完成！[/bold green]")
    console.print(f"耗时: {duration:.2f} 秒")
    console.print(f"总计处理: {success_count} 个文件")
    console.print(f"当前知识库片段总数: {ks.get_stats()}")

if __name__ == "__main__":
    # 引入 Panel 只是为了好看，需要 lazy import
    from rich.panel import Panel
    train()