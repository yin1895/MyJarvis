# tools.py
import subprocess
import webbrowser
import datetime
import requests
from config import Config
from playwright.sync_api import sync_playwright

# ... (open_app, get_current_time, get_weather 保持不变，请保留) ...
# 为了方便，我把 get_weather 和 open_app 简略写在这，你需要确保文件里有它们

def open_app(app_name):
    # ... (保持原样) ...
    app_name = app_name.lower()
    try:
        if "记事本" in app_name:
            subprocess.Popen("notepad.exe")
            return True, "记事本打开啦"
        elif "计算器" in app_name:
            subprocess.Popen("calc.exe")
            return True, "计算器来喽"
        elif "浏览器" in app_name:
            webbrowser.open("https://www.bilibili.com") 
            return True, "浏览器打开了，正在前往 Bilibili"
        else:
            return False, f"抱歉主人，我还没学会怎么打开 {app_name}"
    except Exception as e:
        return False, f"打开失败了: {e}"

def get_current_time():
    now = datetime.datetime.now()
    return now.strftime("%H点%M分")

def get_weather(city="Beijing"):
    try:
        # 英文城市名兼容 (简单处理)
        if "北京" in city: city = "Beijing"
        elif "上海" in city: city = "Shanghai"
        elif "广州" in city: city = "Guangzhou"
        elif "深圳" in city: city = "Shenzhen"
        
        url = f"https://wttr.in/{city}?format=3"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return True, f"今日天气：{response.text.strip()}"
        else:
            return False, "天气服务好像连不通呢。"
    except Exception as e:
        return False, f"查询天气失败了: {e}"

# === 重构后的搜索函数 ===
def search_web(query):
    """使用 Playwright 模拟真实浏览器搜索 (抗造版)"""
    print(f"[系统]: 正在调用浏览器引擎搜索 '{query}' ...")
    
    try:
        with sync_playwright() as p:
            launch_args = {
                "headless": False,  # 【调试】设为 False 可以看到浏览器弹出，方便排错
                "args": ["--start-maximized"]
            }
            
            # 配置代理
            if Config.PROXY_ENABLED and Config.PROXY_URL:
                launch_args["proxy"] = {"server": Config.PROXY_URL}

            browser = p.chromium.launch(**launch_args)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()

            try:
                # 访问 DDG
                page.goto("https://duckduckgo.com/", timeout=30000)
                
                # 输入并搜索
                page.wait_for_selector('input[name="q"]', state="visible")
                page.fill('input[name="q"]', query)
                page.press('input[name="q"]', 'Enter')
                
                # 【关键修改】不再死等 article，而是等待页面加载完成
                # domcontentloaded 比 networkidle 更快，但也够用了
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=15000)
                    # 再稍微等一下 JS 渲染
                    page.wait_for_timeout(2000) 
                except:
                    print("[调试]: 页面加载等待超时，尝试强行读取...")

                # 【核心逻辑】尝试提取内容，如果失败则启用暴力兜底
                content = page.evaluate("""() => {
                    const items = document.querySelectorAll('article');
                    let results = [];
                    
                    // 1. 尝试标准结果
                    if (items.length > 0) {
                        for (let i = 0; i < Math.min(items.length, 4); i++) {
                            let title = items[i].querySelector('h2')?.innerText || '无标题';
                            let snippet = items[i].querySelector('[data-result="snippet"]')?.innerText || items[i].innerText;
                            results.push(`【结果${i+1}】${title}\\n${snippet}`);
                        }
                        return results.join('\\n----------------\\n');
                    }
                    
                    // 2. 如果没找到 article (比如只有天气组件)，直接抓正文文本
                    // 排除掉导航栏等无用信息，尽量抓主容器
                    const main = document.querySelector('#react-layout') || document.body;
                    return "【非标准结果模式】\\n" + main.innerText.substring(0, 3000); 
                }""")
                
                browser.close()
                return True, content

            except Exception as inner_e:
                browser.close()
                raise inner_e

    except Exception as e:
        print(f"[Playwright Error]: {e}")
        return False, f"浏览器搜索出错了: {e}"