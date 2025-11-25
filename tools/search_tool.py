# Jarvis Cortex Protocol - Search Tool
# tools/search_tool.py

"""
SearchTool: Web search using DuckDuckGo via Playwright.

Migrated from: agents/search_agent.py + tools.py
Risk Level: MODERATE (network access, browser automation)

Features:
- Real browser simulation via Playwright
- Proxy support
- Anti-detection measures
"""

from typing import Optional, List
from pydantic import BaseModel, Field

from core.tools.base import BaseTool, RiskLevel, ToolResult


class WebSearchInput(BaseModel):
    """Input schema for web search."""
    query: str = Field(
        ...,
        description="搜索关键词",
        examples=["Python 最新版本", "今日新闻"]
    )
    max_results: int = Field(
        default=4,
        ge=1,
        le=10,
        description="返回的最大结果数量"
    )


class WebSearchTool(BaseTool[WebSearchInput]):
    """
    Search the web using DuckDuckGo via Playwright browser automation.
    
    Features:
    - Uses real browser to avoid bot detection
    - Supports proxy configuration
    - Extracts structured search results
    - Falls back to raw text extraction if structured parsing fails
    
    Note: Requires Playwright to be installed and configured.
    """
    
    name = "web_search"
    description = "使用 DuckDuckGo 搜索网络信息。适用于获取实时新闻、查询事实、了解最新动态。"
    risk_level = RiskLevel.MODERATE
    InputSchema = WebSearchInput
    tags = ["search", "web", "information"]
    
    def __init__(self, proxy_url: Optional[str] = None, headless: bool = True):
        """
        Initialize the search tool.
        
        Args:
            proxy_url: Optional proxy URL (e.g., "http://127.0.0.1:7890")
            headless: Whether to run browser in headless mode
        """
        super().__init__()
        self.proxy_url = proxy_url
        self.headless = headless
        
        # Try to import config for proxy settings
        try:
            from config import Config
            if Config.PROXY_ENABLED and Config.PROXY_URL:
                self.proxy_url = Config.PROXY_URL
        except ImportError:
            pass
    
    def execute(self, params: WebSearchInput) -> ToolResult:
        """Execute the web search."""
        query = params.query.strip()
        max_results = params.max_results
        
        if not query:
            return ToolResult(
                success=False,
                error="搜索关键词不能为空"
            )
        
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return ToolResult(
                success=False,
                error="Playwright 未安装。请运行: pip install playwright && playwright install chromium"
            )
        
        try:
            with sync_playwright() as p:
                # Configure browser launch
                launch_args = {
                    "headless": self.headless,
                    "args": ["--start-maximized"]
                }
                
                if self.proxy_url:
                    launch_args["proxy"] = {"server": self.proxy_url}
                
                browser = p.chromium.launch(**launch_args)
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
                page = context.new_page()
                
                try:
                    # Navigate to DuckDuckGo
                    page.goto("https://duckduckgo.com/", timeout=30000)
                    
                    # Enter search query
                    page.wait_for_selector('input[name="q"]', state="visible")
                    page.fill('input[name="q"]', query)
                    page.press('input[name="q"]', 'Enter')
                    
                    # Wait for results to load
                    try:
                        page.wait_for_load_state("domcontentloaded", timeout=15000)
                        page.wait_for_timeout(2000)  # Wait for JS rendering
                    except Exception:
                        pass  # Continue even if timeout
                    
                    # Extract results
                    results = page.evaluate(f"""() => {{
                        const items = document.querySelectorAll('article');
                        let results = [];
                        
                        if (items.length > 0) {{
                            for (let i = 0; i < Math.min(items.length, {max_results}); i++) {{
                                let title = items[i].querySelector('h2')?.innerText || '无标题';
                                let link = items[i].querySelector('a')?.href || '';
                                let snippet = items[i].querySelector('[data-result="snippet"]')?.innerText || 
                                             items[i].innerText.substring(0, 300);
                                results.push({{
                                    title: title,
                                    link: link,
                                    snippet: snippet.substring(0, 300)
                                }});
                            }}
                            return {{ type: 'structured', results: results }};
                        }}
                        
                        // Fallback: extract raw text
                        const main = document.querySelector('#react-layout') || document.body;
                        return {{ 
                            type: 'raw', 
                            content: main.innerText.substring(0, 3000) 
                        }};
                    }}""")
                    
                    browser.close()
                    
                    if results.get("type") == "structured":
                        return ToolResult(
                            success=True,
                            data={
                                "query": query,
                                "results": results["results"]
                            }
                        )
                    else:
                        return ToolResult(
                            success=True,
                            data={
                                "query": query,
                                "raw_content": results.get("content", "无法提取内容")
                            },
                            metadata={"extraction_mode": "raw"}
                        )
                        
                except Exception as e:
                    browser.close()
                    raise e
                    
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"搜索失败: {str(e)}",
                metadata={"exception_type": type(e).__name__}
            )


# === Quick Info Tools (Safe, No Browser) ===

class GetTimeInput(BaseModel):
    """Input for getting current time."""
    timezone: str = Field(
        default="Asia/Shanghai",
        description="时区 (IANA 格式，如 'Asia/Shanghai', 'UTC')"
    )


class GetTimeTool(BaseTool[GetTimeInput]):
    """Get current time in specified timezone."""
    
    name = "get_time"
    description = "获取当前时间"
    risk_level = RiskLevel.SAFE
    InputSchema = GetTimeInput
    tags = ["utility", "time"]
    
    def execute(self, params: GetTimeInput) -> ToolResult:
        from datetime import datetime
        
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(params.timezone)
            current_time = datetime.now(tz)
            
            return ToolResult(
                success=True,
                data={
                    "time": current_time.strftime("%H点%M分"),
                    "datetime": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "timezone": params.timezone
                }
            )
        except Exception as e:
            # Fallback to local time
            current_time = datetime.now()
            return ToolResult(
                success=True,
                data={
                    "time": current_time.strftime("%H点%M分"),
                    "datetime": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "timezone": "local"
                },
                metadata={"warning": f"时区 {params.timezone} 无效，使用本地时间"}
            )


class GetWeatherInput(BaseModel):
    """Input for getting weather information."""
    city: str = Field(
        default="Beijing",
        description="城市名称 (英文或中文)"
    )


class GetWeatherTool(BaseTool[GetWeatherInput]):
    """Get current weather for a city using wttr.in API."""
    
    name = "get_weather"
    description = "获取指定城市的天气信息"
    risk_level = RiskLevel.SAFE
    InputSchema = GetWeatherInput
    tags = ["utility", "weather"]
    
    # City name mapping (Chinese to English)
    CITY_MAP = {
        "北京": "Beijing",
        "上海": "Shanghai",
        "广州": "Guangzhou",
        "深圳": "Shenzhen",
        "杭州": "Hangzhou",
        "成都": "Chengdu",
        "武汉": "Wuhan",
        "西安": "Xian",
        "南京": "Nanjing",
        "重庆": "Chongqing",
    }
    
    def execute(self, params: GetWeatherInput) -> ToolResult:
        import requests
        
        city = params.city.strip()
        
        # Map Chinese city names to English
        city_en = self.CITY_MAP.get(city, city)
        
        try:
            url = f"https://wttr.in/{city_en}?format=3"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                weather_text = response.text.strip()
                return ToolResult(
                    success=True,
                    data={
                        "city": city,
                        "weather": weather_text
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    error=f"天气服务返回错误: HTTP {response.status_code}"
                )
                
        except requests.Timeout:
            return ToolResult(
                success=False,
                error="天气服务请求超时"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"查询天气失败: {str(e)}"
            )
