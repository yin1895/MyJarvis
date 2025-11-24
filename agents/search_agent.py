from agents.base import BaseAgent
import tools

class SearchAgent(BaseAgent):
    def run(self, query: str) -> str:
        success, raw_result = tools.search_web(query)
        
        if not success:
            return f"搜索失败了: {raw_result}"
            
        # 让 LLM 总结
        prompt = [
            {"role": "system", "content": "你是一个信息整合助手。请阅读以下搜索结果，并进行客观、简洁的总结。不要使用 Markdown 格式。"},
            {"role": "user", "content": f"搜索关键词：{query}\n\n搜索结果：\n{raw_result[:3000]}"} # Limit context
        ]
        
        return self._call_llm(prompt)
