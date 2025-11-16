from langchain_deepseek import ChatDeepSeek
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

llm = ChatDeepSeek(model="deepseek-chat")
search = TavilySearch(max_results=5)





