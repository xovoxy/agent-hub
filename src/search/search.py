from langchain_deepseek import ChatDeepSeek
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
import os

llm = ChatDeepSeek(
    model=os.getenv(
        "DEEPSEEK_MODEL",
        os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-v4-flash"),
    ),
    extra_body={"thinking": {"type": "disabled"}},
)
search = TavilySearch(max_results=5)


