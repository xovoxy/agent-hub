from langchain_core.prompts import ChatPromptTemplate
from src.utils import llm, reason_llm
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field


tavily_search = TavilySearch(max_results=5)

class NewsInfo(BaseModel):
    title: str = Field(description="The title of the news article")
    content: str = Field(description="The final, in-depth interpretation report, don't include the title in the content")
    newstype: str = Field(description="The type of news: 'ai_news' for AI资讯类, 'model' for 模型类, 'ai_product' for AI产品类")

class ClassifyAgent:
    def __init__(self):
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", """
你是一个专业的AI新闻分类专家。你的任务是根据新闻内容判断新闻类型。

新闻类型定义：
1. ai_news (AI资讯类): 关于AI行业动态、政策、公司新闻、市场趋势等资讯类内容
2. model (模型类): 关于AI模型发布、模型技术、模型性能、模型对比等技术性内容
3. ai_product (AI产品类): 关于具体的AI产品、AI应用、产品功能、产品发布等内容
4. other (其他类): 不属于上述三类的其他内容

请仔细阅读新闻内容，判断其属于哪种类型，只返回类型标识符：ai_news、model、ai_product 或 other。
不要返回其他内容，只返回类型标识符。
                """),
                ("human", "请对以下新闻进行分类：\n{news}"),
            ]
        )

    def classify(self, input: str) -> str:
        messages = self.prompt_template.format_messages(news=input)
        try:
            response = llm.invoke(messages)
            classification = response.content.strip().lower()
        except Exception as e:
            # 如果 llm.invoke 失败，返回默认分类
            print(f"Error invoking llm: {e}")
            return "other"
        
        # 确保返回的是有效的分类
        valid_types = ["ai_news", "model", "ai_product", "other"]
        if classification in valid_types:
            return classification
        # 如果返回的不是标准格式，尝试提取
        for valid_type in valid_types:
            if valid_type in classification:
                return valid_type
        # 默认返回 ai_news
        return "ai_news"

class AINewsAgent:
    def __init__(self):
        self.agent = create_agent(
            model=reason_llm,
            tools=[tavily_search],
            response_format=NewsInfo,
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", """
# Role
AI技术与市场情报官，具备全网信息检索与整合能力。

# Workflow
1. **阅读与提炼**：快速阅读新闻，提取核心事件。
2. **搜索与验证**：(如需)使用搜索工具查找该事件的背景信息。
3. **整合输出**：结合新闻本体与搜索结果，输出趋势分析。

# Goals
识别该事件在AI产业链中的位置（如：算力层、模型层、应用层）及潜在价值。

# Output Rules
- 直接输出结果，无需描述思考过程。
- 全文中文，严格限制在150字以内。

# Input News
{news}

# Output json format
{{"title": "...","content": "...","newstype": "..."}}
                """),
                ("human", "{news}"),
            ]
        )

    def parse(self, input: str, news_type: str):
        messages = self.prompt_template.format_messages(news=input)
        try:
            result = self.agent.invoke({"messages": messages})
            news_info = result["structured_response"]
            news_info.newstype = news_type
            return news_info
        except Exception as e:
            # 如果 agent.invoke 失败，返回 None
            print(f"Error invoking agent: {e}")
            return None

class ModelAgent:
    def __init__(self):
        self.agent = create_agent(
            model=reason_llm,
            tools=[tavily_search],
            response_format=NewsInfo,
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", """
# Role
AI技术主笔，善于将枯燥的技术参数转化为通俗易懂的市场洞察。

# Task
请阅读【模型新闻】，利用搜索工具补全核心参数（如参数量、上下文、开源协议），输出一份结构清晰、阅读流畅的分析简报。

# Focus (关注点)
1. **技术规格**：模型规模、架构特点、开源/闭源状态。
2. **实力定位**：在基准测试中对标谁（如GPT-4o/Claude 3.5）？胜出还是追平？
3. **行业涟漪**：对开发者成本、应用落地或竞争格局的具体影响。

# Constraints
- **字数**：严格控制在200字以内。
- **风格**：口语化专业叙述，拒绝机械罗列，确保逻辑通顺。

# Input News
{news}

# Output json format
{{"title": "...","content": "...","newstype": "..."}}
                """),
                ("human", "{news}"),
            ]
        )

    def parse(self, input: str, news_type: str):
        messages = self.prompt_template.format_messages(news=input)
        try:
            result = self.agent.invoke({"messages": messages})
            news_info = result["structured_response"]
            news_info.newstype = news_type
            return news_info
        except Exception as e:
            # 如果 agent.invoke 失败，返回 None
            print(f"Error invoking agent: {e}")
            return None

class AIProductAgent:
    def __init__(self):
        self.agent = create_agent(
            model=reason_llm,
            tools=[tavily_search],
            response_format=NewsInfo,
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", """
# Role
兼具新闻敏感度与产品思维的AI观察员。

# Task
请阅读【AI产品新闻】，利用搜索工具补充细节。你需要将“新闻事件”与“产品价值”紧密结合，输出一份既有时效性又有深度的简报。

# Analysis Logic (逻辑流)
1. **新闻焦点 (What's New)**：准确概括新闻核心事件。
2. **场景穿透 (So What)**：该更新具体解决了什么**新**痛点？或扩展了什么**新**使用场景？
3. **竞争/市场 (Market)**：该动作反映了怎样的市场趋势或竞争策略？

# Constraints
- **字数**：200字以内。
- **风格**：叙事流畅，逻辑紧凑，确保“新闻”与“产品”不割裂。

# Input News
{news}

# Output json format
{{"title": "...","content": "...","newstype": "..."}}
                """),
                ("human", "{news}"),
            ]
        )

    def parse(self, input: str, news_type: str):
        messages = self.prompt_template.format_messages(news=input)
        try:
            result = self.agent.invoke({"messages": messages})
            news_info = result["structured_response"]
            news_info.newstype = news_type
            return news_info
        except Exception as e:
            # 如果 agent.invoke 失败，返回 None
            print(f"Error invoking agent: {e}")
            return None

class ParseAgent:
    def __init__(self):
        self.classify_agent = ClassifyAgent()
        self.ai_news_agent = AINewsAgent()
        self.model_agent = ModelAgent()
        self.ai_product_agent = AIProductAgent()

    def parse(self, input: str):
        # 首先进行分类
        news_type = self.classify_agent.classify(input)
        
        # 根据分类结果路由到对应的 agent
        if news_type == "ai_news":
            return self.ai_news_agent.parse(input, news_type)
        elif news_type == "model":
            return self.model_agent.parse(input, news_type)
        elif news_type == "ai_product":
            return self.ai_product_agent.parse(input, news_type)
        elif news_type == "other":
            return None
        else:
            return None