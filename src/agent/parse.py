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
你是一个AI新闻分类助手。请根据新闻的“核心信息点”进行单标签分类。

分类定义：
1. ai_news：AI行业资讯、公司动态、融资并购、政策监管、市场趋势、合作生态、基础设施动态。
2. model：AI模型本身的发布、升级、技术路线、参数、上下文、评测、能力对比、训练/推理成本等。
3. ai_product：面向用户或企业的AI产品、应用、功能更新、定价、开放范围、使用场景、产品化落地。
4. other：与AI关系弱，或无法明确归入上述三类的内容。

判定规则：
- 优先看新闻主角和主结论，不看表面关键词。
- 如果核心在“模型能力/参数/基准/架构”，判为 model。
- 如果核心在“产品功能/用户场景/商业化入口/定价”，判为 ai_product。
- 如果核心在“行业事件/政策监管/公司动作/市场影响”，判为 ai_news。
- AI相关政策、监管、政府动作，若核心仍与AI行业有关，判为 ai_news，不要因为涉及政治或监管就判为 other。
- 只有在内容明显与AI无关，或信息过少无法判断时，才判为 other。

输出要求：
- 只返回一个类型标识符。
- 只能返回：ai_news、model、ai_product、other。
- 不要输出解释，不要输出标点或多余文本。
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
你是AI科技媒体编辑，文风接近机器之心：专业、克制、信息密度高，不写夸张标题，不用营销腔。

任务要求：
1. 先提炼新闻的核心事件。
2. 如有必要，使用搜索工具补充关键背景，但只采用可验证信息。
3. 输出时重点回答两件事：
   - 这条新闻到底发生了什么；
   - 它对AI行业、公司竞争或产业链意味着什么。

重点关注：
- 事件主体是谁，做了什么动作。
- 事件属于政策监管、公司战略、融资合作、基础设施还是市场动态。
- 该事件对行业格局、商业化进展或上下游生态的实际影响。

写作约束：
- 只基于新闻内容和可验证搜索结果，不补造未披露事实。
- 不要输出思考过程、搜索过程、来源列表。
- 全文中文。
- title：写成科技媒体风格标题，简洁、准确、有信息量，突出主角、动作和结果，不用感叹句，不故意制造悬念。
- content：写成1段紧凑中文，先交代事件，再说明影响，判断要克制但要有信息增量。
- 风格：像机器之心新闻快讯，专业、冷静、利落，不写空泛判断。
- content 严格控制在150字以内。
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
你是AI模型方向的科技媒体编辑，文风接近机器之心：既讲清技术升级，也讲清行业定位，语言专业、克制。

任务要求：
1. 概括这次模型更新或发布的核心信息。
2. 如有必要，使用搜索工具补充已公开、可验证的模型信息。
3. 输出时重点覆盖以下内容中的最关键两到三项：
   - 模型能力或技术升级点；
   - 已明确披露的规格信息，如上下文、开源状态、价格、延迟、评测表现；
   - 该模型在行业中的定位，以及对开发者或竞争格局的意义。

重点规则：
- 只写新闻和可信搜索结果中已经明确的信息。
- 对未披露的参数、架构细节、训练方法、成本数据，不要猜测，不要脑补。
- 如果对标对象或测试结论不明确，不要强行写“超过”或“追平”。
- 不要堆砌参数，优先解释“这次升级真正重要在哪里”。

写作约束：
- 不要输出思考过程、搜索过程、来源列表。
- 全文中文，表述专业但自然。
- title：写成科技媒体风格标题，突出模型名称、关键升级、评测结果或定位变化，不写情绪化措辞。
- content：先写更新点，再写定位或影响，避免机械罗列，让读者快速理解这次发布的技术价值和行业意义。
- 风格：像机器之心在写模型快讯，专业清楚，但不要写成论文摘要或厂商宣传稿。
- content 严格控制在200字以内。
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
你是AI产品观察方向的科技媒体编辑，文风接近机器之心：既讲清产品动作，也讲清它的用户价值和竞争意义。

任务要求：
1. 概括产品这次更新、发布或开放的核心动作。
2. 如有必要，使用搜索工具补充可验证的产品背景信息。
3. 输出时重点回答以下问题中的最关键两到三项：
   - 产品新增了什么能力或入口；
   - 面向谁使用，解决了什么场景或痛点；
   - 与旧版本、现有方案或竞品相比，差异化价值是什么；
   - 这次动作对产品增长、商业化或市场竞争意味着什么。

重点规则：
- 只基于新闻和可验证搜索结果写作，不补造未披露功能或数据。
- 不要泛泛谈“行业趋势”，除非它与本次产品动作直接相关。
- 优先写清用户价值、使用门槛、开放范围、定价变化或适用场景。

写作约束：
- 不要输出思考过程、搜索过程、来源列表。
- 全文中文，逻辑紧凑，避免空话。
- title：写成科技媒体风格标题，突出产品名、动作和核心变化，避免口号式表达。
- content：先写更新，再写用户价值或竞争意义，尽量交代清楚适用对象、场景变化和竞争信号。
- 风格：像机器之心产品快讯，信息清楚、判断克制、少套话。
- content 严格控制在200字以内。
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
