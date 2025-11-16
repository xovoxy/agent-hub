from dataclasses import dataclass
from langchain_core.prompts import ChatPromptTemplate
from src.utils import llm, reason_llm
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field


tavily_search = TavilySearch(max_results=5)

class NewsInfo(BaseModel):
    title: str = Field(description="The title of the news article")
    content: str = Field(description="The final, in-depth interpretation report, don't include the title in the content")

class ParseAgent:
    def __init__(self):
        self.agent = create_agent(
            model=reason_llm,
            tools=[tavily_search],
            debug=True,
            response_format=NewsInfo,
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", """
                Role and Directive

You are an expert news analyst and fact-checker. Your task is to receive the news article below, execute a rigorous analysis process internally, and deliver only a final, coherent, insightful, and fact-checked interpretation report.

Internal Workflow (Do not show any of this in your output)

Silent Identification: Internally identify the article's core claims, key individuals, organizations, data points, and potential biases.

Web Search for Verification & Enrichment:

Fact-Check: Verify the key facts and data presented in the article.

Contextual Research: Search for the historical background, root causes, and relevant policies related to the event.

Multiple Perspectives: Find different viewpoints or more detailed coverage from other reputable sources and experts.

Synthesis and Analysis: Compare and integrate your external search findings with the original article's content to form a comprehensive and objective understanding that goes beyond the surface-level reporting.

News Article for Analysis

{news}

Final Output Requirement

Output in Chinese.

Output the Interpretation Only. Absolutely do not show your internal workflow, the keywords you identified, your search steps, or raw search findings.

Content Structure: Your report should naturally integrate the following aspects into a fluid, well-written essay, not a list of bullet points:

The Core Event and Its Significance: State the essence of the event and why it is important.

Deeper Context and Background: Explain the underlying reasons and historical context for the event (this is where the value of your web search should be evident).

Potential Impact and Outlook: Analyze the likely short-term and long-term consequences of the event.

Tone: Professional, objective, and highly analytical.
                """),
                ("human", "{news}"),
            ]
        )

    def parse(self, input: str):
        # Format the prompt with the news input
        messages = self.prompt_template.format_messages(news=input)
        
        # Invoke the agent with the formatted messages
        result = self.agent.invoke({"messages": messages})
        
        # Extract the final response content
        return result["structured_response"]
        
        