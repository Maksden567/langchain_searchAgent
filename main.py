from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

load_dotenv() 

tavily = TavilyClient()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro-preview",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

class Source(BaseModel):
    """A source that agent found during the search."""
    url: str = Field(..., description="The URL of the source")


class AgentResponse(BaseModel):
    """The response from the agent."""

    sources: List[Source] = Field(
        ..., description="A list of sources that the agent found during the search"
    )

@tool("search")
def search(query: str) -> str:
    """
    Tool help function to search the web.
    Args: query: The search query.
    Returns: A string containing the search results.
    """
    print(f"Search results for '{query}'")
    return tavily.search(query=query)

agent = create_agent(
    model=llm,
    tools=[search],
    response_format=AgentResponse,
)

def main():
    print("Hello from langchain-search-agent!")

    result = agent.invoke({"messages": [HumanMessage(content="3 active jobs AI in Ukraine on dou")]})
        
    

    print("Result:", result)


if __name__ == "__main__":
    main()
