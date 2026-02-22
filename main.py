from dotenv import load_dotenv
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
    tools=[search]
)

def main():
    print("Hello from langchain-search-agent!")

    result = agent.invoke({"messages": [HumanMessage(content="Which dollar exchange rate in Ukraine?")]})
        
    

    print("Result:", result)


if __name__ == "__main__":
    main()
