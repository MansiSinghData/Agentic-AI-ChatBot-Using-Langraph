from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage


load_dotenv() 

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

# openai_client = ChatOpenAI(model="gpt-4o-mini", api_key=OPEN_AI_API_KEY)
# groq_client = ChatGroq(model="llama-3.3-70b-versatile",api_key=GROQ_API_KEY)

search_tool=TavilySearchResults(api_key=TAVILY_API_KEY,max_results=2)



def get_response_from_agent(llm_id,query,system_prompt,provider,allow_search=False):
    if provider=="Groq":
        llm=ChatGroq(model=llm_id,api_key=GROQ_API_KEY)
    elif provider=="OpenAI":
        llm=ChatOpenAI(model=llm_id, api_key=OPEN_AI_API_KEY)    

    agent = create_react_agent(
        model=llm,
        tools=[search_tool] if allow_search else [],prompt=system_prompt)

    state={"messages":query}
    response=agent.invoke(state)
    messages=response["messages"]
    ai_message = [message.content for message in messages if isinstance(message, AIMessage)]
    return ai_message[-1]
