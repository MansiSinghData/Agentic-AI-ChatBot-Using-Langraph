from pydantic import BaseModel
from typing import List
from fastapi import FastAPI
from AI_Agent import get_response_from_agent
import uvicorn

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

allowed_model_list=["gpt-4o-mini","llama-3.3-70b-versatile","llama3-70b-8192","mixtral-8x7b-32768"]

app=FastAPI(title="Langraph Agentic AI Chatbot")

@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    Endpoint to handle chat requests. It takes a RequestState object as input and returns a response based on the agent's processing of the input messages."""
    if request.model_name not in allowed_model_list:
        return {"error": "Model not allowed. Please choose from the allowed model list."}
    
    response = get_response_from_agent(
        llm_id=request.model_name,
        query=request.messages,
        system_prompt=request.system_prompt,
        provider=request.model_provider,
        allow_search=request.allow_search
    )
    return {"response": response}


if __name__=='__main__':
    uvicorn.run(app, host="127.0.0.1", port=9000)

