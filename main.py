"""FastAPI server that exposes the Langgraph weather agent over HTTP."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from weather_agent import weather_graph

app = FastAPI(title="Weather Agent API")

# Allow the React dev server (and others) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
def health():
    return {"status":"ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, details="message must not empty")

    result = weather_graph.invoke({"messages":[HumanMessage(content=req.message)]})
    final_message = result["messages"][-1]
    return ChatResponse(reply=final_message.content)