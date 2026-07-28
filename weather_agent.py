"""
LangGraph-based weather agent.

Flow: user message -> LLM (Groq, Google, OpenRouter, OpenAI) decides whether to call weather tool
-> tool executes (Open-Meteo, no API Key needed) -> LLM formats a final answer.
"""

from typing import Annotated, TypedDict

import os
import requests
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Paartly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog", 
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 
}

def get_weather(city: str) -> str:
    """Get the current weather for a given city name.
    
    Args:
        city: The name of the city, e.g. "Hyderabad" or "London"
    """
    geo_resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count":1},
        timeout=10
    )
    geo_data = geo_resp.json()
    if not geo_data.get("results"):
        return f"Could not find a location matching '{city}'."

    loc = geo_data["results"][-1]
    lat, lon = loc["latitude"],loc["longitude"]
    resolved_name = f"{loc['name']}, {loc.get('country', '')}"

    w_resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current":"temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "temperature_unit":"celsius"
        },
        timeout=10,
    )
    current = w_resp.json().get("current",{})
    condition = WEATHER_CODES.get(current.get("weather_code"),"Unknown")

    return (
        f"Weather in {resolved_name}: {condition}. "
        f"Temperature {current.get('temperature_2m')}C,"
        f"Humidity {current.get('relative_humidity_2m')}%, "
        f"Wind {current.get('wind_speed_10m')} km/h."

    )

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph():
    groq_model = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
    llm = ChatGroq(model=groq_model, temperature=0)
    tools = [get_weather]
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: AgentState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages":[response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()

weather_graph = build_graph()

