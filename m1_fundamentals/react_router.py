import os
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# 1. Definición del Estado
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. Definición de la Tool
@tool
def get_weather(city: str) -> str:
    """Consulta el clima de una ciudad dada."""
    return f"El clima actual en {city} es soleado con 22°C."

tools = [get_weather]

# 3. Configuración del LLM (compatible con Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools)

# 4. Nodos
def router_node(state: AgentState):
    """Evalúa la entrada del usuario y decide si requiere herramientas."""
    return {}

def react_agent_node(state: AgentState):
    """Nodo principal del agente ReAct que decide responder o invocar herramientas."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

# 5. Bordes Condicionales (Conditional Edges)
def route_decision(state: AgentState) -> Literal["react_agent", "simple_chat"]:
    last_msg = state["messages"][-1].content.lower()
    if any(word in last_msg for word in ["clima", "weather", "temperatura"]):
        return "react_agent"
    return "simple_chat"

def should_continue(state: AgentState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

def simple_chat_node(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 6. Construcción del Grafo
builder = StateGraph(AgentState)

builder.add_node("router", router_node)
builder.add_node("react_agent", react_agent_node)
builder.add_node("tools", tool_node)
builder.add_node("simple_chat", simple_chat_node)

builder.add_edge(START, "router")
builder.add_conditional_edges("router", route_decision)
builder.add_conditional_edges("react_agent", should_continue)
builder.add_edge("tools", "react_agent")
builder.add_edge("simple_chat", END)

graph = builder.compile()

# Pruebas de ejecución
if __name__ == "__main__":
    print("--- Prueba 1: Ruta ReAct con Tool ---")
    res1 = graph.invoke({"messages": [HumanMessage("¿Cuál es el clima en Nariño?")]})
    print("Resultado:", res1["messages"][-1].content)

    print("\n--- Prueba 2: Ruta Chat Simple ---")
    res2 = graph.invoke({"messages": [HumanMessage("Dime un chiste corto de Ingeniero de Datos.")]})
    print("Resultado:", res2["messages"][-1].content)