import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, trim_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

# 1. Estado con Reducer (add_messages acumula/combina los mensajes)
class MemoryState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def chatbot_node(state: MemoryState):
    # 2. Trim/Manejo de mensajes: Mantiene los últimos N mensajes para optimizar tokens
    trimmed_msgs = trim_messages(
        state["messages"],
        max_tokens=1000,
        strategy="last",
        token_counter=len, # Contador simple por número de mensajes para el MVP
        start_on="human",
    )
    
    response = llm.invoke(trimmed_msgs)
    return {"messages": [response]}

# 3. Grafo con Checkpointer para Persistencia
builder = StateGraph(MemoryState)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Inyectamos el Checkpointer
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Pruebas de Persistencia por Thread ID
if __name__ == "__main__":
    config_thread_1 = {"configurable": {"thread_id": "usuario_jean"}}
    config_thread_2 = {"configurable": {"thread_id": "usuario_maria"}}

    print("--- Sesión Jean: Mensaje 1 ---")
    r1 = graph.invoke({"messages": [HumanMessage("Hola, me llamo Jean y vivo en Colombia.")]}, config=config_thread_1)
    print("Bot:", r1["messages"][-1].content)

    print("\n--- Sesión María: Mensaje 1 (Hilo Independiente) ---")
    r2 = graph.invoke({"messages": [HumanMessage("Hola, cuál es mi nombre?")]}, config=config_thread_2)
    print("Bot:", r2["messages"][-1].content)

    print("\n--- Sesión Jean: Mensaje 2 (Demostrando Persistencia de Memoria) ---")
    r3 = graph.invoke({"messages": [HumanMessage("¿Recuerdas cómo me llamo y dónde vivo?")]}, config=config_thread_1)
    print("Bot:", r3["messages"][-1].content)