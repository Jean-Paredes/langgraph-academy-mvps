import os
import warnings
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.runtime import get_store  # Importante para acceder al store

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

class LongTermState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def memory_node(state: LongTermState):
    # Recuperamos el store del contexto de ejecución de LangGraph
    store = get_store()
    user_id = state["user_id"]
    namespace = ("users", user_id, "memories")
    
    # 1. Recuperar preferencias guardadas previamente
    memories = store.search(namespace)
    pref_text = "\n".join([m.value.get("preference", "") for m in memories]) if memories else "Sin preferencias guardadas."
    
    # 2. Generar respuesta con contexto de largo plazo
    prompt = f"Preferencias del usuario conocidas:\n{pref_text}\n\nResponde a: {state['messages'][-1].content}"
    response = llm.invoke(prompt)
    
    # 3. Guardar nueva preferencia si se menciona en el mensaje
    last_msg = state["messages"][-1].content.lower()
    if "me gusta" in last_msg or "prefiero" in last_msg:
        store.put(namespace, key="pref_key", value={"preference": state["messages"][-1].content})
        
    return {"messages": [response]}

builder = StateGraph(LongTermState)
builder.add_node("agent", memory_node)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

checkpointer = InMemorySaver()
in_memory_store = InMemoryStore()

graph = builder.compile(checkpointer=checkpointer, store=in_memory_store)

if __name__ == "__main__":
    # Hilo 1: El usuario comparte un gusto
    config1 = {"configurable": {"thread_id": "thread_1"}}
    print("--- Sesión 1 (Hilo 1): Guardando preferencia ---")
    r1 = graph.invoke(
        {"messages": [HumanMessage("Hola, me gusta programar en Python con café.")], "user_id": "usr_101"},
        config=config1
    )
    print("Bot:", r1["messages"][-1].content)

    # Hilo 2: Se inicia un nuevo chat independiente, pero se recupera la memoria cross-thread por user_id
    config2 = {"configurable": {"thread_id": "thread_2"}}
    print("\n--- Sesión 2 (Hilo 2 - Nuevo Chat): Recuperando memoria de largo plazo ---")
    r2 = graph.invoke(
        {"messages": [HumanMessage("¿Recuerdas qué bebida o lenguaje prefiero?")], "user_id": "usr_101"},
        config=config2
    )
    print("Bot:", r2["messages"][-1].content)