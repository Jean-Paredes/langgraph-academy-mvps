import os
import warnings
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv
import operator
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

class HitlState(TypedDict):
    query: str
    approval_status: str
    result: str

def process_node(state: HitlState) -> Command[Literal["execute_action", END]]:
    query = state["query"]
    
    # 1. Interrupción si la acción es crítica
    admin_decision = interrupt({
        "question": f"¿Apruebas ejecutar la acción crítica para la consulta '{query}'?",
        "allowed": ["approve", "reject"]
    })
    
    if admin_decision == "approve":
        return Command(
            update={"approval_status": "approved"},
            goto="execute_action"
        )
    else:
        return Command(
            update={"approval_status": "rejected", "result": "Operación cancelada por el usuario."},
            goto=END
        )

def execute_action_node(state: HitlState) -> HitlState:
    return {"result": f"Acción ejecutada con éxito para: {state['query']}"}

# Construcción del grafo
builder = StateGraph(HitlState)
builder.add_node("process", process_node)
builder.add_node("execute_action", execute_action_node)

builder.add_edge(START, "process")
builder.add_edge("execute_action", END)

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "hitl_thread_1"}}
    
    print("--- 1. Iniciando ejecución (Provocando Interrupt) ---")
    res1 = graph.invoke({"query": "Eliminar registros obsoletos"}, config=config)
    
    if "__interrupt__" in res1:
        interrupt_info = res1["__interrupt__"][0].value
        print(f"⚠️ PAUSA DETECTADA: {interrupt_info['question']}")
        
        print("\n--- 2. Reanudando ejecución (Aprobación del humano) ---")
        res2 = graph.invoke(Command(resume="approve"), config=config)
        print("Resultado Final:", res2["result"])