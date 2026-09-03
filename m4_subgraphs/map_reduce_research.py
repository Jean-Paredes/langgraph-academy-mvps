import os
import warnings
from typing import Annotated, TypedDict
import operator
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

# --- Sub-grafo: Analista Individual (Map) ---
class SubGraphState(TypedDict):
    topic: str
    analysis: str

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def analyze_topic_node(state: SubGraphState) -> SubGraphState:
    response = llm.invoke(f"Proporciona un dato clave sobre el siguiente tema: {state['topic']}")
    return {"analysis": response.content}

sub_builder = StateGraph(SubGraphState)
sub_builder.add_node("analyze", analyze_topic_node)
sub_builder.add_edge(START, "analyze")
sub_builder.add_edge("analyze", END)
sub_graph = sub_builder.compile()

# --- Grafo Principal: Coordinador (Reduce) ---
class MainState(TypedDict):
    topics: list[str]
    analyses: Annotated[list[str], operator.add]
    final_report: str

def map_analyses_node(state: MainState) -> MainState:
    # Simulación de llamadas en paralelo al sub-grafo para cada tema
    results = []
    for t in state["topics"]:
        sub_res = sub_graph.invoke({"topic": t})
        results.append(f"[{t.upper()}]: {sub_res['analysis']}")
    return {"analyses": results}

def reduce_report_node(state: MainState) -> MainState:
    combined_analyses = "\n".join(state["analyses"])
    summary = llm.invoke(f"Resume los siguientes análisis en un párrafo:\n{combined_analyses}")
    return {"final_report": summary.content}

main_builder = StateGraph(MainState)
main_builder.add_node("map_node", map_analyses_node)
main_builder.add_node("reduce_node", reduce_report_node)

main_builder.add_edge(START, "map_node")
main_builder.add_edge("map_node", reduce_node)
main_builder.add_edge("reduce_node", END)

main_graph = main_builder.compile()

if __name__ == "__main__":
    print("--- Ejecutando Map-Reduce Research Assistant ---")
    input_data = {"topics": ["Inteligencia Artificial", "Computación Cuántica", "Biotecnología"]}
    res = main_graph.invoke(input_data)
    
    print("\nResultados del Map (Análisis individuales):")
    for a in res["analyses"]:
        print(f"- {a}\n")
        
    print("Reporte Consolidado (Reduce):")
    print(res["final_report"])