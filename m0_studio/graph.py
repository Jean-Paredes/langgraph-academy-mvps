from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    message: str

def hello_node(state: State) -> State:
    return {"message": f"Hello, {state['message']}! Welcome to LangGraph Studio."}

builder = StateGraph(State)
builder.add_node("hello", hello_node)
builder.add_edge(START, "hello")
builder.add_edge("hello", END)

graph = builder.compile()