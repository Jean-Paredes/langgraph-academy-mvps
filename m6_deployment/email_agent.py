import os
import uuid
import warnings
from typing import Literal, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

# Esquemátizado Pydantic para structured_output con Gemini
class EmailClassification(BaseModel):
    intent: Literal["question", "bug", "billing", "feature", "complex"] = Field(
        description="Intención principal del correo"
    )
    urgency: Literal["low", "medium", "high", "critical"] = Field(
        description="Nivel de urgencia"
    )
    topic: str = Field(description="Tema principal")
    summary: str = Field(description="Resumen corto")

class EmailAgentState(TypedDict):
    email_content: str
    sender_email: str
    email_id: str
    classification: dict | None
    ticket_id: str | None
    search_results: list[str] | None
    draft_response: str | None

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def read_email(state: EmailAgentState) -> EmailAgentState:
    """Extrae y valida el contenido inicial."""
    return {}

def classify_intent(state: EmailAgentState) -> EmailAgentState:
    """Clasifica el correo de forma estructurada."""
    structured_llm = llm.with_structured_output(EmailClassification)
    prompt = f"Analiza este correo:\nContenido: {state['email_content']}\nRemitente: {state['sender_email']}"
    res = structured_llm.invoke(prompt)
    return {"classification": res.dict()}

def search_documentation(state: EmailAgentState) -> EmailAgentState:
    """Busca información en la base de conocimientos."""
    return {"search_results": ["Doc 1: Política de reembolsos", "Doc 2: Preguntas frecuentes"]}

def bug_tracking(state: EmailAgentState) -> EmailAgentState:
    """Genera un ticket de soporte si aplica."""
    return {"ticket_id": f"BUG_{uuid.uuid4().hex[:6]}"}

def write_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    """Genera el borrador y enruta según urgencia."""
    classification = state.get("classification", {})
    prompt = f"Redacta una respuesta breve y profesional para:\n{state['email_content']}"
    res = llm.invoke(prompt)
    
    needs_review = classification.get("urgency") in ["high", "critical"] or classification.get("intent") == "billing"
    goto = "human_review" if needs_review else "send_reply"
    
    return Command(
        update={"draft_response": res.content},
        goto=goto
    )

def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
    """Pausa execution mediante interrupt para supervisión humana."""
    decision = interrupt({
        "action": "Aprobar o editar respuesta",
        "email_id": state["email_id"],
        "draft": state.get("draft_response")
    })
    
    if isinstance(decision, dict) and decision.get("approved"):
        return Command(
            update={"draft_response": decision.get("edited_response", state["draft_response"])},
            goto="send_reply"
        )
    return Command(update={}, goto=END)

def send_reply(state: EmailAgentState) -> EmailAgentState:
    """Envía la respuesta final."""
    print(f"📧 Respuesta enviada: {state.get('draft_response')[:50]}...")
    return {}

# Construcción del Grafo
builder = StateGraph(EmailAgentState)
builder.add_node("read_email", read_email)
builder.add_node("classify_intent", classify_intent)
builder.add_node("search_documentation", search_documentation)
builder.add_node("bug_tracking", bug_tracking)
builder.add_node("write_response", write_response)
builder.add_node("human_review", human_review)
builder.add_node("send_reply", send_reply)

builder.add_edge(START, "read_email")
builder.add_edge("read_email", "classify_intent")
builder.add_edge("classify_intent", "search_documentation")
builder.add_edge("classify_intent", "bug_tracking")
builder.add_edge("search_documentation", "write_response")
builder.add_edge("bug_tracking", "write_response")
builder.add_edge("send_reply", END)

# Checkpointer requerido para despliegue y persistencia por hilo
memory = InMemorySaver()
app = builder.compile(checkpointer=memory)