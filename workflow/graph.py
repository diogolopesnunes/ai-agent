from typing_extensions import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from classifier.service import UrgencyClassifier
from generation.answer_service import AnswerService
from rag.context_budget import build_context
from rag.retriever import KnowledgeRetriever


class SupportState(TypedDict, total=False):
    trace_id: str
    question: str
    urgency: str
    classifier_confidence: float
    classifier_probabilities: dict
    retrieved_documents: list[dict]
    context: str
    context_tokens: int
    proposed_action: str | None
    requires_approval: bool
    answer: dict
    usage: dict | None
    error: str | None


classifier = UrgencyClassifier()
retriever = KnowledgeRetriever()
answer_service = AnswerService()


def validate_input(state: SupportState) -> SupportState:
    question = state.get("question", "").strip()

    if not question:
        return {"error": "A pergunta não pode estar vazia."}

    if len(question) > 4000:
        return {"error": "A pergunta ultrapassou o limite permitido."}

    return {
        "trace_id": state.get("trace_id") or uuid4().hex,
        "question": question,
        "error": None,
    }


def classify_urgency(state: SupportState) -> SupportState:
    if state.get("error"):
        return {}

    result = classifier.predict(state["question"])

    return {
        "urgency": result["urgencia"],
        "classifier_confidence": result["confianca"],
        "classifier_probabilities": result["probabilidades"],
    }


def retrieve_knowledge(state: SupportState) -> SupportState:
    if state.get("error"):
        return {}

    documents = retriever.search(state["question"])
    context, selected, token_count = build_context(documents)

    return {
        "retrieved_documents": selected,
        "context": context,
        "context_tokens": token_count,
    }


def decide_action(state: SupportState) -> SupportState:
    if state.get("error"):
        return {}

    confidence = state.get("classifier_confidence", 0.0)
    urgency = state.get("urgency", "baixa")
    documents = state.get("retrieved_documents", [])

    requires_human = confidence < 0.65 or urgency == "alta"

    if not documents:
        return {
            "requires_approval": False,
            "proposed_action": None,
        }

    if urgency == "alta":
        return {
            "requires_approval": True,
            "proposed_action": "create_ticket",
        }

    return {
        "requires_approval": requires_human,
        "proposed_action": None,
    }


def generate_answer(state: SupportState) -> SupportState:
    if state.get("error"):
        return {}

    documents = state.get("retrieved_documents", [])

    if not documents:
        return {
            "answer": {
                "answer": (
                    "Não encontrei evidência suficiente na base disponível. "
                    "O caso deve ser revisado por uma pessoa."
                ),
                "sources": [],
                "urgency": state.get("urgency", "indefinida"),
                "classifier_confidence": state.get(
                    "classifier_confidence",
                    0.0,
                ),
                "needs_human_review": True,
                "proposed_action": None,
            },
            "usage": None,
        }

    parsed, usage = answer_service.generate(
        question=state["question"],
        context=state["context"],
        sources=documents,
        urgency=state["urgency"],
        confidence=state["classifier_confidence"],
        proposed_action=state.get("proposed_action"),
    )

    return {
        "answer": parsed.model_dump(),
        "usage": usage.model_dump() if usage else None,
    }


def route_after_validation(state: SupportState) -> str:
    return "failure" if state.get("error") else "continue"


def failure_response(state: SupportState) -> SupportState:
    return {
        "answer": {
            "answer": state.get("error", "Falha desconhecida."),
            "sources": [],
            "urgency": "indefinida",
            "classifier_confidence": 0.0,
            "needs_human_review": True,
            "proposed_action": None,
        }
    }


builder = StateGraph(SupportState)
builder.add_node("validate", validate_input)
builder.add_node("classify", classify_urgency)
builder.add_node("retrieve", retrieve_knowledge)
builder.add_node("decide", decide_action)
builder.add_node("generate", generate_answer)
builder.add_node("failure", failure_response)

builder.add_edge(START, "validate")
builder.add_conditional_edges(
    "validate",
    route_after_validation,
    {
        "continue": "classify",
        "failure": "failure",
    },
)
builder.add_edge("classify", "retrieve")
builder.add_edge("retrieve", "decide")
builder.add_edge("decide", "generate")
builder.add_edge("generate", END)
builder.add_edge("failure", END)

support_graph = builder.compile()