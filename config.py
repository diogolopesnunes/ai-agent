from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    api_base: str
    openai_api_key: str
    model_name: str
    embedding_model: str
    rag_top_k: int
    rag_min_score: float
    max_context_tokens: int
    classifier_confidence_min: float


def load_settings() -> Settings:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não foi configurada.")

    api_base = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

    settings = Settings(
        api_base=api_base,
        openai_api_key=api_key,
        model_name=os.getenv("OPENAI_MODEL", "gpt-5.4-nano"),
        embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        ),
        rag_top_k=int(os.getenv("RAG_TOP_K", "5")),
        rag_min_score=float(os.getenv("RAG_MIN_SCORE", "0.45")),
        max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "3500")),
        classifier_confidence_min=float(
            os.getenv("CLASSIFIER_CONFIDENCE_MIN", "0.65")
        ),
    )

    # Export to environment so libraries (langchain, openai) pick up the base and key
    if settings.api_base:
        os.environ.setdefault("OPENAI_API_BASE", settings.api_base)
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

    return settings