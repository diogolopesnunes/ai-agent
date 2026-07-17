from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import load_settings

CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "easy_support_docs"


class KnowledgeRetriever:
    def __init__(self) -> None:
        settings = load_settings()

        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                "Índice não encontrado. Execute rag/index_documents.py."
            )

        self.settings = settings
        self.store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            embedding_function=OpenAIEmbeddings(
                model=settings.embedding_model
            ),
        )

    def search(self, query: str) -> list[dict]:
        results = self.store.similarity_search_with_relevance_scores(
            query,
            k=self.settings.rag_top_k,
        )

        filtered = []

        for document, score in results:
            if score < self.settings.rag_min_score:
                continue

            filtered.append({
                "content": document.page_content,
                "score": float(score),
                "source": document.metadata.get(
                    "source_name",
                    document.metadata.get("source", "desconhecida"),
                ),
                "chunk_id": document.metadata.get("chunk_id"),
                "metadata": document.metadata,
            })

        return filtered