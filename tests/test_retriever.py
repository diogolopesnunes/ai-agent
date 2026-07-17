from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.retriever import KnowledgeRetriever


@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings to avoid needing real environment variables."""
    with patch("rag.retriever.load_settings") as mock_load:
        settings = MagicMock()
        settings.embedding_model = "text-embedding-3-small"
        settings.rag_top_k = 5
        settings.rag_min_score = 0.45
        mock_load.return_value = settings
        yield settings


@pytest.fixture
def mock_chroma_dir():
    """Ensure Path.exists() returns True for all tests that need it."""
    with patch.object(Path, "exists", return_value=True):
        yield


def test_retriever_raises_error_when_index_missing():
    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="Índice não encontrado"):
            KnowledgeRetriever()


def test_search_returns_empty_list_when_no_results(mock_chroma_dir, mock_settings):
    with patch("rag.retriever.Chroma") as chroma_cls:
        store = MagicMock()
        store.similarity_search_with_relevance_scores.return_value = []
        chroma_cls.return_value = store

        retriever = KnowledgeRetriever()
        results = retriever.search("consulta sem resultados")

    assert results == []


def test_search_filters_by_relevance_score(mock_chroma_dir, mock_settings):
    doc_high = MagicMock()
    doc_high.page_content = "Documento relevante"
    doc_high.metadata = {"source_name": "manual.pdf", "chunk_id": 1}

    doc_low = MagicMock()
    doc_low.page_content = "Documento irrelevante"
    doc_low.metadata = {"source_name": "manual.pdf", "chunk_id": 2}

    with patch("rag.retriever.Chroma") as chroma_cls:
        store = MagicMock()
        store.similarity_search_with_relevance_scores.return_value = [
            (doc_high, 0.85),
            (doc_low, 0.20),  # below min_score of 0.45
        ]
        chroma_cls.return_value = store

        retriever = KnowledgeRetriever()
        results = retriever.search("consulta de teste")

    assert len(results) == 1
    assert results[0]["content"] == "Documento relevante"
    assert results[0]["score"] == 0.85


def test_search_returns_all_results_above_threshold(mock_chroma_dir, mock_settings):
    docs = [
        (MagicMock(page_content="Resultado A", metadata={"source_name": "doc1.md", "chunk_id": 0}), 0.9),
        (MagicMock(page_content="Resultado B", metadata={"source_name": "doc2.md", "chunk_id": 1}), 0.75),
        (MagicMock(page_content="Resultado C", metadata={"source_name": "doc3.md", "chunk_id": 2}), 0.5),
    ]

    with patch("rag.retriever.Chroma") as chroma_cls:
        store = MagicMock()
        store.similarity_search_with_relevance_scores.return_value = docs
        chroma_cls.return_value = store

        retriever = KnowledgeRetriever()
        results = retriever.search("consulta")

    assert len(results) == 3


def test_search_respects_top_k_parameter(mock_chroma_dir, mock_settings):
    mock_settings.rag_top_k = 2

    docs = [
        (MagicMock(page_content="A", metadata={"source_name": "a.md", "chunk_id": 0}), 0.9),
        (MagicMock(page_content="B", metadata={"source_name": "b.md", "chunk_id": 1}), 0.8),
        (MagicMock(page_content="C", metadata={"source_name": "c.md", "chunk_id": 2}), 0.7),
    ]

    with patch("rag.retriever.Chroma") as chroma_cls:
        store = MagicMock()
        store.similarity_search_with_relevance_scores.return_value = docs[:2]
        chroma_cls.return_value = store

        retriever = KnowledgeRetriever()
        results = retriever.search("consulta")

    assert len(results) == 2
    assert results[0]["content"] == "A"
    assert results[1]["content"] == "B"


def test_search_uses_source_name_metadata(mock_chroma_dir, mock_settings):
    doc = MagicMock()
    doc.page_content = "Conteudo do documento"
    doc.metadata = {"source_name": "politica_senhas.md", "chunk_id": 3}

    with patch("rag.retriever.Chroma") as chroma_cls:
        store = MagicMock()
        store.similarity_search_with_relevance_scores.return_value = [(doc, 0.95)]
        chroma_cls.return_value = store

        retriever = KnowledgeRetriever()
        results = retriever.search("politica de senhas")

    assert results[0]["source"] == "politica_senhas.md"
    assert results[0]["chunk_id"] == 3


def test_search_falls_back_to_source_metadata(mock_chroma_dir, mock_settings):
    doc = MagicMock()
    doc.page_content = "Conteudo"
    doc.metadata = {"source": "backup_source.txt", "chunk_id": 5}

    with patch("rag.retriever.Chroma") as chroma_cls:
        store = MagicMock()
        store.similarity_search_with_relevance_scores.return_value = [(doc, 0.8)]
        chroma_cls.return_value = store

        retriever = KnowledgeRetriever()
        results = retriever.search("consulta")

    assert results[0]["source"] == "backup_source.txt"


def test_search_uses_desconhecida_when_no_source_in_metadata(mock_chroma_dir, mock_settings):
    doc = MagicMock()
    doc.page_content = "Conteudo sem metadata de origem"
    doc.metadata = {"chunk_id": 7}

    with patch("rag.retriever.Chroma") as chroma_cls:
        store = MagicMock()
        store.similarity_search_with_relevance_scores.return_value = [(doc, 0.6)]
        chroma_cls.return_value = store

        retriever = KnowledgeRetriever()
        results = retriever.search("consulta")

    assert results[0]["source"] == "desconhecida"


def test_search_returns_full_metadata(mock_chroma_dir, mock_settings):
    doc = MagicMock()
    doc.page_content = "Conteudo completo"
    doc.metadata = {"source_name": "doc.pdf", "chunk_id": 10, "page": 5}

    with patch("rag.retriever.Chroma") as chroma_cls:
        store = MagicMock()
        store.similarity_search_with_relevance_scores.return_value = [(doc, 0.7)]
        chroma_cls.return_value = store

        retriever = KnowledgeRetriever()
        results = retriever.search("consulta")

    assert results[0]["metadata"]["source_name"] == "doc.pdf"
    assert results[0]["metadata"]["chunk_id"] == 10
    assert results[0]["metadata"]["page"] == 5


def test_search_score_is_float(mock_chroma_dir, mock_settings):
    doc = MagicMock()
    doc.page_content = "Conteudo"
    doc.metadata = {"source_name": "doc.md", "chunk_id": 1}

    with patch("rag.retriever.Chroma") as chroma_cls:
        store = MagicMock()
        store.similarity_search_with_relevance_scores.return_value = [(doc, 0.753)]
        chroma_cls.return_value = store

        retriever = KnowledgeRetriever()
        results = retriever.search("consulta")

    assert isinstance(results[0]["score"], float)
    assert results[0]["score"] == 0.753