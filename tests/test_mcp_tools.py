from unittest.mock import MagicMock, patch

import pytest

from mcp_server.server import create_ticket, get_ticket


@pytest.fixture(autouse=True)
def mock_db():
    """Mock the database connection for all tests in this module.

    The server uses ``with get_connection() as connection:``, so the mock
    must support the context-manager protocol.
    """
    conn_mock = MagicMock()
    conn_mock.__enter__.return_value = conn_mock
    conn_mock.__exit__.return_value = None
    with patch("mcp_server.server.get_connection", return_value=conn_mock):
        yield conn_mock


def test_create_ticket_success(mock_db):
    result = create_ticket(
        title="Sistema fora do ar",
        description="Todos os usuarios estao sem acesso ao sistema principal",
        urgency="alta",
    )

    assert result["id"].startswith("TKT-")
    assert result["status"] == "aberto"
    assert result["urgency"] == "alta"


def test_create_ticket_title_too_short():
    with pytest.raises(ValueError, match="O título deve possuir pelo menos 5 caracteres"):
        create_ticket(title="abc", description="Descricao valida para o ticket", urgency="baixa")


def test_create_ticket_description_too_short():
    with pytest.raises(ValueError, match="A descrição deve possuir pelo menos 10 caracteres"):
        create_ticket(title="Titulo valido", description="curta", urgency="media")


def test_create_ticket_invalid_urgency():
    with pytest.raises(ValueError, match="Urgência inválida"):
        create_ticket(
            title="Titulo valido",
            description="Descricao valida para o ticket de teste",
            urgency="urgente",
        )


@pytest.mark.parametrize("urgency", ["baixa", "media", "alta"])
def test_create_ticket_valid_urgencies(mock_db, urgency):
    result = create_ticket(
        title="Problema no sistema",
        description="Usuarios relatam lentidao no modulo de relatorios",
        urgency=urgency,
    )

    assert result["urgency"] == urgency
    assert result["status"] == "aberto"


def test_create_ticket_strips_whitespace(mock_db):
    result = create_ticket(
        title="  Sistema fora do ar  ",
        description="  Todos os usuarios estao sem acesso  ",
        urgency="  ALTA  ",
    )

    assert result["urgency"] == "alta"


def test_get_ticket_found(mock_db):
    # Simulate a sqlite3.Row-like object that can be passed to dict()
    row = {
        "id": "TKT-ABC12345",
        "title": "Sistema fora do ar",
        "description": "Descricao do problema",
        "urgency": "alta",
        "status": "aberto",
        "created_at": "2024-01-01 10:00:00",
    }
    mock_db.execute.return_value.fetchone.return_value = row

    result = get_ticket("TKT-ABC12345")

    assert result["found"] is True
    assert result["id"] == "TKT-ABC12345"
    assert result["title"] == "Sistema fora do ar"
    assert result["urgency"] == "alta"


def test_get_ticket_not_found(mock_db):
    mock_db.execute.return_value.fetchone.return_value = None

    result = get_ticket("TKT-INEXISTENTE")

    assert result["found"] is False
    assert result["id"] == "TKT-INEXISTENTE"


def test_get_ticket_strips_input_for_query(mock_db):
    mock_db.execute.return_value.fetchone.return_value = None

    result = get_ticket("  TKT-12345  ")

    # Verify the query was called with the stripped id
    call_args = mock_db.execute.call_args[0]
    assert call_args[1] == ("TKT-12345",)
    # The function returns the original unstripped id when not found
    assert result["id"] == "  TKT-12345  "
    assert result["found"] is False
