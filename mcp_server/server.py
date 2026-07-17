import logging
from pathlib import Path
import sqlite3
import sys
from uuid import uuid4

from mcp.server.fastmcp import FastMCP

DB_PATH = Path("data/support.db")

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(message)s",
)

mcp = FastMCP("easy-support-mcp")


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                urgency TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


@mcp.tool()
def create_ticket(title: str, description: str, urgency: str) -> dict:
    """Cria um chamado após aprovação explícita no cliente."""
    title = title.strip()
    description = description.strip()
    urgency = urgency.strip().lower()

    if len(title) < 5:
        raise ValueError("O título deve possuir pelo menos 5 caracteres.")

    if len(description) < 10:
        raise ValueError("A descrição deve possuir pelo menos 10 caracteres.")

    if urgency not in {"baixa", "media", "alta"}:
        raise ValueError("Urgência inválida.")

    ticket_id = f"TKT-{uuid4().hex[:8].upper()}"

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO tickets (id, title, description, urgency, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ticket_id, title, description, urgency, "aberto"),
        )

    logging.info("ticket_created id=%s urgency=%s", ticket_id, urgency)

    return {
        "id": ticket_id,
        "status": "aberto",
        "urgency": urgency,
    }


@mcp.tool()
def get_ticket(ticket_id: str) -> dict:
    """Consulta um chamado pelo identificador."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM tickets WHERE id = ?",
            (ticket_id.strip(),),
        ).fetchone()

    if row is None:
        return {"found": False, "id": ticket_id}

    return {"found": True, **dict(row)}


@mcp.resource("support://policy/escalation")
def escalation_policy() -> str:
    return (
        "Chamados de urgência alta devem ser revisados por uma pessoa "
        "antes da abertura e do acionamento da equipe técnica."
    )


if __name__ == "__main__":
    initialize_database()
    mcp.run(transport="stdio")