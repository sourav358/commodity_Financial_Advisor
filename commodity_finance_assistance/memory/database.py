
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from commodity_finance_assistance.config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_database_path() -> Path:
    """Resolve the configured SQLite database path."""
    database_url = load_config()["database"]["url"]
    prefix = "sqlite:///"

    if not database_url.startswith(prefix):
        raise ValueError(
            "Expected a SQLite URL beginning with sqlite:///"
        )

    path = Path(database_url[len(prefix):])

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection and always close it."""
    connection = sqlite3.connect(get_database_path())
    connection.row_factory = sqlite3.Row

    try:
        connection.execute("PRAGMA foreign_keys = ON")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    """Create conversation tables if they do not exist."""
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL
                    CHECK (role IN ('system', 'user', 'assistant', 'tool')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(conversation_id)
                    ON DELETE CASCADE
            )
            """
        )
