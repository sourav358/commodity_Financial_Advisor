
from uuid import uuid4

from commodity_finance_assistance.memory.database import get_connection


class LocalMemory:
    """Persist conversation messages in a local SQLite database."""

    def create_conversation(self, title: str = "") -> str:
        conversation_id = str(uuid4())

        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO conversations (conversation_id, title)
                VALUES (?, ?)
                """,
                (conversation_id, title),
            )

        return conversation_id

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role: {role}")

        with get_connection() as connection:
            exists = connection.execute(
                "SELECT 1 FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()

            if exists is None:
                raise ValueError("Conversation does not exist")

            connection.execute(
                """
                INSERT INTO messages (conversation_id, role, content)
                VALUES (?, ?, ?)
                """,
                (conversation_id, role, content),
            )

    def get_messages(
        self,
        conversation_id: str,
        limit: int = 20,
    ) -> list[dict[str, str]]:
        if limit < 1:
            raise ValueError("limit must be positive")

        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT role, content
                FROM (
                    SELECT id, role, content
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (conversation_id, limit),
            ).fetchall()

        return [
            {"role": row["role"], "content": row["content"]}
            for row in rows
        ]
