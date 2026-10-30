
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
                """
                SELECT 1
                FROM conversations
                WHERE conversation_id = ?
                """,
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

    def list_conversations(
        self,
        limit: int = 20,
    ) -> list[dict]:
        """List conversations, most recently active first."""
        if limit < 1:
            raise ValueError("limit must be positive")

        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    c.conversation_id,
                    c.title,
                    c.created_at,
                    COUNT(m.id) AS message_count,
                    MAX(m.created_at) AS last_message_at
                FROM conversations AS c
                LEFT JOIN messages AS m
                    ON m.conversation_id = c.conversation_id
                GROUP BY
                    c.conversation_id,
                    c.title,
                    c.created_at
                ORDER BY
                    COALESCE(MAX(m.id), 0) DESC,
                    c.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            {
                "conversation_id": row["conversation_id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "message_count": row["message_count"],
                "last_message_at": row["last_message_at"],
            }
            for row in rows
        ]

    def get_conversation(
        self,
        conversation_id: str,
    ) -> dict | None:
        """Get conversation details, or None if it doesn't exist."""
        with get_connection() as connection:
            row = connection.execute(
                """
                SELECT conversation_id, title, created_at
                FROM conversations
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()

        if row is None:
            return None

        return dict(row)
